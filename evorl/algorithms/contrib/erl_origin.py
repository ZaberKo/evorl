import copy
import logging
import math
import time
from typing import Any
from typing_extensions import Self  # pytype: disable=not-supported-yet]
from functools import partial
from omegaconf import DictConfig

import chex
import flashbax
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
import orbax.checkpoint as ocp

from evorl.distributed import agent_gradient_update, tree_pmean
from evorl.metrics import MetricBase, metricfield, EvaluateMetric
from evorl.types import PyTreeDict, State, LossDict
from evorl.utils import running_statistics
from evorl.utils.jax_utils import tree_stop_gradient, tree_set
from evorl.utils.rl_toolkits import soft_target_update, flatten_rollout_trajectory
from evorl.utils.flashbax_utils import get_buffer_size
from evorl.evaluator import Evaluator
from evorl.sample_batch import SampleBatch
from evorl.agent import Agent, AgentState, RandomAgent
from evorl.envs import create_env, AutoresetMode, Box, Env
from evorl.workflows import Workflow
from evorl.rollout import rollout
from evorl.recorders import get_1d_array_statistics, add_prefix
from evorl.ec.optimizers import ERLGA, EvoOptimizer, ECState

from ..td3 import TD3Agent, TD3NetworkParams
from ..offpolicy_utils import clean_trajectory, skip_replay_buffer_state
from ..erl.episode_collector import EpisodeCollector
from ..erl.utils import flatten_pop_rollout_episode


logger = logging.getLogger(__name__)


class TD3TrainMetric(MetricBase):
    critic_loss: chex.Array
    actor_loss: chex.Array
    num_updates_per_iter: int = 0
    raw_loss_dict: LossDict = metricfield(
        default_factory=PyTreeDict, reduce_fn=tree_pmean
    )


class POPTrainMetric(MetricBase):
    rb_size: int
    pop_episode_returns: chex.Array
    pop_episode_lengths: chex.Array
    rl_episode_returns: chex.Array | None = None
    rl_episode_lengths: chex.Array | None = None
    rl_metrics: MetricBase | None = None
    per_iter_time_cost: float = 0.0
    ec_info: PyTreeDict = metricfield(default_factory=PyTreeDict)


class WorkflowMetric(MetricBase):
    sampled_timesteps: chex.Array = jnp.zeros((), dtype=jnp.uint32)
    sampled_episodes: chex.Array = jnp.zeros((), dtype=jnp.uint32)
    rl_sampled_timesteps: chex.Array = jnp.zeros((), dtype=jnp.uint32)
    iterations: chex.Array = jnp.zeros((), dtype=jnp.uint32)


class ERLWorkflow(Workflow):
    """
    EC: n actors
    RL: k actors + k critics + 1 replay buffer.
    """

    def __init__(
        self,
        env: Env,
        agent: Agent,
        optimizer: optax.GradientTransformation,
        ec_optimizer: EvoOptimizer,
        ec_collector: EpisodeCollector,
        rl_collector: EpisodeCollector,
        evaluator: Evaluator,  # to evaluate the pop-mean actor
        replay_buffer: Any,
        config: DictConfig,
    ):
        super().__init__(config)
        self.env = env
        self.agent = agent
        self.optimizer = optimizer
        self.ec_optimizer = ec_optimizer
        self.ec_collector = ec_collector
        self.rl_collector = rl_collector
        self.evaluator = evaluator
        self.replay_buffer = replay_buffer

        self.devices = jax.local_devices()[:1]

    @classmethod
    def name(cls):
        return "ERL-Origin"

    @classmethod
    def build_from_config(
        cls,
        config: DictConfig,
        enable_multi_devices: bool = False,
        enable_jit: bool = True,
    ) -> Self:
        config = copy.deepcopy(config)  # avoid in-place modification

        devices = jax.local_devices()

        if enable_multi_devices or len(devices) > 1:
            raise NotImplementedError("Multi-devices is not supported yet.")

        if enable_jit:
            cls.enable_jit()

        workflow = cls._build_from_config(config)

        return workflow

    @classmethod
    def _build_from_config(cls, config: DictConfig):
        """
        return workflow
        """

        # env for rl&ec rollout
        env = create_env(
            config.env.env_name,
            config.env.env_type,
            episode_length=config.env.max_episode_steps,
            parallel=config.num_envs,
            autoreset_mode=AutoresetMode.DISABLED,
            record_ori_obs=True,
        )

        assert isinstance(
            env.action_space, Box
        ), "Only continue action space is supported."

        agent = TD3Agent(
            num_critics=config.agent_network.num_critics,
            norm_layer_type=config.agent_network.norm_layer_type,
            critic_hidden_layer_sizes=config.agent_network.critic_hidden_layer_sizes,
            actor_hidden_layer_sizes=config.agent_network.actor_hidden_layer_sizes,
            discount=config.discount,
            exploration_epsilon=config.exploration_epsilon,
            policy_noise=config.policy_noise,
            clip_policy_noise=config.clip_policy_noise,
            critics_in_actor_loss=config.critics_in_actor_loss,
        )

        if (
            config.optimizer.grad_clip_norm is not None
            and config.optimizer.grad_clip_norm > 0
        ):
            optimizer = optax.chain(
                optax.clip_by_global_norm(config.optimizer.grad_clip_norm),
                optax.adam(config.optimizer.lr),
            )
        else:
            optimizer = optax.adam(config.optimizer.lr)

        ec_optimizer = ERLGA(
            pop_size=config.pop_size,
            num_elites=config.num_elites,
            weight_max_magnitude=config.weight_max_magnitude,
            mut_strength=config.mut_strength,
            num_mutation_frac=config.num_mutation_frac,
            super_mut_strength=config.super_mut_strength,
            super_mut_prob=config.super_mut_prob,
            reset_prob=config.reset_prob,
            vec_relative_prob=config.vec_relative_prob,
            num_crossover_frac=config.num_crossover_frac,
        )

        if config.fitness_with_exploration:
            action_fn = agent.compute_actions
        else:
            action_fn = agent.evaluate_actions

        ec_collector = EpisodeCollector(
            env_step_fn=env.step,
            env_reset_fn=env.reset,
            action_fn=action_fn,
            num_envs=config.num_envs,
            max_episode_steps=config.env.max_episode_steps,
            env_extra_fields=("ori_obs", "termination"),
        )

        if config.rl_exploration:
            action_fn = agent.compute_actions
        else:
            action_fn = agent.evaluate_actions

        rl_collector = EpisodeCollector(
            env_step_fn=env.step,
            env_reset_fn=env.reset,
            action_fn=action_fn,
            num_envs=config.num_envs,
            max_episode_steps=config.env.max_episode_steps,
            env_extra_fields=("ori_obs", "termination"),
        )

        replay_buffer = flashbax.make_item_buffer(
            max_length=config.replay_buffer_capacity,
            min_length=config.batch_size,
            sample_batch_size=config.batch_size,
            add_batches=True,
        )

        # to evaluate the pop-mean actor
        eval_env = create_env(
            config.env.env_name,
            config.env.env_type,
            episode_length=config.env.max_episode_steps,
            parallel=config.num_eval_envs,
            autoreset_mode=AutoresetMode.DISABLED,
        )

        evaluator = Evaluator(
            env=eval_env,
            agent=agent,
            max_episode_steps=config.env.max_episode_steps,
        )

        workflow = cls(
            env,
            agent,
            optimizer,
            ec_optimizer,
            ec_collector,
            rl_collector,
            evaluator,
            replay_buffer,
            config,
        )

        workflow.agent_state_pytree_axes = AgentState(
            params=0,
            obs_preprocessor_state=None,
        )

        workflow._rl_update_fn = jax.jit(
            _build_rl_update_fn(
                agent, optimizer, config, workflow.agent_state_pytree_axes
            )
        )

        return workflow

    def setup(self, key: chex.PRNGKey) -> State:
        """
        obs_preprocessor_state update strategy: only updated at _postsetup_replaybuffer(), then fixed during the training.
        """
        key, agent_key, rb_key = jax.random.split(key, 3)

        # agent_state: [num_rl_agents, ...]
        # ec_opt_state.pop: [pop_size, ...]
        agent_state, opt_state, ec_opt_state = self._setup_agent_and_optimizer(
            agent_key
        )

        workflow_metrics = WorkflowMetric()

        replay_buffer_state = self._setup_replaybuffer(rb_key)

        # =======================

        state = State(
            key=key,
            metrics=workflow_metrics,
            agent_state=agent_state,
            opt_state=opt_state,
            ec_opt_state=ec_opt_state,
            replay_buffer_state=replay_buffer_state,
        )

        if self.config.random_timesteps > 0:
            logger.info("Start replay buffer post-setup")
            state = self._postsetup_replaybuffer(state)
            logger.info("Complete replay buffer post-setup")

        return state

    def _setup_agent_and_optimizer(
        self, key: chex.PRNGKey
    ) -> tuple[AgentState, chex.ArrayTree, ECState]:
        agent_key, pop_agent_key, ec_key = jax.random.split(key, 3)

        # agent for RL
        agent_state = jax.vmap(self.agent.init, in_axes=(None, None, 0))(
            self.env.obs_space,
            self.env.action_space,
            jax.random.split(agent_key, self.config.num_rl_agents),
        )

        # all agents will share the same obs_preprocessor_state
        if agent_state.obs_preprocessor_state is not None:
            agent_state = agent_state.replace(
                obs_preprocessor_state=jtu.tree_map(
                    lambda x: x[0], agent_state.obs_preprocessor_state
                )
            )

        pop_actor_params = jax.vmap(self.agent.actor_network.init, in_axes=(0, None))(
            jax.random.split(pop_agent_key, self.config.pop_size),
            jnp.zeros((1, self.env.obs_space.shape[0])),
        )

        ec_opt_state = self.ec_optimizer.init(pop_actor_params, ec_key)

        opt_state = PyTreeDict(
            actor=jax.vmap(self.optimizer.init)(agent_state.params.actor_params),
            critic=jax.vmap(self.optimizer.init)(agent_state.params.critic_params),
        )

        return agent_state, opt_state, ec_opt_state

    def _setup_replaybuffer(self, key: chex.PRNGKey) -> chex.ArrayTree:
        action_space = self.env.action_space
        obs_space = self.env.obs_space

        # create dummy data to initialize the replay buffer
        dummy_action = jnp.zeros(action_space.shape)
        dummy_obs = jnp.zeros(obs_space.shape)

        dummy_reward = jnp.zeros(())
        dummy_done = jnp.zeros(())

        dummy_sample_batch = SampleBatch(
            obs=dummy_obs,
            actions=dummy_action,
            rewards=dummy_reward,
            # next_obs=dummy_obs,
            # dones=dummy_done,
            extras=PyTreeDict(
                policy_extras=PyTreeDict(),
                env_extras=PyTreeDict(
                    {"ori_obs": dummy_obs, "termination": dummy_done}
                ),
            ),
        )
        replay_buffer_state = self.replay_buffer.init(dummy_sample_batch)

        return replay_buffer_state

    def _postsetup_replaybuffer(self, state: State) -> State:
        action_space = self.env.action_space
        obs_space = self.env.obs_space
        config = self.config

        replay_buffer_state = state.replay_buffer_state
        agent_state = state.agent_state

        # We need a separate autoreset env to fill the replay buffer
        env = create_env(
            config.env.env_name,
            config.env.env_type,
            episode_length=config.env.max_episode_steps,
            parallel=config.num_envs,
            autoreset_mode=AutoresetMode.NORMAL,
            record_ori_obs=True,
        )

        # ==== fill random transitions ====

        key, env_key, rollout_key = jax.random.split(state.key, num=3)
        random_agent = RandomAgent()
        random_agent_state = random_agent.init(
            obs_space, action_space, jax.random.PRNGKey(0)
        )
        rollout_length = config.random_timesteps // config.num_envs

        env_state = env.reset(env_key)
        trajectory, env_state = rollout(
            env_fn=env.step,
            action_fn=random_agent.compute_actions,
            env_state=env_state,
            agent_state=random_agent_state,
            key=rollout_key,
            rollout_length=rollout_length,
            env_extra_fields=("ori_obs", "termination"),
        )

        # [T, B, ...] -> [T*B, ...]
        trajectory = clean_trajectory(trajectory)
        trajectory = flatten_rollout_trajectory(trajectory)
        trajectory = tree_stop_gradient(trajectory)

        replay_buffer_state = self.replay_buffer.add(replay_buffer_state, trajectory)

        if agent_state.obs_preprocessor_state is not None and rollout_length > 0:
            agent_state = agent_state.replace(
                obs_preprocessor_state=running_statistics.update(
                    agent_state.obs_preprocessor_state, trajectory.obs
                )
            )

        sampled_timesteps = jnp.uint32(rollout_length * config.num_envs)
        # Since we sample from autoreset env, this metric might not be accurate:
        sampled_episodes = trajectory.dones.sum()

        workflow_metrics = state.metrics.replace(
            sampled_timesteps=state.metrics.sampled_timesteps + sampled_timesteps,
            sampled_episodes=state.metrics.sampled_episodes + sampled_episodes,
        )

        return state.replace(
            key=key,
            metrics=workflow_metrics,
            agent_state=agent_state,
            replay_buffer_state=replay_buffer_state,
        )

    def _ec_rollout(self, agent_state, key):
        eval_metrics, trajectory = jax.vmap(
            self.ec_collector.evaluate,
            in_axes=(self.agent_state_pytree_axes, None, 0),
        )(
            agent_state,
            self.config.episodes_for_fitness,
            jax.random.split(key, self.config.pop_size),
        )

        trajectory = clean_trajectory(trajectory)
        # [#pop, T, B, ...] -> [T, #pop*B, ...]
        trajectory = flatten_pop_rollout_episode(trajectory)
        trajectory = tree_stop_gradient(trajectory)

        return eval_metrics, trajectory

    def _rl_rollout(self, agent_state, key):
        eval_metrics, trajectory = jax.vmap(
            self.rl_collector.evaluate,
            in_axes=(self.agent_state_pytree_axes, None, 0),
        )(
            agent_state,
            self.config.rollout_episodes,
            jax.random.split(key, self.config.num_rl_agents),
        )

        trajectory = clean_trajectory(trajectory)
        # [num_rl_agents, T, B, ...] -> [T, num_rl_agents*B, ...]
        trajectory = flatten_pop_rollout_episode(trajectory)
        trajectory = tree_stop_gradient(trajectory)

        return eval_metrics, trajectory

    def _rl_update(self, agent_state, opt_state, replay_buffer_state, num_updates, key):
        def _sample_fn(key):
            return self.replay_buffer.sample(replay_buffer_state, key).experience

        def _sample_and_update_fn(carry, unused_t):
            key, agent_state, opt_state, _ = carry

            key, rb_key, learn_key = jax.random.split(key, 3)

            rb_key = jax.random.split(key, self.config.actor_update_interval)
            sample_batches = jax.vmap(_sample_fn)(rb_key)

            (agent_state, opt_state), train_info = self._rl_update_fn(
                agent_state, opt_state, sample_batches, learn_key
            )

            return (key, agent_state, opt_state, train_info), None

        # unlike erl-ga, since num_updates is large, we only use the last train_info
        init_train_info = (
            jnp.zeros(()),
            jnp.zeros(()),
            PyTreeDict(
                critic_loss=jnp.zeros((self.config.num_rl_agents,)),
                q_value=jnp.zeros((self.config.num_rl_agents,)),
            ),
            PyTreeDict(actor_loss=jnp.zeros((self.config.num_rl_agents,))),
        )

        (_, agent_state, opt_state, train_info), _ = jax.lax.scan(
            _sample_and_update_fn,
            (key, agent_state, opt_state, init_train_info),
            (),
            length=num_updates,
        )

        critic_loss, actor_loss, critic_loss_dict, actor_loss_dict = train_info

        # smoothed td3 metrics
        td3_metrics = TD3TrainMetric(
            actor_loss=actor_loss,
            critic_loss=critic_loss,
            num_updates_per_iter=num_updates,
            raw_loss_dict=PyTreeDict({**critic_loss_dict, **actor_loss_dict}),
        )

        return td3_metrics, agent_state, opt_state

    def _ec_update(self, ec_opt_state, pop_actor_params, fitnesses):
        ec_opt_state = self.ec_optimizer.tell(ec_opt_state, pop_actor_params, fitnesses)
        return ec_opt_state

    def _add_to_replay_buffer(self, replay_buffer_state, trajectory, episode_lengths):
        # trajectory [T,B,...]
        # episode_lengths [B]

        def concat_valid(x):
            # x: [T, B, ...]
            return jnp.concatenate(
                [x[:t, i] for i, t in enumerate(episode_lengths)], axis=0
            )

        valid_trajectory = jtu.tree_map(concat_valid, trajectory)

        replay_buffer_state = self.replay_buffer.add(
            replay_buffer_state, valid_trajectory
        )

        return replay_buffer_state

    def step(self, state: State) -> tuple[MetricBase, State]:
        """
        the basic step function for the workflow to update agent
        """
        pop_size = self.config.pop_size
        agent_state = state.agent_state
        opt_state = state.opt_state
        ec_opt_state = state.ec_opt_state
        replay_buffer_state = state.replay_buffer_state
        iterations = state.metrics.iterations + 1

        pop_actor_params = ec_opt_state.pop

        sampled_timesteps = jnp.zeros((), dtype=jnp.uint32)
        sampled_episodes = jnp.zeros((), dtype=jnp.uint32)

        key, rb_sample_key, ec_rollout_key, rl_rollout_key, ec_key, learn_key = (
            jax.random.split(state.key, num=6)
        )

        # ======== EC update ========
        # the trajectory [#pop, T, B, ...]
        # metrics: [#pop, B]
        pop_agent_state = agent_state.replace(
            params=TD3NetworkParams(
                actor_params=pop_actor_params,
                target_actor_params=pop_actor_params,
                critic_params=None,
                target_critic_params=None,
            )
        )
        ec_eval_metrics, ec_trajectory = self._ec_rollout(
            pop_agent_state, ec_rollout_key
        )

        fitnesses = ec_eval_metrics.episode_returns.mean(axis=-1)

        replay_buffer_state = self._add_to_replay_buffer(
            replay_buffer_state,
            ec_trajectory,
            ec_eval_metrics.episode_lengths.flatten(),
        )

        ec_opt_state = self._ec_update(ec_opt_state, pop_actor_params, fitnesses)

        # calculate the number of timestep
        sampled_timesteps += ec_eval_metrics.episode_lengths.sum().astype(jnp.uint32)
        sampled_episodes += jnp.uint32(self.config.episodes_for_fitness * pop_size)

        train_metrics = POPTrainMetric(
            rb_size=0,  # dummy
            pop_episode_lengths=ec_eval_metrics.episode_lengths.mean(-1),
            pop_episode_returns=ec_eval_metrics.episode_returns.mean(-1),
        )

        # ======== RL update ========
        if iterations > self.config.warmup_iters:
            rl_eval_metrics, rl_trajectory = self._rl_rollout(
                agent_state, rl_rollout_key
            )

            replay_buffer_state = self._add_to_replay_buffer(
                replay_buffer_state,
                rl_trajectory,
                rl_eval_metrics.episode_lengths.flatten(),
            )

            rl_sampled_timesteps = rl_eval_metrics.episode_lengths.sum().astype(
                jnp.uint32
            )
            sampled_timesteps += rl_sampled_timesteps
            sampled_episodes += jnp.uint32(
                self.config.num_rl_agents * self.config.rollout_episodes
            )

            # new:
            total_timesteps = state.metrics.sampled_timesteps + sampled_timesteps
            num_updates = (
                math.ceil(total_timesteps * self.config.rl_updates_frac_per_iter)
                // self.config.actor_update_interval
            )

            td3_metrics, agent_state, opt_state = self._rl_update(
                agent_state, opt_state, replay_buffer_state, num_updates, learn_key
            )

            if iterations % self.config.rl_injection_interval == 0:
                ec_opt_state = self._rl_injection(agent_state, ec_opt_state, fitnesses)

            train_metrics = train_metrics.replace(
                rl_episode_lengths=rl_eval_metrics.episode_lengths.mean(-1),
                rl_episode_returns=rl_eval_metrics.episode_returns.mean(-1),
                rl_metrics=td3_metrics,
            )

        else:
            rl_sampled_timesteps = jnp.zeros((), dtype=jnp.uint32)

        train_metrics = train_metrics.replace(
            rb_size=get_buffer_size(replay_buffer_state),
        )

        # iterations is the number of updates of the agent
        workflow_metrics = state.metrics.replace(
            sampled_timesteps=state.metrics.sampled_timesteps + sampled_timesteps,
            sampled_episodes=state.metrics.sampled_episodes + sampled_episodes,
            rl_sampled_timesteps=state.metrics.rl_sampled_timesteps
            + rl_sampled_timesteps,
            iterations=iterations,
        )

        state = state.replace(
            key=key,
            metrics=workflow_metrics,
            agent_state=agent_state,
            replay_buffer_state=replay_buffer_state,
            ec_opt_state=ec_opt_state,
            opt_state=opt_state,
        )

        return train_metrics, state

    def _rl_injection(self, agent_state, ec_opt_state, fitnesses):
        pop_actor_params = ec_opt_state.pop
        # replace EC worst individuals
        worst_indices = fitnesses.argsort()[: self.config.num_rl_agents]
        rl_actor_params = agent_state.params.actor_params

        chex.assert_tree_shape_prefix(rl_actor_params, (self.config.num_rl_agents,))

        ec_opt_state = ec_opt_state.replace(
            pop=tree_set(
                pop_actor_params,
                rl_actor_params,
                worst_indices,
                unique_indices=True,
            )
        )

        return ec_opt_state

    def evaluate(self, state: State) -> tuple[MetricBase, State]:
        key, eval_key = jax.random.split(state.key, num=2)

        _vmap_evaluate = jax.vmap(
            partial(self.evaluator.evaluate, num_episodes=self.config.eval_episodes),
            in_axes=(self.agent_state_pytree_axes, 0),
        )

        # [num_rl_agents, #episodes]
        raw_eval_metrics = _vmap_evaluate(
            state.agent_state, jax.random.split(eval_key, self.config.num_rl_agents)
        )

        eval_metrics = EvaluateMetric(
            episode_returns=raw_eval_metrics.episode_returns.mean(-1),
            episode_lengths=raw_eval_metrics.episode_lengths.mean(-1),
        )

        state = state.replace(key=key)
        return eval_metrics, state

    def learn(self, state: State) -> State:
        num_iters = math.ceil(
            self.config.total_episodes
            / (self.config.episodes_for_fitness * self.config.pop_size)
        )

        for i in range(state.metrics.iterations, num_iters):
            iters = i + 1
            start_t = time.perf_counter()
            train_metrics, state = self.step(state)
            per_iter_time_cost = time.perf_counter() - start_t
            train_metrics = train_metrics.replace(per_iter_time_cost=per_iter_time_cost)

            workflow_metrics = state.metrics

            workflow_metrics_dict = workflow_metrics.to_local_dict()
            self.recorder.write(workflow_metrics_dict, iters)

            train_metrics_dict = train_metrics.to_local_dict()
            train_metrics_dict["pop_episode_returns"] = get_1d_array_statistics(
                train_metrics_dict["pop_episode_returns"], histogram=True
            )

            train_metrics_dict["pop_episode_lengths"] = get_1d_array_statistics(
                train_metrics_dict["pop_episode_lengths"], histogram=True
            )

            if train_metrics_dict["rl_metrics"] is not None:
                if self.config.num_rl_agents > 1:
                    train_metrics_dict["rl_episode_lengths"] = get_1d_array_statistics(
                        train_metrics_dict["rl_episode_lengths"], histogram=True
                    )
                    train_metrics_dict["rl_episode_returns"] = get_1d_array_statistics(
                        train_metrics_dict["rl_episode_returns"], histogram=True
                    )
                    train_metrics_dict["rl_metrics"]["actor_loss"] /= (
                        self.config.num_rl_agents
                    )
                    train_metrics_dict["rl_metrics"]["critic_loss"] /= (
                        self.config.num_rl_agents
                    )
                    train_metrics_dict["rl_metrics"]["raw_loss_dict"] = jtu.tree_map(
                        get_1d_array_statistics,
                        train_metrics_dict["rl_metrics"]["raw_loss_dict"],
                    )
                else:
                    train_metrics_dict["rl_episode_lengths"] = train_metrics_dict[
                        "rl_episode_lengths"
                    ].squeeze(-1)
                    train_metrics_dict["rl_episode_returns"] = train_metrics_dict[
                        "rl_episode_returns"
                    ].squeeze(-1)

            self.recorder.write(train_metrics_dict, iters)

            if iters % self.config.eval_interval == 0:
                eval_metrics, state = self.evaluate(state)

                eval_metrics_dict = eval_metrics.to_local_dict()
                if self.config.num_rl_agents > 1:
                    eval_metrics_dict = jtu.tree_map(
                        partial(get_1d_array_statistics, histogram=True),
                        eval_metrics_dict,
                    )

                self.recorder.write(add_prefix(eval_metrics_dict, "eval"), iters)

            saved_state = state
            if not self.config.save_replay_buffer:
                saved_state = skip_replay_buffer_state(saved_state)

            self.checkpoint_manager.save(iters, args=ocp.args.StandardSave(state))

        return state

    @classmethod
    def enable_jit(cls) -> None:
        """
        Do not jit replay buffer add
        """
        cls._rl_rollout = jax.jit(cls._rl_rollout, static_argnums=(0,))
        # cls._rl_update = jax.jit(cls._rl_update, static_argnums=(0,))
        cls._ec_rollout = jax.jit(cls._ec_rollout, static_argnums=(0,))
        cls._ec_update = jax.jit(cls._ec_update, static_argnums=(0,))

        cls.evaluate = jax.jit(cls.evaluate, static_argnums=(0,))
        cls._postsetup_replaybuffer = jax.jit(
            cls._postsetup_replaybuffer, static_argnums=(0,)
        )


def _build_rl_update_fn(
    agent: Agent,
    optimizer: optax.GradientTransformation,
    config: DictConfig,
    agent_state_pytree_axes: AgentState,
):
    num_rl_agents = config.num_rl_agents

    def critic_loss_fn(agent_state, sample_batch, key):
        # loss on a single critic with multiple actors
        # sample_batch: (B, ...)

        loss_dict = jax.vmap(
            agent.critic_loss, in_axes=(agent_state_pytree_axes, None, 0)
        )(agent_state, sample_batch, jax.random.split(key, num_rl_agents))

        loss = config.loss_weights.critic_loss * loss_dict.critic_loss.sum()

        return loss, loss_dict

    def actor_loss_fn(agent_state, sample_batch, key):
        # loss on a single actor
        # different actor shares same sample_batch (B, ...) input
        loss_dict = jax.vmap(
            agent.actor_loss, in_axes=(agent_state_pytree_axes, None, 0)
        )(agent_state, sample_batch, jax.random.split(key, num_rl_agents))

        loss = config.loss_weights.actor_loss * loss_dict.actor_loss.sum()

        return loss, loss_dict

    critic_update_fn = agent_gradient_update(
        critic_loss_fn,
        optimizer,
        has_aux=True,
        attach_fn=lambda agent_state, critic_params: agent_state.replace(
            params=agent_state.params.replace(critic_params=critic_params)
        ),
        detach_fn=lambda agent_state: agent_state.params.critic_params,
    )

    actor_update_fn = agent_gradient_update(
        actor_loss_fn,
        optimizer,
        has_aux=True,
        attach_fn=lambda agent_state, actor_params: agent_state.replace(
            params=agent_state.params.replace(actor_params=actor_params)
        ),
        detach_fn=lambda agent_state: agent_state.params.actor_params,
    )

    def _update_fn(agent_state, opt_state, sample_batches, key):
        critic_opt_state = opt_state.critic
        actor_opt_state = opt_state.actor

        key, critic_key, actor_key = jax.random.split(key, num=3)

        critic_sample_batches = jax.tree_map(lambda x: x[:-1], sample_batches)
        last_sample_batch = jax.tree_map(lambda x: x[-1], sample_batches)

        if config.actor_update_interval - 1 > 0:

            def _update_critic_fn(carry, sample_batch):
                key, agent_state, critic_opt_state = carry

                key, critic_key = jax.random.split(key)

                (critic_loss, critic_loss_dict), agent_state, critic_opt_state = (
                    critic_update_fn(
                        critic_opt_state, agent_state, sample_batch, critic_key
                    )
                )

                return (key, agent_state, critic_opt_state), None

            key, critic_multiple_update_key = jax.random.split(key)

            (_, agent_state, critic_opt_state), _ = jax.lax.scan(
                _update_critic_fn,
                (
                    critic_multiple_update_key,
                    agent_state,
                    critic_opt_state,
                ),
                critic_sample_batches,
                length=config.actor_update_interval - 1,
            )

        (critic_loss, critic_loss_dict), agent_state, critic_opt_state = (
            critic_update_fn(
                critic_opt_state, agent_state, last_sample_batch, critic_key
            )
        )

        (actor_loss, actor_loss_dict), agent_state, actor_opt_state = actor_update_fn(
            actor_opt_state, agent_state, last_sample_batch, actor_key
        )

        # not need vmap
        target_actor_params = soft_target_update(
            agent_state.params.target_actor_params,
            agent_state.params.actor_params,
            config.tau,
        )
        target_critic_params = soft_target_update(
            agent_state.params.target_critic_params,
            agent_state.params.critic_params,
            config.tau,
        )
        agent_state = agent_state.replace(
            params=agent_state.params.replace(
                target_actor_params=target_actor_params,
                target_critic_params=target_critic_params,
            )
        )

        opt_state = opt_state.replace(actor=actor_opt_state, critic=critic_opt_state)

        return (
            (agent_state, opt_state),
            (critic_loss, actor_loss, critic_loss_dict, actor_loss_dict),
        )

    return _update_fn
