import math
from omegaconf import DictConfig


import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
import orbax.checkpoint as ocp

from evorl.replay_buffers import ReplayBuffer
from evorl.metrics import MetricBase
from evorl.types import State
from evorl.utils.jax_utils import (
    right_shift_with_padding,
)
from evorl.evaluators import Evaluator, EpisodeCollector
from evorl.agent import AgentState
from evorl.envs import create_env, AutoresetMode
from evorl.recorders import get_1d_array_statistics, add_prefix
from evorl.ec.optimizers import OpenES, ExponentialScheduleSpec

from ..offpolicy_utils import skip_replay_buffer_state
from ..td3 import (
    make_mlp_td3_agent,
    TD3NetworkParams,
    DUMMY_TD3_TRAINMETRIC,
)
from .cemrl_base import POPTrainMetric
from .cemrl import (
    build_rl_update_fn,
    replace_actor_params,
    CEMRLWorkflow,
)


class CEMRLOpenESWorkflow(CEMRLWorkflow):
    """
    1 critic + n actors + 1 replay buffer.
    We use shard_map to split and parallel the population.
    """

    @classmethod
    def name(cls):
        return "CEMRL-OpenES"

    @classmethod
    def _build_from_config(cls, config: DictConfig):
        """
        return workflow
        """

        # env for one actor
        env = create_env(
            config.env.env_name,
            config.env.env_type,
            episode_length=config.env.max_episode_steps,
            parallel=config.num_envs,
            autoreset_mode=AutoresetMode.DISABLED,
            record_ori_obs=True,
        )

        agent = make_mlp_td3_agent(
            action_space=env.action_space,
            norm_layer_type=config.agent_network.norm_layer_type,
            num_critics=config.agent_network.num_critics,
            critic_hidden_layer_sizes=config.agent_network.critic_hidden_layer_sizes,
            actor_hidden_layer_sizes=config.agent_network.actor_hidden_layer_sizes,
            discount=config.discount,
            exploration_epsilon=config.exploration_epsilon,
            policy_noise=config.policy_noise,
            clip_policy_noise=config.clip_policy_noise,
            critics_in_actor_loss=config.critics_in_actor_loss,
            normalize_obs=config.normalize_obs,
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

        ec_optimizer = OpenES(
            pop_size=config.pop_size,
            lr_schedule=ExponentialScheduleSpec(**config.ec_lr),
            noise_std_schedule=ExponentialScheduleSpec(**config.ec_noise_std),
            mirror_sampling=config.mirror_sampling,
        )

        if config.fitness_with_exploration:
            action_fn = agent.compute_actions
        else:
            action_fn = agent.evaluate_actions

        collector = EpisodeCollector(
            env=env,
            action_fn=action_fn,
            max_episode_steps=config.env.max_episode_steps,
            env_extra_fields=("ori_obs", "termination"),
        )

        replay_buffer = ReplayBuffer(
            capacity=config.replay_buffer_capacity,
            min_sample_timesteps=config.batch_size,
            sample_batch_size=config.batch_size,
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
            action_fn=agent.evaluate_actions,
            max_episode_steps=config.env.max_episode_steps,
        )

        agent_state_vmap_axes = AgentState(
            params=TD3NetworkParams(
                critic_params=None,
                actor_params=0,
                target_critic_params=None,
                target_actor_params=0,
            ),
            obs_preprocessor_state=None,
        )

        workflow = cls(
            env,
            agent,
            agent_state_vmap_axes,
            optimizer,
            ec_optimizer,
            collector,
            evaluator,
            replay_buffer,
            config,
        )

        workflow._rl_update_fn = build_rl_update_fn(
            agent, optimizer, config, workflow.agent_state_vmap_axes
        )

        return workflow

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

        pop_actor_params = agent_state.params.actor_params

        key, rollout_key, perm_key, learn_key = jax.random.split(state.key, num=4)

        # ======= CEM Sample ========
        pop_actor_params, ec_opt_state = self.ec_optimizer.ask(ec_opt_state)

        # ======== RL update ========
        def _dummy_rl_update(
            agent_state, opt_state, replay_buffer_state, pop_actor_params, learn_key
        ):
            td3_metrics = DUMMY_TD3_TRAINMETRIC.replace(
                raw_loss_dict=jtu.tree_map(
                    lambda x: jnp.broadcast_to(
                        x, (self.config.num_learning_offspring, *x.shape)
                    ),
                    DUMMY_TD3_TRAINMETRIC.raw_loss_dict,
                )
            )
            return td3_metrics, pop_actor_params, agent_state, opt_state

        td3_metrics, pop_actor_params, agent_state, opt_state = jax.lax.cond(
            iterations > self.config.warmup_iters,
            self._rl_update,
            _dummy_rl_update,
            agent_state,
            opt_state,
            replay_buffer_state,
            pop_actor_params,
            learn_key,
        )

        # ======== CEM update ========
        pop_agent_state = replace_actor_params(agent_state, pop_actor_params)

        # the trajectory [T, #pop*B, ...]
        # metrics: [#pop, B]
        eval_metrics, trajectory = self._rollout(pop_agent_state, rollout_key)

        fitnesses = eval_metrics.episode_returns.mean(axis=-1)

        mask = jnp.logical_not(right_shift_with_padding(trajectory.dones, 1))
        trajectory = trajectory.replace(dones=None)
        trajectory, mask = jtu.tree_map(
            lambda x: jax.lax.collapse(x, 0, 2),
            (trajectory, mask),
        )

        replay_buffer_state = self.replay_buffer.add(
            replay_buffer_state, trajectory, mask
        )

        ec_metrics, ec_opt_state = self.ec_optimizer.tell(ec_opt_state, fitnesses)

        train_metrics = POPTrainMetric(
            rb_size=replay_buffer_state.buffer_size,
            pop_episode_lengths=eval_metrics.episode_lengths.mean(-1),
            pop_episode_returns=eval_metrics.episode_returns.mean(-1),
            rl_metrics=td3_metrics,
            ec_info=ec_metrics,
        )

        # calculate the number of timestep
        sampled_timesteps = eval_metrics.episode_lengths.sum().astype(jnp.uint32)
        sampled_episodes = jnp.uint32(self.config.episodes_for_fitness * pop_size)

        # iterations is the number of updates of the agent

        workflow_metrics = state.metrics.replace(
            sampled_timesteps=state.metrics.sampled_timesteps + sampled_timesteps,
            sampled_episodes=state.metrics.sampled_episodes + sampled_episodes,
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

    def learn(self, state: State) -> State:
        num_iters = math.ceil(
            (self.config.total_episodes - state.metrics.sampled_episodes)
            / (self.config.episodes_for_fitness * self.config.pop_size)
        )

        for i in range(state.metrics.iterations, num_iters + state.metrics.iterations):
            iters = i + 1
            train_metrics, state = self.step(state)
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
                train_metrics_dict["rl_metrics"]["actor_loss"] /= (
                    self.config.num_learning_offspring
                )
                train_metrics_dict["rl_metrics"]["raw_loss_dict"] = jtu.tree_map(
                    get_1d_array_statistics,
                    train_metrics_dict["rl_metrics"]["raw_loss_dict"],
                )

            self.recorder.write(train_metrics_dict, iters)

            if iters % self.config.eval_interval == 0:
                eval_metrics, state = self.evaluate(state)

                self.recorder.write(
                    add_prefix(eval_metrics.to_local_dict(), "eval"), iters
                )

            saved_state = state
            if not self.config.save_replay_buffer:
                saved_state = skip_replay_buffer_state(saved_state)

            self.checkpoint_manager.save(iters, args=ocp.args.StandardSave(state))

        return state
