import logging
import math
from collections.abc import Sequence
from functools import partial
from typing import Any

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
import orbax.checkpoint as ocp
from omegaconf import DictConfig

from evorl.distributed import agent_gradient_update, psum, tree_unpmap
from evorl.distribution import get_categorical_dist, get_tanh_norm_dist
from evorl.envs import Env, EnvState, create_env
from evorl.evaluator import Evaluator
from evorl.metrics import TrainMetric
from evorl.networks import make_policy_network, make_v_network
from evorl.rollout import env_step
from evorl.sample_batch import SampleBatch
from evorl.types import (
    MISSING_REWARD,
    Action,
    LossDict,
    Params,
    PolicyExtraInfo,
    PyTreeData,
    PyTreeDict,
    State,
    pytree_field,
)
from evorl.utils import running_statistics
from evorl.utils.jax_utils import rng_split, tree_stop_gradient
from evorl.utils.toolkits import (
    average_episode_discount_return,
    compute_gae,
    flatten_rollout_trajectory,
)
from evorl.workflows import OnPolicyRLWorkflow

from ..agent import Agent, AgentState
from ..ppo import PPOAgent, rollout, PPOWorkflow

logger = logging.getLogger(__name__)


class ParamPPOWorkflow(PPOWorkflow):
    @classmethod
    def name(cls):
        return "ParamPPO"

    def setup(self, key: chex.PRNGKey) -> State:
        state = super().setup(key)

        return state.replace(
            hp_state=PyTreeDict(
                actor_loss_weight=1.0,
                critic_loss_weight=0.5,
                entropy_loss_weight=-0.01,
            )
        )

    def step(self, state: State) -> tuple[TrainMetric, State]:
        key, rollout_key, learn_key = jax.random.split(state.key, num=3)

        # trajectory: [T, #envs, ...]
        trajectory, env_state = rollout(
            self.env,
            self.agent,
            state.env_state,
            state.agent_state,
            rollout_key,
            rollout_length=self.config.rollout_length,
            discount=self.config.discount,
            env_extra_fields=("last_obs", "episode_return"),
        )

        agent_state = state.agent_state
        if agent_state.obs_preprocessor_state is not None:
            agent_state = agent_state.replace(
                obs_preprocessor_state=running_statistics.update(
                    agent_state.obs_preprocessor_state,
                    trajectory.obs,
                    pmap_axis_name=self.pmap_axis_name,
                )
            )

        train_episode_return = average_episode_discount_return(
            trajectory.extras.env_extras.episode_return,
            trajectory.dones,
            pmap_axis_name=self.pmap_axis_name,
        )

        # ======== compute GAE =======
        last_obs = trajectory.extras.env_extras.last_obs
        _obs = jnp.concatenate([trajectory.obs, last_obs[-1:]], axis=0)
        # concat [values, bootstrap_value]
        vs = self.agent.compute_values(state.agent_state, SampleBatch(obs=_obs))
        v_targets, advantages = compute_gae(
            rewards=trajectory.rewards,  # peb_rewards
            values=vs,
            dones=trajectory.dones,
            gae_lambda=self.config.gae_lambda,
            discount=self.config.discount,
        )
        trajectory.extras.v_targets = jax.lax.stop_gradient(v_targets)
        trajectory.extras.advantages = jax.lax.stop_gradient(advantages)
        # [T,B,...] -> [T*B,...]
        trajectory = tree_stop_gradient(flatten_rollout_trajectory(trajectory))
        # ============================

        def loss_fn(agent_state, sample_batch, key):
            # learn all data from trajectory
            loss_dict = self.agent.loss(agent_state, sample_batch, key)
            loss_weights = dict(
                actor_loss=state.hp_state.actor_loss_weight,
                critic_loss=state.hp_state.critic_loss_weight,
                actor_entropy_loss=state.hp_state.entropy_loss_weight,
            )
            loss = jnp.zeros(())
            for loss_key in loss_weights.keys():
                loss += loss_weights[loss_key] * loss_dict[loss_key]

            return loss, loss_dict

        update_fn = agent_gradient_update(
            loss_fn, self.optimizer, pmap_axis_name=self.pmap_axis_name, has_aux=True
        )

        num_minibatches = (
            self.config.rollout_length
            * self.config.num_envs
            // self.config.minibatch_size
        )

        def _get_shuffled_minibatch(perm_key, x):
            x = jax.random.permutation(perm_key, x)[
                : num_minibatches * self.config.minibatch_size
            ]
            return x.reshape(num_minibatches, -1, *x.shape[1:])

        def minibatch_step(carry, trajectory):
            opt_state, agent_state, key = carry
            key, learn_key = jax.random.split(key)

            (loss, loss_dict), agent_state, opt_state = update_fn(
                opt_state, agent_state, trajectory, learn_key
            )

            return (opt_state, agent_state, key), (loss, loss_dict)

        def epoch_step(carry, _):
            opt_state, agent_state, key = carry
            perm_key, learn_key = jax.random.split(key, num=2)

            (opt_state, agent_state, key), (loss_list, loss_dict_list) = jax.lax.scan(
                minibatch_step,
                (opt_state, agent_state, learn_key),
                jtu.tree_map(partial(_get_shuffled_minibatch, perm_key), trajectory),
                length=num_minibatches,
            )

            return (opt_state, agent_state, key), (loss_list, loss_dict_list)

        # loss_list: [reuse_rollout_epochs, num_minibatches]
        (opt_state, agent_state, _), (loss_list, loss_dict_list) = jax.lax.scan(
            epoch_step,
            (state.opt_state, agent_state, learn_key),
            None,
            length=self.config.reuse_rollout_epochs,
        )

        loss = loss_list.mean()
        loss_dict = jtu.tree_map(jnp.mean, loss_dict_list)

        # ======== update metrics ========

        sampled_timesteps = psum(
            jnp.uint32(self.config.rollout_length * self.config.num_envs),
            axis_name=self.pmap_axis_name,
        )

        workflow_metrics = state.metrics.replace(
            sampled_timesteps=state.metrics.sampled_timesteps + sampled_timesteps,
            iterations=state.metrics.iterations + 1,
        ).all_reduce(pmap_axis_name=self.pmap_axis_name)

        train_metrics = TrainMetric(
            train_episode_return=train_episode_return,
            loss=loss,
            raw_loss_dict=loss_dict,
        ).all_reduce(pmap_axis_name=self.pmap_axis_name)

        return train_metrics, state.replace(
            key=key,
            metrics=workflow_metrics,
            agent_state=agent_state,
            env_state=env_state,
            opt_state=opt_state,
        )
