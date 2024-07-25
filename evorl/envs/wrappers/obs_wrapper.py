import chex
import jax
import jax.numpy as jnp

from evorl.types import Action, Observation

from ..env import Env, EnvState
from ..space import Box, Space
from .wrapper import Wrapper


class ObsFlattenWrapper(Wrapper):
    def __init__(self, env: Env):
        super().__init__(env)

        self.obs_ndim = len(env.obs_space.shape)

    def _flatten_obs(self, state: EnvState) -> EnvState:
        start_idx = state.obs.ndim - self.obs_ndim
        state = state.replace(obs=jax.lax.collapse(state.obs, start_idx))

        if "last_obs" in state.info:
            state.info.last_obs = jax.lax.collapse(state.info.last_obs, start_idx)

        return state

    def reset(self, key: chex.PRNGKey) -> EnvState:
        state = self.env.reset(key)
        return self._flatten_obs(state)

    def step(self, state: EnvState, action: Action) -> EnvState:
        state = self.env.step(state, action)
        return self._flatten_obs(state)

    @property
    def obs_space(self) -> Space:
        ori_obs_space = self.env.obs_space
        return Box(low=jnp.ravel(ori_obs_space.low), high=jnp.ravel(ori_obs_space.high))
