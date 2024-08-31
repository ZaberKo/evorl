import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from evorl.sample_batch import SampleBatch
from evorl.types import MISSING_REWARD

from .jax_utils import is_jitted


def compute_episode_length(
    dones: chex.Array,  # [T, B]
) -> chex.Array:
    """
    dones: should be collected from episodic trajectory
    """
    # [B]
    prev_dones = dones[:-1].astype(jnp.int32)
    return (1 - prev_dones).sum(axis=0) + 1


def compute_discount_return(
    rewards: chex.Array,  # [T, B]
    dones: chex.Array,  # [T, B]
    discount: float = 1.0,
) -> chex.Array:
    """
    For episodic trajectory
    """

    def _compute_discount_return(discount_return, x_t):
        # G_t := r_t + γ * G_{t+1}
        reward_t, discount_t = x_t
        discount_return = reward_t + discount_return * discount_t

        return discount_return, None

    # [#envs]
    discount_return = jnp.zeros_like(rewards[0])

    discount_return, _ = jax.lax.scan(
        _compute_discount_return,
        discount_return,
        (rewards, (1 - dones) * discount),
        reverse=True,
        unroll=16,
    )

    return discount_return  # [B]


def compute_gae(
    rewards: jax.Array,  # [T, B]
    values: jax.Array,  # [T+1, B]
    dones: jax.Array,  # [T, B]
    gae_lambda: float = 1.0,
    discount: float = 0.99,
) -> tuple[jax.Array, jax.Array]:
    """
    Calculates the Generalized Advantage Estimation (GAE).

    Args:
        rewards: A float32 tensor of shape [T, B] containing rewards generated by
          following the behaviour policy.
        values: A float32 tensor of shape [T+1, B] with the value function estimates
          wrt. the target policy. values[T] is the bootstrap_value
        dones: A float32 tensor of shape [T, B] with truncation signal.
        gae_lambda: Mix between 1-step (gae_lambda=0) and n-step (gae_lambda=1).
        discount: TD discount.

    Returns:
        A float32 tensor of shape [T, B]. Can be used as target to
          train a baseline (V(x_t) - vs_t)^2.
        A float32 tensor of shape [T, B] of advantages.
    """
    rewards_shape = rewards.shape
    chex.assert_shape(values, (rewards_shape[0] + 1, *rewards_shape[1:]))

    deltas = rewards + discount * (1 - dones) * values[1:] - values[:-1]

    bootstrap_gae = jnp.zeros_like(values[0])

    def _compute_gae(gae_t_plus_1, x_t):
        delta_t, factor_t = x_t
        gae_t = delta_t + factor_t * gae_t_plus_1

        return gae_t, gae_t

    _, advantages = jax.lax.scan(
        _compute_gae,
        bootstrap_gae,
        (deltas, discount * gae_lambda * (1 - dones)),
        reverse=True,
        unroll=16,
    )

    lambda_retruns = advantages + values[:-1]

    return lambda_retruns, advantages


def shuffle_sample_batch(sample_batch: SampleBatch, key: chex.PRNGKey):
    return jtu.tree_map(lambda x: jax.random.permutation(key, x), sample_batch)


def soft_target_update(target_params, source_params, tau: float):
    """
    Perform soft update of target network

    Args:
        target_params: target network parameters
        source_params: source network parameters
        tau: interpolation factor

    Returns:
        updated target network parameters
    """

    return jtu.tree_map(
        lambda target, source: tau * source + (1 - tau) * target,
        target_params,
        source_params,
    )


def flatten_rollout_trajectory(trajectory: SampleBatch):
    """
    Flatten the trajectory from [T, B, ...] to [T*B, ...]
    """
    return jtu.tree_map(lambda x: jax.lax.collapse(x, 0, 2), trajectory)


def average_episode_discount_return(
    episode_discount_return: jax.Array,  # [T,B]
    dones: jax.Array,  # [T,B]
    pmap_axis_name: str | None = None,
) -> jax.Array:
    cnt = dones.sum()
    episode_discount_return_sum = (episode_discount_return * dones).sum()

    if pmap_axis_name is not None:
        episode_discount_return_sum = jax.lax.psum(
            episode_discount_return_sum, pmap_axis_name
        )
        cnt = jax.lax.psum(cnt, pmap_axis_name)

    return jnp.where(
        jnp.isclose(cnt, 0),
        jnp.full_like(episode_discount_return_sum, MISSING_REWARD),
        episode_discount_return_sum / cnt,
    )


def approximate_kl(logratio: jax.Array, mode="k3", axis=-1) -> jax.Array:
    """
    Approximate KL divergence by K3 estimator (no bias, low variance)
    http://joschu.net/blog/kl-approx.html

    Args:
        logratio: ratio of p(x)/q(x), where x are sampled from q(x)
    Returns:
        approx_kl: approximated KL(q||p) (Forward KL)
    """
    ratio = jnp.exp(logratio)

    if mode == "k1":
        approx_kl = -jnp.mean(logratio, axis=axis)
    elif mode == "k2":
        approx_kl = jnp.mean(0.5 * jnp.square(logratio), axis=axis)
    elif mode == "k3":
        approx_kl = jnp.mean((ratio - 1) * logratio, axis=axis)
    return approx_kl


def fold_multi_steps(step_fn, num_steps):
    def _multi_steps(state):
        def _step(state, unused_t):
            train_metrics, state = step_fn(state)
            return state, train_metrics

        state, train_metrics_arr = jax.lax.scan(_step, state, (), length=num_steps)

        return train_metrics_arr, state

    if is_jitted(step_fn):
        _multi_steps = jax.jit(_multi_steps)

    return _multi_steps
