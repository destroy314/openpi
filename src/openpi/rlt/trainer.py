import dataclasses
import logging
from typing import Any

import flax.nnx as nnx
import flax.traverse_util
import jax
import jax.numpy as jnp
import optax
import numpy as np

from openpi.rlt import actor_critic as _actor_critic
from openpi.shared import nnx_utils
from openpi.training import utils as training_utils


@dataclasses.dataclass(frozen=True)
class OnlineRLTConfig:
    discount: float = 0.99
    actor_bc_weight: float = 0.1
    reference_dropout: float = 0.5
    actor_update_interval: int = 2
    target_tau: float = 0.005


@dataclasses.dataclass
class CriticTrainState:
    step: int
    params: Any
    target_params: Any
    opt_state: Any
    tx: optax.GradientTransformation = dataclasses.field(repr=False)


def actor_filter() -> nnx.filterlib.Filter:
    return nnx_utils.PathRegex(".*rlt_actor.*")


def init_policy_state(model, learning_rate: float) -> training_utils.TrainState:
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate))
    params = nnx.state(model)
    return training_utils.TrainState(
        step=jnp.asarray(0),
        params=params,
        model_def=nnx.graphdef(model),
        tx=tx,
        opt_state=tx.init(params.filter(actor_filter())),
        ema_decay=None,
        ema_params=None,
    )


def init_critic_state(
    rng: jax.Array,
    state_dim: int,
    action_dim: int,
    *,
    hidden_dim: int = 256,
    learning_rate: float = 3e-4,
) -> tuple[_actor_critic.TwinCritic, CriticTrainState]:
    critic = _actor_critic.TwinCritic(hidden_dim=hidden_dim)
    params = critic.init(rng, jnp.ones((1, state_dim), dtype=jnp.float32), jnp.ones((1, action_dim), dtype=jnp.float32))[
        "params"
    ]
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate))
    state = CriticTrainState(
        step=0,
        params=params,
        target_params=params,
        opt_state=tx.init(params),
        tx=tx,
    )
    return critic, state


def _apply_reference_dropout(reference_action: jax.Array, rng: jax.Array, prob: float) -> jax.Array:
    if prob <= 0:
        return reference_action
    drop_mask = jax.random.bernoulli(rng, prob, (reference_action.shape[0], 1))
    return jnp.where(drop_mask, 0.0, reference_action)


def actor_mean(model, state: jax.Array, reference_action: jax.Array) -> jax.Array:
    mean, _ = model.rlt_actor(state, reference_action)
    return mean


def actor_sample(
    model,
    state: jax.Array,
    reference_action: jax.Array,
    rng: jax.Array,
    *,
    deterministic: bool = False,
) -> jax.Array:
    mean, std = model.rlt_actor(state, reference_action)
    delta = mean - reference_action
    jax.debug.print(
        "actor_sample | state     : mean={m:.4f} std={s:.4f} max_abs={mx:.4f}",
        m=jnp.mean(state), s=jnp.std(state), mx=jnp.max(jnp.abs(state)),
    )
    jax.debug.print(
        "actor_sample | ref_action: mean={m:.4f} std={s:.4f} max_abs={mx:.4f}",
        m=jnp.mean(reference_action), s=jnp.std(reference_action), mx=jnp.max(jnp.abs(reference_action)),
    )
    jax.debug.print(
        "actor_sample | delta     : mean={m:.4f} std={s:.4f} max_abs={mx:.4f}",
        m=jnp.mean(delta), s=jnp.std(delta), mx=jnp.max(jnp.abs(delta)),
    )
    jax.debug.print(
        "actor_sample | mean_out  : mean={m:.4f} std={s:.4f} max_abs={mx:.4f}",
        m=jnp.mean(mean), s=jnp.std(mean), mx=jnp.max(jnp.abs(mean)),
    )
    if deterministic:
        return mean
    noise = jax.random.normal(rng, mean.shape, dtype=mean.dtype)
    return mean + noise * std


def critic_step(
    critic_model: _actor_critic.TwinCritic,
    critic_state: CriticTrainState,
    policy_state: training_utils.TrainState,
    batch,
    rng: jax.Array,
    config: OnlineRLTConfig,
) -> tuple[CriticTrainState, dict[str, jax.Array]]:
    model = nnx.merge(policy_state.model_def, policy_state.params)
    next_action = actor_sample(model, batch.next_state, batch.next_reference_action, rng)
    target_q1, target_q2 = critic_model.apply({"params": critic_state.target_params}, batch.next_state, next_action)
    bootstrap_discount = jnp.power(config.discount, batch.bootstrap_steps.astype(jnp.float32))
    target_q = batch.reward + bootstrap_discount * (1.0 - batch.done) * jnp.minimum(target_q1, target_q2)

    def loss_fn(params):
        q1, q2 = critic_model.apply({"params": params}, batch.state, batch.action)
        loss = jnp.mean(jnp.square(q1 - target_q) + jnp.square(q2 - target_q))
        return loss, (q1, q2)

    (loss, (q1, q2)), grads = jax.value_and_grad(loss_fn, has_aux=True)(critic_state.params)
    grad_norm = optax.global_norm(grads)
    logging.info(
        "critic_step | target_q: mean=%.4f std=%.4f max_abs=%.4f",
        float(jnp.mean(target_q)), float(jnp.std(target_q)), float(jnp.max(jnp.abs(target_q))),
    )
    logging.info(
        "critic_step | q1      : mean=%.4f std=%.4f max_abs=%.4f",
        float(jnp.mean(q1)), float(jnp.std(q1)), float(jnp.max(jnp.abs(q1))),
    )
    logging.info(
        "critic_step | loss=%.4f grad_norm=%.4f",
        float(loss), float(grad_norm),
    )
    updates, new_opt_state = critic_state.tx.update(grads, critic_state.opt_state, critic_state.params)
    new_params = optax.apply_updates(critic_state.params, updates)
    new_target_params = optax.incremental_update(new_params, critic_state.target_params, config.target_tau)
    new_state = CriticTrainState(
        step=critic_state.step + 1,
        params=new_params,
        target_params=new_target_params,
        opt_state=new_opt_state,
        tx=critic_state.tx,
    )
    info = {
        "critic_loss": loss,
        "critic_q1": jnp.mean(q1),
        "critic_q2": jnp.mean(q2),
    }
    return new_state, info


def actor_step(
    critic_model: _actor_critic.TwinCritic,
    critic_state: CriticTrainState,
    policy_state: training_utils.TrainState,
    batch,
    rng: jax.Array,
    config: OnlineRLTConfig,
) -> tuple[training_utils.TrainState, dict[str, jax.Array]]:
    model = nnx.merge(policy_state.model_def, policy_state.params)

    def loss_fn(model_with_actor, rngs):
        dropout_rng, sample_rng = rngs
        dropped_reference = _apply_reference_dropout(batch.reference_action, dropout_rng, config.reference_dropout)
        actions = actor_sample(model_with_actor, batch.state, dropped_reference, sample_rng)
        q1, q2 = critic_model.apply({"params": critic_state.params}, batch.state, actions)
        q = jnp.minimum(q1, q2)
        bc_penalty = jnp.mean(jnp.square(actions - batch.reference_action), axis=-1)
        return jnp.mean(-q + config.actor_bc_weight * bc_penalty)

    diff_state = nnx.DiffState(0, actor_filter())
    rngs = jax.random.split(rng, 2)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, rngs)
    actor_grads = grads.filter(actor_filter())
    grad_norm = optax.global_norm(actor_grads.to_pure_dict())
    logging.info(
        "actor_step | actor_loss=%.4f grad_norm=%.4f",
        float(loss), float(grad_norm),
    )
    params = policy_state.params.filter(actor_filter())
    updates, new_opt_state = policy_state.tx.update(grads, policy_state.opt_state, params)
    new_actor_params = optax.apply_updates(params, updates)
    nnx.update(model, new_actor_params)
    new_params = nnx.state(model)
    new_state = dataclasses.replace(
        policy_state,
        step=policy_state.step + 1,
        params=new_params,
        opt_state=new_opt_state,
    )
    info = {"actor_loss": loss}
    return new_state, info


def bundle_policy_state(state: training_utils.TrainState) -> dict[str, Any]:
    """Bundle only the actor params and optimizer state.

    The VLA backbone is frozen and reloaded from the Stage-1 checkpoint on every
    run, so we only need to persist the small RL-specific weights.
    """
    return {
        "step": jnp.asarray(state.step),
        "actor_params": state.params.filter(actor_filter()).to_pure_dict(),
        "opt_state": state.opt_state,
    }


def restore_policy_state(
    model,
    bundle: dict[str, Any],
    learning_rate: float,
) -> training_utils.TrainState:
    """Restore a Stage-2 policy state.

    The VLA backbone is taken from `model` (already loaded from the Stage-1
    checkpoint by the caller).  Only the actor weights are overwritten from
    `bundle`.
    """
    state = init_policy_state(model, learning_rate)
    # Rebuild a temporary model so we can do a targeted actor-only update.
    tmp_model = nnx.merge(state.model_def, state.params)
    actor_params = nnx.state(tmp_model).filter(actor_filter())
    actor_params.replace_by_pure_dict(bundle["actor_params"])
    nnx.update(tmp_model, actor_params)
    new_params = nnx.state(tmp_model)
    return dataclasses.replace(state, step=bundle["step"], params=new_params, opt_state=bundle["opt_state"])


def bundle_critic_state(state: CriticTrainState) -> dict[str, Any]:
    return {
        "step": jnp.asarray(state.step),
        "params": state.params,
        "target_params": state.target_params,
        "opt_state": state.opt_state,
    }


def restore_critic_state(
    critic_model: _actor_critic.TwinCritic,
    bundle: dict[str, Any],
    learning_rate: float,
) -> CriticTrainState:
    params = bundle["params"]
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate))
    del critic_model
    return CriticTrainState(
        step=int(bundle["step"]),
        params=params,
        target_params=bundle["target_params"],
        opt_state=bundle["opt_state"],
        tx=tx,
    )


def flatten_params(params: dict[str, Any]) -> dict[str, Any]:
    return flax.traverse_util.flatten_dict(params, sep="/")
