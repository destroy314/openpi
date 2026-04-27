import dataclasses
import logging
import pathlib
from typing import Any

import flax.nnx as nnx
import flax.traverse_util
import jax
import jax.numpy as jnp
import optax

from openpi.rlt import actor_critic as _actor_critic
from openpi.shared import nnx_utils


_ACTOR_SAMPLE_DEBUG = False
_ACTOR_SAMPLE_DEBUG_FILE: pathlib.Path | None = None
_ACTOR_SAMPLE_LOGGER = logging.getLogger("openpi.rlt.actor_sample")


@dataclasses.dataclass(frozen=True)
class OnlineRLTConfig:
    discount: float = 0.99
    actor_bc_weight: float = 1.0
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


@dataclasses.dataclass
class ActorTrainState:
    step: int
    params: Any
    opt_state: Any
    tx: optax.GradientTransformation = dataclasses.field(repr=False)


def set_actor_sample_debug(enabled: bool, *, log_path: str | pathlib.Path | None = None) -> None:
    global _ACTOR_SAMPLE_DEBUG
    global _ACTOR_SAMPLE_DEBUG_FILE
    _ACTOR_SAMPLE_DEBUG = enabled
    _ACTOR_SAMPLE_DEBUG_FILE = pathlib.Path(log_path) if log_path is not None else None
    _ACTOR_SAMPLE_LOGGER.setLevel(logging.INFO)
    _ACTOR_SAMPLE_LOGGER.propagate = False
    _ACTOR_SAMPLE_LOGGER.handlers.clear()
    if not enabled:
        return
    if _ACTOR_SAMPLE_DEBUG_FILE is not None:
        _ACTOR_SAMPLE_DEBUG_FILE.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(_ACTOR_SAMPLE_DEBUG_FILE, encoding="utf-8")
    else:
        handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    _ACTOR_SAMPLE_LOGGER.addHandler(handler)


def _emit_actor_sample_stats(
    debug_label: str,
    debug_step: int,
    state_mean: float,
    state_std: float,
    state_max_abs: float,
    reference_mean: float,
    reference_std: float,
    reference_max_abs: float,
    delta_mean: float,
    delta_std: float,
    delta_max_abs: float,
    mean_out_mean: float,
    mean_out_std: float,
    mean_out_max_abs: float,
) -> None:
    prefix = []
    if debug_label:
        prefix.append(debug_label)
    if debug_step >= 0:
        prefix.append(f"learner_step={debug_step}")
    prefix_text = f"[{' '.join(prefix)}] " if prefix else ""
    lines = [
        f"{prefix_text}actor_sample | state     : mean={state_mean:.4f} std={state_std:.4f} max_abs={state_max_abs:.4f}",
        f"{prefix_text}actor_sample | ref_action: mean={reference_mean:.4f} std={reference_std:.4f} max_abs={reference_max_abs:.4f}",
        f"{prefix_text}actor_sample | delta     : mean={delta_mean:.4f} std={delta_std:.4f} max_abs={delta_max_abs:.4f}",
        f"{prefix_text}actor_sample | mean_out  : mean={mean_out_mean:.4f} std={mean_out_std:.4f} max_abs={mean_out_max_abs:.4f}",
    ]
    for line in lines:
        _ACTOR_SAMPLE_LOGGER.info(line)


def log_debug(message: str, *args: Any) -> None:
    _ACTOR_SAMPLE_LOGGER.info(message, *args)


def _maybe_log_actor_sample_stats(
    state: jax.Array,
    reference_action: jax.Array,
    mean: jax.Array,
    *,
    debug_step: jax.Array | int | None = None,
    debug_label: str = "",
) -> None:
    if not _ACTOR_SAMPLE_DEBUG:
        return
    delta = mean - reference_action
    callback = lambda *args: _emit_actor_sample_stats(debug_label, *args)
    state_mean = jnp.mean(state)
    state_std = jnp.std(state)
    state_max_abs = jnp.max(jnp.abs(state))
    reference_mean = jnp.mean(reference_action)
    reference_std = jnp.std(reference_action)
    reference_max_abs = jnp.max(jnp.abs(reference_action))
    delta_mean = jnp.mean(delta)
    delta_std = jnp.std(delta)
    delta_max_abs = jnp.max(jnp.abs(delta))
    mean_out_mean = jnp.mean(mean)
    mean_out_std = jnp.std(mean)
    mean_out_max_abs = jnp.max(jnp.abs(mean))
    debug_step_value = jnp.asarray(-1 if debug_step is None else debug_step)
    jax.debug.callback(
        callback,
        debug_step_value,
        state_mean,
        state_std,
        state_max_abs,
        reference_mean,
        reference_std,
        reference_max_abs,
        delta_mean,
        delta_std,
        delta_max_abs,
        mean_out_mean,
        mean_out_std,
        mean_out_max_abs,
    )


def actor_filter() -> nnx.filterlib.Filter:
    return nnx_utils.PathRegex(".*rlt_actor.*")


def extract_actor_params(model) -> dict[str, Any]:
    return nnx.state(model).filter(actor_filter()).to_pure_dict()["rlt_actor"]


def init_actor_state(
    rng: jax.Array,
    state_dim: int,
    action_dim: int,
    *,
    hidden_dim: int = 256,
    learning_rate: float = 3e-4,
) -> tuple[_actor_critic.GaussianActor, ActorTrainState]:
    actor = _actor_critic.GaussianActor(action_dim=action_dim, hidden_dim=hidden_dim)
    params = actor.init(
        rng,
        jnp.ones((1, state_dim), dtype=jnp.float32),
        jnp.ones((1, action_dim), dtype=jnp.float32),
    )["params"]
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate))
    state = ActorTrainState(
        step=0,
        params=params,
        opt_state=tx.init(params),
        tx=tx,
    )
    return actor, state


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


def actor_sample_params(
    actor_model: _actor_critic.GaussianActor,
    actor_params: Any,
    state: jax.Array,
    reference_action: jax.Array,
    rng: jax.Array,
    *,
    deterministic: bool = False,
    debug_step: jax.Array | int | None = None,
    debug_label: str = "",
) -> jax.Array:
    mean, std = actor_model.apply({"params": actor_params}, state, reference_action)
    _maybe_log_actor_sample_stats(
        state,
        reference_action,
        mean,
        debug_step=debug_step,
        debug_label=debug_label,
    )
    if deterministic:
        return mean
    noise = jax.random.normal(rng, mean.shape, dtype=mean.dtype)
    return mean + noise * std


def critic_step_with_actor(
    actor_model: _actor_critic.GaussianActor,
    actor_state: ActorTrainState,
    critic_model: _actor_critic.TwinCritic,
    critic_state: CriticTrainState,
    batch,
    rng: jax.Array,
    config: OnlineRLTConfig,
) -> tuple[CriticTrainState, dict[str, jax.Array]]:
    next_action = actor_sample_params(
        actor_model,
        actor_state.params,
        batch.next_state,
        batch.next_reference_action,
        rng,
        debug_step=actor_state.step,
        debug_label="critic_step",
    )
    target_q1, target_q2 = critic_model.apply({"params": critic_state.target_params}, batch.next_state, next_action)
    bootstrap_discount = jnp.power(config.discount, batch.bootstrap_steps.astype(jnp.float32))
    target_q = batch.reward + bootstrap_discount * (1.0 - batch.done) * jnp.minimum(target_q1, target_q2)

    def loss_fn(params):
        q1, q2 = critic_model.apply({"params": params}, batch.state, batch.action)
        loss = jnp.mean(jnp.square(q1 - target_q) + jnp.square(q2 - target_q))
        return loss, (q1, q2)

    (loss, (q1, q2)), grads = jax.value_and_grad(loss_fn, has_aux=True)(critic_state.params)
    grad_norm = optax.global_norm(grads)
    log_debug(
        "critic_step | target_q: mean=%.4f std=%.4f max_abs=%.4f",
        float(jnp.mean(target_q)), float(jnp.std(target_q)), float(jnp.max(jnp.abs(target_q))),
    )
    log_debug(
        "critic_step | q1      : mean=%.4f std=%.4f max_abs=%.4f",
        float(jnp.mean(q1)), float(jnp.std(q1)), float(jnp.max(jnp.abs(q1))),
    )
    log_debug(
        "critic_step | q2      : mean=%.4f std=%.4f max_abs=%.4f",
        float(jnp.mean(q2)), float(jnp.std(q2)), float(jnp.max(jnp.abs(q2))),
    )
    log_debug(
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


def actor_step_with_actor(
    actor_model: _actor_critic.GaussianActor,
    critic_model: _actor_critic.TwinCritic,
    critic_state: CriticTrainState,
    actor_state: ActorTrainState,
    batch,
    rng: jax.Array,
    config: OnlineRLTConfig,
) -> tuple[ActorTrainState, dict[str, jax.Array]]:
    def loss_fn(params, rngs):
        dropout_rng, sample_rng = rngs
        dropped_reference = _apply_reference_dropout(batch.reference_action, dropout_rng, config.reference_dropout)
        actions = actor_sample_params(
            actor_model,
            params,
            batch.state,
            dropped_reference,
            sample_rng,
            debug_step=actor_state.step,
            debug_label="learner_actor",
        )
        q1, q2 = critic_model.apply({"params": critic_state.params}, batch.state, actions)
        q = jnp.minimum(q1, q2)
        bc_penalty = jnp.mean(jnp.square(actions - batch.reference_action), axis=-1)
        return jnp.mean(-q + config.actor_bc_weight * bc_penalty)

    rngs = jax.random.split(rng, 2)
    loss, grads = jax.value_and_grad(loss_fn)(actor_state.params, rngs)
    grad_norm = optax.global_norm(grads)
    log_debug(
        "actor_step | actor_loss=%.4f grad_norm=%.4f",
        float(loss), float(grad_norm),
    )
    updates, new_opt_state = actor_state.tx.update(grads, actor_state.opt_state, actor_state.params)
    new_params = optax.apply_updates(actor_state.params, updates)
    new_state = ActorTrainState(
        step=actor_state.step + 1,
        params=new_params,
        opt_state=new_opt_state,
        tx=actor_state.tx,
    )
    info = {"actor_loss": loss}
    return new_state, info


def bundle_actor_params(state: ActorTrainState) -> dict[str, Any]:
    return {
        "step": jnp.asarray(state.step),
        "actor_params": state.params,
    }


def apply_actor_params(state: ActorTrainState, bundle: dict[str, Any]) -> ActorTrainState:
    return dataclasses.replace(state, step=int(bundle["step"]), params=bundle["actor_params"])


def bundle_actor_train_state(state: ActorTrainState) -> dict[str, Any]:
    return {
        "step": jnp.asarray(state.step),
        "actor_params": state.params,
        "opt_state": state.opt_state,
    }


def restore_actor_state(
    actor_model: _actor_critic.GaussianActor,
    bundle: dict[str, Any],
    learning_rate: float,
) -> ActorTrainState:
    del actor_model
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate))
    return ActorTrainState(
        step=int(bundle["step"]),
        params=bundle["actor_params"],
        opt_state=bundle["opt_state"],
        tx=tx,
    )


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
