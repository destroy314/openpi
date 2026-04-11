import dataclasses
from typing import Any

import flax.nnx as nnx
import flax.traverse_util
import jax
import jax.numpy as jnp
import optax

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


def critic_step(
    critic_model: _actor_critic.TwinCritic,
    critic_state: CriticTrainState,
    policy_state: training_utils.TrainState,
    batch,
    config: OnlineRLTConfig,
) -> tuple[CriticTrainState, dict[str, jax.Array]]:
    model = nnx.merge(policy_state.model_def, policy_state.params)
    next_action = actor_mean(model, batch.next_state, batch.next_reference_action)
    target_q1, target_q2 = critic_model.apply({"params": critic_state.target_params}, batch.next_state, next_action)
    bootstrap_discount = jnp.power(config.discount, batch.bootstrap_steps.astype(jnp.float32))
    target_q = batch.reward + bootstrap_discount * (1.0 - batch.done) * jnp.minimum(target_q1, target_q2)

    def loss_fn(params):
        q1, q2 = critic_model.apply({"params": params}, batch.state, batch.action)
        loss = jnp.mean(jnp.square(q1 - target_q) + jnp.square(q2 - target_q))
        return loss, (q1, q2)

    (loss, (q1, q2)), grads = jax.value_and_grad(loss_fn, has_aux=True)(critic_state.params)
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

    def loss_fn(model_with_actor, dropout_rng):
        dropped_reference = _apply_reference_dropout(batch.reference_action, dropout_rng, config.reference_dropout)
        actions = actor_mean(model_with_actor, batch.state, dropped_reference)
        q1, q2 = critic_model.apply({"params": critic_state.params}, batch.state, actions)
        q = jnp.minimum(q1, q2)
        bc_penalty = jnp.mean(jnp.square(actions - batch.reference_action), axis=-1)
        return jnp.mean(-q + config.actor_bc_weight * bc_penalty)

    diff_state = nnx.DiffState(0, actor_filter())
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, rng)
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
    return {
        "step": jnp.asarray(state.step),
        "params": state.params.to_pure_dict(),
        "opt_state": state.opt_state,
    }


def restore_policy_state(
    model,
    bundle: dict[str, Any],
    learning_rate: float,
) -> training_utils.TrainState:
    state = init_policy_state(model, learning_rate)
    pure_params = bundle["params"]
    params = nnx.state(model)
    params.replace_by_pure_dict(pure_params)
    return dataclasses.replace(state, step=bundle["step"], params=params, opt_state=bundle["opt_state"])


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
