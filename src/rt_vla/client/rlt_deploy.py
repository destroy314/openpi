from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from openpi.rlt import RLT_ACTION_DIM
from openpi.rlt import RLT_PROPRIO_DIM
from openpi.rlt import RLT_RL_TOKEN_DIM
from openpi.rlt import checkpointing as _checkpointing
from openpi.rlt import trainer as _trainer
from openpi.training import config as _config


@dataclasses.dataclass(frozen=True)
class DeployActorConfig:
    config_name: str
    checkpoint_dir: pathlib.Path
    action_horizon: int = 10
    actor_lr: float = 3e-4
    seed: int = 0


@dataclasses.dataclass(frozen=True)
class DeployActor:
    actor_model: Any
    actor_state: Any
    action_horizon: int


def load_deploy_actor(config: DeployActorConfig) -> DeployActor:
    action_horizon = int(config.action_horizon)
    if action_horizon <= 0:
        raise ValueError(f"action_horizon must be positive, got {action_horizon}")
    train_config = _config.get_config(config.config_name)
    state_dim = RLT_RL_TOKEN_DIM + RLT_PROPRIO_DIM
    action_dim = action_horizon * RLT_ACTION_DIM
    hidden_dim = int(train_config.model.rlt_actor_hidden_dim)
    actor_model, _ = _trainer.init_actor_state(
        jax.random.key(int(config.seed)),
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        learning_rate=float(config.actor_lr),
    )
    actor_state = _trainer.restore_actor_state(
        actor_model,
        _checkpointing.restore_bundle(config.checkpoint_dir, "policy_state"),
        float(config.actor_lr),
    )
    return DeployActor(actor_model=actor_model, actor_state=actor_state, action_horizon=action_horizon)


def sample_deploy_action(
    deploy_actor: DeployActor,
    *,
    rlt_state: np.ndarray,
    reference_action: np.ndarray,
    rng: jax.Array | None = None,
    deterministic: bool = True,
) -> np.ndarray:
    state = np.asarray(rlt_state, dtype=np.float32).reshape(-1)
    if state.shape != (RLT_RL_TOKEN_DIM + RLT_PROPRIO_DIM,):
        raise ValueError(
            f"rlt_state must have shape ({RLT_RL_TOKEN_DIM + RLT_PROPRIO_DIM},), got {state.shape}"
        )
    reference = np.asarray(reference_action, dtype=np.float32)
    if reference.shape == (deploy_actor.action_horizon, RLT_ACTION_DIM):
        reference = reference.reshape(-1)
    expected_action_dim = deploy_actor.action_horizon * RLT_ACTION_DIM
    if reference.shape != (expected_action_dim,):
        raise ValueError(f"reference_action must have shape ({expected_action_dim},), got {reference.shape}")
    if rng is None:
        rng = jax.random.key(0)
    action = _trainer.actor_sample_params(
        deploy_actor.actor_model,
        deploy_actor.actor_state.params,
        jnp.asarray(state, dtype=jnp.float32)[None, ...],
        jnp.asarray(reference, dtype=jnp.float32)[None, ...],
        rng,
        deterministic=deterministic,
        debug_step=deploy_actor.actor_state.step,
        debug_label="deploy",
    )
    return np.asarray(action[0], dtype=np.float32).reshape(deploy_actor.action_horizon, RLT_ACTION_DIM)
