from __future__ import annotations

import numpy as np

from openpi import transforms as _transforms
from openpi.shared.normalize import NormStats


def slice_norm_stats(stats: NormStats, dim: int) -> NormStats:
    """Truncate a NormStats to the first `dim` elements along the last axis."""
    return NormStats(
        mean=stats.mean[..., :dim],
        std=stats.std[..., :dim],
        q01=stats.q01[..., :dim] if stats.q01 is not None else None,
        q99=stats.q99[..., :dim] if stats.q99 is not None else None,
    )


def airbot_delta_action_mask() -> tuple[bool, ...]:
    return _transforms.make_bool_mask(6, -1, 6, -1)


def action_transform_state(state: np.ndarray, env_dim: int) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.shape[-1] < env_dim:
        raise ValueError(f"state must have at least {env_dim} dims, got {state.shape[-1]}")
    return state[:env_dim].copy()


def denormalize_action(
    norm_stats: dict[str, NormStats],
    use_quantiles: bool,
    actions: np.ndarray,
    *,
    state: np.ndarray | None = None,
    use_delta_joint_actions: bool = False,
) -> np.ndarray:
    """Convert model-space actions to real joint-position space."""
    outputs = {"actions": np.asarray(actions, dtype=np.float32).copy()}
    env_dim = outputs["actions"].shape[-1]
    action_stats = {"actions": slice_norm_stats(norm_stats["actions"], env_dim)}
    outputs = _transforms.Unnormalize(action_stats, use_quantiles=use_quantiles)(outputs)
    if use_delta_joint_actions:
        if state is None:
            raise ValueError("state is required to convert delta actions back to absolute actions.")
        outputs["state"] = action_transform_state(state, env_dim)
        outputs = _transforms.AbsoluteActions(airbot_delta_action_mask())(outputs)
    return np.asarray(outputs["actions"], dtype=np.float32)


def normalize_action(
    norm_stats: dict[str, NormStats],
    use_quantiles: bool,
    actions: np.ndarray,
    *,
    state: np.ndarray | None = None,
    use_delta_joint_actions: bool = False,
) -> np.ndarray:
    """Convert real joint-position actions to model space."""
    outputs = {"actions": np.asarray(actions, dtype=np.float32).copy()}
    env_dim = outputs["actions"].shape[-1]
    if use_delta_joint_actions:
        if state is None:
            raise ValueError("state is required to convert absolute actions into delta actions.")
        outputs["state"] = action_transform_state(state, env_dim)
        outputs = _transforms.DeltaActions(airbot_delta_action_mask())(outputs)
    action_stats = {"actions": slice_norm_stats(norm_stats["actions"], env_dim)}
    outputs = _transforms.Normalize(action_stats, use_quantiles=use_quantiles)(outputs)
    return np.asarray(outputs["actions"], dtype=np.float32)


_slice_norm_stats = slice_norm_stats
_airbot_delta_action_mask = airbot_delta_action_mask
_action_transform_state = action_transform_state
_denormalize_action = denormalize_action
_normalize_action = normalize_action


__all__ = [
    "action_transform_state",
    "airbot_delta_action_mask",
    "denormalize_action",
    "normalize_action",
    "slice_norm_stats",
    "_action_transform_state",
    "_airbot_delta_action_mask",
    "_denormalize_action",
    "_normalize_action",
    "_slice_norm_stats",
]
