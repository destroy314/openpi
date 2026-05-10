from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
import dataclasses
from typing import Any
import uuid

import jax
import jax.numpy as jnp
import numpy as np

from openpi.rlt import OPENPI_IMAGE_KEYS
from openpi.rlt import RLT_ACTION_DIM
from openpi.rlt import RLT_PROPRIO_DIM
from openpi.rlt import RLT_RL_TOKEN_DIM
from openpi.rlt import ReplayItem
from openpi.rlt import RLTInferRequest
from openpi.rlt import RLTInferResponse
from openpi.rlt import RLTObservation
from openpi.rlt import RLTStatusResponse
from openpi.rlt import RLTTokenRequest
from openpi.rlt import RLTTokenResponse
from openpi.rlt import action_space as _action_space
from openpi.rlt import trainer as _trainer

Transport = Callable[[str, Mapping[str, Any] | None], Mapping[str, Any]]


@dataclasses.dataclass(frozen=True)
class PlanWindow:
    feature_id: str
    plan_start_step: int
    rl_token: np.ndarray
    reference_plan_norm: np.ndarray
    reference_plan_real: np.ndarray
    prefix_valid: bool
    server_infer_time_s: float | None = None
    debug: dict[str, Any] = dataclasses.field(default_factory=dict)
    reference_plan_norm_sha256: str = dataclasses.field(init=False)
    reference_plan_real_sha256: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        rl_token = np.asarray(self.rl_token, dtype=np.float32).reshape(-1)
        reference_plan_norm = np.asarray(self.reference_plan_norm, dtype=np.float32)
        reference_plan_real = np.asarray(self.reference_plan_real, dtype=np.float32)
        if rl_token.shape != (RLT_RL_TOKEN_DIM,):
            raise ValueError(f"rl_token must have shape ({RLT_RL_TOKEN_DIM},), got {rl_token.shape}")
        if reference_plan_norm.ndim != 2:
            raise ValueError(f"reference_plan_norm must be rank-2, got {reference_plan_norm.shape}")
        if reference_plan_norm.shape[1] != RLT_ACTION_DIM:
            raise ValueError(f"reference_plan_norm must have width {RLT_ACTION_DIM}, got {reference_plan_norm.shape}")
        if reference_plan_real.shape != reference_plan_norm.shape:
            raise ValueError(
                "reference_plan_real must match reference_plan_norm shape, "
                f"got {reference_plan_real.shape} vs {reference_plan_norm.shape}"
            )
        object.__setattr__(self, "rl_token", rl_token)
        object.__setattr__(self, "reference_plan_norm", reference_plan_norm)
        object.__setattr__(self, "reference_plan_real", reference_plan_real)
        object.__setattr__(self, "debug", dict(self.debug))
        object.__setattr__(self, "reference_plan_norm_sha256", _array_sha256(reference_plan_norm))
        object.__setattr__(self, "reference_plan_real_sha256", _array_sha256(reference_plan_real))

    @property
    def horizon(self) -> int:
        return int(self.reference_plan_norm.shape[0])

    @property
    def action_dim(self) -> int:
        return int(self.reference_plan_norm.shape[1])

    def contains_chunk(self, *, step: int, action_horizon: int) -> bool:
        offset = int(step) - int(self.plan_start_step)
        return offset >= 0 and (offset + int(action_horizon)) <= self.horizon

    def slice_norm_chunk(self, *, step: int, action_horizon: int) -> np.ndarray:
        return self._slice_chunk(self.reference_plan_norm, step=step, action_horizon=action_horizon)

    def slice_real_chunk(self, *, step: int, action_horizon: int) -> np.ndarray:
        return self._slice_chunk(self.reference_plan_real, step=step, action_horizon=action_horizon)

    def _slice_chunk(self, plan: np.ndarray, *, step: int, action_horizon: int) -> np.ndarray:
        offset = int(step) - int(self.plan_start_step)
        end = offset + int(action_horizon)
        if offset < 0 or end > plan.shape[0]:
            raise ValueError(
                f"step={step} with action_horizon={action_horizon} is outside plan window "
                f"[{self.plan_start_step}, {self.plan_start_step + self.horizon})"
            )
        return np.asarray(plan[offset:end], dtype=np.float32).reshape(-1)


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def build_rlt_state(rl_token: np.ndarray, proprio: np.ndarray) -> np.ndarray:
    rl_token = np.asarray(rl_token, dtype=np.float32).reshape(-1)
    proprio = np.asarray(proprio, dtype=np.float32).reshape(-1)
    if rl_token.shape != (RLT_RL_TOKEN_DIM,):
        raise ValueError(f"rl_token must have shape ({RLT_RL_TOKEN_DIM},), got {rl_token.shape}")
    if proprio.shape != (RLT_PROPRIO_DIM,):
        raise ValueError(f"proprio must have shape ({RLT_PROPRIO_DIM},), got {proprio.shape}")
    return np.concatenate([rl_token, proprio], axis=0).astype(np.float32)


def should_request_new_plan(
    *,
    step: int,
    action_horizon: int,
    replan_horizon: int,
    current_plan: PlanWindow | None,
) -> bool:
    if current_plan is None:
        return True
    if not current_plan.contains_chunk(step=step, action_horizon=action_horizon):
        return True
    plan_offset = int(step) - int(current_plan.plan_start_step)
    return plan_offset > 0 and plan_offset % int(replan_horizon) == 0


def build_contract_observation(
    *,
    images: Mapping[str, bytes],
    state: np.ndarray,
    prompt: str,
    timestamp: float,
) -> RLTObservation:
    reverse_image_keys = {openpi_key: rt_vla_key for rt_vla_key, openpi_key in OPENPI_IMAGE_KEYS.items()}
    normalized_images = {
        reverse_image_keys.get(key, key): value
        for key, value in dict(images).items()
    }
    return RLTObservation(
        images=normalized_images,
        state=np.asarray(state, dtype=np.float32),
        prompt=prompt,
        timestamp=float(timestamp),
    )


@dataclasses.dataclass(frozen=True)
class RLTFeatureClient:
    transport: Transport
    infer_endpoint: str = "/rlt/infer"
    token_endpoint: str = "/rlt/token"
    status_endpoint: str = "/rlt/status"
    request_id_factory: Callable[[], str] = dataclasses.field(
        default_factory=lambda: lambda: f"req-{uuid.uuid4().hex}"
    )

    def infer(
        self,
        *,
        episode_id: str,
        observation: RLTObservation,
        diffusion_steps: int,
        reference_horizon: int,
    ) -> RLTInferResponse:
        request = RLTInferRequest(
            request_id=self.request_id_factory(),
            episode_id=episode_id,
            observation=observation,
            diffusion_steps=diffusion_steps,
            reference_horizon=reference_horizon,
        )
        return RLTInferResponse.from_dict(dict(self.transport(self.infer_endpoint, request.to_dict())))

    def token(
        self,
        *,
        episode_id: str,
        observation: RLTObservation,
    ) -> RLTTokenResponse:
        request = RLTTokenRequest(
            request_id=self.request_id_factory(),
            episode_id=episode_id,
            observation=observation,
        )
        return RLTTokenResponse.from_dict(dict(self.transport(self.token_endpoint, request.to_dict())))

    def status(self) -> RLTStatusResponse:
        return RLTStatusResponse.from_dict(dict(self.transport(self.status_endpoint, None)))


@dataclasses.dataclass(frozen=True)
class ExecutionSummary:
    executed_actions: np.ndarray
    intervened_mask: np.ndarray
    timestamps: list[float]
    records: list[dict[str, Any]]
    action_sources: list[str] = dataclasses.field(default_factory=list)


_REPLAY_ACTION_SOURCE_PRIORITY = (
    "pre_mpc_action",
    "pre_smooth_action",
    "command_action",
    "executed_action",
    "post_mpc_action",
    "post_smooth_action",
)


def _extract_replay_action(record: Mapping[str, Any]) -> tuple[list[float], str]:
    for source in _REPLAY_ACTION_SOURCE_PRIORITY:
        value = record.get(source)
        if value is not None:
            return [float(v) for v in value], source
    telemetry = record.get("telemetry", [])
    if isinstance(telemetry, list):
        by_source = {
            item.get("source"): item.get("action")
            for item in telemetry
            if isinstance(item, Mapping) and item.get("action") is not None
        }
        for source in _REPLAY_ACTION_SOURCE_PRIORITY:
            value = by_source.get(source)
            if value is not None:
                return [float(v) for v in value], source
    if record.get("action") is not None and record.get("source") in _REPLAY_ACTION_SOURCE_PRIORITY:
        return [float(v) for v in record["action"]], str(record["source"])
    raise KeyError(
        "execution record does not contain a replay action; expected one of "
        f"{_REPLAY_ACTION_SOURCE_PRIORITY}"
    )


def actor_infer_action_chunk(
    actor_model: Any,
    actor_state: Any,
    rng: jax.Array,
    *,
    rlt_state: np.ndarray,
    reference_action: np.ndarray,
) -> np.ndarray:
    action = _trainer.actor_sample_params(
        actor_model,
        actor_state.params,
        jnp.asarray(rlt_state, dtype=jnp.float32)[None, ...],
        jnp.asarray(reference_action, dtype=jnp.float32)[None, ...],
        rng,
        debug_step=actor_state.step,
        debug_label="collector",
    )
    return np.asarray(action[0], dtype=np.float32)


def summarize_execution_records(
    records: list[dict[str, Any]],
    *,
    intervened_mask: np.ndarray | None = None,
) -> ExecutionSummary:
    if not records:
        raise ValueError("records must not be empty")
    extracted = [_extract_replay_action(record) for record in records]
    executed_actions = np.asarray([action for action, _source in extracted], dtype=np.float32)
    action_sources = [source for _action, source in extracted]
    if intervened_mask is None:
        intervened_mask_array = np.zeros((executed_actions.shape[0],), dtype=bool)
    else:
        intervened_mask_array = np.asarray(intervened_mask, dtype=bool).reshape(-1)
        if intervened_mask_array.shape != (executed_actions.shape[0],):
            raise ValueError(
                "intervened_mask length must match executed actions, "
                f"got {intervened_mask_array.shape[0]} vs {executed_actions.shape[0]}"
            )
    return ExecutionSummary(
        executed_actions=executed_actions,
        intervened_mask=intervened_mask_array,
        timestamps=[float(record["timestamp"]) for record in records],
        records=[dict(record) for record in records],
        action_sources=action_sources,
    )


def discounted_return(reward_history: list[float], *, discount: float) -> float:
    return float(sum((float(discount) ** offset) * float(reward) for offset, reward in enumerate(reward_history)))


def build_replay_item(
    *,
    state: np.ndarray,
    executed_action_chunk_real: np.ndarray,
    reference_action_chunk_real: np.ndarray,
    action_base_state: np.ndarray,
    next_state: np.ndarray,
    next_reference_action_real: np.ndarray,
    next_action_base_state: np.ndarray,
    reward_history: list[float],
    discount: float,
    norm_stats: dict[str, Any],
    use_quantiles: bool,
    use_delta_joint_actions: bool,
    done: bool,
    env_step: int,
    intervened_mask: np.ndarray | None = None,
) -> ReplayItem:
    executed_action_chunk_real = np.asarray(executed_action_chunk_real, dtype=np.float32)
    reference_action_chunk_real = np.asarray(reference_action_chunk_real, dtype=np.float32).copy()
    if intervened_mask is not None:
        intervened_mask_array = np.asarray(intervened_mask, dtype=bool).reshape(-1)
        if intervened_mask_array.shape != (executed_action_chunk_real.shape[0],):
            raise ValueError(
                "intervened_mask length must match executed action steps, "
                f"got {intervened_mask_array.shape[0]} vs {executed_action_chunk_real.shape[0]}"
            )
        if np.any(intervened_mask_array):
            reference_action_chunk_real[intervened_mask_array] = executed_action_chunk_real[intervened_mask_array]
    action_chunk = _action_space.normalize_action(
        norm_stats,
        use_quantiles,
        executed_action_chunk_real,
        state=action_base_state,
        use_delta_joint_actions=use_delta_joint_actions,
    )
    reference_chunk = _action_space.normalize_action(
        norm_stats,
        use_quantiles,
        np.asarray(reference_action_chunk_real, dtype=np.float32),
        state=action_base_state,
        use_delta_joint_actions=use_delta_joint_actions,
    )
    next_reference_action = _action_space.normalize_action(
        norm_stats,
        use_quantiles,
        np.asarray(next_reference_action_real, dtype=np.float32),
        state=next_action_base_state,
        use_delta_joint_actions=use_delta_joint_actions,
    )
    return ReplayItem(
        state=np.asarray(state, dtype=np.float32).reshape(-1),
        action=np.asarray(action_chunk, dtype=np.float32).reshape(-1),
        reference_action=np.asarray(reference_chunk, dtype=np.float32).reshape(-1),
        reward=discounted_return(reward_history, discount=discount),
        next_state=np.asarray(next_state, dtype=np.float32).reshape(-1),
        next_reference_action=np.asarray(next_reference_action, dtype=np.float32).reshape(-1),
        bootstrap_steps=len(reward_history),
        done=bool(done),
        env_step=int(env_step),
    )
