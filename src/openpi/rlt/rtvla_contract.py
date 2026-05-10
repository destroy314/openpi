from __future__ import annotations

import dataclasses
from typing import Any
from typing import Mapping
from typing import Sequence

import numpy as np


RLT_PROTOCOL_VERSION = "rlt-rtvla-v2-batch-a"
RLT_STATE_DIM = 14
RLT_PROPRIO_DIM = 28
RLT_RL_TOKEN_DIM = 256
RLT_ACTION_DIM = 14
RT_VLA_IMAGE_KEYS = ("high", "left_hand", "right_hand")
OPENPI_IMAGE_KEYS = {
    "high": "cam_high",
    "left_hand": "cam_left_wrist",
    "right_hand": "cam_right_wrist",
}


def _expect_mapping(data: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(data)!r}")
    return data


def _expect_keys(data: Mapping[str, Any], *, name: str, required: Sequence[str], optional: Sequence[str] = ()) -> None:
    allowed = set(required) | set(optional)
    missing = [key for key in required if key not in data]
    extra = sorted(set(data) - allowed)
    if missing:
        raise ValueError(f"{name} is missing required keys: {missing}")
    if extra:
        raise ValueError(f"{name} has unexpected keys: {extra}")


def _coerce_float(value: object, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a float-compatible value, got {value!r}") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}")
    return result


def _coerce_bool(value: object, *, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise TypeError(f"{name} must be a bool, got {type(value)!r}")


def _coerce_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an int, got bool")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be an int-compatible value, got {value!r}") from exc
    return result


def _coerce_vector(
    value: object,
    *,
    name: str,
    expected_length: int | None = None,
) -> list[float]:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError(f"{name} must be rank-1, got shape {array.shape}")
    if expected_length is not None and array.shape[0] != expected_length:
        raise ValueError(f"{name} must have length {expected_length}, got {array.shape[0]}")
    return array.astype(np.float32).tolist()


def _coerce_matrix(
    value: object,
    *,
    name: str,
    expected_cols: int | None = None,
) -> list[list[float]]:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must be rank-2, got shape {array.shape}")
    if expected_cols is not None and array.shape[1] != expected_cols:
        raise ValueError(f"{name} must have width {expected_cols}, got {array.shape[1]}")
    return array.astype(np.float32).tolist()


def _coerce_shape(value: object, *, name: str, rank: int) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a sequence of ints, got {type(value)!r}")
    shape = [_coerce_int(v, name=f"{name}[{index}]") for index, v in enumerate(value)]
    if len(shape) != rank:
        raise ValueError(f"{name} must have rank {rank}, got {len(shape)}")
    return shape


def _coerce_image_bytes(value: object, *, name: str) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    raise TypeError(f"{name} must be bytes-like, got {type(value)!r}")


def _coerce_images(value: object, *, name: str) -> dict[str, bytes]:
    data = _expect_mapping(value, name=name)
    _expect_keys(data, name=name, required=RT_VLA_IMAGE_KEYS)
    return {
        key: _coerce_image_bytes(data[key], name=f"{name}.{key}")
        for key in RT_VLA_IMAGE_KEYS
    }


@dataclasses.dataclass(frozen=True)
class RLTObservation:
    images: dict[str, bytes]
    state: list[float]
    prompt: str
    timestamp: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "images", _coerce_images(self.images, name="observation.images"))
        object.__setattr__(self, "state", _coerce_vector(self.state, name="observation.state", expected_length=RLT_STATE_DIM))
        if not isinstance(self.prompt, str):
            raise TypeError(f"observation.prompt must be str, got {type(self.prompt)!r}")
        object.__setattr__(self, "timestamp", _coerce_float(self.timestamp, name="observation.timestamp"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "images": dict(self.images),
            "state": list(self.state),
            "prompt": self.prompt,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RLTObservation":
        payload = _expect_mapping(data, name="observation")
        _expect_keys(payload, name="observation", required=("images", "state", "prompt", "timestamp"))
        return cls(
            images=payload["images"],
            state=payload["state"],
            prompt=payload["prompt"],
            timestamp=payload["timestamp"],
        )


@dataclasses.dataclass(frozen=True)
class RLTInferRequest:
    request_id: str
    episode_id: str
    observation: RLTObservation
    diffusion_steps: int
    reference_horizon: int

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str):
            raise TypeError(f"request_id must be str, got {type(self.request_id)!r}")
        if not isinstance(self.episode_id, str):
            raise TypeError(f"episode_id must be str, got {type(self.episode_id)!r}")
        if not isinstance(self.observation, RLTObservation):
            raise TypeError(f"observation must be RLTObservation, got {type(self.observation)!r}")
        diffusion_steps = _coerce_int(self.diffusion_steps, name="diffusion_steps")
        reference_horizon = _coerce_int(self.reference_horizon, name="reference_horizon")
        if diffusion_steps <= 0:
            raise ValueError(f"diffusion_steps must be positive, got {diffusion_steps}")
        if reference_horizon <= 0:
            raise ValueError(f"reference_horizon must be positive, got {reference_horizon}")
        object.__setattr__(self, "diffusion_steps", diffusion_steps)
        object.__setattr__(self, "reference_horizon", reference_horizon)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "episode_id": self.episode_id,
            "observation": self.observation.to_dict(),
            "diffusion_steps": self.diffusion_steps,
            "reference_horizon": self.reference_horizon,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RLTInferRequest":
        payload = _expect_mapping(data, name="infer_request")
        _expect_keys(
            payload,
            name="infer_request",
            required=("request_id", "episode_id", "observation", "diffusion_steps", "reference_horizon"),
        )
        return cls(
            request_id=payload["request_id"],
            episode_id=payload["episode_id"],
            observation=RLTObservation.from_dict(payload["observation"]),
            diffusion_steps=payload["diffusion_steps"],
            reference_horizon=payload["reference_horizon"],
        )


@dataclasses.dataclass(frozen=True)
class RLTInferDebug:
    rl_token_norm: float
    reference_plan_norm: float
    feature_shape: list[int]
    reference_plan_shape: list[int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "rl_token_norm", _coerce_float(self.rl_token_norm, name="debug.rl_token_norm"))
        object.__setattr__(
            self,
            "reference_plan_norm",
            _coerce_float(self.reference_plan_norm, name="debug.reference_plan_norm"),
        )
        feature_shape = _coerce_shape(self.feature_shape, name="debug.feature_shape", rank=1)
        reference_plan_shape = _coerce_shape(self.reference_plan_shape, name="debug.reference_plan_shape", rank=2)
        if feature_shape[0] != RLT_RL_TOKEN_DIM:
            raise ValueError(f"debug.feature_shape must be [{RLT_RL_TOKEN_DIM}], got {feature_shape}")
        if reference_plan_shape[1] != RLT_ACTION_DIM:
            raise ValueError(f"debug.reference_plan_shape must end with {RLT_ACTION_DIM}, got {reference_plan_shape}")
        object.__setattr__(self, "feature_shape", feature_shape)
        object.__setattr__(self, "reference_plan_shape", reference_plan_shape)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rl_token_norm": self.rl_token_norm,
            "reference_plan_norm": self.reference_plan_norm,
            "feature_shape": list(self.feature_shape),
            "reference_plan_shape": list(self.reference_plan_shape),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RLTInferDebug":
        payload = _expect_mapping(data, name="infer_debug")
        _expect_keys(
            payload,
            name="infer_debug",
            required=("rl_token_norm", "reference_plan_norm", "feature_shape", "reference_plan_shape"),
        )
        return cls(
            rl_token_norm=payload["rl_token_norm"],
            reference_plan_norm=payload["reference_plan_norm"],
            feature_shape=payload["feature_shape"],
            reference_plan_shape=payload["reference_plan_shape"],
        )


@dataclasses.dataclass(frozen=True)
class RLTInferResponse:
    request_id: str
    feature_id: str
    rl_token: list[float]
    reference_plan_norm: list[list[float]]
    reference_plan_list: list[list[float]]
    prefix_valid: bool
    server_infer_time_s: float
    debug: RLTInferDebug

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str):
            raise TypeError(f"request_id must be str, got {type(self.request_id)!r}")
        if not isinstance(self.feature_id, str):
            raise TypeError(f"feature_id must be str, got {type(self.feature_id)!r}")
        object.__setattr__(self, "rl_token", _coerce_vector(self.rl_token, name="rl_token", expected_length=RLT_RL_TOKEN_DIM))
        reference_plan_norm = _coerce_matrix(
            self.reference_plan_norm,
            name="reference_plan_norm",
            expected_cols=RLT_ACTION_DIM,
        )
        reference_plan_list = _coerce_matrix(
            self.reference_plan_list,
            name="reference_plan_list",
            expected_cols=RLT_ACTION_DIM,
        )
        if len(reference_plan_norm) != len(reference_plan_list):
            raise ValueError(
                "reference_plan_norm and reference_plan_list must have the same length, "
                f"got {len(reference_plan_norm)} and {len(reference_plan_list)}"
            )
        if not isinstance(self.debug, RLTInferDebug):
            raise TypeError(f"debug must be RLTInferDebug, got {type(self.debug)!r}")
        expected_plan_shape = [len(reference_plan_norm), RLT_ACTION_DIM]
        if self.debug.reference_plan_shape != expected_plan_shape:
            raise ValueError(
                "debug.reference_plan_shape must match reference_plan_norm, "
                f"got {self.debug.reference_plan_shape} vs {expected_plan_shape}"
            )
        object.__setattr__(self, "reference_plan_norm", reference_plan_norm)
        object.__setattr__(self, "reference_plan_list", reference_plan_list)
        object.__setattr__(self, "prefix_valid", _coerce_bool(self.prefix_valid, name="prefix_valid"))
        object.__setattr__(self, "server_infer_time_s", _coerce_float(self.server_infer_time_s, name="server_infer_time_s"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "feature_id": self.feature_id,
            "rl_token": list(self.rl_token),
            "reference_plan_norm": [list(row) for row in self.reference_plan_norm],
            "reference_plan_list": [list(row) for row in self.reference_plan_list],
            "prefix_valid": self.prefix_valid,
            "server_infer_time_s": self.server_infer_time_s,
            "debug": self.debug.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RLTInferResponse":
        payload = _expect_mapping(data, name="infer_response")
        _expect_keys(
            payload,
            name="infer_response",
            required=(
                "request_id",
                "feature_id",
                "rl_token",
                "reference_plan_norm",
                "reference_plan_list",
                "prefix_valid",
                "server_infer_time_s",
                "debug",
            ),
        )
        return cls(
            request_id=payload["request_id"],
            feature_id=payload["feature_id"],
            rl_token=payload["rl_token"],
            reference_plan_norm=payload["reference_plan_norm"],
            reference_plan_list=payload["reference_plan_list"],
            prefix_valid=payload["prefix_valid"],
            server_infer_time_s=payload["server_infer_time_s"],
            debug=RLTInferDebug.from_dict(payload["debug"]),
        )


@dataclasses.dataclass(frozen=True)
class RLTTokenDebug:
    rl_token_norm: float
    feature_shape: list[int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "rl_token_norm", _coerce_float(self.rl_token_norm, name="debug.rl_token_norm"))
        feature_shape = _coerce_shape(self.feature_shape, name="debug.feature_shape", rank=1)
        if feature_shape[0] != RLT_RL_TOKEN_DIM:
            raise ValueError(f"debug.feature_shape must be [{RLT_RL_TOKEN_DIM}], got {feature_shape}")
        object.__setattr__(self, "feature_shape", feature_shape)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rl_token_norm": self.rl_token_norm,
            "feature_shape": list(self.feature_shape),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RLTTokenDebug":
        payload = _expect_mapping(data, name="token_debug")
        _expect_keys(payload, name="token_debug", required=("rl_token_norm", "feature_shape"))
        return cls(
            rl_token_norm=payload["rl_token_norm"],
            feature_shape=payload["feature_shape"],
        )


@dataclasses.dataclass(frozen=True)
class RLTTokenRequest:
    request_id: str
    episode_id: str
    observation: RLTObservation

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str):
            raise TypeError(f"request_id must be str, got {type(self.request_id)!r}")
        if not isinstance(self.episode_id, str):
            raise TypeError(f"episode_id must be str, got {type(self.episode_id)!r}")
        if not isinstance(self.observation, RLTObservation):
            raise TypeError(f"observation must be RLTObservation, got {type(self.observation)!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "episode_id": self.episode_id,
            "observation": self.observation.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RLTTokenRequest":
        payload = _expect_mapping(data, name="token_request")
        _expect_keys(payload, name="token_request", required=("request_id", "episode_id", "observation"))
        return cls(
            request_id=payload["request_id"],
            episode_id=payload["episode_id"],
            observation=RLTObservation.from_dict(payload["observation"]),
        )


@dataclasses.dataclass(frozen=True)
class RLTTokenResponse:
    request_id: str
    feature_id: str
    rl_token: list[float]
    prefix_valid: bool
    server_token_time_s: float
    debug: RLTTokenDebug

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str):
            raise TypeError(f"request_id must be str, got {type(self.request_id)!r}")
        if not isinstance(self.feature_id, str):
            raise TypeError(f"feature_id must be str, got {type(self.feature_id)!r}")
        object.__setattr__(self, "rl_token", _coerce_vector(self.rl_token, name="rl_token", expected_length=RLT_RL_TOKEN_DIM))
        object.__setattr__(self, "prefix_valid", _coerce_bool(self.prefix_valid, name="prefix_valid"))
        object.__setattr__(self, "server_token_time_s", _coerce_float(self.server_token_time_s, name="server_token_time_s"))
        if not isinstance(self.debug, RLTTokenDebug):
            raise TypeError(f"debug must be RLTTokenDebug, got {type(self.debug)!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "feature_id": self.feature_id,
            "rl_token": list(self.rl_token),
            "prefix_valid": self.prefix_valid,
            "server_token_time_s": self.server_token_time_s,
            "debug": self.debug.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RLTTokenResponse":
        payload = _expect_mapping(data, name="token_response")
        _expect_keys(
            payload,
            name="token_response",
            required=("request_id", "feature_id", "rl_token", "prefix_valid", "server_token_time_s", "debug"),
        )
        return cls(
            request_id=payload["request_id"],
            feature_id=payload["feature_id"],
            rl_token=payload["rl_token"],
            prefix_valid=payload["prefix_valid"],
            server_token_time_s=payload["server_token_time_s"],
            debug=RLTTokenDebug.from_dict(payload["debug"]),
        )


@dataclasses.dataclass(frozen=True)
class RLTServerRuntimeContract:
    protocol_version: str = RLT_PROTOCOL_VERSION
    stage1_feature_backend: str = "jax"
    server_runs_actor: bool = False
    server_runs_critic: bool = False
    server_runs_learner: bool = False
    server_builds_replay: bool = False
    server_saves_stage2_checkpoint: bool = False
    stage2_checkpoint_owner: str = "client"

    def __post_init__(self) -> None:
        if self.stage2_checkpoint_owner != "client":
            raise ValueError(
                "Batch A freezes Stage 2 checkpoint ownership on the client, "
                f"got {self.stage2_checkpoint_owner!r}"
            )
        if self.server_runs_actor or self.server_runs_critic or self.server_runs_learner:
            raise ValueError("Batch A freezes the server as a Stage-1-only feature/reference service.")
        if self.server_builds_replay:
            raise ValueError("Batch A freezes replay construction on the client.")
        if self.server_saves_stage2_checkpoint:
            raise ValueError("Batch A freezes Stage 2 checkpoint persistence on the client.")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class RLTClientRuntimeContract:
    actor_runtime_location: str = "client_main_process_threads"
    learner_runtime_location: str = "client_learner_process"
    sample_queue_message_types: tuple[str, ...] = ("ReplayItem", "StopSignal")
    policy_queue_message_types: tuple[str, ...] = ("PolicyUpdate",)
    status_queue_message_types: tuple[str, ...] = ("LearnerInit", "LearnerStats", "LearnerError")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class RLTStatusResponse:
    status: str
    server: RLTServerRuntimeContract = dataclasses.field(default_factory=RLTServerRuntimeContract)

    def __post_init__(self) -> None:
        if not isinstance(self.status, str):
            raise TypeError(f"status must be str, got {type(self.status)!r}")
        if not isinstance(self.server, RLTServerRuntimeContract):
            raise TypeError(f"server must be RLTServerRuntimeContract, got {type(self.server)!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "server": self.server.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RLTStatusResponse":
        payload = _expect_mapping(data, name="status_response")
        _expect_keys(payload, name="status_response", required=("status",), optional=("server",))
        server_payload = payload.get("server", {})
        server = (
            server_payload
            if isinstance(server_payload, RLTServerRuntimeContract)
            else RLTServerRuntimeContract(**dict(_expect_mapping(server_payload, name="status_response.server")))
        )
        return cls(status=payload["status"], server=server)


@dataclasses.dataclass
class ReplayItem:
    state: np.ndarray
    action: np.ndarray
    reference_action: np.ndarray
    reward: float
    next_state: np.ndarray
    next_reference_action: np.ndarray
    bootstrap_steps: int
    done: bool
    env_step: int


@dataclasses.dataclass(frozen=True)
class StopSignal:
    final_step: int


@dataclasses.dataclass
class LearnerInit:
    start_step: int
    actor_params: dict[str, Any]


@dataclasses.dataclass
class PolicyUpdate:
    actor_params: dict[str, Any]


@dataclasses.dataclass
class LearnerError:
    message: str
    traceback: str


@dataclasses.dataclass
class LearnerStats:
    learner_step: int
    env_step: int
    replay_size: int
    latest_checkpoint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "learner_step": int(self.learner_step),
            "env_step": int(self.env_step),
            "replay_size": int(self.replay_size),
            "latest_checkpoint": self.latest_checkpoint,
        }
