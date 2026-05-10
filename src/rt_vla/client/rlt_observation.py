from __future__ import annotations

from collections.abc import Mapping
import dataclasses
from typing import Any, Literal

import numpy as np

RT_VLA_TO_OPENPI_IMAGE_KEYS = {
    "high": "cam_high",
    "left_hand": "cam_left_wrist",
    "right_hand": "cam_right_wrist",
}
RLT_STATE_DIM = 14
RLT_PROPRIO_DIM = 28
VelocitySource = Literal["sdk", "finite_difference"]

_DEFAULT_VELOCITY_KEYS = ("velocity", "joint_velocity", "state_velocity")
_DEFAULT_TIMESTAMP_KEYS = ("timestamp", "state_timestamp", "observation_timestamp", "ts")


@dataclasses.dataclass(frozen=True)
class VelocityFilterConfig:
    clip_abs: float | None = 10.0
    low_pass_alpha: float | None = 1.0
    min_dt: float = 1e-6
    max_timestamp_skew_sec: float = 0.02

    def __post_init__(self) -> None:
        if self.clip_abs is not None and self.clip_abs <= 0:
            raise ValueError(f"clip_abs must be positive when set, got {self.clip_abs}")
        if self.low_pass_alpha is not None and not 0.0 < self.low_pass_alpha <= 1.0:
            raise ValueError(f"low_pass_alpha must be in (0, 1], got {self.low_pass_alpha}")
        if self.min_dt <= 0:
            raise ValueError(f"min_dt must be positive, got {self.min_dt}")
        if self.max_timestamp_skew_sec < 0:
            raise ValueError(f"max_timestamp_skew_sec must be >= 0, got {self.max_timestamp_skew_sec}")


@dataclasses.dataclass(frozen=True)
class ObservationAdapterConfig:
    velocity_source: VelocitySource
    filter: VelocityFilterConfig = dataclasses.field(default_factory=VelocityFilterConfig)
    image_key_map: Mapping[str, str] = dataclasses.field(
        default_factory=lambda: dict(RT_VLA_TO_OPENPI_IMAGE_KEYS)
    )
    state_key: str = "state"
    timestamp_keys: tuple[str, ...] = _DEFAULT_TIMESTAMP_KEYS
    velocity_keys: tuple[str, ...] = _DEFAULT_VELOCITY_KEYS

    def __post_init__(self) -> None:
        if self.velocity_source not in {"sdk", "finite_difference"}:
            raise ValueError(
                f"velocity_source must be 'sdk' or 'finite_difference', got {self.velocity_source!r}"
            )
        if len(self.timestamp_keys) == 0:
            raise ValueError("timestamp_keys must not be empty")
        if len(self.velocity_keys) == 0:
            raise ValueError("velocity_keys must not be empty")


@dataclasses.dataclass(frozen=True)
class AdaptedObservation:
    images: dict[str, Any]
    state: np.ndarray
    proprio: np.ndarray
    timestamp: float
    velocity: np.ndarray
    velocity_source: VelocitySource


def remap_images(
    images: Mapping[str, Any],
    image_key_map: Mapping[str, str] = RT_VLA_TO_OPENPI_IMAGE_KEYS,
) -> dict[str, Any]:
    if not isinstance(images, Mapping):
        raise TypeError(f"images must be a mapping, got {type(images)!r}")
    missing = [key for key in image_key_map if key not in images]
    if missing:
        raise ValueError(f"observer images missing required keys: {missing}")
    return {openpi_key: images[rtvla_key] for rtvla_key, openpi_key in image_key_map.items()}


def _coerce_state(raw_state: Any, *, name: str) -> np.ndarray:
    state = np.asarray(raw_state, dtype=np.float32)
    if state.ndim != 1:
        raise ValueError(f"{name} must be rank-1, got shape {state.shape}")
    if state.shape[0] != RLT_STATE_DIM:
        raise ValueError(f"{name} must have length {RLT_STATE_DIM}, got {state.shape[0]}")
    return state


def _extract_timestamp(data: Mapping[str, Any], *, timestamp_keys: tuple[str, ...], context: str) -> float:
    for key in timestamp_keys:
        if key in data:
            value = float(data[key])
            if not np.isfinite(value):
                raise ValueError(f"{context}.{key} must be finite, got {value!r}")
            return value
    raise KeyError(f"{context} must contain one of timestamp keys {timestamp_keys}")


def _parse_velocity_payload(
    raw_value: Any,
    *,
    observer_data: Mapping[str, Any],
    state_timestamp: float,
    key: str,
    timestamp_keys: tuple[str, ...],
) -> tuple[Any, float]:
    if not isinstance(raw_value, Mapping):
        timestamp_key = f"{key}_timestamp"
        if timestamp_key in observer_data:
            timestamp = float(observer_data[timestamp_key])
            if not np.isfinite(timestamp):
                raise ValueError(f"observer.{timestamp_key} must be finite, got {timestamp!r}")
            return raw_value, timestamp
        return raw_value, state_timestamp
    for value_key in ("value", "velocity", "data"):
        if value_key in raw_value:
            return raw_value[value_key], _extract_timestamp(
                raw_value, timestamp_keys=timestamp_keys, context=f"observer.{key}"
            )
    raise KeyError(f"observer.{key} mapping must contain one of ('value', 'velocity', 'data')")


def _low_pass_filter(
    velocity: np.ndarray,
    previous_velocity: np.ndarray | None,
    *,
    alpha: float | None,
) -> np.ndarray:
    if alpha is None or alpha >= 1.0 or previous_velocity is None:
        return velocity
    return previous_velocity + alpha * (velocity - previous_velocity)


class RLTObservationAdapter:
    def __init__(self, config: ObservationAdapterConfig) -> None:
        self._config = config
        self._previous_state: np.ndarray | None = None
        self._previous_timestamp: float | None = None
        self._previous_velocity: np.ndarray | None = None

    def reset(self) -> None:
        self._previous_state = None
        self._previous_timestamp = None
        self._previous_velocity = None

    def adapt(self, observer_data: Mapping[str, Any]) -> AdaptedObservation:
        if not isinstance(observer_data, Mapping):
            raise TypeError(f"observer_data must be a mapping, got {type(observer_data)!r}")

        raw_images = observer_data.get("images", observer_data)
        images = remap_images(raw_images, image_key_map=self._config.image_key_map)
        state = _coerce_state(observer_data[self._config.state_key], name=f"observer.{self._config.state_key}")
        timestamp = _extract_timestamp(
            observer_data,
            timestamp_keys=self._config.timestamp_keys,
            context="observer",
        )

        if self._config.velocity_source == "sdk":
            for key in self._config.velocity_keys:
                if key not in observer_data:
                    continue
                raw_value, velocity_timestamp = _parse_velocity_payload(
                    observer_data[key],
                    observer_data=observer_data,
                    state_timestamp=timestamp,
                    key=key,
                    timestamp_keys=self._config.timestamp_keys,
                )
                velocity = _coerce_state(raw_value, name=f"observer.{key}")
                skew = abs(velocity_timestamp - timestamp)
                if skew > self._config.filter.max_timestamp_skew_sec:
                    raise ValueError(
                        f"observer.{key} timestamp skew {skew:.6f}s exceeds "
                        f"{self._config.filter.max_timestamp_skew_sec:.6f}s"
                    )
                break
            else:
                available = sorted(observer_data.keys())
                raise KeyError(
                    "velocity_source='sdk' requires an explicit velocity field; "
                    f"looked for {self._config.velocity_keys}, available keys: {available}"
                )
        else:
            velocity = self._finite_difference_velocity(state, timestamp)

        velocity = self._post_process_velocity(velocity)
        proprio = np.concatenate([state, velocity], axis=0).astype(np.float32)
        if proprio.shape[0] != RLT_PROPRIO_DIM:
            raise AssertionError(f"proprio must have length {RLT_PROPRIO_DIM}, got {proprio.shape[0]}")

        self._previous_state = state.copy()
        self._previous_timestamp = timestamp
        self._previous_velocity = velocity.copy()
        return AdaptedObservation(
            images=images,
            state=state,
            proprio=proprio,
            timestamp=timestamp,
            velocity=velocity,
            velocity_source=self._config.velocity_source,
        )

    def _finite_difference_velocity(self, state: np.ndarray, timestamp: float) -> np.ndarray:
        if self._previous_state is None or self._previous_timestamp is None:
            return np.zeros((RLT_STATE_DIM,), dtype=np.float32)
        dt = timestamp - self._previous_timestamp
        if dt <= 0:
            raise ValueError(
                f"observer.timestamp must be strictly increasing for finite_difference, got dt={dt:.6f}"
            )
        dt = max(dt, self._config.filter.min_dt)
        return ((state - self._previous_state) / dt).astype(np.float32)

    def _post_process_velocity(self, velocity: np.ndarray) -> np.ndarray:
        filtered = velocity.astype(np.float32)
        if self._config.filter.clip_abs is not None:
            filtered = np.clip(filtered, -self._config.filter.clip_abs, self._config.filter.clip_abs)
        filtered = _low_pass_filter(
            filtered,
            self._previous_velocity,
            alpha=self._config.filter.low_pass_alpha,
        )
        return filtered.astype(np.float32)
