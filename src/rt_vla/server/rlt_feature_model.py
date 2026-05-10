from __future__ import annotations

import dataclasses
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
import pathlib
import threading
from typing import Any

import cv2
import jax
import numpy as np

from openpi import transforms as _transforms
from openpi.policies import policy_config as _policy_config
from openpi.rlt import action_space as _action_space
from openpi.rlt import OPENPI_IMAGE_KEYS
from openpi.rlt import RLT_ACTION_DIM
from openpi.rlt import RLTObservation
from openpi.rlt import RLT_RL_TOKEN_DIM
from openpi.shared import download as _download
from openpi.shared import nnx_utils
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config


@dataclass(frozen=True)
class RLTFeatureModelObservation:
    images: dict[str, bytes]
    state: np.ndarray
    prompt: str
    timestamp: float

    @classmethod
    def from_contract(cls, observation: RLTObservation) -> "RLTFeatureModelObservation":
        return cls(
            images={OPENPI_IMAGE_KEYS[key]: value for key, value in observation.images.items()},
            state=np.asarray(observation.state, dtype=np.float32),
            prompt=observation.prompt,
            timestamp=float(observation.timestamp),
        )


@dataclass(frozen=True)
class RLTFeatureModelOutput:
    rl_token: np.ndarray
    prefix_valid: bool = True
    reference_plan_norm: np.ndarray | None = None
    reference_plan_list: np.ndarray | None = None

    def __post_init__(self) -> None:
        rl_token = np.asarray(self.rl_token, dtype=np.float32)
        if rl_token.shape != (RLT_RL_TOKEN_DIM,):
            raise ValueError(f"rl_token must have shape ({RLT_RL_TOKEN_DIM},), got {rl_token.shape}")
        object.__setattr__(self, "rl_token", rl_token)

        if self.reference_plan_norm is None:
            plan_norm = None
        else:
            plan_norm = np.asarray(self.reference_plan_norm, dtype=np.float32)
            if plan_norm.ndim != 2 or plan_norm.shape[1] != RLT_ACTION_DIM:
                raise ValueError(
                    "reference_plan_norm must have shape (horizon, 14), "
                    f"got {plan_norm.shape}"
                )
        if self.reference_plan_list is None:
            plan_list = None
        else:
            plan_list = np.asarray(self.reference_plan_list, dtype=np.float32)
            if plan_list.ndim != 2 or plan_list.shape[1] != RLT_ACTION_DIM:
                raise ValueError(
                    "reference_plan_list must have shape (horizon, 14), "
                    f"got {plan_list.shape}"
                )

        if (plan_norm is None) != (plan_list is None):
            raise ValueError("reference_plan_norm and reference_plan_list must either both be set or both be omitted")
        if plan_norm is not None and plan_list is not None and plan_norm.shape != plan_list.shape:
            raise ValueError(
                "reference_plan_norm and reference_plan_list must have the same shape, "
                f"got {plan_norm.shape} and {plan_list.shape}"
            )

        object.__setattr__(self, "reference_plan_norm", plan_norm)
        object.__setattr__(self, "reference_plan_list", plan_list)

    def require_reference_plan(self) -> tuple[np.ndarray, np.ndarray]:
        if self.reference_plan_norm is None or self.reference_plan_list is None:
            raise ValueError("reference plan was requested but the feature model did not return one")
        return self.reference_plan_norm, self.reference_plan_list


class BaseRLTFeatureModel(ABC):
    @abstractmethod
    def infer_features(
        self,
        observation: RLTFeatureModelObservation,
        *,
        diffusion_steps: int,
        reference_horizon: int,
    ) -> RLTFeatureModelOutput:
        raise NotImplementedError

    @abstractmethod
    def infer_token(
        self,
        observation: RLTFeatureModelObservation,
    ) -> RLTFeatureModelOutput:
        raise NotImplementedError


class DeferredJaxRLTFeatureModel(BaseRLTFeatureModel):
    def __init__(
        self,
        *,
        config_name: str,
        checkpoint_dir: str | pathlib.Path,
        default_prompt: str | None = None,
        seed: int = 0,
        backend_name: str = "jax",
    ) -> None:
        self.backend_name = backend_name
        self.config_name = config_name
        self.checkpoint_dir = pathlib.Path(_download.maybe_download(str(checkpoint_dir)))
        self.default_prompt = default_prompt
        self.seed = int(seed)

        train_config = _config.get_config(self.config_name)
        stage1_policy_config = _build_stage1_policy_config(train_config)
        self._policy = _policy_config.create_trained_policy(
            stage1_policy_config,
            self.checkpoint_dir,
            default_prompt=self.default_prompt,
        )
        self._encode_prefix = nnx_utils.module_jit(self._policy.model.encode_prefix)
        self._compute_rl_token = nnx_utils.module_jit(self._policy.model.compute_rl_token)
        self._sample_reference_actions = nnx_utils.module_jit(self._policy.model.sample_reference_actions)

        self._model_action_horizon = int(stage1_policy_config.model.action_horizon)
        self._env_action_dim = int(stage1_policy_config.model.rlt_env_action_dim)
        data_config = stage1_policy_config.data.create(stage1_policy_config.assets_dirs, stage1_policy_config.model)
        self._use_quantiles = bool(data_config.use_quantile_norm)
        self._use_delta_joint_actions = any(
            isinstance(transform, _transforms.DeltaActions) for transform in data_config.data_transforms.inputs
        )
        self._norm_stats = (
            _checkpoints.load_norm_stats(self.checkpoint_dir / "assets", data_config.asset_id)
            if data_config.asset_id is not None
            else None
        )
        self._rng = jax.random.key(self.seed)
        self._rng_lock = threading.Lock()

    def infer_features(
        self,
        observation: RLTFeatureModelObservation,
        *,
        diffusion_steps: int,
        reference_horizon: int,
    ) -> RLTFeatureModelOutput:
        if reference_horizon <= 0:
            raise ValueError(f"reference_horizon must be positive, got {reference_horizon}")
        if reference_horizon > self._model_action_horizon:
            raise ValueError(
                f"reference_horizon {reference_horizon} exceeds model action horizon {self._model_action_horizon}"
            )

        inputs, batched_observation = self._policy.prepare_observation(_contract_observation_to_policy_input(observation))
        _, _, prefix_rlt_mask, prefix_out, _ = self._encode_prefix(batched_observation)
        rl_token, _ = self._compute_rl_token(prefix_out, prefix_rlt_mask)
        sample_rng = self._next_rng()
        reference_actions = self._sample_reference_actions(sample_rng, batched_observation, num_steps=diffusion_steps)
        reference_plan_norm = np.asarray(
            reference_actions[0, :reference_horizon, : self._env_action_dim],
            dtype=np.float32,
        )
        reference_plan_list = _action_space.denormalize_action(
            self._norm_stats,
            self._use_quantiles,
            reference_plan_norm,
            state=np.asarray(inputs["state"], dtype=np.float32),
            use_delta_joint_actions=self._use_delta_joint_actions,
        )
        return RLTFeatureModelOutput(
            rl_token=np.asarray(rl_token[0], dtype=np.float32),
            prefix_valid=True,
            reference_plan_norm=reference_plan_norm,
            reference_plan_list=np.asarray(reference_plan_list, dtype=np.float32),
        )

    def infer_token(
        self,
        observation: RLTFeatureModelObservation,
    ) -> RLTFeatureModelOutput:
        _, batched_observation = self._policy.prepare_observation(_contract_observation_to_policy_input(observation))
        _, _, prefix_rlt_mask, prefix_out, _ = self._encode_prefix(batched_observation)
        rl_token, _ = self._compute_rl_token(prefix_out, prefix_rlt_mask)
        return RLTFeatureModelOutput(rl_token=np.asarray(rl_token[0], dtype=np.float32), prefix_valid=True)

    def _next_rng(self) -> jax.Array:
        with self._rng_lock:
            self._rng, sample_rng = jax.random.split(self._rng)
        return sample_rng


@dataclass(frozen=True)
class HybridRTCRLTFeatureModel(BaseRLTFeatureModel):
    """Use JAX for RL token extraction and an RTC adapter for reference actions."""

    token_model: BaseRLTFeatureModel
    reference_adapter: Any
    norm_stats: dict[str, Any]
    use_quantiles: bool = False
    use_delta_joint_actions: bool = False
    backend_name: str = "hybrid_triton_jax_token"

    def infer_features(
        self,
        observation: RLTFeatureModelObservation,
        *,
        diffusion_steps: int,
        reference_horizon: int,
    ) -> RLTFeatureModelOutput:
        del diffusion_steps
        if reference_horizon <= 0:
            raise ValueError(f"reference_horizon must be positive, got {reference_horizon}")
        token_output = self.token_model.infer_token(observation)
        reference_plan_list = np.asarray(
            self.reference_adapter.infer_actions(_feature_observation_to_rtc_payload(observation)),
            dtype=np.float32,
        )
        if reference_plan_list.ndim != 2 or reference_plan_list.shape[1] != RLT_ACTION_DIM:
            raise ValueError(
                "RTC reference adapter must return actions with shape (horizon, 14), "
                f"got {reference_plan_list.shape}"
            )
        if reference_plan_list.shape[0] < reference_horizon:
            raise ValueError(
                "RTC reference adapter returned too few actions for requested horizon: "
                f"{reference_plan_list.shape[0]} < {reference_horizon}"
            )
        reference_plan_list = reference_plan_list[:reference_horizon]
        reference_plan_norm = _action_space.normalize_action(
            self.norm_stats,
            self.use_quantiles,
            reference_plan_list,
            state=np.asarray(observation.state, dtype=np.float32),
            use_delta_joint_actions=self.use_delta_joint_actions,
        )
        return RLTFeatureModelOutput(
            rl_token=token_output.rl_token,
            prefix_valid=token_output.prefix_valid,
            reference_plan_norm=np.asarray(reference_plan_norm, dtype=np.float32),
            reference_plan_list=reference_plan_list,
        )

    def infer_token(
        self,
        observation: RLTFeatureModelObservation,
    ) -> RLTFeatureModelOutput:
        return self.token_model.infer_token(observation)


def _build_stage1_policy_config(train_config):
    model_config = train_config.model
    if getattr(model_config, "use_rlt", False) and getattr(model_config, "rlt_actor_enabled", False):
        model_config = dataclasses.replace(model_config, rlt_actor_enabled=False)
    data_config = train_config.data
    if getattr(data_config, "require_proprio", None) is not None and getattr(data_config, "require_proprio", False):
        data_config = dataclasses.replace(data_config, require_proprio=False)
    return dataclasses.replace(train_config, model=model_config, data=data_config)


def _decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("failed to decode observation image bytes")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _contract_observation_to_policy_input(observation: RLTFeatureModelObservation) -> dict[str, object]:
    return {
        "images": {key: _decode_image_bytes(value) for key, value in observation.images.items()},
        "state": np.asarray(observation.state, dtype=np.float32),
        "prompt": observation.prompt,
    }


def _feature_observation_to_rtc_payload(observation: RLTFeatureModelObservation) -> dict[str, object]:
    reverse_image_keys = {openpi_key: rt_vla_key for rt_vla_key, openpi_key in OPENPI_IMAGE_KEYS.items()}
    return {
        "images": {
            reverse_image_keys.get(key, key): value
            for key, value in observation.images.items()
        },
        "action": [np.asarray(observation.state, dtype=np.float32).tolist()],
        "state": np.asarray(observation.state, dtype=np.float32).tolist(),
        "prompt": observation.prompt,
        "timestamp": float(observation.timestamp),
    }
