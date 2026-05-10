from __future__ import annotations

import dataclasses

import cv2
import numpy as np
import pytest

from openpi.rlt import OPENPI_IMAGE_KEYS
from openpi.rlt import RLT_ACTION_DIM
from openpi.rlt import RLTInferResponse
from openpi.rlt import RLTObservation
from openpi.rlt import RLTStatusResponse
from openpi.rlt import RLTTokenResponse
from openpi.rlt import RLT_RL_TOKEN_DIM
from openpi.rlt import RLT_STATE_DIM
import rt_vla.server.rlt_feature_model as feature_model_module
from rt_vla.server.rlt_feature_model import BaseRLTFeatureModel
from rt_vla.server.rlt_feature_model import DeferredJaxRLTFeatureModel
from rt_vla.server.rlt_feature_model import HybridRTCRLTFeatureModel
from rt_vla.server.rlt_feature_model import RLTFeatureModelObservation
from rt_vla.server.rlt_feature_model import RLTFeatureModelOutput
from rt_vla.server.rlt_feature_server import RLTFeatureServer
from openpi.shared.normalize import NormStats


class RecordingFeatureModel(BaseRLTFeatureModel):
    def __init__(self) -> None:
        self.last_observation: RLTFeatureModelObservation | None = None
        self.last_diffusion_steps: int | None = None
        self.last_reference_horizon: int | None = None

    def infer_features(
        self,
        observation: RLTFeatureModelObservation,
        *,
        diffusion_steps: int,
        reference_horizon: int,
    ) -> RLTFeatureModelOutput:
        self.last_observation = observation
        self.last_diffusion_steps = diffusion_steps
        self.last_reference_horizon = reference_horizon
        reference_plan_norm = np.arange(
            reference_horizon * RLT_ACTION_DIM,
            dtype=np.float32,
        ).reshape(reference_horizon, RLT_ACTION_DIM)
        reference_plan_list = reference_plan_norm + 1000.0
        return RLTFeatureModelOutput(
            rl_token=np.arange(RLT_RL_TOKEN_DIM, dtype=np.float32),
            prefix_valid=True,
            reference_plan_norm=reference_plan_norm,
            reference_plan_list=reference_plan_list,
        )

    def infer_token(
        self,
        observation: RLTFeatureModelObservation,
    ) -> RLTFeatureModelOutput:
        self.last_observation = observation
        return RLTFeatureModelOutput(
            rl_token=np.full((RLT_RL_TOKEN_DIM,), 3.0, dtype=np.float32),
            prefix_valid=False,
        )


class _RecordingRTCReferenceAdapter:
    def __init__(self, actions: np.ndarray) -> None:
        self.actions = np.asarray(actions, dtype=np.float32)
        self.last_payload: dict | None = None

    def infer_actions(self, payload: dict) -> list[list[float]]:
        self.last_payload = payload
        return self.actions.tolist()


def _make_observation() -> RLTObservation:
    return RLTObservation(
        images={
            "high": b"high-image",
            "left_hand": b"left-image",
            "right_hand": b"right-image",
        },
        state=np.arange(RLT_STATE_DIM, dtype=np.float32),
        prompt="pick up block",
        timestamp=12.5,
    )


def test_infer_route_returns_protocol_response_and_maps_image_keys():
    model = RecordingFeatureModel()
    server = RLTFeatureServer(
        feature_model=model,
        clock=iter([10.0, 10.25]).__next__,
        feature_id_factory=lambda: "feature-infer",
    )

    response = server.dispatch(
        "/rlt/infer",
        {
            "request_id": "req-1",
            "episode_id": "ep-1",
            "observation": _make_observation().to_dict(),
            "diffusion_steps": 7,
            "reference_horizon": 5,
        },
    )

    parsed = RLTInferResponse.from_dict(response)
    assert parsed.feature_id == "feature-infer"
    assert parsed.debug.feature_shape == [RLT_RL_TOKEN_DIM]
    assert parsed.debug.reference_plan_shape == [5, RLT_ACTION_DIM]
    assert "action_list" not in response
    assert "raw_action_list" not in response
    assert model.last_diffusion_steps == 7
    assert model.last_reference_horizon == 5
    assert model.last_observation is not None
    assert set(model.last_observation.images) == set(OPENPI_IMAGE_KEYS.values())
    assert model.last_observation.images["cam_high"] == b"high-image"
    np.testing.assert_allclose(model.last_observation.state, np.arange(RLT_STATE_DIM, dtype=np.float32))


def test_token_route_returns_token_only_protocol_response():
    model = RecordingFeatureModel()
    server = RLTFeatureServer(
        feature_model=model,
        clock=iter([4.0, 4.1]).__next__,
        feature_id_factory=lambda: "feature-token",
    )

    response = server.dispatch(
        "/rlt/token",
        {
            "request_id": "req-2",
            "episode_id": "ep-2",
            "observation": _make_observation().to_dict(),
        },
    )

    parsed = RLTTokenResponse.from_dict(response)
    assert parsed.feature_id == "feature-token"
    assert parsed.prefix_valid is False
    assert "reference_plan_norm" not in response
    assert "reference_plan_list" not in response
    assert parsed.debug.feature_shape == [RLT_RL_TOKEN_DIM]


def test_hybrid_rtc_feature_model_uses_jax_token_and_rtc_reference_actions():
    token_model = RecordingFeatureModel()
    reference_actions = 1.0 + np.arange(8 * RLT_ACTION_DIM, dtype=np.float32).reshape(8, RLT_ACTION_DIM)
    reference_adapter = _RecordingRTCReferenceAdapter(reference_actions)
    model = HybridRTCRLTFeatureModel(
        token_model=token_model,
        reference_adapter=reference_adapter,
        norm_stats={
            "actions": NormStats(
                mean=np.full((RLT_ACTION_DIM,), 1.0, dtype=np.float32),
                std=np.full((RLT_ACTION_DIM,), 2.0, dtype=np.float32),
                q01=np.full((RLT_ACTION_DIM,), -1.0, dtype=np.float32),
                q99=np.full((RLT_ACTION_DIM,), 1.0, dtype=np.float32),
            )
        },
    )
    observation = RLTFeatureModelObservation.from_contract(_make_observation())

    result = model.infer_features(observation, diffusion_steps=99, reference_horizon=5)

    np.testing.assert_allclose(result.rl_token, np.full((RLT_RL_TOKEN_DIM,), 3.0, dtype=np.float32))
    assert result.prefix_valid is False
    np.testing.assert_allclose(result.reference_plan_list, reference_actions[:5])
    np.testing.assert_allclose(result.reference_plan_norm, (reference_actions[:5] - 1.0) / 2.0, rtol=1e-5)
    assert reference_adapter.last_payload is not None
    assert set(reference_adapter.last_payload["images"]) == {"high", "left_hand", "right_hand"}
    assert reference_adapter.last_payload["action"] == [np.arange(RLT_STATE_DIM, dtype=np.float32).tolist()]


def test_hybrid_rtc_feature_model_rejects_short_reference_plan():
    model = HybridRTCRLTFeatureModel(
        token_model=RecordingFeatureModel(),
        reference_adapter=_RecordingRTCReferenceAdapter(np.zeros((2, RLT_ACTION_DIM), dtype=np.float32)),
        norm_stats={
            "actions": NormStats(
                mean=np.zeros((RLT_ACTION_DIM,), dtype=np.float32),
                std=np.ones((RLT_ACTION_DIM,), dtype=np.float32),
            )
        },
    )
    observation = RLTFeatureModelObservation.from_contract(_make_observation())

    with pytest.raises(ValueError, match="too few actions"):
        model.infer_features(observation, diffusion_steps=1, reference_horizon=3)


def test_status_route_returns_fixed_runtime_contract():
    server = RLTFeatureServer(feature_model=RecordingFeatureModel(), status="ready")

    response = server.dispatch("/rlt/status")

    parsed = RLTStatusResponse.from_dict(response)
    assert parsed.status == "ready"
    assert parsed.server.stage1_feature_backend == "jax"
    assert parsed.server.stage2_checkpoint_owner == "client"
    assert parsed.server.server_runs_learner is False


def test_dispatch_rejects_unknown_route():
    server = RLTFeatureServer(feature_model=RecordingFeatureModel())

    with pytest.raises(KeyError, match="unknown RLT feature route"):
        server.dispatch("/rlt/unknown")


@dataclasses.dataclass(frozen=True)
class _FakeModelConfig:
    use_rlt: bool = True
    rlt_actor_enabled: bool = True
    action_horizon: int = 50
    rlt_env_action_dim: int = 14


@dataclasses.dataclass(frozen=True)
class _FakeTransforms:
    inputs: list = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True)
class _FakeCreatedDataConfig:
    asset_id: str | None = "airbot"
    use_quantile_norm: bool = False
    data_transforms: _FakeTransforms = dataclasses.field(default_factory=_FakeTransforms)


@dataclasses.dataclass(frozen=True)
class _FakeDataConfigFactory:
    require_proprio: bool = True

    def create(self, assets_dirs, model_config):
        del assets_dirs, model_config
        return _FakeCreatedDataConfig(data_transforms=_FakeTransforms())


@dataclasses.dataclass(frozen=True)
class _FakeTrainConfig:
    model: _FakeModelConfig
    data: _FakeDataConfigFactory
    assets_dirs: str = "/tmp/assets"


def _encode_png(rgb: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    return encoded.tobytes()


def test_deferred_jax_feature_model_loads_stage1_policy_and_emits_real_outputs(monkeypatch: pytest.MonkeyPatch):
    created: dict[str, object] = {}
    fake_train_config = _FakeTrainConfig(model=_FakeModelConfig(), data=_FakeDataConfigFactory())

    class _FakePolicyModel:
        def encode_prefix(self, observation):
            created["encoded_observation"] = observation
            return None, None, np.ones((1, 1), dtype=np.bool_), np.ones((1, 3, 4), dtype=np.float32), None

        def compute_rl_token(self, prefix_out, prefix_rlt_mask):
            del prefix_out, prefix_rlt_mask
            return np.full((1, RLT_RL_TOKEN_DIM), 2.0, dtype=np.float32), None

        def sample_reference_actions(self, rng, observation, *, num_steps):
            created["sample_rng"] = rng
            created["sample_observation"] = observation
            created["diffusion_steps"] = num_steps
            return np.full((1, 50, 14), 3.0, dtype=np.float32)

    class _FakePolicy:
        def __init__(self) -> None:
            self.model = _FakePolicyModel()

        def prepare_observation(self, obs: dict):
            created["prepared_inputs"] = obs
            return {"state": np.asarray(obs["state"], dtype=np.float32)}, {"batched": True}

    def _fake_create_trained_policy(train_config, checkpoint_dir, *, default_prompt=None):
        created["train_config"] = train_config
        created["checkpoint_dir"] = checkpoint_dir
        created["default_prompt"] = default_prompt
        return _FakePolicy()

    def _fake_denormalize_action(norm_stats, use_quantiles, actions, *, state=None, use_delta_joint_actions=False):
        created["denorm_norm_stats"] = norm_stats
        created["denorm_use_quantiles"] = use_quantiles
        created["denorm_state"] = np.asarray(state, dtype=np.float32)
        created["denorm_delta"] = use_delta_joint_actions
        return np.asarray(actions, dtype=np.float32) + 5.0

    monkeypatch.setattr(feature_model_module._config, "get_config", lambda name: fake_train_config)
    monkeypatch.setattr(feature_model_module._policy_config, "create_trained_policy", _fake_create_trained_policy)
    monkeypatch.setattr(feature_model_module._checkpoints, "load_norm_stats", lambda *_args, **_kwargs: {"actions": "stats"})
    monkeypatch.setattr(feature_model_module._download, "maybe_download", lambda path: path)
    monkeypatch.setattr(feature_model_module.nnx_utils, "module_jit", lambda fn: fn)
    monkeypatch.setattr(feature_model_module._action_space, "denormalize_action", _fake_denormalize_action)

    model = DeferredJaxRLTFeatureModel(
        config_name="pi05_airbot_rlt",
        checkpoint_dir="/tmp/fake-checkpoint",
        default_prompt="pick up block",
        seed=7,
    )
    observation = RLTFeatureModelObservation(
        images={
            "cam_high": _encode_png(np.full((4, 4, 3), 16, dtype=np.uint8)),
            "cam_left_wrist": _encode_png(np.full((4, 4, 3), 32, dtype=np.uint8)),
            "cam_right_wrist": _encode_png(np.full((4, 4, 3), 48, dtype=np.uint8)),
        },
        state=np.arange(14, dtype=np.float32),
        prompt="transfer block",
        timestamp=1.0,
    )

    result = model.infer_features(observation, diffusion_steps=9, reference_horizon=6)

    assert created["checkpoint_dir"] == model.checkpoint_dir
    assert created["default_prompt"] == "pick up block"
    assert created["train_config"].model.rlt_actor_enabled is False
    assert created["train_config"].data.require_proprio is False
    assert created["diffusion_steps"] == 9
    assert created["prepared_inputs"]["prompt"] == "transfer block"
    assert set(created["prepared_inputs"]["images"]) == {"cam_high", "cam_left_wrist", "cam_right_wrist"}
    assert created["prepared_inputs"]["images"]["cam_high"].shape == (4, 4, 3)
    np.testing.assert_allclose(result.rl_token, np.full((RLT_RL_TOKEN_DIM,), 2.0, dtype=np.float32))
    assert result.reference_plan_norm is not None and result.reference_plan_norm.shape == (6, 14)
    np.testing.assert_allclose(result.reference_plan_norm, np.full((6, 14), 3.0, dtype=np.float32))
    assert result.reference_plan_list is not None and result.reference_plan_list.shape == (6, 14)
    np.testing.assert_allclose(result.reference_plan_list, np.full((6, 14), 8.0, dtype=np.float32))
    np.testing.assert_allclose(created["denorm_state"], np.arange(14, dtype=np.float32))
    assert created["denorm_norm_stats"] == {"actions": "stats"}
    assert created["denorm_use_quantiles"] is False
    assert created["denorm_delta"] is False


def test_deferred_jax_feature_model_validates_reference_horizon(monkeypatch: pytest.MonkeyPatch):
    fake_train_config = _FakeTrainConfig(model=_FakeModelConfig(action_horizon=4), data=_FakeDataConfigFactory())

    class _FakePolicyModel:
        def encode_prefix(self, observation):
            del observation
            return None, None, np.ones((1, 1), dtype=np.bool_), np.ones((1, 1, 1), dtype=np.float32), None

        def compute_rl_token(self, prefix_out, prefix_rlt_mask):
            del prefix_out, prefix_rlt_mask
            return np.zeros((1, RLT_RL_TOKEN_DIM), dtype=np.float32), None

        def sample_reference_actions(self, rng, observation, *, num_steps):
            del rng, observation, num_steps
            return np.zeros((1, 4, 14), dtype=np.float32)

    class _FakePolicy:
        def __init__(self) -> None:
            self.model = _FakePolicyModel()

        def prepare_observation(self, obs: dict):
            return {"state": np.asarray(obs["state"], dtype=np.float32)}, {"batched": True}

    monkeypatch.setattr(feature_model_module._config, "get_config", lambda name: fake_train_config)
    monkeypatch.setattr(feature_model_module._policy_config, "create_trained_policy", lambda *args, **kwargs: _FakePolicy())
    monkeypatch.setattr(feature_model_module._checkpoints, "load_norm_stats", lambda *_args, **_kwargs: {"actions": "stats"})
    monkeypatch.setattr(feature_model_module._download, "maybe_download", lambda path: path)
    monkeypatch.setattr(feature_model_module.nnx_utils, "module_jit", lambda fn: fn)
    monkeypatch.setattr(
        feature_model_module._action_space,
        "denormalize_action",
        lambda *args, **kwargs: np.zeros((4, 14), dtype=np.float32),
    )

    model = DeferredJaxRLTFeatureModel(config_name="pi05_airbot_rlt", checkpoint_dir="/tmp/fake-checkpoint")
    observation = RLTFeatureModelObservation(
        images={
            "cam_high": _encode_png(np.zeros((2, 2, 3), dtype=np.uint8)),
            "cam_left_wrist": _encode_png(np.zeros((2, 2, 3), dtype=np.uint8)),
            "cam_right_wrist": _encode_png(np.zeros((2, 2, 3), dtype=np.uint8)),
        },
        state=np.zeros((14,), dtype=np.float32),
        prompt="test",
        timestamp=0.0,
    )

    with pytest.raises(ValueError, match="exceeds model action horizon"):
        model.infer_features(observation, diffusion_steps=1, reference_horizon=5)
