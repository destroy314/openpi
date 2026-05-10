from __future__ import annotations

from types import SimpleNamespace
import threading

import jax
import numpy as np
import pytest

from openpi.rlt import trainer
from openpi.shared.normalize import NormStats
from rt_vla.client.config import ClientConfig
from rt_vla.client.config import Config
from rt_vla.client.config import ExecutorConfig
from rt_vla.client.config import FeedbackConfig
from rt_vla.client.config import ObserverConfig
from rt_vla.client.config import RLTConfig
from rt_vla.client.config import RLTLearnerConfig
from rt_vla.client.config import RLTObservationConfig
from rt_vla.client.config import RLTServiceConfig
from rt_vla.client.config import VisualizationConfig
from rt_vla.client.rlt_actor_runtime import RLTFeatureClient
from rt_vla.client.rlt_local_client import TrainingLoopResult
from rt_vla.client.rlt_local_client_test import _FakeExecutor
from rt_vla.client.rlt_local_client_test import _FakeFeatureModel
from rt_vla.client.rlt_local_client_test import _FakeLearnerRuntime
from rt_vla.client.rlt_local_client_test import _FakeObserver
from rt_vla.client.rlt_training_runtime import RLTTrainingRuntime
from rt_vla.server.rlt_feature_server import RLTFeatureServer


def _make_config() -> Config:
    return Config(
        client=ClientConfig(
            infer_url="http://127.0.0.1:8000",
            endpoint="/infer",
            timeout_s=1.0,
            run_duration_s=1.0,
        ),
        observer=ObserverConfig(
            name="mock",
            image_size=(224, 224),
            fps=30,
            state_dim=14,
            airbot_host="localhost",
            airbot_left_port=50051,
            airbot_right_port=50053,
            top_camera_id="top",
            left_camera_id="left",
            right_camera_id="right",
            enable_cameras=False,
        ),
        executor=ExecutorConfig(name="raw_action"),
        visualization=VisualizationConfig(output_dir="/tmp/rt-vla", enable_recording=False),
        feedback=FeedbackConfig(name="noop"),
        rlt=RLTConfig(
            service=RLTServiceConfig(),
            observation=RLTObservationConfig(
                prompt="pick up block",
                velocity_source="sdk",
                diffusion_steps=7,
                reference_horizon=5,
                action_horizon=2,
                chunk_stride=2,
                vla_replan_horizon_scale=2,
            ),
            learner=RLTLearnerConfig(),
        ),
    )


def test_training_runtime_runs_and_updates_state() -> None:
    server = RLTFeatureServer(feature_model=_FakeFeatureModel(), feature_id_factory=lambda: "feature-runtime")
    actor_model, actor_state = trainer.init_actor_state(
        jax.random.key(0),
        state_dim=284,
        action_dim=28,
        hidden_dim=8,
        learning_rate=1e-3,
    )
    norm_stats = {
        "actions": NormStats(
            mean=np.zeros((14,), dtype=np.float32),
            std=np.ones((14,), dtype=np.float32),
            q01=np.full((14,), -1.0, dtype=np.float32),
            q99=np.full((14,), 1.0, dtype=np.float32),
        )
    }
    runtime = RLTTrainingRuntime(
        config=_make_config(),
        local_client=__import__("rt_vla.client.rlt_local_client", fromlist=["RLTLocalClient"]).RLTLocalClient(
            config=_make_config(),
            feature_client=RLTFeatureClient(transport=server.dispatch, request_id_factory=lambda: "req-runtime"),
            observer=_FakeObserver(),
        ),
        actor_model=actor_model,
        actor_state=actor_state,
        executor=_FakeExecutor(),
        learner_runtime=_FakeLearnerRuntime(),
        feedback_provider=__import__(
            "rt_vla.client.feedback", fromlist=["NoopFeedbackProvider"]
        ).NoopFeedbackProvider(),
        norm_stats=norm_stats,
        use_quantiles=False,
        use_delta_joint_actions=False,
        action_horizon=2,
        control_dt_s=0.02,
        apply_actor_params=trainer.apply_actor_params,
        prompt="pick up block",
    )

    result = runtime.run(num_iterations=2, rng=jax.random.key(5))

    assert len(result.iterations) == 2
    assert runtime.actor_state.step == actor_state.step + 2
    assert runtime.env_step == 4
    assert runtime.status()["actor_step"] == actor_state.step + 2
    assert runtime.status()["env_step"] == 4
    assert runtime.status()["actor_collector"]["started"] is True
    assert runtime.status()["actor_collector"]["running"] is False


def test_training_runtime_start_exposes_running_actor_collector_thread() -> None:
    started = threading.Event()
    release = threading.Event()
    actor_state = SimpleNamespace(step=3)

    class _BlockingLocalClient:
        observer = None

        def run_training_loop(self, **kwargs):
            assert kwargs["initial_env_step"] == 5
            started.set()
            if not release.wait(timeout=5.0):
                raise RuntimeError("timed out waiting for test release")
            return TrainingLoopResult(iterations=[], actor_state=actor_state, final_env_step=7)

    runtime = RLTTrainingRuntime(
        config=_make_config(),
        local_client=_BlockingLocalClient(),
        actor_model=object(),
        actor_state=actor_state,
        executor=_FakeExecutor(),
        learner_runtime=_FakeLearnerRuntime(),
        feedback_provider=object(),
        norm_stats={},
        use_quantiles=False,
        use_delta_joint_actions=False,
        action_horizon=1,
        control_dt_s=0.02,
        apply_actor_params=lambda state, params: state,
        env_step=5,
    )

    runtime.start(num_iterations=1, rng=jax.random.key(9))
    assert started.wait(timeout=2.0)
    status = runtime.status()
    assert status["actor_collector"]["started"] is True
    assert status["actor_collector"]["running"] is True
    assert status["actor_collector"]["thread_name"] == "rlt-actor-collector"

    release.set()
    result = runtime.join(timeout_s=5.0)

    assert result is not None
    assert runtime.env_step == 7
    assert runtime.status()["actor_collector"]["running"] is False


def test_build_feedback_provider_supports_noop() -> None:
    from rt_vla.client.builders import build_feedback_provider

    provider = build_feedback_provider(_make_config())

    reward, terminate, reason = provider.consume_step_feedback()
    assert reward == 0.0
    assert not terminate
    assert reason is None


def test_build_feedback_provider_rejects_leader_override_without_keyboard() -> None:
    from rt_vla.client.builders import build_feedback_provider

    cfg = _make_config()
    cfg.feedback = FeedbackConfig(name="noop", enable_leader_override=True)

    with pytest.raises(ValueError, match="requires feedback.name='keyboard'"):
        build_feedback_provider(cfg)


def test_build_feedback_provider_wires_keyboard_leader_override(monkeypatch) -> None:
    from rt_vla.client import builders

    class _FakeLeaderOverride:
        def __init__(self, *, left_leader_port: int, right_leader_port: int) -> None:
            self.left_leader_port = left_leader_port
            self.right_leader_port = right_leader_port

    class _FakeKeyboardProvider:
        def __init__(self, *, config, action_override_callback=None) -> None:
            self.config = config
            self.action_override_callback = action_override_callback

    monkeypatch.setattr(builders, "LeaderArmActionOverride", _FakeLeaderOverride)
    monkeypatch.setattr(builders, "KeyboardFeedbackProvider", _FakeKeyboardProvider)

    cfg = _make_config()
    cfg.feedback = FeedbackConfig(
        name="keyboard",
        enable_leader_override=True,
        left_leader_port=50050,
        right_leader_port=50052,
    )

    provider = builders.build_feedback_provider(cfg)

    assert provider.config.intervention_toggle_key == "s"
    assert provider.action_override_callback.left_leader_port == 50050
    assert provider.action_override_callback.right_leader_port == 50052
