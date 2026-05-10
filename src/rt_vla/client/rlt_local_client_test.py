from __future__ import annotations

import jax
import numpy as np
import pytest

from openpi.rlt import trainer
from openpi.shared.normalize import NormStats
from rt_vla.client.config import ClientConfig
from rt_vla.client.config import Config
from rt_vla.client.config import ExecutorConfig
from rt_vla.client.config import ObserverConfig
from rt_vla.client.config import RLTConfig
from rt_vla.client.config import RLTLearnerConfig
from rt_vla.client.config import RLTObservationConfig
from rt_vla.client.config import RLTServiceConfig
from rt_vla.client.config import VisualizationConfig
from rt_vla.client.rlt_actor_runtime import RLTFeatureClient
import rt_vla.client.rlt_local_client as local_client_module
from rt_vla.client.rlt_local_client import RLTLocalClient
from rt_vla.client.rlt_local_client import validate_plan_window_config
from rt_vla.server.rlt_feature_model import BaseRLTFeatureModel
from rt_vla.server.rlt_feature_model import RLTFeatureModelObservation
from rt_vla.server.rlt_feature_model import RLTFeatureModelOutput
from rt_vla.server.rlt_feature_server import RLTFeatureServer


class _FakeObserver:
    def __init__(self) -> None:
        self._step = 0

    def get_state_observation(self) -> dict:
        self._step += 1
        return {
            "state": np.arange(14, dtype=np.float32) + self._step,
            "joint_velocity": np.ones((14,), dtype=np.float32),
            "timestamp": 12.0 + 0.1 * self._step,
        }

    def get_image_observation(self) -> dict:
        return {
            "images": {
                "high": b"high",
                "left_hand": b"left",
                "right_hand": b"right",
            },
            "timestamp": 12.0,
        }


class _FakeExecutor:
    def execute_training_chunk(
        self,
        action_chunk,
        *,
        plan_id: str,
        control_dt_s: float,
        before_step_callback=None,
        after_step_callback=None,
    ):
        del control_dt_s
        records = []
        for index, action in enumerate(action_chunk):
            if before_step_callback is not None:
                override = before_step_callback(index, list(action))
                if override is not None:
                    action = list(override)
            records.append(
                {
                    "plan_id": plan_id,
                    "chunk_index": index,
                    "timestamp": 20.0 + index * 0.02,
                    "command_action": list(action),
                    "executed_action": list(action),
                    "raw_action": list(action),
                    "source": "execute_training_chunk",
                    "telemetry": [],
                }
            )
            if after_step_callback is not None:
                after_step_callback(index, records[-1])
        return records


class _FakeLearnerRuntime:
    def __init__(self) -> None:
        self.samples = []

    def queue_sample(self, item) -> None:
        self.samples.append(item)

    def check_status(self) -> None:
        return

    def sync_policy_state(self, actor_state, *, apply_actor_params, sync_logger=None):
        del sync_logger
        return apply_actor_params(actor_state, {"step": actor_state.step + 1, "actor_params": actor_state.params})


class _FakeFeatureModel(BaseRLTFeatureModel):
    def infer_features(
        self,
        observation: RLTFeatureModelObservation,
        *,
        diffusion_steps: int,
        reference_horizon: int,
    ) -> RLTFeatureModelOutput:
        del observation, diffusion_steps
        return RLTFeatureModelOutput(
            rl_token=np.full((256,), 2.0, dtype=np.float32),
            prefix_valid=True,
            reference_plan_norm=np.full((reference_horizon, 14), 0.5, dtype=np.float32),
            reference_plan_list=np.full((reference_horizon, 14), 1.5, dtype=np.float32),
        )

    def infer_token(self, observation: RLTFeatureModelObservation) -> RLTFeatureModelOutput:
        del observation
        return RLTFeatureModelOutput(
            rl_token=np.full((256,), 3.0, dtype=np.float32),
            prefix_valid=True,
        )


class _ScriptedFeedbackProvider:
    def __init__(self, feedbacks: list[tuple[float, bool, str | None]]) -> None:
        self._feedbacks = list(feedbacks)
        self.before_episode_calls = 0

    def before_episode(self) -> None:
        self.before_episode_calls += 1

    def override_action(self, policy_action: np.ndarray, current_state: np.ndarray):
        del current_state
        return policy_action, False

    def consume_step_feedback(self) -> tuple[float, bool, str | None]:
        if self._feedbacks:
            return self._feedbacks.pop(0)
        return 0.0, False, None

    def close(self) -> None:
        return


class _InterveningFeedbackProvider:
    def __init__(self) -> None:
        self.override_calls = 0

    def before_episode(self) -> None:
        return

    def override_action(self, policy_action: np.ndarray, current_state: np.ndarray):
        del current_state
        self.override_calls += 1
        if self.override_calls == 2:
            return np.full_like(policy_action, 99.0), True
        return policy_action, False

    def consume_step_feedback(self) -> tuple[float, bool, str | None]:
        return 2.0, True, "operator_terminate"

    def close(self) -> None:
        return


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


def test_validate_plan_window_config_rejects_plan_that_cannot_cover_replan_window() -> None:
    with pytest.raises(ValueError, match="reference_horizon is too small"):
        validate_plan_window_config(
            action_horizon=10,
            chunk_stride=10,
            reference_horizon=19,
            replan_horizon=20,
        )


def test_validate_plan_window_config_allows_subchunk_stride_when_plan_covers_window() -> None:
    validate_plan_window_config(
        action_horizon=10,
        chunk_stride=2,
        reference_horizon=50,
        replan_horizon=10,
    )


def test_local_client_collects_and_requests_plan() -> None:
    def transport(path: str, payload: dict | None):
        if path == "/rlt/infer":
            return {
                "request_id": payload["request_id"],
                "feature_id": "feature-1",
                "rl_token": [2.0] * 256,
                "reference_plan_norm": [[0.0] * 14 for _ in range(5)],
                "reference_plan_list": [[1.0] * 14 for _ in range(5)],
                "prefix_valid": True,
                "server_infer_time_s": 0.01,
                "debug": {
                    "rl_token_norm": 32.0,
                    "reference_plan_norm": 0.0,
                    "feature_shape": [256],
                    "reference_plan_shape": [5, 14],
                },
            }
        raise AssertionError(f"unexpected path {path}")

    client = RLTLocalClient(
        config=_make_config(),
        feature_client=RLTFeatureClient(transport=transport, request_id_factory=lambda: "req-1"),
        observer=_FakeObserver(),
    )

    collected, plan = client.request_plan(episode_id="episode-1")

    assert collected.prompt == "pick up block"
    assert collected.contract_observation.prompt == "pick up block"
    assert collected.rlt_state.shape == (284,)
    np.testing.assert_allclose(collected.rlt_state[:256], 2.0)
    assert plan.feature_id == "feature-1"
    assert plan.contains_chunk(step=0, action_horizon=5)


def test_local_client_requests_token() -> None:
    def transport(path: str, payload: dict | None):
        if path == "/rlt/token":
            return {
                "request_id": payload["request_id"],
                "feature_id": "feature-2",
                "rl_token": [3.0] * 256,
                "prefix_valid": False,
                "server_token_time_s": 0.02,
                "debug": {
                    "rl_token_norm": 48.0,
                    "feature_shape": [256],
                },
            }
        raise AssertionError(f"unexpected path {path}")

    client = RLTLocalClient(
        config=_make_config(),
        feature_client=RLTFeatureClient(transport=transport, request_id_factory=lambda: "req-2"),
        observer=_FakeObserver(),
    )

    collected, response = client.request_token(episode_id="episode-1", prompt="override")

    assert collected.prompt == "override"
    assert response.feature_id == "feature-2"
    np.testing.assert_allclose(collected.rlt_state[:256], 3.0)


def test_local_client_executes_single_training_chunk_and_builds_replay_item() -> None:
    def transport(path: str, payload: dict | None):
        if path == "/rlt/infer":
            return {
                "request_id": payload["request_id"],
                "feature_id": "feature-1",
                "rl_token": [2.0] * 256,
                "reference_plan_norm": [[0.1 * (row + col) for col in range(14)] for row in range(5)],
                "reference_plan_list": [[0.2 * (row + col) for col in range(14)] for row in range(5)],
                "prefix_valid": True,
                "server_infer_time_s": 0.01,
                "debug": {
                    "rl_token_norm": 32.0,
                    "reference_plan_norm": 1.0,
                    "feature_shape": [256],
                    "reference_plan_shape": [5, 14],
                },
            }
        if path == "/rlt/token":
            return {
                "request_id": payload["request_id"],
                "feature_id": "feature-2",
                "rl_token": [3.0] * 256,
                "prefix_valid": True,
                "server_token_time_s": 0.02,
                "debug": {
                    "rl_token_norm": 48.0,
                    "feature_shape": [256],
                },
            }
        raise AssertionError(f"unexpected path {path}")

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
    client = RLTLocalClient(
        config=_make_config(),
        feature_client=RLTFeatureClient(transport=transport, request_id_factory=lambda: "req-train"),
        observer=_FakeObserver(),
    )

    result = client.execute_training_chunk_once(
        episode_id="episode-1",
        actor_model=actor_model,
        actor_state=actor_state,
        rng=jax.random.key(1),
        executor=_FakeExecutor(),
        action_horizon=2,
        control_dt_s=0.02,
        norm_stats=norm_stats,
        use_quantiles=False,
        use_delta_joint_actions=False,
        reward_history=[1.0, 0.5],
        env_step=2,
    )

    assert result.plan.feature_id == "feature-1"
    assert result.execution.executed_actions.shape == (2, 14)
    assert result.replay_item.state.shape == (284,)
    assert result.replay_item.next_state.shape == (284,)
    assert result.replay_item.action.shape == (28,)
    assert result.replay_item.reference_action.shape == (28,)
    assert result.replay_item.next_reference_action.shape == (28,)
    assert result.replay_item.bootstrap_steps == 2
    assert result.replay_item.reward == 1.5


def test_training_chunk_denormalizes_actions_and_records_intervention(monkeypatch: pytest.MonkeyPatch) -> None:
    def transport(path: str, payload: dict | None):
        if path == "/rlt/infer":
            return {
                "request_id": payload["request_id"],
                "feature_id": "feature-intervene",
                "rl_token": [2.0] * 256,
                "reference_plan_norm": [[0.0] * 14 for _ in range(5)],
                "reference_plan_list": [[10.0] * 14 for _ in range(5)],
                "prefix_valid": True,
                "server_infer_time_s": 0.01,
                "debug": {
                    "rl_token_norm": 32.0,
                    "reference_plan_norm": 0.0,
                    "feature_shape": [256],
                    "reference_plan_shape": [5, 14],
                },
            }
        if path == "/rlt/token":
            return {
                "request_id": payload["request_id"],
                "feature_id": "feature-token",
                "rl_token": [3.0] * 256,
                "prefix_valid": True,
                "server_token_time_s": 0.02,
                "debug": {
                    "rl_token_norm": 48.0,
                    "feature_shape": [256],
                },
            }
        raise AssertionError(f"unexpected path {path}")

    def fake_actor_infer_action_chunk(*_args, **_kwargs):
        return np.stack(
            [
                np.ones((14,), dtype=np.float32),
                np.full((14,), 2.0, dtype=np.float32),
            ],
            axis=0,
        ).reshape(-1)

    monkeypatch.setattr(local_client_module, "actor_infer_action_chunk", fake_actor_infer_action_chunk)
    norm_stats = {
        "actions": NormStats(
            mean=np.full((14,), 10.0, dtype=np.float32),
            std=np.full((14,), 2.0, dtype=np.float32),
            q01=np.full((14,), -1.0, dtype=np.float32),
            q99=np.full((14,), 1.0, dtype=np.float32),
        )
    }
    client = RLTLocalClient(
        config=_make_config(),
        feature_client=RLTFeatureClient(transport=transport, request_id_factory=lambda: "req-intervene"),
        observer=_FakeObserver(),
    )

    result = client.execute_training_chunk_once(
        episode_id="episode-1",
        actor_model=object(),
        actor_state=object(),
        rng=jax.random.key(6),
        executor=_FakeExecutor(),
        action_horizon=2,
        control_dt_s=0.02,
        norm_stats=norm_stats,
        use_quantiles=False,
        use_delta_joint_actions=False,
        env_step=0,
        feedback_provider=_InterveningFeedbackProvider(),
    )

    np.testing.assert_allclose(result.command_action_chunk[0], 12.0)
    np.testing.assert_allclose(result.command_action_chunk[1], 99.0)
    np.testing.assert_array_equal(result.execution.intervened_mask, [False, True])
    assert result.replay_item.done is True
    assert result.terminal_reason == "operator_terminate"
    assert result.replay_item.reward == 4.0
    action = result.replay_item.action.reshape(2, 14)
    reference_action = result.replay_item.reference_action.reshape(2, 14)
    np.testing.assert_allclose(action[0], 1.0)
    np.testing.assert_allclose(action[1], 44.5, rtol=1e-5)
    np.testing.assert_allclose(reference_action[0], 0.0)
    np.testing.assert_allclose(reference_action[1], action[1])


def test_training_chunk_consumes_feedback_reward_per_executed_step() -> None:
    server = RLTFeatureServer(feature_model=_FakeFeatureModel(), feature_id_factory=lambda: "feature-reward")
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
    client = RLTLocalClient(
        config=_make_config(),
        feature_client=RLTFeatureClient(transport=server.dispatch, request_id_factory=lambda: "req-reward"),
        observer=_FakeObserver(),
    )

    result = client.execute_training_chunk_once(
        episode_id="episode-1",
        actor_model=actor_model,
        actor_state=actor_state,
        rng=jax.random.key(7),
        executor=_FakeExecutor(),
        action_horizon=2,
        control_dt_s=0.02,
        norm_stats=norm_stats,
        use_quantiles=False,
        use_delta_joint_actions=False,
        env_step=0,
        feedback_provider=_ScriptedFeedbackProvider(
            [
                (1.0, False, None),
                (3.0, True, "operator_terminate"),
            ]
        ),
    )

    assert result.replay_item.reward == 4.0
    assert result.replay_item.done is True
    assert result.terminal_reason == "operator_terminate"


def test_local_client_runs_training_iteration_and_queues_replay() -> None:
    def transport(path: str, payload: dict | None):
        if path == "/rlt/infer":
            return {
                "request_id": payload["request_id"],
                "feature_id": "feature-1",
                "rl_token": [2.0] * 256,
                "reference_plan_norm": [[0.1 * (row + col) for col in range(14)] for row in range(5)],
                "reference_plan_list": [[0.2 * (row + col) for col in range(14)] for row in range(5)],
                "prefix_valid": True,
                "server_infer_time_s": 0.01,
                "debug": {
                    "rl_token_norm": 32.0,
                    "reference_plan_norm": 1.0,
                    "feature_shape": [256],
                    "reference_plan_shape": [5, 14],
                },
            }
        if path == "/rlt/token":
            return {
                "request_id": payload["request_id"],
                "feature_id": "feature-2",
                "rl_token": [3.0] * 256,
                "prefix_valid": True,
                "server_token_time_s": 0.02,
                "debug": {
                    "rl_token_norm": 48.0,
                    "feature_shape": [256],
                },
            }
        raise AssertionError(f"unexpected path {path}")

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
    learner_runtime = _FakeLearnerRuntime()
    client = RLTLocalClient(
        config=_make_config(),
        feature_client=RLTFeatureClient(transport=transport, request_id_factory=lambda: "req-iter"),
        observer=_FakeObserver(),
    )

    result = client.run_training_iteration(
        episode_id="episode-1",
        actor_model=actor_model,
        actor_state=actor_state,
        rng=jax.random.key(2),
        executor=_FakeExecutor(),
        learner_runtime=learner_runtime,
        action_horizon=2,
        control_dt_s=0.02,
        norm_stats=norm_stats,
        use_quantiles=False,
        use_delta_joint_actions=False,
        reward_history=[1.0, 0.5],
        env_step=2,
        apply_actor_params=trainer.apply_actor_params,
    )

    assert len(learner_runtime.samples) == 1
    assert learner_runtime.samples[0] == result.chunk.replay_item
    assert result.actor_state.step == actor_state.step + 1


def test_local_client_training_iteration_with_feature_server_dispatch() -> None:
    server = RLTFeatureServer(feature_model=_FakeFeatureModel(), feature_id_factory=lambda: "feature-server")
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
    learner_runtime = _FakeLearnerRuntime()
    client = RLTLocalClient(
        config=_make_config(),
        feature_client=RLTFeatureClient(transport=server.dispatch, request_id_factory=lambda: "req-e2e"),
        observer=_FakeObserver(),
    )

    result = client.run_training_iteration(
        episode_id="episode-1",
        actor_model=actor_model,
        actor_state=actor_state,
        rng=jax.random.key(3),
        executor=_FakeExecutor(),
        learner_runtime=learner_runtime,
        action_horizon=2,
        control_dt_s=0.02,
        norm_stats=norm_stats,
        use_quantiles=False,
        use_delta_joint_actions=False,
        reward_history=[0.25, 0.75],
        env_step=2,
        apply_actor_params=trainer.apply_actor_params,
    )

    assert result.chunk.plan.feature_id == "feature-server"
    assert result.chunk.token_response.feature_id == "feature-server"
    assert len(learner_runtime.samples) == 1
    assert result.chunk.replay_item.reward == 1.0


def test_local_client_runs_training_loop_until_feedback_terminate() -> None:
    server = RLTFeatureServer(feature_model=_FakeFeatureModel(), feature_id_factory=lambda: "feature-loop")
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
    learner_runtime = _FakeLearnerRuntime()
    feedback_provider = _ScriptedFeedbackProvider(
        [
            (0.0, False, None),
            (1.0, True, "operator_terminate"),
            (0.0, False, None),
        ]
    )
    client = RLTLocalClient(
        config=_make_config(),
        feature_client=RLTFeatureClient(transport=server.dispatch, request_id_factory=lambda: "req-loop"),
        observer=_FakeObserver(),
    )

    result = client.run_training_loop(
        episode_id="episode-1",
        actor_model=actor_model,
        actor_state=actor_state,
        rng=jax.random.key(4),
        executor=_FakeExecutor(),
        learner_runtime=learner_runtime,
        feedback_provider=feedback_provider,
        num_iterations=5,
        action_horizon=2,
        control_dt_s=0.02,
        norm_stats=norm_stats,
        use_quantiles=False,
        use_delta_joint_actions=False,
        apply_actor_params=trainer.apply_actor_params,
    )

    assert feedback_provider.before_episode_calls == 1
    assert len(result.iterations) == 1
    assert len(learner_runtime.samples) == 1
    assert result.iterations[0].chunk.replay_item.reward == 1.0
    assert result.iterations[0].chunk.replay_item.done is True
    assert result.terminal_reason == "operator_terminate"
    assert result.final_env_step == 2
    assert result.actor_state.step == actor_state.step + 1


def test_training_loop_emits_stride_aligned_overlapping_replay_items() -> None:
    server = RLTFeatureServer(feature_model=_FakeFeatureModel(), feature_id_factory=lambda: "feature-stride")
    actor_model, actor_state = trainer.init_actor_state(
        jax.random.key(0),
        state_dim=284,
        action_dim=56,
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
    cfg = _make_config()
    cfg.rlt.observation.reference_horizon = 8
    cfg.rlt.observation.action_horizon = 4
    cfg.rlt.observation.chunk_stride = 2
    cfg.rlt.observation.vla_replan_horizon_scale = 1
    learner_runtime = _FakeLearnerRuntime()
    client = RLTLocalClient(
        config=cfg,
        feature_client=RLTFeatureClient(transport=server.dispatch, request_id_factory=lambda: "req-stride"),
        observer=_FakeObserver(),
    )

    result = client.run_training_loop(
        episode_id="episode-1",
        actor_model=actor_model,
        actor_state=actor_state,
        rng=jax.random.key(8),
        executor=_FakeExecutor(),
        learner_runtime=learner_runtime,
        feedback_provider=_ScriptedFeedbackProvider([(0.0, False, None)] * 8),
        num_iterations=2,
        action_horizon=4,
        control_dt_s=0.02,
        norm_stats=norm_stats,
        use_quantiles=False,
        use_delta_joint_actions=False,
        apply_actor_params=trainer.apply_actor_params,
    )

    assert result.final_env_step == 8
    assert [item.env_step for item in learner_runtime.samples] == [0, 2, 4]
    assert result.iterations[0].chunk.emitted_env_steps == (0,)
    assert result.iterations[1].chunk.emitted_env_steps == (2, 4)
    assert [item.bootstrap_steps for item in learner_runtime.samples] == [4, 4, 4]


def test_training_loop_reuses_plan_until_replan_boundary() -> None:
    calls: list[str] = []
    infer_count = 0

    def transport(path: str, payload: dict | None):
        nonlocal infer_count
        calls.append(path)
        if path == "/rlt/infer":
            base = 100.0 * infer_count
            feature_id = f"feature-{infer_count}"
            infer_count += 1
            reference_horizon = payload["reference_horizon"]
            return {
                "request_id": payload["request_id"],
                "feature_id": feature_id,
                "rl_token": [2.0 + base] * 256,
                "reference_plan_norm": [
                    [base + 0.1 * row + 0.01 * col for col in range(14)]
                    for row in range(reference_horizon)
                ],
                "reference_plan_list": [
                    [base + row + 0.01 * col for col in range(14)]
                    for row in range(reference_horizon)
                ],
                "prefix_valid": True,
                "server_infer_time_s": 0.01,
                "debug": {
                    "rl_token_norm": 32.0,
                    "reference_plan_norm": 1.0,
                    "feature_shape": [256],
                    "reference_plan_shape": [reference_horizon, 14],
                },
            }
        if path == "/rlt/token":
            return {
                "request_id": payload["request_id"],
                "feature_id": "token-feature",
                "rl_token": [3.0] * 256,
                "prefix_valid": True,
                "server_token_time_s": 0.02,
                "debug": {
                    "rl_token_norm": 48.0,
                    "feature_shape": [256],
                },
            }
        raise AssertionError(f"unexpected path {path}")

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
    client = RLTLocalClient(
        config=_make_config(),
        feature_client=RLTFeatureClient(transport=transport, request_id_factory=lambda: "req-plan"),
        observer=_FakeObserver(),
    )

    result = client.run_training_loop(
        episode_id="episode-1",
        actor_model=actor_model,
        actor_state=actor_state,
        rng=jax.random.key(5),
        executor=_FakeExecutor(),
        learner_runtime=_FakeLearnerRuntime(),
        feedback_provider=_ScriptedFeedbackProvider([(0.0, False, None)] * 3),
        num_iterations=3,
        action_horizon=2,
        control_dt_s=0.02,
        norm_stats=norm_stats,
        use_quantiles=False,
        use_delta_joint_actions=False,
        apply_actor_params=trainer.apply_actor_params,
    )

    assert calls == ["/rlt/infer", "/rlt/token", "/rlt/infer", "/rlt/token"]
    assert [iteration.chunk.plan.feature_id for iteration in result.iterations] == [
        "feature-0",
        "feature-0",
        "feature-1",
    ]
    np.testing.assert_allclose(
        result.iterations[1].chunk.replay_item.reference_action.reshape(2, 14)[:, 0],
        [2.0, 3.0],
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        result.iterations[1].chunk.replay_item.next_reference_action.reshape(2, 14)[:, 0],
        [100.0, 101.0],
        rtol=1e-5,
    )
