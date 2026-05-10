from __future__ import annotations

import numpy as np

from openpi.rlt import RLTStatusResponse
from rt_vla.client.rlt_actor_runtime import PlanWindow
from rt_vla.client.rlt_actor_runtime import RLTFeatureClient
from rt_vla.client.rlt_actor_runtime import build_contract_observation
from rt_vla.client.rlt_actor_runtime import build_rlt_state
from rt_vla.client.rlt_actor_runtime import should_request_new_plan


def test_build_rlt_state_concatenates_token_and_proprio() -> None:
    state = build_rlt_state(np.ones((256,), dtype=np.float32), np.full((28,), 2.0, dtype=np.float32))

    assert state.shape == (284,)
    np.testing.assert_allclose(state[:256], 1.0)
    np.testing.assert_allclose(state[256:], 2.0)


def test_plan_window_slices_chunks_and_detects_coverage() -> None:
    plan = PlanWindow(
        feature_id="feature-1",
        plan_start_step=10,
        rl_token=np.zeros((256,), dtype=np.float32),
        reference_plan_norm=np.arange(50 * 14, dtype=np.float32).reshape(50, 14),
        reference_plan_real=np.arange(50 * 14, dtype=np.float32).reshape(50, 14) + 1000.0,
        prefix_valid=True,
    )

    assert plan.contains_chunk(step=12, action_horizon=10)
    assert not plan.contains_chunk(step=45, action_horizon=20)
    np.testing.assert_allclose(
        plan.slice_norm_chunk(step=12, action_horizon=2),
        np.arange(2 * 14, 4 * 14, dtype=np.float32),
    )


def test_should_request_new_plan_checks_coverage_and_replan_boundary() -> None:
    plan = PlanWindow(
        feature_id="feature-1",
        plan_start_step=0,
        rl_token=np.zeros((256,), dtype=np.float32),
        reference_plan_norm=np.zeros((50, 14), dtype=np.float32),
        reference_plan_real=np.zeros((50, 14), dtype=np.float32),
        prefix_valid=True,
    )

    assert should_request_new_plan(step=0, action_horizon=10, replan_horizon=50, current_plan=None)
    assert not should_request_new_plan(step=0, action_horizon=10, replan_horizon=50, current_plan=plan)
    assert should_request_new_plan(step=50, action_horizon=10, replan_horizon=50, current_plan=plan)
    assert not should_request_new_plan(step=11, action_horizon=10, replan_horizon=50, current_plan=plan)


def test_feature_client_uses_contract_roundtrip() -> None:
    calls: list[tuple[str, dict | None]] = []

    def transport(path: str, payload: dict | None):
        calls.append((path, payload))
        if path == "/rlt/infer":
            return {
                "request_id": payload["request_id"],
                "feature_id": "feature-1",
                "rl_token": [0.0] * 256,
                "reference_plan_norm": [[0.0] * 14 for _ in range(5)],
                "reference_plan_list": [[1.0] * 14 for _ in range(5)],
                "prefix_valid": True,
                "server_infer_time_s": 0.01,
                "debug": {
                    "rl_token_norm": 0.0,
                    "reference_plan_norm": 0.0,
                    "feature_shape": [256],
                    "reference_plan_shape": [5, 14],
                },
            }
        if path == "/rlt/token":
            return {
                "request_id": payload["request_id"],
                "feature_id": "feature-2",
                "rl_token": [1.0] * 256,
                "prefix_valid": False,
                "server_token_time_s": 0.02,
                "debug": {
                    "rl_token_norm": 16.0,
                    "feature_shape": [256],
                },
            }
        return RLTStatusResponse(status="ok").to_dict()

    client = RLTFeatureClient(transport=transport, request_id_factory=lambda: "req-1")
    observation = build_contract_observation(
        images={"high": b"h", "left_hand": b"l", "right_hand": b"r"},
        state=np.zeros((14,), dtype=np.float32),
        prompt="pick up block",
        timestamp=1.0,
    )

    infer_response = client.infer(
        episode_id="episode-1",
        observation=observation,
        diffusion_steps=7,
        reference_horizon=5,
    )
    token_response = client.token(episode_id="episode-1", observation=observation)
    status_response = client.status()

    assert infer_response.feature_id == "feature-1"
    assert token_response.feature_id == "feature-2"
    assert status_response.status == "ok"
    assert [path for path, _ in calls] == ["/rlt/infer", "/rlt/token", "/rlt/status"]
