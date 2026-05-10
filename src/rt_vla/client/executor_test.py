from __future__ import annotations

import numpy as np
import pytest

import rt_vla.client.executor as executor_module
from rt_vla.client.executor import OnDeviceMpcExecutor
from rt_vla.client.executor import RawActionExecutor
from rt_vla.client.rlt_actor_runtime import summarize_execution_records


class _RecordingActuator:
    def __init__(self) -> None:
        self.applied: list[np.ndarray] = []
        self.closed = False

    def apply(self, action: np.ndarray) -> None:
        self.applied.append(np.asarray(action, dtype=np.float32).reshape(-1).copy())

    def close(self) -> None:
        self.closed = True


class _FakeClock:
    def __init__(self, start: float = 100.0) -> None:
        self.now = float(start)
        self.sleep_calls: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, duration: float) -> None:
        duration = float(duration)
        self.sleep_calls.append(duration)
        self.now += duration


def test_execute_training_chunk_returns_lineage_and_per_step_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _FakeClock(start=100.0)
    monkeypatch.setattr(executor_module.time, "time", clock.time)
    monkeypatch.setattr(executor_module.time, "sleep", clock.sleep)
    actuator = _RecordingActuator()
    executor = RawActionExecutor(
        infer_fixed_dims=[1],
        infer_fixed_values=[9.0],
        command_fixed_dims=[2],
        command_fixed_values=[7.0],
        actuator=actuator,
    )

    records = executor.execute_training_chunk(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        plan_id="plan-123",
        control_dt_s=0.05,
    )

    assert [record["plan_id"] for record in records] == ["plan-123", "plan-123"]
    assert [record["chunk_index"] for record in records] == [0, 1]
    assert [record["timestamp"] for record in records] == pytest.approx([100.0, 100.05])
    assert records[0]["raw_action"] == [1.0, 2.0, 3.0]
    assert records[0]["command_action"] == [1.0, 9.0, 3.0]
    assert records[0]["executed_action"] == [1.0, 9.0, 7.0]
    assert records[0]["source"] == "execute_training_chunk"
    assert [item["source"] for item in records[0]["telemetry"]] == [
        "raw_action",
        "command_action",
        "executed_action",
    ]
    assert [item["action"] for item in records[0]["telemetry"]] == [
        [1.0, 2.0, 3.0],
        [1.0, 9.0, 3.0],
        [1.0, 9.0, 7.0],
    ]
    np.testing.assert_allclose(actuator.applied[0], [1.0, 9.0, 7.0])
    np.testing.assert_allclose(actuator.applied[1], [4.0, 9.0, 7.0])
    assert clock.sleep_calls == pytest.approx([0.05])
    assert executor._exec_step_count == 2
    assert executor._last_executed_action == [4.0, 9.0, 6.0]


def test_execute_training_chunk_invokes_step_callbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _FakeClock(start=120.0)
    monkeypatch.setattr(executor_module.time, "time", clock.time)
    monkeypatch.setattr(executor_module.time, "sleep", clock.sleep)
    actuator = _RecordingActuator()
    executor = RawActionExecutor(actuator=actuator)
    before_calls: list[tuple[int, list[float]]] = []
    after_calls: list[tuple[int, list[float]]] = []

    def before_step(index: int, raw_action: list[float]) -> list[float]:
        before_calls.append((index, list(raw_action)))
        if index == 1:
            return [8.0, 8.0]
        return raw_action

    def after_step(index: int, record: dict) -> None:
        after_calls.append((index, list(record["command_action"])))

    records = executor.execute_training_chunk(
        [[1.0, 2.0], [3.0, 4.0]],
        plan_id="callback-plan",
        control_dt_s=0.05,
        before_step_callback=before_step,
        after_step_callback=after_step,
    )

    assert before_calls == [(0, [1.0, 2.0]), (1, [3.0, 4.0])]
    assert after_calls == [(0, [1.0, 2.0]), (1, [8.0, 8.0])]
    assert records[1]["raw_action"] == [8.0, 8.0]
    np.testing.assert_allclose(actuator.applied[1], [8.0, 8.0])


def test_heartbeat_step_behavior_remains_stable_without_servo(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _FakeClock(start=10.0)
    monkeypatch.setattr(executor_module.time, "time", clock.time)
    actuator = _RecordingActuator()
    executor = RawActionExecutor(
        command_fixed_dims=[1],
        command_fixed_values=[8.0],
        actuator=actuator,
    )
    with executor._action_queue_lock:
        executor._action_queue.append([1.0, 2.0])
        executor._raw_action_queue.append([1.0, 2.0])

    interval_s, records = executor.heartbeat_step(current_state=[0.0, 0.0], state_timestamp=9.5)

    assert interval_s == pytest.approx(0.01)
    assert [record["source"] for record in records] == [
        "state",
        "raw_pre_smooth_action",
        "pre_smooth_action",
        "post_smooth_action",
    ]
    assert records[0]["timestamp"] == pytest.approx(9.5)
    assert records[1]["action"] == [1.0, 2.0]
    assert records[2]["action"] == [1.0, 2.0]
    assert records[3]["action"] == [1.0, 8.0]
    np.testing.assert_allclose(actuator.applied[0], [1.0, 8.0])
    assert executor._heartbeat_interval_ms == pytest.approx(10.0)
    assert executor._exec_step_count == 1


def test_ondevice_mpc_training_chunk_records_pre_mpc_replay_attribution(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _FakeClock(start=200.0)
    monkeypatch.setattr(executor_module.time, "time", clock.time)
    monkeypatch.setattr(executor_module.time, "sleep", clock.sleep)
    monkeypatch.setattr(executor_module.time, "monotonic", clock.time)
    monkeypatch.setattr(
        executor_module,
        "AcadosPlanner",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("planner disabled in unit test")),
    )
    actuator = _RecordingActuator()
    executor = OnDeviceMpcExecutor(
        planner_dims=(0, 1),
        state_delay_s=0.0,
        infer_fixed_dims=[1],
        infer_fixed_values=[9.0],
        command_fixed_dims=[2],
        command_fixed_values=[7.0],
        actuator=actuator,
    )

    records = executor.execute_training_chunk(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        plan_id="mpc-plan",
        control_dt_s=0.02,
    )

    assert [record["source"] for record in records] == [
        "execute_training_chunk_mpc",
        "execute_training_chunk_mpc",
    ]
    assert [record["plan_id"] for record in records] == ["mpc-plan", "mpc-plan"]
    assert [record["chunk_index"] for record in records] == [0, 1]
    assert records[0]["raw_action"] == [1.0, 2.0, 3.0]
    assert records[0]["pre_mpc_action"] == [1.0, 9.0, 3.0]
    assert records[0]["post_mpc_action"] == [1.0, 9.0, 3.0]
    assert records[0]["executed_action"] == [1.0, 9.0, 7.0]
    assert records[0]["attribution"] == {
        "replay_action_source": "pre_mpc_action",
        "post_mpc_source": "post_mpc_action",
    }
    assert {item["source"] for item in records[0]["telemetry"]} >= {
        "raw_pre_mpc_action",
        "pre_mpc_action",
        "post_mpc_action",
        "executed_action",
    }
    summary = summarize_execution_records(records)
    np.testing.assert_allclose(summary.executed_actions[0], [1.0, 9.0, 3.0])
    assert summary.action_sources == ["pre_mpc_action", "pre_mpc_action"]
    np.testing.assert_allclose(actuator.applied[0], [1.0, 9.0, 7.0])
    np.testing.assert_allclose(actuator.applied[1], [4.0, 9.0, 7.0])


def test_execution_summary_can_use_post_processed_telemetry_without_top_level_action() -> None:
    summary = summarize_execution_records(
        [
            {
                "timestamp": 1.0,
                "telemetry": [
                    {"source": "pre_mpc_action", "action": [1.0, 2.0]},
                    {"source": "post_mpc_action", "action": [3.0, 4.0]},
                ],
            },
            {
                "timestamp": 2.0,
                "telemetry": [
                    {"source": "pre_smooth_action", "action": [5.0, 6.0]},
                    {"source": "post_smooth_action", "action": [7.0, 8.0]},
                ],
            },
        ]
    )

    np.testing.assert_allclose(summary.executed_actions, [[1.0, 2.0], [5.0, 6.0]])
    assert summary.action_sources == ["pre_mpc_action", "pre_smooth_action"]
