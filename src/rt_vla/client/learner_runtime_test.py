from __future__ import annotations

import queue
import time

import pytest

from openpi.rlt.rtvla_contract import LearnerError
from openpi.rlt.rtvla_contract import LearnerInit
from openpi.rlt.rtvla_contract import LearnerStats
from openpi.rlt.rtvla_contract import StopSignal
from rt_vla.client.learner_runtime import LearnerProcessConfig
from rt_vla.client.learner_runtime import LearnerProcessHandle
from rt_vla.client.learner_runtime import publish_policy_update


def _worker_with_policy_updates(
    expected_final_step: int,
    sample_queue,
    policy_queue,
    status_queue,
) -> None:
    publish_policy_update(policy_queue, {"step": 12})
    publish_policy_update(policy_queue, {"step": 13})
    status_queue.put(LearnerInit(start_step=3, actor_params={"step": 11}))
    message = sample_queue.get(timeout=5.0)
    if not isinstance(message, StopSignal) or message.final_step != expected_final_step:
        status_queue.put(
            LearnerError(
                message=f"unexpected stop signal: {message!r}",
                traceback="",
            )
        )


def _worker_with_stats(sample_queue, policy_queue, status_queue) -> None:
    del policy_queue
    status_queue.put(LearnerInit(start_step=0, actor_params={"step": 0}))
    status_queue.put(LearnerStats(learner_step=2, env_step=4, replay_size=3, latest_checkpoint="/tmp/4"))
    message = sample_queue.get(timeout=5.0)
    if not isinstance(message, StopSignal):
        status_queue.put(LearnerError(message=f"unexpected message: {message!r}", traceback=""))


def _worker_with_async_error(sample_queue, policy_queue, status_queue) -> None:
    del policy_queue
    status_queue.put(LearnerInit(start_step=0, actor_params={"step": 0}))
    status_queue.put(LearnerError(message="async boom", traceback="traceback"))
    message = sample_queue.get(timeout=5.0)
    if not isinstance(message, StopSignal):
        status_queue.put(LearnerError(message=f"unexpected message: {message!r}", traceback=""))


def _worker_error_before_init(sample_queue, policy_queue, status_queue) -> None:
    del sample_queue, policy_queue
    status_queue.put(LearnerError(message="init boom", traceback="traceback"))


def _worker_crashes_before_init(sample_queue, policy_queue, status_queue) -> None:
    del sample_queue, policy_queue, status_queue
    raise SystemExit(7)


def _worker_crashes_after_init(sample_queue, policy_queue, status_queue) -> None:
    del sample_queue, policy_queue
    status_queue.put(LearnerInit(start_step=0, actor_params={"step": 0}))
    raise SystemExit(7)


def test_publish_policy_update_replaces_stale_message() -> None:
    policy_queue: queue.Queue = queue.Queue(maxsize=1)

    publish_policy_update(policy_queue, {"step": 1})
    publish_policy_update(policy_queue, {"step": 2})

    message = policy_queue.get_nowait()
    assert message.actor_params["step"] == 2
    with pytest.raises(queue.Empty):
        policy_queue.get_nowait()


def test_learner_process_handle_syncs_latest_policy_update_and_shutdowns() -> None:
    learner_runtime = LearnerProcessHandle.spawn(
        target=_worker_with_policy_updates,
        target_args=(9,),
        config=LearnerProcessConfig(sample_queue_size=1, process_name="learner-runtime-test"),
    )

    init_message = learner_runtime.wait_for_init()
    assert init_message.start_step == 3
    assert init_message.actor_params["step"] == 11

    updated_state = learner_runtime.sync_policy_state(
        {"step": -1},
        apply_actor_params=lambda _state, actor_params: dict(actor_params),
    )
    assert updated_state["step"] == 13

    graceful_stop_sent = learner_runtime.shutdown(final_step=9, graceful_stop_sent=False, join_timeout_s=5.0)
    assert graceful_stop_sent
    assert learner_runtime.exitcode == 0


def test_learner_process_handle_tracks_latest_stats() -> None:
    learner_runtime = LearnerProcessHandle.spawn(
        target=_worker_with_stats,
        target_args=(),
        config=LearnerProcessConfig(sample_queue_size=1, process_name="learner-runtime-test"),
    )

    try:
        init_message = learner_runtime.wait_for_init()
        assert init_message.start_step == 0
        stats = None
        deadline = time.time() + 5.0
        while stats is None and time.time() < deadline:
            stats = learner_runtime.latest_stats()
            time.sleep(0.01)
        assert stats is not None
        assert stats.to_dict() == {
            "learner_step": 2,
            "env_step": 4,
            "replay_size": 3,
            "latest_checkpoint": "/tmp/4",
        }
    finally:
        learner_runtime.shutdown(final_step=4, graceful_stop_sent=False, join_timeout_s=5.0)


def test_learner_process_handle_surfaces_async_worker_error() -> None:
    learner_runtime = LearnerProcessHandle.spawn(
        target=_worker_with_async_error,
        target_args=(),
        config=LearnerProcessConfig(sample_queue_size=1, process_name="learner-runtime-test"),
    )

    try:
        init_message = learner_runtime.wait_for_init()
        assert init_message.start_step == 0

        deadline = time.time() + 5.0
        status_error: RuntimeError | None = None
        while True:
            try:
                learner_runtime.check_status()
            except RuntimeError as exc:
                status_error = exc
                break
            if time.time() >= deadline:
                raise AssertionError("learner_runtime.check_status() did not surface the async worker error")
            time.sleep(0.01)
        assert "async boom" in str(status_error)
    finally:
        learner_runtime.shutdown(final_step=0, graceful_stop_sent=False, join_timeout_s=5.0)


def test_learner_process_handle_wait_for_init_surfaces_worker_error() -> None:
    learner_runtime = LearnerProcessHandle.spawn(
        target=_worker_error_before_init,
        target_args=(),
        config=LearnerProcessConfig(sample_queue_size=1, process_name="learner-runtime-test"),
    )

    with pytest.raises(RuntimeError, match="init boom"):
        learner_runtime.wait_for_init()

    learner_runtime.shutdown(final_step=0, graceful_stop_sent=False, join_timeout_s=5.0)


def test_learner_process_handle_wait_for_init_surfaces_crash_without_status() -> None:
    learner_runtime = LearnerProcessHandle.spawn(
        target=_worker_crashes_before_init,
        target_args=(),
        config=LearnerProcessConfig(sample_queue_size=1, process_name="learner-runtime-test"),
    )

    with pytest.raises(RuntimeError, match="exited with code 7"):
        learner_runtime.wait_for_init(timeout_s=5.0)

    learner_runtime.shutdown(final_step=0, graceful_stop_sent=False, join_timeout_s=5.0)


def test_learner_process_handle_check_status_surfaces_crash_without_status() -> None:
    learner_runtime = LearnerProcessHandle.spawn(
        target=_worker_crashes_after_init,
        target_args=(),
        config=LearnerProcessConfig(sample_queue_size=1, process_name="learner-runtime-test"),
    )

    try:
        init_message = learner_runtime.wait_for_init(timeout_s=5.0)
        assert init_message.start_step == 0

        deadline = time.time() + 5.0
        status_error: RuntimeError | None = None
        while True:
            try:
                learner_runtime.check_status()
            except RuntimeError as exc:
                status_error = exc
                break
            if time.time() >= deadline:
                raise AssertionError("learner_runtime.check_status() did not surface process exit")
            time.sleep(0.01)
        assert "exited with code 7" in str(status_error)
    finally:
        learner_runtime.shutdown(final_step=0, graceful_stop_sent=False, join_timeout_s=5.0)
