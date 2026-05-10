from __future__ import annotations

from collections.abc import Callable
import dataclasses
import logging
import multiprocessing as mp
import queue
import time
from typing import Any, TypeVar

from openpi.rlt.rtvla_contract import LearnerError
from openpi.rlt.rtvla_contract import LearnerInit
from openpi.rlt.rtvla_contract import LearnerStats
from openpi.rlt.rtvla_contract import PolicyUpdate
from openpi.rlt.rtvla_contract import StopSignal

PolicyStateT = TypeVar("PolicyStateT")


@dataclasses.dataclass(frozen=True)
class LearnerProcessConfig:
    sample_queue_size: int
    policy_queue_size: int = 1
    process_name: str = "rlt-learner"
    start_method: str = "spawn"


class LearnerProcessHandle:
    def __init__(
        self,
        *,
        process: mp.Process,
        sample_queue: mp.queues.Queue,
        policy_queue: mp.queues.Queue,
        status_queue: mp.queues.Queue,
    ) -> None:
        self._process = process
        self._sample_queue = sample_queue
        self._policy_queue = policy_queue
        self._status_queue = status_queue
        self._latest_stats: LearnerStats | None = None

    @classmethod
    def spawn(
        cls,
        *,
        target: Callable[..., None],
        target_args: tuple[Any, ...],
        config: LearnerProcessConfig,
    ) -> LearnerProcessHandle:
        mp_context = mp.get_context(config.start_method)
        sample_queue = mp_context.Queue(maxsize=config.sample_queue_size)
        policy_queue = mp_context.Queue(maxsize=config.policy_queue_size)
        status_queue = mp_context.Queue()
        process = mp_context.Process(
            target=target,
            args=(*target_args, sample_queue, policy_queue, status_queue),
            name=config.process_name,
        )
        process.start()
        return cls(
            process=process,
            sample_queue=sample_queue,
            policy_queue=policy_queue,
            status_queue=status_queue,
        )

    @property
    def sample_queue(self) -> mp.queues.Queue:
        return self._sample_queue

    @property
    def exitcode(self) -> int | None:
        return self._process.exitcode

    def queue_sample(self, item: Any) -> None:
        self._sample_queue.put(item)

    def _raise_if_unexpected_exit(self, *, allow_clean_exit: bool = True) -> None:
        exitcode = self._process.exitcode
        if exitcode is not None and (exitcode != 0 or not allow_clean_exit):
            raise RuntimeError(f"Learner process exited with code {exitcode}")

    def wait_for_init(self, *, timeout_s: float | None = 60.0) -> LearnerInit:
        deadline = None if timeout_s is None else time.monotonic() + float(timeout_s)
        while True:
            try:
                message = self._status_queue.get(timeout=0.1)
            except queue.Empty as exc:
                self._raise_if_unexpected_exit(allow_clean_exit=False)
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("Timed out waiting for learner process init") from exc
                continue
            if isinstance(message, LearnerInit):
                return message
            if isinstance(message, LearnerStats):
                self._latest_stats = message
                continue
            if isinstance(message, LearnerError):
                raise RuntimeError(f"Learner process failed: {message.message}\n{message.traceback}")

    def check_status(self) -> None:
        while True:
            try:
                message = self._status_queue.get_nowait()
            except queue.Empty:
                self._raise_if_unexpected_exit()
                return
            if isinstance(message, LearnerError):
                raise RuntimeError(f"Learner process failed: {message.message}\n{message.traceback}")
            if isinstance(message, LearnerStats):
                self._latest_stats = message

    def latest_stats(self) -> LearnerStats | None:
        self.check_status()
        return self._latest_stats

    def drain_policy_updates(self) -> PolicyUpdate | None:
        latest: PolicyUpdate | None = None
        while True:
            try:
                message = self._policy_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(message, PolicyUpdate):
                latest = message
        return latest

    def sync_policy_state(
        self,
        policy_state: PolicyStateT,
        *,
        apply_actor_params: Callable[[PolicyStateT, dict[str, Any]], PolicyStateT],
        sync_logger: logging.Logger | None = None,
    ) -> PolicyStateT:
        latest = self.drain_policy_updates()
        if latest is None:
            return policy_state
        if sync_logger is not None:
            sync_logger.info("received_actor_step=%d", int(latest.actor_params["step"]))
        return apply_actor_params(policy_state, latest.actor_params)

    def shutdown(
        self,
        *,
        final_step: int,
        graceful_stop_sent: bool,
        join_timeout_s: float = 60.0,
    ) -> bool:
        if self._process.pid is None:
            return graceful_stop_sent
        if self._process.is_alive() and not graceful_stop_sent:
            self._sample_queue.put(StopSignal(final_step=max(final_step, 0)))
            graceful_stop_sent = True
        self._process.join(timeout=join_timeout_s)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=join_timeout_s)
        return graceful_stop_sent


def publish_policy_update(policy_queue: Any, actor_params: dict[str, Any]) -> None:
    update = PolicyUpdate(actor_params=actor_params)
    while True:
        try:
            policy_queue.put_nowait(update)
            return
        except queue.Full:
            try:
                policy_queue.get_nowait()
            except queue.Empty:
                continue
