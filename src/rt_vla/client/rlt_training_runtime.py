from __future__ import annotations

import dataclasses
import threading
from typing import Any

import jax

from .config import Config
from .rlt_local_client import RLTLocalClient
from .rlt_local_client import TrainingLoopResult


@dataclasses.dataclass
class RLTTrainingRuntime:
    config: Config
    local_client: RLTLocalClient
    actor_model: Any
    actor_state: Any
    executor: Any
    learner_runtime: Any
    feedback_provider: Any
    norm_stats: dict[str, Any]
    use_quantiles: bool
    use_delta_joint_actions: bool
    action_horizon: int
    control_dt_s: float
    apply_actor_params: Any
    episode_id: str = "episode-0"
    prompt: str | None = None
    env_step: int = 0
    _actor_collector_thread: threading.Thread | None = dataclasses.field(default=None, init=False, repr=False)
    _result_box: dict[str, TrainingLoopResult] = dataclasses.field(default_factory=dict, init=False, repr=False)
    _error_box: dict[str, BaseException] = dataclasses.field(default_factory=dict, init=False, repr=False)
    _thread_lock: threading.Lock = dataclasses.field(default_factory=threading.Lock, init=False, repr=False)

    def run(self, *, num_iterations: int, rng: jax.Array) -> TrainingLoopResult:
        return self.run_with_artifacts(num_iterations=num_iterations, rng=rng)

    def start(
        self,
        *,
        num_iterations: int,
        rng: jax.Array,
        transition_recorder: Any = None,
    ) -> None:
        with self._thread_lock:
            if self._actor_collector_thread is not None and self._actor_collector_thread.is_alive():
                raise RuntimeError("RLT actor-collector thread is already running")
            result_box: dict[str, TrainingLoopResult] = {}
            error_box: dict[str, BaseException] = {}
            self._result_box = result_box
            self._error_box = error_box

            def _run_actor_collector() -> None:
                try:
                    result_box["result"] = self.local_client.run_training_loop(
                        episode_id=self.episode_id,
                        actor_model=self.actor_model,
                        actor_state=self.actor_state,
                        rng=rng,
                        executor=self.executor,
                        learner_runtime=self.learner_runtime,
                        feedback_provider=self.feedback_provider,
                        num_iterations=num_iterations,
                        action_horizon=self.action_horizon,
                        control_dt_s=self.control_dt_s,
                        norm_stats=self.norm_stats,
                        use_quantiles=self.use_quantiles,
                        use_delta_joint_actions=self.use_delta_joint_actions,
                        apply_actor_params=self.apply_actor_params,
                        prompt=self.prompt,
                        transition_recorder=transition_recorder,
                        initial_env_step=self.env_step,
                    )
                except BaseException as exc:
                    error_box["error"] = exc

            actor_collector = threading.Thread(
                target=_run_actor_collector,
                name="rlt-actor-collector",
            )
            self._actor_collector_thread = actor_collector
            actor_collector.start()

    def join(self, timeout_s: float | None = None) -> TrainingLoopResult | None:
        actor_collector = self._actor_collector_thread
        if actor_collector is None:
            raise RuntimeError("RLT actor-collector thread has not been started")
        actor_collector.join(timeout=timeout_s)
        if actor_collector.is_alive():
            return None
        if "error" in self._error_box:
            raise self._error_box["error"]
        result = self._result_box["result"]
        self.actor_state = result.actor_state
        self.env_step = result.final_env_step
        return result

    def run_with_artifacts(
        self,
        *,
        num_iterations: int,
        rng: jax.Array,
        transition_recorder: Any = None,
    ) -> TrainingLoopResult:
        self.start(num_iterations=num_iterations, rng=rng, transition_recorder=transition_recorder)
        result = self.join()
        if result is None:
            raise RuntimeError("RLT actor-collector thread did not finish")
        return result

    def status(self) -> dict[str, Any]:
        learner_stats = None
        latest_stats = getattr(self.learner_runtime, "latest_stats", None)
        if callable(latest_stats):
            stats = latest_stats()
            learner_stats = stats.to_dict() if stats is not None and hasattr(stats, "to_dict") else stats
        actor_collector = self._actor_collector_thread
        return {
            "episode_id": self.episode_id,
            "env_step": int(self.env_step),
            "actor_step": int(getattr(self.actor_state, "step", 0)),
            "actor_collector": {
                "started": actor_collector is not None,
                "running": bool(actor_collector is not None and actor_collector.is_alive()),
                "thread_name": actor_collector.name if actor_collector is not None else None,
                "error": str(self._error_box["error"]) if "error" in self._error_box else None,
            },
            "learner": learner_stats,
        }

    def close(self) -> None:
        if hasattr(self.feedback_provider, "close"):
            self.feedback_provider.close()
        if hasattr(self.executor, "close"):
            self.executor.close()
        observer = getattr(self.local_client, "observer", None)
        if observer is not None and hasattr(observer, "close"):
            observer.close()
