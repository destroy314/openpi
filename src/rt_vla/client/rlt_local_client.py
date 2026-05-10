from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import inspect
from typing import Any

import jax
import numpy as np

from openpi.rlt import ReplayItem
from openpi.rlt import RLTTokenResponse
from openpi.rlt import action_space as _action_space

from .config import Config
from .rlt_actor_runtime import ExecutionSummary
from .rlt_actor_runtime import PlanWindow
from .rlt_actor_runtime import RLTFeatureClient
from .rlt_actor_runtime import actor_infer_action_chunk
from .rlt_actor_runtime import build_contract_observation
from .rlt_actor_runtime import build_replay_item
from .rlt_actor_runtime import build_rlt_state
from .rlt_actor_runtime import should_request_new_plan
from .rlt_actor_runtime import summarize_execution_records
from .rlt_observation import AdaptedObservation
from .rlt_observation import ObservationAdapterConfig
from .rlt_observation import RLTObservationAdapter
from .rlt_observation import VelocityFilterConfig


@dataclasses.dataclass(frozen=True)
class CollectedObservation:
    prompt: str
    adapted: AdaptedObservation
    raw_observation: dict[str, Any]
    rlt_state: np.ndarray
    contract_observation: Any


@dataclasses.dataclass(frozen=True)
class TrainingChunkResult:
    collected: CollectedObservation
    plan: PlanWindow
    next_plan: PlanWindow
    next_collected: CollectedObservation
    token_response: RLTTokenResponse | None
    actor_action_chunk: np.ndarray
    command_action_chunk: np.ndarray
    execution: ExecutionSummary
    chunk_env_step: int
    replay_item: ReplayItem
    replay_items: tuple[ReplayItem, ...] = dataclasses.field(default_factory=tuple)
    emitted_env_steps: tuple[int, ...] = dataclasses.field(default_factory=tuple)
    executed_steps: int = 0
    terminal_reason: str | None = None


@dataclasses.dataclass(frozen=True)
class TrainingIterationResult:
    chunk: TrainingChunkResult
    actor_state: Any


@dataclasses.dataclass(frozen=True)
class TrainingLoopResult:
    iterations: list[TrainingIterationResult]
    actor_state: Any
    final_env_step: int = 0
    terminal_reason: str | None = None


@dataclasses.dataclass
class CollectorReplayState:
    current_plan: PlanWindow | None = None
    plans_by_start: dict[int, PlanWindow] = dataclasses.field(default_factory=dict)
    collected_by_step: dict[int, CollectedObservation] = dataclasses.field(default_factory=dict)
    state_history: dict[int, np.ndarray] = dataclasses.field(default_factory=dict)
    action_base_state_history: dict[int, np.ndarray] = dataclasses.field(default_factory=dict)
    action_history: dict[int, np.ndarray] = dataclasses.field(default_factory=dict)
    intervention_history: dict[int, bool] = dataclasses.field(default_factory=dict)
    reward_history: dict[int, float] = dataclasses.field(default_factory=dict)
    pending_chunks: list[int] = dataclasses.field(default_factory=list)


def merge_observations(state_observation: Mapping[str, Any], image_observation: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(image_observation)
    merged.update(dict(state_observation))
    timestamp = state_observation.get("timestamp", state_observation.get("state_timestamp"))
    if timestamp is not None:
        merged["timestamp"] = timestamp
    return merged


def build_observation_adapter(cfg: Config) -> RLTObservationAdapter:
    observation_cfg = cfg.rlt.observation
    if observation_cfg.velocity_source not in {"sdk", "finite_difference"}:
        raise ValueError(
            "rlt.observation.velocity_source must be configured explicitly as 'sdk' or 'finite_difference'; "
            f"got {observation_cfg.velocity_source!r}"
        )
    return RLTObservationAdapter(
        ObservationAdapterConfig(
            velocity_source=observation_cfg.velocity_source,
            filter=VelocityFilterConfig(),
        )
    )


def compute_replan_horizon(*, action_horizon: int, vla_replan_horizon_scale: int) -> int:
    action_horizon = int(action_horizon)
    vla_replan_horizon_scale = int(vla_replan_horizon_scale)
    if action_horizon <= 0:
        raise ValueError(f"action_horizon must be positive, got {action_horizon}")
    if vla_replan_horizon_scale <= 0:
        raise ValueError(f"vla_replan_horizon_scale must be positive, got {vla_replan_horizon_scale}")
    return action_horizon * vla_replan_horizon_scale


def validate_plan_window_config(
    *,
    action_horizon: int,
    chunk_stride: int,
    reference_horizon: int,
    replan_horizon: int,
) -> None:
    action_horizon = int(action_horizon)
    chunk_stride = int(chunk_stride)
    reference_horizon = int(reference_horizon)
    replan_horizon = int(replan_horizon)
    if chunk_stride <= 0:
        raise ValueError(f"chunk_stride must be positive, got {chunk_stride}")
    if action_horizon % chunk_stride != 0:
        raise ValueError(f"chunk_stride ({chunk_stride}) must divide action_horizon ({action_horizon})")
    required_horizon = replan_horizon + (action_horizon - chunk_stride)
    if reference_horizon < required_horizon:
        raise ValueError(
            "reference_horizon is too small for the requested (vla_replan_horizon_scale, chunk_stride): "
            f"need reference_horizon >= replan_horizon + (action_horizon - chunk_stride) = "
            f"{replan_horizon} + ({action_horizon} - {chunk_stride}) = {required_horizon}, "
            f"but reference_horizon is {reference_horizon}."
        )


def apply_feedback_overrides(
    action_chunk: np.ndarray,
    *,
    current_state: np.ndarray,
    feedback_provider: Any | None,
) -> tuple[np.ndarray, np.ndarray]:
    action_chunk_array = np.asarray(action_chunk, dtype=np.float32)
    if feedback_provider is None:
        return action_chunk_array, np.zeros((action_chunk_array.shape[0],), dtype=bool)
    overridden_actions: list[np.ndarray] = []
    intervened: list[bool] = []
    for action in action_chunk_array:
        override_action, was_intervened = feedback_provider.override_action(action, current_state)
        overridden_actions.append(np.asarray(override_action, dtype=np.float32).reshape(action.shape))
        intervened.append(bool(was_intervened))
    return np.stack(overridden_actions, axis=0).astype(np.float32), np.asarray(intervened, dtype=bool)


def _executor_supports_step_callbacks(executor: Any) -> bool:
    try:
        signature = inspect.signature(executor.execute_training_chunk)
    except (AttributeError, TypeError, ValueError):
        return False
    return "before_step_callback" in signature.parameters and "after_step_callback" in signature.parameters


def _append_pending_once(pending_chunks: list[int], step: int) -> None:
    step = int(step)
    if step not in pending_chunks:
        pending_chunks.append(step)
        pending_chunks.sort()


def _latest_collected_state(replay_state: CollectorReplayState, *, step: int, fallback: np.ndarray) -> np.ndarray:
    candidate_steps = [candidate for candidate in replay_state.collected_by_step if candidate <= int(step)]
    if not candidate_steps:
        return np.asarray(fallback, dtype=np.float32)
    collected = replay_state.collected_by_step[max(candidate_steps)]
    return np.asarray(collected.adapted.state, dtype=np.float32)


def _resolve_plan_for_step(
    plans_by_start: dict[int, PlanWindow],
    *,
    step: int,
    action_horizon: int,
) -> PlanWindow:
    candidates = [
        (plan_start, plan)
        for plan_start, plan in plans_by_start.items()
        if plan_start <= int(step) and plan.contains_chunk(step=int(step), action_horizon=int(action_horizon))
    ]
    if not candidates:
        raise ValueError(f"No cached VLA reference plan can cover step={step} horizon={action_horizon}")
    return max(candidates, key=lambda item: item[0])[1]


def _record_collected_observation(
    replay_state: CollectorReplayState,
    *,
    step: int,
    collected: CollectedObservation,
) -> None:
    step = int(step)
    replay_state.collected_by_step[step] = collected
    replay_state.state_history[step] = np.asarray(collected.rlt_state, dtype=np.float32).reshape(-1)
    replay_state.action_base_state_history[step] = np.asarray(collected.adapted.state, dtype=np.float32).reshape(-1)


def _discounted_return_by_step(
    reward_history: dict[int, float],
    *,
    start_step: int,
    bootstrap_steps: int,
    discount: float,
) -> list[float]:
    rewards: list[float] = []
    for offset in range(int(bootstrap_steps)):
        step = int(start_step) + offset
        if step not in reward_history:
            raise ValueError(f"missing reward for env step {step}")
        rewards.append(float(reward_history[step]))
    return rewards


def _build_replay_item_from_history(
    replay_state: CollectorReplayState,
    *,
    start_step: int,
    current_step: int,
    action_horizon: int,
    terminal: bool,
    discount: float,
    norm_stats: dict[str, Any],
    use_quantiles: bool,
    use_delta_joint_actions: bool,
) -> ReplayItem:
    available_steps = int(current_step) - int(start_step)
    if available_steps <= 0:
        raise ValueError(f"cannot build replay item for start_step={start_step} at current_step={current_step}")
    if available_steps < int(action_horizon) and not terminal:
        raise ValueError(
            f"replay item at step {start_step} only has {available_steps} executed steps; "
            "it is not ready until a full action horizon is available or terminal=True"
        )
    bootstrap_steps = min(int(action_horizon), available_steps)
    if start_step not in replay_state.state_history:
        raise ValueError(f"missing RLT state for env step {start_step}")
    next_step = int(start_step) + bootstrap_steps
    if next_step not in replay_state.state_history:
        raise ValueError(f"missing next RLT state for env step {next_step}")
    plan = _resolve_plan_for_step(replay_state.plans_by_start, step=start_step, action_horizon=action_horizon)
    next_plan = _resolve_plan_for_step(replay_state.plans_by_start, step=next_step, action_horizon=action_horizon)
    reference_chunk_real = plan.slice_real_chunk(step=start_step, action_horizon=action_horizon).reshape(
        action_horizon, -1
    )
    action_chunk_real = reference_chunk_real.copy()
    intervened_mask = np.zeros((action_horizon,), dtype=bool)
    for offset in range(bootstrap_steps):
        step = int(start_step) + offset
        if step not in replay_state.action_history:
            raise ValueError(f"missing executed action for env step {step}")
        action_chunk_real[offset] = replay_state.action_history[step]
        if replay_state.intervention_history.get(step, False):
            reference_chunk_real[offset] = replay_state.action_history[step]
            intervened_mask[offset] = True
    done = bool(terminal and next_step >= int(current_step))
    return build_replay_item(
        state=replay_state.state_history[start_step],
        executed_action_chunk_real=action_chunk_real,
        reference_action_chunk_real=reference_chunk_real,
        action_base_state=replay_state.action_base_state_history[start_step],
        next_state=replay_state.state_history[next_step],
        next_reference_action_real=next_plan.slice_real_chunk(step=next_step, action_horizon=action_horizon).reshape(
            action_horizon, -1
        ),
        next_action_base_state=replay_state.action_base_state_history[next_step],
        reward_history=_discounted_return_by_step(
            replay_state.reward_history,
            start_step=start_step,
            bootstrap_steps=bootstrap_steps,
            discount=discount,
        ),
        discount=discount,
        norm_stats=norm_stats,
        use_quantiles=use_quantiles,
        use_delta_joint_actions=use_delta_joint_actions,
        done=done,
        intervened_mask=intervened_mask,
        env_step=start_step,
    )


def _flush_ready_replay_items(
    replay_state: CollectorReplayState,
    *,
    current_step: int,
    action_horizon: int,
    terminal: bool,
    discount: float,
    norm_stats: dict[str, Any],
    use_quantiles: bool,
    use_delta_joint_actions: bool,
) -> tuple[ReplayItem, ...]:
    emitted: list[ReplayItem] = []
    while replay_state.pending_chunks:
        start_step = replay_state.pending_chunks[0]
        available_steps = int(current_step) - int(start_step)
        if available_steps <= 0:
            break
        if available_steps < int(action_horizon) and not terminal:
            break
        emitted.append(
            _build_replay_item_from_history(
                replay_state,
                start_step=start_step,
                current_step=current_step,
                action_horizon=action_horizon,
                terminal=terminal,
                discount=discount,
                norm_stats=norm_stats,
                use_quantiles=use_quantiles,
                use_delta_joint_actions=use_delta_joint_actions,
            )
        )
        replay_state.pending_chunks.pop(0)
    return tuple(emitted)


@dataclasses.dataclass
class RLTLocalClient:
    config: Config
    feature_client: RLTFeatureClient
    observer: Any
    observation_adapter: RLTObservationAdapter | None = None

    def __post_init__(self) -> None:
        if self.observation_adapter is None:
            self.observation_adapter = build_observation_adapter(self.config)

    def collect_observation(self, *, prompt: str | None = None) -> CollectedObservation:
        state_observation = self.observer.get_state_observation()
        image_observation = self.observer.get_image_observation()
        merged = merge_observations(state_observation, image_observation)
        adapted = self.observation_adapter.adapt(merged)
        prompt_value = self.config.rlt.observation.prompt if prompt is None else prompt
        contract_observation = build_contract_observation(
            images=adapted.images,
            state=adapted.state,
            prompt=prompt_value,
            timestamp=adapted.timestamp,
        )
        return CollectedObservation(
            prompt=prompt_value,
            adapted=adapted,
            raw_observation=merged,
            rlt_state=build_rlt_state(np.zeros((256,), dtype=np.float32), adapted.proprio),
            contract_observation=contract_observation,
        )

    def request_plan(
        self,
        *,
        episode_id: str,
        plan_start_step: int = 0,
        prompt: str | None = None,
        collected: CollectedObservation | None = None,
    ) -> tuple[CollectedObservation, PlanWindow]:
        if collected is None:
            collected = self.collect_observation(prompt=prompt)
        response = self.feature_client.infer(
            episode_id=episode_id,
            observation=collected.contract_observation,
            diffusion_steps=self.config.rlt.observation.diffusion_steps,
            reference_horizon=self.config.rlt.observation.reference_horizon,
        )
        plan = PlanWindow(
            feature_id=response.feature_id,
            plan_start_step=int(plan_start_step),
            rl_token=np.asarray(response.rl_token, dtype=np.float32),
            reference_plan_norm=np.asarray(response.reference_plan_norm, dtype=np.float32),
            reference_plan_real=np.asarray(response.reference_plan_list, dtype=np.float32),
            prefix_valid=response.prefix_valid,
            server_infer_time_s=response.server_infer_time_s,
            debug=response.debug.to_dict(),
        )
        return dataclasses.replace(collected, rlt_state=build_rlt_state(plan.rl_token, collected.adapted.proprio)), plan

    def request_token(
        self,
        *,
        episode_id: str,
        prompt: str | None = None,
        collected: CollectedObservation | None = None,
    ) -> tuple[CollectedObservation, RLTTokenResponse]:
        if collected is None:
            collected = self.collect_observation(prompt=prompt)
        response = self.feature_client.token(
            episode_id=episode_id,
            observation=collected.contract_observation,
        )
        return (
            dataclasses.replace(
                collected,
                rlt_state=build_rlt_state(np.asarray(response.rl_token), collected.adapted.proprio),
            ),
            response,
        )

    def execute_training_chunk_once(
        self,
        *,
        episode_id: str,
        actor_model: Any,
        actor_state: Any,
        rng: jax.Array,
        executor: Any,
        action_horizon: int,
        control_dt_s: float,
        norm_stats: dict[str, Any],
        use_quantiles: bool,
        use_delta_joint_actions: bool,
        env_step: int,
        reward_history: list[float] | None = None,
        current_plan: PlanWindow | None = None,
        done: bool = False,
        feedback_provider: Any | None = None,
        prompt: str | None = None,
        replay_state: CollectorReplayState | None = None,
    ) -> TrainingChunkResult:
        observation_cfg = self.config.rlt.observation
        chunk_stride = int(observation_cfg.chunk_stride)
        replan_horizon = compute_replan_horizon(
            action_horizon=action_horizon,
            vla_replan_horizon_scale=observation_cfg.vla_replan_horizon_scale,
        )
        validate_plan_window_config(
            action_horizon=action_horizon,
            chunk_stride=chunk_stride,
            reference_horizon=observation_cfg.reference_horizon,
            replan_horizon=replan_horizon,
        )
        chunk_step = int(env_step)
        if replay_state is None:
            replay_state = CollectorReplayState()
        collected = self.collect_observation(prompt=prompt)
        active_plan = replay_state.current_plan if current_plan is None else current_plan
        if should_request_new_plan(
            step=chunk_step,
            action_horizon=action_horizon,
            replan_horizon=replan_horizon,
            current_plan=active_plan,
        ):
            collected, plan = self.request_plan(
                episode_id=episode_id,
                plan_start_step=chunk_step,
                prompt=prompt,
                collected=collected,
            )
        else:
            plan = active_plan
            collected = dataclasses.replace(
                collected,
                rlt_state=build_rlt_state(plan.rl_token, collected.adapted.proprio),
            )
        replay_state.current_plan = plan
        replay_state.plans_by_start[int(plan.plan_start_step)] = plan
        _record_collected_observation(replay_state, step=chunk_step, collected=collected)
        _append_pending_once(replay_state.pending_chunks, chunk_step)

        reference_chunk = plan.slice_norm_chunk(step=chunk_step, action_horizon=action_horizon)
        actor_action_chunk = actor_infer_action_chunk(
            actor_model,
            actor_state,
            rng,
            rlt_state=collected.rlt_state,
            reference_action=reference_chunk,
        ).reshape(action_horizon, -1)
        command_action_chunk = _action_space.denormalize_action(
            norm_stats,
            use_quantiles,
            actor_action_chunk,
            state=collected.adapted.state,
            use_delta_joint_actions=use_delta_joint_actions,
        )
        command_action_chunk = np.asarray(command_action_chunk, dtype=np.float32)
        submitted_action_chunk = command_action_chunk.copy()
        step_intervened = np.zeros((action_horizon,), dtype=bool)
        per_step_rewards: dict[int, float] = {}
        callback_observations: dict[int, CollectedObservation] = {}
        terminal_reason = None

        def _prepare_observation_for_step(
            step: int,
            observed: CollectedObservation,
        ) -> tuple[CollectedObservation, PlanWindow, RLTTokenResponse | None]:
            active = replay_state.current_plan
            token: RLTTokenResponse | None = None
            if should_request_new_plan(
                step=step,
                action_horizon=action_horizon,
                replan_horizon=replan_horizon,
                current_plan=active,
            ):
                observed, next_plan = self.request_plan(
                    episode_id=episode_id,
                    plan_start_step=step,
                    prompt=prompt,
                    collected=observed,
                )
                replay_state.current_plan = next_plan
                replay_state.plans_by_start[int(next_plan.plan_start_step)] = next_plan
            else:
                observed, token = self.request_token(
                    episode_id=episode_id,
                    prompt=prompt,
                    collected=observed,
                )
                next_plan = active
            _record_collected_observation(replay_state, step=step, collected=observed)
            return observed, next_plan, token

        def _before_step_callback(chunk_index: int, raw_action: list[float]) -> list[float]:
            step = chunk_step + int(chunk_index)
            policy_action = np.asarray(raw_action, dtype=np.float32)
            if feedback_provider is None:
                return policy_action.tolist()
            current_state = _latest_collected_state(
                replay_state,
                step=step,
                fallback=collected.adapted.state,
            )
            override_action, was_intervened = feedback_provider.override_action(policy_action, current_state)
            override_array = np.asarray(override_action, dtype=np.float32).reshape(policy_action.shape)
            submitted_action_chunk[int(chunk_index)] = override_array
            step_intervened[int(chunk_index)] = bool(was_intervened)
            return override_array.tolist()

        def _after_step_callback(chunk_index: int, _record: Mapping[str, Any]) -> None:
            nonlocal done, terminal_reason
            step = chunk_step + int(chunk_index)
            if reward_history is None:
                if feedback_provider is None:
                    reward = 0.0
                    terminate = False
                    reason = None
                else:
                    reward, terminate, reason = feedback_provider.consume_step_feedback()
                per_step_rewards[step] = float(reward)
                if terminate:
                    done = True
                    terminal_reason = reason
            offset = int(chunk_index) + 1
            if offset % chunk_stride == 0 or offset == action_horizon:
                callback_observations[chunk_step + offset] = self.collect_observation(prompt=prompt)

        supports_callbacks = _executor_supports_step_callbacks(executor)
        if supports_callbacks:
            records = executor.execute_training_chunk(
                command_action_chunk.tolist(),
                plan_id=plan.feature_id,
                control_dt_s=control_dt_s,
                before_step_callback=_before_step_callback,
                after_step_callback=_after_step_callback,
            )
        else:
            if chunk_stride < action_horizon:
                raise ValueError(
                    "executor.execute_training_chunk must support step callbacks when chunk_stride < action_horizon "
                    "so intermediate observations and rewards can be aligned"
                )
            submitted_action_chunk, step_intervened = apply_feedback_overrides(
                command_action_chunk,
                current_state=collected.adapted.state,
                feedback_provider=feedback_provider,
            )
            records = executor.execute_training_chunk(
                submitted_action_chunk.tolist(),
                plan_id=plan.feature_id,
                control_dt_s=control_dt_s,
            )
        execution = summarize_execution_records(records, intervened_mask=step_intervened[: len(records)])
        executed_steps = int(execution.executed_actions.shape[0])
        current_step = chunk_step + executed_steps
        if reward_history is None and not supports_callbacks:
            for offset in range(executed_steps):
                step = chunk_step + offset
                if feedback_provider is None:
                    reward = 0.0
                    terminate = False
                    reason = None
                else:
                    reward, terminate, reason = feedback_provider.consume_step_feedback()
                per_step_rewards[step] = float(reward)
                if terminate:
                    done = True
                    terminal_reason = reason
        elif reward_history is not None:
            if len(reward_history) != executed_steps:
                raise ValueError(
                    f"reward_history length ({len(reward_history)}) must match executed steps ({executed_steps})"
                )
            per_step_rewards = {chunk_step + offset: float(reward) for offset, reward in enumerate(reward_history)}

        for offset, executed_action in enumerate(execution.executed_actions):
            step = chunk_step + offset
            replay_state.action_history[step] = np.asarray(executed_action, dtype=np.float32).copy()
            replay_state.intervention_history[step] = bool(execution.intervened_mask[offset])
            replay_state.reward_history[step] = float(per_step_rewards.get(step, 0.0))

        token_response = None
        next_plan = replay_state.current_plan
        next_collected: CollectedObservation | None = None
        sampled_offsets = list(range(chunk_stride, executed_steps + 1, chunk_stride))
        if executed_steps not in sampled_offsets:
            sampled_offsets.append(executed_steps)
        for offset in sampled_offsets:
            sample_step = chunk_step + int(offset)
            observed = callback_observations.get(sample_step)
            if observed is None:
                if offset != executed_steps:
                    raise ValueError(
                        f"missing intermediate observation for chunk_stride sample at env step {sample_step}; "
                        "executor step callbacks are required"
                    )
                observed = self.collect_observation(prompt=prompt)
            observed, observed_plan, observed_token = _prepare_observation_for_step(sample_step, observed)
            _append_pending_once(replay_state.pending_chunks, sample_step)
            if sample_step == current_step:
                next_collected = observed
                next_plan = observed_plan
                token_response = observed_token
        if next_collected is None:
            raise RuntimeError("failed to collect next observation after executing training chunk")

        replay_items = _flush_ready_replay_items(
            replay_state,
            current_step=current_step,
            action_horizon=action_horizon,
            terminal=bool(done),
            discount=float(observation_cfg.discount),
            norm_stats=norm_stats,
            use_quantiles=use_quantiles,
            use_delta_joint_actions=use_delta_joint_actions,
        )
        if not replay_items:
            raise RuntimeError("no replay item was emitted after executing a full atomic chunk")
        replay_item = replay_items[0]
        return TrainingChunkResult(
            collected=collected,
            plan=plan,
            next_plan=next_plan,
            next_collected=next_collected,
            token_response=token_response,
            actor_action_chunk=actor_action_chunk,
            command_action_chunk=submitted_action_chunk,
            execution=execution,
            chunk_env_step=chunk_step,
            replay_item=replay_item,
            replay_items=replay_items,
            emitted_env_steps=tuple(int(item.env_step) for item in replay_items),
            executed_steps=executed_steps,
            terminal_reason=terminal_reason,
        )

    def run_training_iteration(
        self,
        *,
        episode_id: str,
        actor_model: Any,
        actor_state: Any,
        rng: jax.Array,
        executor: Any,
        learner_runtime: Any,
        action_horizon: int,
        control_dt_s: float,
        norm_stats: dict[str, Any],
        use_quantiles: bool,
        use_delta_joint_actions: bool,
        env_step: int,
        reward_history: list[float] | None = None,
        current_plan: PlanWindow | None = None,
        done: bool = False,
        feedback_provider: Any | None = None,
        apply_actor_params: Any,
        prompt: str | None = None,
        sync_logger: Any = None,
        replay_state: CollectorReplayState | None = None,
    ) -> TrainingIterationResult:
        chunk = self.execute_training_chunk_once(
            episode_id=episode_id,
            actor_model=actor_model,
            actor_state=actor_state,
            rng=rng,
            executor=executor,
            action_horizon=action_horizon,
            control_dt_s=control_dt_s,
            norm_stats=norm_stats,
            use_quantiles=use_quantiles,
            use_delta_joint_actions=use_delta_joint_actions,
            reward_history=reward_history,
            env_step=env_step,
            current_plan=current_plan,
            done=done,
            feedback_provider=feedback_provider,
            prompt=prompt,
            replay_state=replay_state,
        )
        for replay_item in chunk.replay_items:
            if hasattr(learner_runtime, "queue_sample"):
                learner_runtime.queue_sample(replay_item)
            else:
                learner_runtime.put(replay_item)
        if hasattr(learner_runtime, "check_status"):
            learner_runtime.check_status()
        if hasattr(learner_runtime, "sync_policy_state"):
            actor_state = learner_runtime.sync_policy_state(
                actor_state,
                apply_actor_params=apply_actor_params,
                sync_logger=sync_logger,
            )
        return TrainingIterationResult(chunk=chunk, actor_state=actor_state)

    def run_training_loop(
        self,
        *,
        episode_id: str,
        actor_model: Any,
        actor_state: Any,
        rng: jax.Array,
        executor: Any,
        learner_runtime: Any,
        feedback_provider: Any,
        num_iterations: int,
        action_horizon: int,
        control_dt_s: float,
        norm_stats: dict[str, Any],
        use_quantiles: bool,
        use_delta_joint_actions: bool,
        apply_actor_params: Any,
        prompt: str | None = None,
        sync_logger: Any = None,
        transition_recorder: Any = None,
        initial_env_step: int = 0,
    ) -> TrainingLoopResult:
        if num_iterations <= 0:
            raise ValueError(f"num_iterations must be positive, got {num_iterations}")
        feedback_provider.before_episode()
        results: list[TrainingIterationResult] = []
        current_actor_state = actor_state
        env_step = int(initial_env_step)
        current_plan: PlanWindow | None = None
        replay_state = CollectorReplayState(current_plan=current_plan)
        terminal_reason = None
        for _ in range(num_iterations):
            rng, iter_rng = jax.random.split(rng)
            result = self.run_training_iteration(
                episode_id=episode_id,
                actor_model=actor_model,
                actor_state=current_actor_state,
                rng=iter_rng,
                executor=executor,
                learner_runtime=learner_runtime,
                action_horizon=action_horizon,
                control_dt_s=control_dt_s,
                norm_stats=norm_stats,
                use_quantiles=use_quantiles,
                use_delta_joint_actions=use_delta_joint_actions,
                env_step=env_step,
                reward_history=None,
                current_plan=current_plan,
                feedback_provider=feedback_provider,
                apply_actor_params=apply_actor_params,
                prompt=prompt,
                sync_logger=sync_logger,
                replay_state=replay_state,
            )
            if transition_recorder is not None:
                for replay_item in result.chunk.replay_items:
                    transition_recorder.add(replay_item)
            results.append(result)
            current_plan = replay_state.current_plan
            current_actor_state = result.actor_state
            env_step += int(result.chunk.executed_steps)
            if any(replay_item.done for replay_item in result.chunk.replay_items):
                terminal_reason = result.chunk.terminal_reason
                break
        return TrainingLoopResult(
            iterations=results,
            actor_state=current_actor_state,
            final_env_step=env_step,
            terminal_reason=terminal_reason,
        )


def build_feature_metadata(chunk: TrainingChunkResult) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "plan": {
            "feature_id": chunk.plan.feature_id,
            "plan_start_step": int(chunk.plan.plan_start_step),
            "prefix_valid": bool(chunk.plan.prefix_valid),
            "server_infer_time_s": chunk.plan.server_infer_time_s,
            "debug": dict(chunk.plan.debug),
            "reference_plan_shape": list(chunk.plan.reference_plan_norm.shape),
            "reference_plan_norm_sha256": chunk.plan.reference_plan_norm_sha256,
            "reference_plan_real_sha256": chunk.plan.reference_plan_real_sha256,
        }
    }
    if chunk.token_response is not None:
        metadata["next_token"] = {
            "feature_id": chunk.token_response.feature_id,
            "prefix_valid": bool(chunk.token_response.prefix_valid),
            "server_token_time_s": chunk.token_response.server_token_time_s,
            "debug": chunk.token_response.debug.to_dict(),
        }
    elif chunk.next_plan.feature_id != chunk.plan.feature_id:
        metadata["next_plan"] = {
            "feature_id": chunk.next_plan.feature_id,
            "plan_start_step": int(chunk.next_plan.plan_start_step),
            "prefix_valid": bool(chunk.next_plan.prefix_valid),
            "server_infer_time_s": chunk.next_plan.server_infer_time_s,
            "debug": dict(chunk.next_plan.debug),
            "reference_plan_shape": list(chunk.next_plan.reference_plan_norm.shape),
            "reference_plan_norm_sha256": chunk.next_plan.reference_plan_norm_sha256,
            "reference_plan_real_sha256": chunk.next_plan.reference_plan_real_sha256,
        }
    return metadata
