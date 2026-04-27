from __future__ import annotations

import dataclasses
import logging
import multiprocessing as mp
import os
import pathlib
import queue
import signal
import traceback
from typing import Any, Callable

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import tyro

from openpi import transforms as _transforms
from openpi.policies import policy_config as _policy_config
from openpi.rlt import airbot_env as _airbot_env
from openpi.rlt import checkpointing as _checkpointing
from openpi.rlt import replay_buffer as _replay_buffer
from openpi.rlt import repro as _repro
from openpi.rlt import trainer as _trainer
from openpi.shared import nnx_utils
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config


@dataclasses.dataclass
class PlannedChunk:
    step: int
    state: np.ndarray
    action: np.ndarray
    reference_action: np.ndarray


@dataclasses.dataclass
class ReplayItem:
    state: np.ndarray
    action: np.ndarray
    reference_action: np.ndarray
    reward: float
    next_state: np.ndarray
    next_reference_action: np.ndarray
    bootstrap_steps: int
    done: bool
    env_step: int


@dataclasses.dataclass(frozen=True)
class StopSignal:
    final_step: int


@dataclasses.dataclass
class LearnerInit:
    start_step: int
    actor_params: dict[str, Any]


@dataclasses.dataclass
class PolicyUpdate:
    actor_params: dict[str, Any]


@dataclasses.dataclass
class LearnerError:
    message: str
    traceback: str


@dataclasses.dataclass
class Args:
    # Model/data config that defines the deployable PI05-RLT topology.
    config: str = "pi05_airbot_rlt"
    # Stage 1 checkpoint directory containing PI05 + RL token weights.
    init_checkpoint_dir: pathlib.Path = pathlib.Path("checkpoints/pi05_airbot_rlt_token/default/19999")
    # Output directory for Stage 2 checkpoints.
    checkpoint_dir: pathlib.Path = pathlib.Path("checkpoints/pi05_airbot_rlt/online")

    max_env_steps: int = 10_000
    warmup_steps: int = 1_000
    replay_capacity: int = 100_000
    batch_size: int = 256
    utd_ratio: int = 5
    diffusion_steps: int = 10
    chunk_stride: int = 2
    vla_replan_horizon_scale: int = 1
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    save_interval: int = 1_000
    log_interval: int = 100
    overwrite: bool = False
    resume: bool = False
    seed: int = 0
    record_repro: bool = False
    repro_shard_size: int = 64

    online: _trainer.OnlineRLTConfig = dataclasses.field(default_factory=_trainer.OnlineRLTConfig)


def _to_jax_batch(batch: _replay_buffer.TransitionBatch) -> _replay_buffer.TransitionBatch:
    return _replay_buffer.TransitionBatch(
        state=jnp.asarray(batch.state),
        action=jnp.asarray(batch.action),
        reference_action=jnp.asarray(batch.reference_action),
        reward=jnp.asarray(batch.reward),
        next_state=jnp.asarray(batch.next_state),
        next_reference_action=jnp.asarray(batch.next_reference_action),
        bootstrap_steps=jnp.asarray(batch.bootstrap_steps),
        done=jnp.asarray(batch.done),
    )


def _compute_rlt_state(
    policy,
    encode_prefix,
    compute_rl_token,
    observation: dict[str, Any],
) -> np.ndarray:
    inputs, batched_observation = policy.prepare_observation(observation)
    proprio = inputs.get("proprio")
    _, _, prefix_rlt_mask, prefix_out, _ = encode_prefix(batched_observation)
    rl_token, _ = compute_rl_token(prefix_out, prefix_rlt_mask)
    return np.concatenate(
        [
            np.asarray(rl_token[0], dtype=np.float32),
            np.asarray(proprio, dtype=np.float32),
        ],
        axis=-1,
    )


def _compute_reference_plan(
    policy,
    sample_reference_actions,
    rng: jax.Array,
    observation: dict[str, Any],
    *,
    diffusion_steps: int,
    env_action_dim: int,
) -> np.ndarray:
    _, batched_observation = policy.prepare_observation(observation)
    reference_actions = sample_reference_actions(rng, batched_observation, num_steps=diffusion_steps)
    return np.asarray(reference_actions[0, :, :env_action_dim], dtype=np.float32)


def _resolve_reference_chunk(
    reference_plan_history: dict[int, np.ndarray],
    *,
    step: int,
    action_horizon: int,
    replan_origin_step: int,
    vla_replan_horizon: int,
) -> np.ndarray:
    if action_horizon <= 0:
        raise ValueError(f"action_horizon must be positive, got {action_horizon}")
    if vla_replan_horizon <= 0:
        raise ValueError(f"vla_replan_horizon must be positive, got {vla_replan_horizon}")
    if step < replan_origin_step:
        raise ValueError(f"step ({step}) must be >= replan_origin_step ({replan_origin_step})")
    if not reference_plan_history:
        raise ValueError("reference_plan_history must contain at least one VLA plan.")
    del replan_origin_step, vla_replan_horizon

    candidate_steps = [plan_start for plan_start in reference_plan_history if plan_start <= step]
    if not candidate_steps:
        raise ValueError(f"No VLA reference plan available for step {step}.")
    plan_step = max(candidate_steps)
    plan = reference_plan_history[plan_step]
    plan_offset = step - plan_step
    end = plan_offset + action_horizon
    if plan_offset >= plan.shape[0] or end > plan.shape[0]:
        raise ValueError(
            f"Cached VLA plan from step {plan_step} with length {plan.shape[0]} "
            f"cannot provide a contiguous chunk of length {action_horizon} for step {step}."
        )
    return np.asarray(plan[plan_offset:end], dtype=np.float32).reshape(-1)


def _slice_norm_stats(stats, dim: int):
    """Truncate a NormStats to the first `dim` elements along the last axis."""
    from openpi.shared.normalize import NormStats
    return NormStats(
        mean=stats.mean[..., :dim],
        std=stats.std[..., :dim],
        q01=stats.q01[..., :dim] if stats.q01 is not None else None,
        q99=stats.q99[..., :dim] if stats.q99 is not None else None,
    )


def _airbot_delta_action_mask() -> tuple[bool, ...]:
    return _transforms.make_bool_mask(6, -1, 6, -1)


def _action_transform_state(state: np.ndarray, env_dim: int) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.shape[-1] < env_dim:
        raise ValueError(f"state must have at least {env_dim} dims, got {state.shape[-1]}")
    return state[:env_dim].copy()


def _denormalize_action(
    norm_stats,
    use_quantiles: bool,
    actions: np.ndarray,
    *,
    state: np.ndarray | None = None,
    use_delta_joint_actions: bool = False,
) -> np.ndarray:
    """Convert a (steps, env_action_dim) action from model space to real joint-position space."""
    outputs = {"actions": np.asarray(actions, dtype=np.float32).copy()}
    env_dim = outputs["actions"].shape[-1]
    action_stats = {"actions": _slice_norm_stats(norm_stats["actions"], env_dim)}
    outputs = _transforms.Unnormalize(action_stats, use_quantiles=use_quantiles)(outputs)
    if use_delta_joint_actions:
        if state is None:
            raise ValueError("state is required to convert delta actions back to absolute actions.")
        outputs["state"] = _action_transform_state(state, env_dim)
        outputs = _transforms.AbsoluteActions(_airbot_delta_action_mask())(outputs)
    return np.asarray(outputs["actions"], dtype=np.float32)


def _normalize_action(
    norm_stats,
    use_quantiles: bool,
    actions: np.ndarray,
    *,
    state: np.ndarray | None = None,
    use_delta_joint_actions: bool = False,
) -> np.ndarray:
    """Convert a (steps, env_action_dim) action from real joint-position space back to model space."""
    outputs = {"actions": np.asarray(actions, dtype=np.float32).copy()}
    env_dim = outputs["actions"].shape[-1]
    if use_delta_joint_actions:
        if state is None:
            raise ValueError("state is required to convert absolute actions into delta actions.")
        outputs["state"] = _action_transform_state(state, env_dim)
        outputs = _transforms.DeltaActions(_airbot_delta_action_mask())(outputs)
    action_stats = {"actions": _slice_norm_stats(norm_stats["actions"], env_dim)}
    outputs = _transforms.Normalize(action_stats, use_quantiles=use_quantiles)(outputs)
    return np.asarray(outputs["actions"], dtype=np.float32)


def _actor_action(actor_model, actor_state, rng: jax.Array, state: np.ndarray, reference_action: np.ndarray) -> np.ndarray:
    action = _trainer.actor_sample_params(
        actor_model,
        actor_state.params,
        jnp.asarray(state)[None, ...],
        jnp.asarray(reference_action)[None, ...],
        rng,
        debug_step=actor_state.step,
        debug_label="collector",
    )
    return np.asarray(action[0], dtype=np.float32)


def _extract_step_rewards(reward: float, info: dict[str, Any], requested_steps: int) -> list[float]:
    step_rewards = info.get("step_rewards")
    if step_rewards is None:
        if requested_steps == 1:
            return [float(reward)]
        raise ValueError(
            "env.step() must provide per-step rewards in info['step_rewards'] when chunk_stride > 1 "
            "so the script can build chunk-aligned n-step replay entries."
        )
    rewards = np.asarray(step_rewards, dtype=np.float32).reshape(-1).tolist()
    if not rewards:
        raise ValueError("info['step_rewards'] must contain at least one reward.")
    return [float(x) for x in rewards]


def _extract_executed_actions(
    info: dict[str, Any],
    *,
    executed_steps: int,
    fallback_actions: np.ndarray,
) -> np.ndarray:
    executed_actions = info.get("executed_actions")
    fallback = np.asarray(fallback_actions, dtype=np.float32)
    if fallback.ndim != 2:
        raise ValueError(f"fallback_actions must be rank-2, got shape {fallback.shape}")
    if executed_actions is None:
        logging.warning("info['executed_actions'] is None, using fallback actions")
        return fallback[:executed_steps].copy()
    actions = np.asarray(executed_actions, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"info['executed_actions'] must be rank-2, got shape {actions.shape}")
    if actions.shape[0] != executed_steps:
        raise ValueError(
            f"info['executed_actions'] length ({actions.shape[0]}) must match executed_steps ({executed_steps})"
        )
    if actions.shape[1] != fallback.shape[1]:
        raise ValueError(
            f"info['executed_actions'] width ({actions.shape[1]}) must match env action width ({fallback.shape[1]})"
        )
    return actions.copy()


def _extract_intervened_mask(info: dict[str, Any], *, executed_steps: int) -> np.ndarray:
    intervened_mask = info.get("intervened_mask")
    if intervened_mask is None:
        return np.zeros((executed_steps,), dtype=bool)
    mask = np.asarray(intervened_mask, dtype=bool).reshape(-1)
    if mask.shape[0] != executed_steps:
        raise ValueError(f"info['intervened_mask'] length ({mask.shape[0]}) must match executed_steps ({executed_steps})")
    return mask.copy()


def _patch_chunk_with_executed_prefix(
    chunk: PlannedChunk,
    *,
    executed_actions: np.ndarray,
    intervened_mask: np.ndarray,
    action_horizon: int,
    action_dim: int,
    env_action_dim: int,
) -> None:
    if executed_actions.ndim != 2 or executed_actions.shape[1] != env_action_dim:
        raise ValueError(f"executed_actions must have shape (steps, {env_action_dim}), got {executed_actions.shape}")
    if intervened_mask.shape != (executed_actions.shape[0],):
        raise ValueError(
            f"intervened_mask must have shape ({executed_actions.shape[0]},), got {intervened_mask.shape}"
        )

    action_chunk = np.asarray(chunk.action, dtype=np.float32).reshape(action_horizon, action_dim).copy()
    reference_chunk = np.asarray(chunk.reference_action, dtype=np.float32).reshape(action_horizon, action_dim).copy()

    action_chunk[: executed_actions.shape[0], :env_action_dim] = executed_actions
    if np.any(intervened_mask):
        reference_chunk[: executed_actions.shape[0], :env_action_dim][intervened_mask] = executed_actions[intervened_mask]

    chunk.action = action_chunk.reshape(-1)
    chunk.reference_action = reference_chunk.reshape(-1)


def _discounted_return(reward_history: list[float], start_step: int, horizon: int, discount: float) -> float:
    return float(sum((discount**offset) * reward_history[start_step + offset] for offset in range(horizon)))


def _record_action_history(
    action_history: dict[int, np.ndarray],
    intervention_history: dict[int, bool],
    *,
    start_step: int,
    executed_actions: np.ndarray,
    intervened_mask: np.ndarray,
) -> None:
    if executed_actions.ndim != 2:
        raise ValueError(f"executed_actions must be rank-2, got shape {executed_actions.shape}")
    if intervened_mask.shape != (executed_actions.shape[0],):
        raise ValueError(
            f"intervened_mask must have shape ({executed_actions.shape[0]},), got {intervened_mask.shape}"
        )
    for offset, action in enumerate(executed_actions):
        action_history[start_step + offset] = np.asarray(action, dtype=np.float32).copy()
        intervention_history[start_step + offset] = bool(intervened_mask[offset])


def _flush_ready_chunks(
    pending_chunks: list[int],
    state_history: dict[int, np.ndarray],
    reference_plan_history: dict[int, np.ndarray],
    action_history: dict[int, np.ndarray],
    intervention_history: dict[int, bool],
    reward_history: list[float],
    transition_sink: Callable[[ReplayItem], None],
    *,
    current_step: int,
    action_horizon: int,
    replan_origin_step: int,
    vla_replan_horizon: int,
    discount: float,
    terminal: bool = False,
    terminal_state: np.ndarray | None = None,
) -> int:
    emitted = 0
    while pending_chunks:
        chunk_step = pending_chunks[0]
        available_steps = current_step - chunk_step
        if available_steps <= 0:
            break
        if available_steps < action_horizon and not terminal:
            break

        bootstrap_steps = min(action_horizon, available_steps)
        discounted_reward = _discounted_return(reward_history, chunk_step, bootstrap_steps, discount)
        state = state_history[chunk_step]
        reference_chunk = _resolve_reference_chunk(
            reference_plan_history,
            step=chunk_step,
            action_horizon=action_horizon,
            replan_origin_step=replan_origin_step,
            vla_replan_horizon=vla_replan_horizon,
        ).reshape(action_horizon, -1)
        action_chunk = reference_chunk.copy()
        for offset in range(bootstrap_steps):
            step = chunk_step + offset
            executed_action = action_history[step]
            action_chunk[offset] = executed_action
            if intervention_history.get(step, False):
                reference_chunk[offset] = executed_action

        if bootstrap_steps == action_horizon and not terminal:
            next_step = chunk_step + action_horizon
            next_state = state_history[next_step]
            next_reference_action = _resolve_reference_chunk(
                reference_plan_history,
                step=next_step,
                action_horizon=action_horizon,
                replan_origin_step=replan_origin_step,
                vla_replan_horizon=vla_replan_horizon,
            )
            done = False
        else:
            if terminal_state is None:
                raise ValueError("terminal_state is required when flushing truncated chunk transitions.")
            next_state = terminal_state
            next_reference_action = _resolve_reference_chunk(
                reference_plan_history,
                step=current_step,
                action_horizon=action_horizon,
                replan_origin_step=replan_origin_step,
                vla_replan_horizon=vla_replan_horizon,
            )
            done = True

        transition_sink(
            ReplayItem(
                state=state,
                action=action_chunk.reshape(-1),
                reference_action=reference_chunk.reshape(-1),
                reward=discounted_reward,
                next_state=next_state,
                next_reference_action=next_reference_action,
                bootstrap_steps=bootstrap_steps,
                done=done,
                env_step=current_step,
            )
        )
        pending_chunks.pop(0)
        emitted += 1
    return emitted


def _queue_transition(
    sample_queue: mp.queues.Queue,
    item: ReplayItem,
    *,
    transition_recorder: _repro.TransitionRecorder | None = None,
) -> None:
    if transition_recorder is not None:
        transition_recorder.add(item)
    sample_queue.put(item)


def _publish_policy_update(policy_queue: mp.queues.Queue, policy_state) -> None:
    update = PolicyUpdate(actor_params=_trainer.bundle_actor_params(policy_state))
    while True:
        try:
            policy_queue.put_nowait(update)
            return
        except queue.Full:
            try:
                policy_queue.get_nowait()
            except queue.Empty:
                continue


def _drain_policy_updates(policy_queue: mp.queues.Queue, policy_state, *, sync_logger: logging.Logger | None = None):
    latest: PolicyUpdate | None = None
    while True:
        try:
            message = policy_queue.get_nowait()
        except queue.Empty:
            break
        if isinstance(message, PolicyUpdate):
            latest = message
    if latest is None:
        return policy_state
    if sync_logger is not None:
        sync_logger.info("received_actor_step=%d", int(latest.actor_params["step"]))
    return _trainer.apply_actor_params(policy_state, latest.actor_params)


def _log_collector_actions(
    *,
    env_step: int,
    learner_step: int,
    executed_actions_real: np.ndarray,
    intervened_mask: np.ndarray,
) -> None:
    debug_logger = logging.getLogger("openpi.rlt.actor_sample")
    if not debug_logger.handlers:
        return

    array = np.asarray(executed_actions_real, dtype=np.float32)
    flat = array.reshape(-1)
    rendered = np.array2string(
        array,
        separator=", ",
        max_line_width=1000,
        formatter={"float_kind": lambda x: f"{x:.4f}"},
    )

    prefix = f"[collector env_step={env_step} learner_step={learner_step}]"
    debug_logger.info(
        "%s executed_action_real: shape=%s mean=%.4f std=%.4f max_abs=%.4f values=%s intervened_mask=%s",
        prefix,
        array.shape,
        float(flat.mean()),
        float(flat.std()),
        float(np.abs(flat).max()),
        rendered,
        np.asarray(intervened_mask, dtype=bool).astype(np.int32).tolist(),
    )


def _check_learner_status(status_queue: mp.queues.Queue) -> None:
    while True:
        try:
            message = status_queue.get_nowait()
        except queue.Empty:
            return
        if isinstance(message, LearnerError):
            raise RuntimeError(f"Learner process failed: {message.message}\n{message.traceback}")


def _shutdown_learner(
    learner_process: mp.Process,
    sample_queue: mp.queues.Queue,
    *,
    final_step: int,
    graceful_stop_sent: bool,
    join_timeout_s: float = 60.0,
) -> bool:
    if learner_process.pid is None:
        return graceful_stop_sent
    if learner_process.is_alive() and not graceful_stop_sent:
        sample_queue.put(StopSignal(final_step=max(final_step, 0)))
        graceful_stop_sent = True
    learner_process.join(timeout=join_timeout_s)
    if learner_process.is_alive():
        learner_process.terminate()
        learner_process.join(timeout=join_timeout_s)
    return graceful_stop_sent


def _wait_for_learner_init(status_queue: mp.queues.Queue) -> LearnerInit:
    while True:
        message = status_queue.get()
        if isinstance(message, LearnerInit):
            return message
        if isinstance(message, LearnerError):
            raise RuntimeError(f"Learner process failed: {message.message}\n{message.traceback}")


def _learner_main(
    args: Args,
    checkpoint_root: pathlib.Path,
    state_dim: int,
    action_dim: int,
    actor_hidden_dim: int,
    sample_queue: mp.queues.Queue,
    policy_queue: mp.queues.Queue,
    status_queue: mp.queues.Queue,
) -> None:
    try:
        logging.basicConfig(level=logging.INFO, force=True)
        # Let the collector process own Ctrl-C handling so it can enqueue StopSignal
        # and give the learner a chance to flush replay artifacts before exit.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        _trainer.set_actor_sample_debug(True, log_path=checkpoint_root / "actor_sample.log")

        train_config = _config.get_config(args.config)
        data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
        norm_stats = (
            _checkpoints.load_norm_stats(args.init_checkpoint_dir / "assets", data_config.asset_id)
            if data_config.asset_id is not None
            else None
        )

        rng = jax.random.key(args.seed)
        replay = _replay_buffer.ReplayBuffer(args.replay_capacity, state_dim=state_dim, action_dim=action_dim)
        rng, actor_rng = jax.random.split(rng)
        # Stage 2 must start from the new RL actor initialization rather than the
        # Stage 1 actor weights. The small-output-head init was added after the
        # existing Stage 1 checkpoints were trained, so reusing those actor params
        # would silently discard the intended initialization.
        actor_model, actor_state = _trainer.init_actor_state(
            actor_rng,
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=actor_hidden_dim,
            learning_rate=args.actor_lr,
        )
        critic_model, critic_state = _trainer.init_critic_state(
            rng,
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=train_config.model.rlt_actor_hidden_dim,
            learning_rate=args.critic_lr,
        )

        start_step = 0
        last_saved_step = -1
        if args.resume:
            last_step = _checkpointing.latest_step(checkpoint_root)
            if last_step is not None:
                step_dir = checkpoint_root / str(last_step)
                actor_state = _trainer.restore_actor_state(
                    actor_model,
                    _checkpointing.restore_bundle(step_dir, "policy_state"),
                    args.actor_lr,
                )
                critic_state = _trainer.restore_critic_state(
                    critic_model,
                    _checkpointing.restore_bundle(step_dir, "critic_state"),
                    args.critic_lr,
                )
                start_step = last_step + 1
                last_saved_step = last_step
                logging.info("Resumed online RLT state from %s", step_dir)

        if args.record_repro:
            repro_dir = checkpoint_root / "repro"
            _repro.save_initial_state(repro_dir, "policy_state", _trainer.bundle_actor_train_state(actor_state))
            _repro.save_initial_state(repro_dir, "critic_state", _trainer.bundle_critic_state(critic_state))

        status_queue.put(LearnerInit(start_step=start_step, actor_params=_trainer.bundle_actor_params(actor_state)))

        np_rng = np.random.default_rng(args.seed)
        metrics: list[dict[str, float]] = []
        latest_env_step = start_step

        while True:
            message = sample_queue.get()
            if isinstance(message, StopSignal):
                final_step = max(message.final_step, 0)
                _checkpointing.save_checkpoint(
                    checkpoint_root,
                    final_step,
                    policy_state=_trainer.bundle_actor_train_state(actor_state),
                    critic_state=_trainer.bundle_critic_state(critic_state),
                    norm_stats=norm_stats,
                    asset_id=data_config.asset_id,
                )
                return

            if not isinstance(message, ReplayItem):
                raise TypeError(f"Unsupported learner message: {type(message)!r}")

            replay.add(
                state=message.state,
                action=message.action,
                reference_action=message.reference_action,
                reward=message.reward,
                next_state=message.next_state,
                next_reference_action=message.next_reference_action,
                bootstrap_steps=message.bootstrap_steps,
                done=message.done,
            )
            latest_env_step = max(latest_env_step, message.env_step)

            if len(replay) >= args.batch_size and latest_env_step >= args.warmup_steps:
                for _ in range(args.utd_ratio):
                    batch = _to_jax_batch(replay.sample(args.batch_size, rng=np_rng))
                    rng, critic_rng = jax.random.split(rng)
                    critic_state, critic_info = _trainer.critic_step_with_actor(
                        actor_model,
                        actor_state,
                        critic_model,
                        critic_state,
                        batch,
                        critic_rng,
                        args.online,
                    )
                    metrics.append({k: float(np.asarray(v)) for k, v in critic_info.items()})
                    if critic_state.step % args.online.actor_update_interval == 0:
                        rng, actor_rng = jax.random.split(rng)
                        actor_state, actor_info = _trainer.actor_step_with_actor(
                            actor_model,
                            critic_model,
                            critic_state,
                            actor_state,
                            batch,
                            actor_rng,
                            args.online,
                        )
                        metrics.append({k: float(np.asarray(v)) for k, v in actor_info.items()})
                        _publish_policy_update(policy_queue, actor_state)

            if latest_env_step % args.log_interval == 0 and metrics:
                reduced = {
                    key: float(np.mean([entry[key] for entry in metrics if key in entry]))
                    for key in sorted({key for entry in metrics for key in entry})
                }
                logging.info("Learner step %d: %s", latest_env_step, ", ".join(f"{k}={v:.4f}" for k, v in reduced.items()))
                metrics.clear()

            if (
                latest_env_step % args.save_interval == 0
                and latest_env_step > start_step
                and latest_env_step != last_saved_step
            ):
                _checkpointing.save_checkpoint(
                    checkpoint_root,
                    latest_env_step,
                    policy_state=_trainer.bundle_actor_train_state(actor_state),
                    critic_state=_trainer.bundle_critic_state(critic_state),
                    norm_stats=norm_stats,
                    asset_id=data_config.asset_id,
                )
                last_saved_step = latest_env_step
    except Exception as exc:  # pragma: no cover - best effort propagation across processes.
        status_queue.put(LearnerError(message=str(exc), traceback=traceback.format_exc()))
        raise


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)

    train_config = _config.get_config(args.config)
    if not getattr(train_config.model, "use_rlt", False):
        raise ValueError(f"Config {args.config} must enable use_rlt=True.")
    if args.vla_replan_horizon_scale <= 0:
        raise ValueError(
            f"vla_replan_horizon_scale ({args.vla_replan_horizon_scale}) must be a positive integer."
        )
    online_action_horizon = getattr(train_config.model, "rlt_action_horizon", train_config.model.action_horizon)
    online_env_action_dim = getattr(train_config.model, "rlt_env_action_dim", train_config.model.action_dim)
    vla_replan_horizon = online_action_horizon * args.vla_replan_horizon_scale
    if vla_replan_horizon > train_config.model.action_horizon:
        raise ValueError(
            "VLA replan horizon "
            f"({vla_replan_horizon} = {args.vla_replan_horizon_scale} * {online_action_horizon}) "
            f"must be <= model action_horizon ({train_config.model.action_horizon})"
        )
    if online_action_horizon % args.chunk_stride != 0:
        raise ValueError(
            f"chunk_stride ({args.chunk_stride}) must divide online RLT action horizon ({online_action_horizon})"
        )
    # `_resolve_reference_chunk` slices a contiguous length-H window from a single cached VLA plan.
    # For a chunk starting at step s, `plan_offset = s - plan_step` is bounded by `vla_replan_horizon - chunk_stride`
    # (the farthest chunk_stride-aligned start before the next replan boundary), so we need the plan to cover
    # `plan_offset + online_action_horizon` actions, i.e. model.action_horizon must be at least
    # `vla_replan_horizon + (online_action_horizon - chunk_stride)`.
    required_model_horizon = vla_replan_horizon + (online_action_horizon - args.chunk_stride)
    if required_model_horizon > train_config.model.action_horizon:
        raise ValueError(
            "model action_horizon is too small for the requested (vla_replan_horizon_scale, chunk_stride): "
            f"need model.action_horizon >= vla_replan_horizon + (online_action_horizon - chunk_stride) = "
            f"{vla_replan_horizon} + ({online_action_horizon} - {args.chunk_stride}) = {required_model_horizon}, "
            f"but model.action_horizon is {train_config.model.action_horizon}. "
            "Either reduce vla_replan_horizon_scale or increase chunk_stride (up to online_action_horizon)."
        )

    checkpoint_root, _ = _checkpointing.initialize_checkpoint_dir(
        args.checkpoint_dir,
        overwrite=args.overwrite,
        resume=args.resume,
    )
    _trainer.set_actor_sample_debug(True, log_path=checkpoint_root / "actor_sample.log")
    debug_logger = logging.getLogger("openpi.rlt.actor_sample")
    debug_logger.info("online_debug_logger_initialized path=%s", checkpoint_root / "actor_sample.log")
    repro_dir = checkpoint_root / "repro"
    transition_recorder = _repro.TransitionRecorder(repro_dir, shard_size=args.repro_shard_size) if args.record_repro else None

    stage1_policy_config = train_config
    if getattr(train_config.model, "use_rlt", False) and getattr(train_config.model, "rlt_actor_enabled", False):
        # Stage 2 only reuses the Stage 1 backbone/reference policy features. The online actor
        # is always initialized separately below, so skip loading any historical rlt_actor params.
        stage1_policy_config = dataclasses.replace(
            train_config,
            model=dataclasses.replace(train_config.model, rlt_actor_enabled=False),
        )

    policy = _policy_config.create_trained_policy(stage1_policy_config, args.init_checkpoint_dir)
    encode_prefix = nnx_utils.module_jit(policy.model.encode_prefix)
    compute_rl_token = nnx_utils.module_jit(policy.model.compute_rl_token)
    sample_reference_actions = nnx_utils.module_jit(policy.model.sample_reference_actions)
    actor_hidden_dim = train_config.model.rlt_actor_hidden_dim
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    norm_stats = (
        _checkpoints.load_norm_stats(args.init_checkpoint_dir / "assets", data_config.asset_id)
        if data_config.asset_id is not None
        else None
    )

    env_config = _airbot_env.AirbotRLTEnvConfig.from_env()
    env = _airbot_env.AirbotRLTEnv(env_config)

    rng = jax.random.key(args.seed)
    initial_observation = env.reset()
    initial_state = _compute_rlt_state(
        policy,
        encode_prefix,
        compute_rl_token,
        initial_observation,
    )
    rng, plan_rng = jax.random.split(rng)
    initial_reference_plan = _compute_reference_plan(
        policy,
        sample_reference_actions,
        plan_rng,
        initial_observation,
        diffusion_steps=args.diffusion_steps,
        env_action_dim=online_env_action_dim,
    )
    initial_reference = _resolve_reference_chunk(
        {0: initial_reference_plan},
        step=0,
        action_horizon=online_action_horizon,
        replan_origin_step=0,
        vla_replan_horizon=vla_replan_horizon,
    )
    state_dim = int(initial_state.shape[-1])
    action_dim = int(initial_reference.shape[-1])
    rng, actor_rng = jax.random.split(rng)
    actor_model, actor_state = _trainer.init_actor_state(
        actor_rng,
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=actor_hidden_dim,
        learning_rate=args.actor_lr,
    )
    # Keep the local actor on the same fresh Stage 2 initialization as the learner.
    # Do not replace these params from the Stage 1 checkpoint; see the learner-side
    # note above for the historical reason.

    if args.record_repro:
        _repro.write_manifest(
            repro_dir,
            {
                "schema_version": 1,
                "args": _repro.to_jsonable(args),
                "state_dim": state_dim,
                "action_dim": action_dim,
                "actor_hidden_dim": actor_hidden_dim,
                "asset_id": data_config.asset_id,
                "use_quantile_norm": data_config.use_quantile_norm,
                "checkpoint_root": str(checkpoint_root),
                "init_checkpoint_dir": str(args.init_checkpoint_dir.resolve()),
                "env": {
                    "OPENPI_AIRBOT_RLT_CONFIG": os.environ.get("OPENPI_AIRBOT_RLT_CONFIG"),
                    "OPENPI_AIRBOT_PROMPT": os.environ.get("OPENPI_AIRBOT_PROMPT"),
                    "OPENPI_AIRBOT_FAKE_ENV": os.environ.get("OPENPI_AIRBOT_FAKE_ENV"),
                },
                "runtime": _repro.collect_runtime_metadata(),
            },
        )
        _repro.save_resolved_env_config(repro_dir, env_config)
        _repro.save_norm_stats_snapshot(repro_dir, norm_stats, data_config.asset_id)

    mp_context = mp.get_context("spawn")
    sample_queue = mp_context.Queue(maxsize=max(args.batch_size * args.utd_ratio, 1024))
    policy_queue = mp_context.Queue(maxsize=1)
    status_queue = mp_context.Queue()
    learner_process = mp_context.Process(
        target=_learner_main,
        args=(
            args,
            checkpoint_root,
            state_dim,
            action_dim,
            actor_hidden_dim,
            sample_queue,
            policy_queue,
            status_queue,
        ),
        name="rlt-learner",
    )
    learner_process.start()

    init_message = _wait_for_learner_init(status_queue)
    start_step = init_message.start_step
    if args.record_repro:
        manifest = _repro.load_manifest(repro_dir)
        manifest["start_step"] = int(start_step)
        _repro.write_manifest(repro_dir, manifest)
    debug_logger.info("received_initial_actor_step=%d", int(init_message.actor_params["step"]))
    actor_state = _trainer.apply_actor_params(actor_state, init_message.actor_params)
    transition_sink = lambda item: _queue_transition(sample_queue, item, transition_recorder=transition_recorder)

    observation = initial_observation
    metrics: list[dict[str, float]] = []
    reward_history: list[float] = [0.0] * start_step
    current_step = start_step
    state_history: dict[int, np.ndarray] = {current_step: initial_state} if current_step == 0 else {}
    reference_plan_history: dict[int, np.ndarray] = {current_step: initial_reference_plan} if current_step == 0 else {}
    action_history: dict[int, np.ndarray] = {}
    intervention_history: dict[int, bool] = {}
    pending_chunks: list[int] = [current_step] if current_step == 0 else []
    replan_origin_step = current_step
    graceful_stop_sent = False

    try:
        while current_step < args.max_env_steps:
            _check_learner_status(status_queue)
            actor_state = _drain_policy_updates(policy_queue, actor_state, sync_logger=debug_logger)

            if current_step in state_history:
                state = state_history[current_step]
            else:
                state = _compute_rlt_state(
                    policy,
                    encode_prefix,
                    compute_rl_token,
                    observation,
                )
                state_history[current_step] = state
                pending_chunks.append(current_step)

            if (current_step - replan_origin_step) % vla_replan_horizon == 0 and current_step not in reference_plan_history:
                rng, plan_rng = jax.random.split(rng)
                reference_plan_history[current_step] = _compute_reference_plan(
                    policy,
                    sample_reference_actions,
                    plan_rng,
                    observation,
                    diffusion_steps=args.diffusion_steps,
                    env_action_dim=online_env_action_dim,
                )

            reference_action = _resolve_reference_chunk(
                reference_plan_history,
                step=current_step,
                action_horizon=online_action_horizon,
                replan_origin_step=replan_origin_step,
                vla_replan_horizon=vla_replan_horizon,
            )

            if current_step < args.warmup_steps:
                action = reference_action
            else:
                rng, actor_sample_rng = jax.random.split(rng)
                action = _actor_action(actor_model, actor_state, actor_sample_rng, state, reference_action)

            # action / reference_action are in model space; convert to real env actions at the env boundary.
            action_chunk_norm = action.reshape(online_action_horizon, online_env_action_dim)
            action_chunk_real = _denormalize_action(
                norm_stats,
                data_config.use_quantile_norm,
                action_chunk_norm,
                state=observation["state"],
                use_delta_joint_actions=data_config.use_delta_joint_actions,
            )
            next_observation, reward, done, info = env.step(action_chunk_real)
            step_rewards = _extract_step_rewards(reward, info, online_action_horizon)
            # executed_actions returned by env are in real joint-position space; convert them back into
            # the same model space as reference_action before storing them in replay.
            executed_actions_real = _extract_executed_actions(
                info,
                executed_steps=len(step_rewards),
                fallback_actions=action_chunk_real,
            )
            intervened_mask = _extract_intervened_mask(info, executed_steps=len(step_rewards))
            executed_actions_norm = _normalize_action(
                norm_stats,
                data_config.use_quantile_norm,
                executed_actions_real,
                state=observation["state"],
                use_delta_joint_actions=data_config.use_delta_joint_actions,
            )
            _log_collector_actions(
                env_step=current_step,
                learner_step=int(actor_state.step),
                executed_actions_real=executed_actions_real,
                intervened_mask=intervened_mask,
            )
            _record_action_history(
                action_history,
                intervention_history,
                start_step=current_step,
                executed_actions=executed_actions_norm,
                intervened_mask=intervened_mask,
            )
            reward_history.extend(step_rewards)

            step_observations = info.get("step_observations")
            if step_observations is not None:
                if not isinstance(step_observations, list):
                    raise ValueError("info['step_observations'] must be a list when provided.")
                if len(step_observations) != len(step_rewards):
                    raise ValueError(
                        "info['step_observations'] length "
                        f"({len(step_observations)}) must match executed_steps ({len(step_rewards)})"
                    )

            sampled_offsets = list(range(args.chunk_stride, len(step_rewards) + 1, args.chunk_stride))
            for offset in sampled_offsets:
                sample_step = current_step + offset
                if sample_step in state_history:
                    continue
                if offset == len(step_rewards):
                    sample_observation = next_observation
                else:
                    if step_observations is None:
                        raise ValueError(
                            "env.step() must provide info['step_observations'] when chunk_stride < action_horizon "
                            "so the trainer can subsample replay entries from intermediate states."
                        )
                    sample_observation = step_observations[offset - 1]
                state_history[sample_step] = _compute_rlt_state(
                    policy,
                    encode_prefix,
                    compute_rl_token,
                    sample_observation,
                )
                pending_chunks.append(sample_step)
                if (sample_step - replan_origin_step) % vla_replan_horizon == 0:
                    rng, plan_rng = jax.random.split(rng)
                    reference_plan_history[sample_step] = _compute_reference_plan(
                        policy,
                        sample_reference_actions,
                        plan_rng,
                        sample_observation,
                        diffusion_steps=args.diffusion_steps,
                        env_action_dim=online_env_action_dim,
                    )

            current_step += len(step_rewards)

            terminal_state: np.ndarray | None = None
            if done:
                if current_step in state_history:
                    terminal_state = state_history[current_step]
                else:
                    terminal_state = _compute_rlt_state(
                        policy,
                        encode_prefix,
                        compute_rl_token,
                        next_observation,
                    )
                    state_history[current_step] = terminal_state
                if (current_step - replan_origin_step) % vla_replan_horizon == 0 and current_step not in reference_plan_history:
                    rng, plan_rng = jax.random.split(rng)
                    reference_plan_history[current_step] = _compute_reference_plan(
                        policy,
                        sample_reference_actions,
                        plan_rng,
                        next_observation,
                        diffusion_steps=args.diffusion_steps,
                        env_action_dim=online_env_action_dim,
                    )

            _flush_ready_chunks(
                pending_chunks,
                state_history,
                reference_plan_history,
                action_history,
                intervention_history,
                reward_history,
                transition_sink,
                current_step=current_step,
                action_horizon=online_action_horizon,
                replan_origin_step=replan_origin_step,
                vla_replan_horizon=vla_replan_horizon,
                discount=args.online.discount,
                terminal=done,
                terminal_state=terminal_state,
            )

            metrics.append(
                {
                    "reward": reward,
                    "done": float(done),
                    "policy_step": float(np.asarray(actor_state.step)),
                }
            )
            for key, value in info.items():
                if key != "step_rewards" and np.isscalar(value) and not isinstance(value, str):
                    metrics.append({f"env/{key}": float(value)})

            if current_step % args.log_interval == 0 and metrics:
                reduced = {
                    key: float(np.mean([entry[key] for entry in metrics if key in entry]))
                    for key in sorted({key for entry in metrics for key in entry})
                }
                logging.info("Step %d: %s", current_step, ", ".join(f"{k}={v:.4f}" for k, v in reduced.items()))
                metrics.clear()

            if done:
                observation = env.reset()
                action_history.clear()
                intervention_history.clear()
                state_history.clear()
                reference_plan_history.clear()
                pending_chunks.clear()
                replan_origin_step = current_step
            else:
                observation = next_observation

        if pending_chunks:
            final_state = _compute_rlt_state(
                policy,
                encode_prefix,
                compute_rl_token,
                observation,
            )
            state_history[current_step] = final_state
            if (current_step - replan_origin_step) % vla_replan_horizon == 0 and current_step not in reference_plan_history:
                rng, plan_rng = jax.random.split(rng)
                reference_plan_history[current_step] = _compute_reference_plan(
                    policy,
                    sample_reference_actions,
                    plan_rng,
                    observation,
                    diffusion_steps=args.diffusion_steps,
                    env_action_dim=online_env_action_dim,
                )
            _flush_ready_chunks(
                pending_chunks,
                state_history,
                reference_plan_history,
                action_history,
                intervention_history,
                reward_history,
                transition_sink,
                current_step=current_step,
                action_horizon=online_action_horizon,
                replan_origin_step=replan_origin_step,
                vla_replan_horizon=vla_replan_horizon,
                discount=args.online.discount,
            )
            _flush_ready_chunks(
                pending_chunks,
                state_history,
                reference_plan_history,
                action_history,
                intervention_history,
                reward_history,
                transition_sink,
                current_step=current_step,
                action_horizon=online_action_horizon,
                replan_origin_step=replan_origin_step,
                vla_replan_horizon=vla_replan_horizon,
                discount=args.online.discount,
                terminal=True,
                terminal_state=final_state,
            )

        graceful_stop_sent = _shutdown_learner(
            learner_process,
            sample_queue,
            final_step=max(current_step - 1, 0),
            graceful_stop_sent=graceful_stop_sent,
        )
        _check_learner_status(status_queue)
        if learner_process.exitcode not in (0, None):
            raise RuntimeError(f"Learner process exited with code {learner_process.exitcode}")
    finally:
        final_step = max(current_step - 1, 0)
        if transition_recorder is not None:
            transition_recorder.close()
            _repro.save_stop_signal(repro_dir, final_step)
        env.close()
        graceful_stop_sent = _shutdown_learner(
            learner_process,
            sample_queue,
            final_step=final_step,
            graceful_stop_sent=graceful_stop_sent,
        )


if __name__ == "__main__":
    main(tyro.cli(Args))
