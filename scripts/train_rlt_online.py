import dataclasses
import logging
import pathlib
from typing import Any

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
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    save_interval: int = 1_000
    log_interval: int = 100
    overwrite: bool = False
    resume: bool = False
    seed: int = 0

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


def _compute_features(
    policy,
    extract_features,
    rng: jax.Array,
    observation: dict[str, Any],
    *,
    diffusion_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    _, batched_observation = policy.prepare_observation(observation)
    _, _, rlt_state, reference_actions = extract_features(rng, batched_observation, num_steps=diffusion_steps)
    state = np.asarray(rlt_state[0], dtype=np.float32)
    reference = np.asarray(reference_actions[0].reshape(-1), dtype=np.float32)
    return state, reference


def _slice_norm_stats(stats, dim: int):
    """Truncate a NormStats to the first `dim` elements along the last axis."""
    from openpi.shared.normalize import NormStats
    return NormStats(
        mean=stats.mean[..., :dim],
        std=stats.std[..., :dim],
        q01=stats.q01[..., :dim] if stats.q01 is not None else None,
        q99=stats.q99[..., :dim] if stats.q99 is not None else None,
    )


def _denormalize_action(norm_stats, use_quantiles: bool, actions: np.ndarray) -> np.ndarray:
    """Convert a (steps, env_action_dim) action from normalized model space to real joint-position space."""
    if norm_stats is None or "actions" not in norm_stats:
        return actions
    env_dim = actions.shape[-1]
    action_stats = {"actions": _slice_norm_stats(norm_stats["actions"], env_dim)}
    outputs = _transforms.Unnormalize(action_stats, use_quantiles=use_quantiles)({"actions": actions})
    return np.asarray(outputs["actions"], dtype=np.float32)


def _normalize_action(norm_stats, use_quantiles: bool, actions: np.ndarray) -> np.ndarray:
    """Convert a (steps, env_action_dim) action from real joint-position space back to normalized model space."""
    if norm_stats is None or "actions" not in norm_stats:
        return actions
    env_dim = actions.shape[-1]
    action_stats = {"actions": _slice_norm_stats(norm_stats["actions"], env_dim)}
    normalized = _transforms.Normalize(action_stats, use_quantiles=use_quantiles)({"actions": actions})
    return np.asarray(normalized["actions"], dtype=np.float32)


def _actor_action(policy_state, rng: jax.Array, state: np.ndarray, reference_action: np.ndarray) -> np.ndarray:
    model = nnx.merge(policy_state.model_def, policy_state.params)
    action = _trainer.actor_sample(
        model,
        jnp.asarray(state)[None, ...],
        jnp.asarray(reference_action)[None, ...],
        rng,
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
    feature_history: dict[int, tuple[np.ndarray, np.ndarray]],
    action_history: dict[int, np.ndarray],
    intervention_history: dict[int, bool],
    reward_history: list[float],
    replay: _replay_buffer.ReplayBuffer,
    *,
    current_step: int,
    action_horizon: int,
    env_action_dim: int,
    discount: float,
    terminal: bool = False,
    terminal_features: tuple[np.ndarray, np.ndarray] | None = None,
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
        state, reference_action = feature_history[chunk_step]
        reference_chunk = np.asarray(reference_action, dtype=np.float32).reshape(action_horizon, env_action_dim).copy()
        action_chunk = reference_chunk.copy()
        for offset in range(bootstrap_steps):
            step = chunk_step + offset
            executed_action = action_history[step]
            action_chunk[offset] = executed_action
            if intervention_history.get(step, False):
                reference_chunk[offset] = executed_action

        if bootstrap_steps == action_horizon and not terminal:
            next_state, next_reference_action = feature_history[chunk_step + action_horizon]
            done = False
        else:
            if terminal_features is None:
                raise ValueError("terminal_features are required when flushing truncated chunk transitions.")
            next_state, next_reference_action = terminal_features
            done = True

        replay.add(
            state=state,
            action=action_chunk.reshape(-1),
            reference_action=reference_chunk.reshape(-1),
            reward=discounted_reward,
            next_state=next_state,
            next_reference_action=next_reference_action,
            bootstrap_steps=bootstrap_steps,
            done=done,
        )
        pending_chunks.pop(0)
        emitted += 1
    return emitted


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)

    train_config = _config.get_config(args.config)
    if not getattr(train_config.model, "use_rlt", False):
        raise ValueError(f"Config {args.config} must enable use_rlt=True.")
    online_action_horizon = getattr(train_config.model, "rlt_action_horizon", train_config.model.action_horizon)
    online_env_action_dim = getattr(train_config.model, "rlt_env_action_dim", train_config.model.action_dim)
    if online_action_horizon % args.chunk_stride != 0:
        raise ValueError(
            f"chunk_stride ({args.chunk_stride}) must divide online RLT action horizon ({online_action_horizon})"
        )

    checkpoint_root, resuming = _checkpointing.initialize_checkpoint_dir(
        args.checkpoint_dir,
        overwrite=args.overwrite,
        resume=args.resume,
    )

    policy = _policy_config.create_trained_policy(train_config, args.init_checkpoint_dir)
    extract_features = nnx_utils.module_jit(policy.model.extract_rlt_features)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    norm_stats = (
        _checkpoints.load_norm_stats(args.init_checkpoint_dir / "assets", data_config.asset_id)
        if data_config.asset_id is not None
        else None
    )

    env = _airbot_env.AirbotRLTEnv(_airbot_env.AirbotRLTEnvConfig.from_env())

    rng = jax.random.key(args.seed)
    rng, feature_rng = jax.random.split(rng)
    initial_observation = env.reset()
    initial_state, initial_reference = _compute_features(
        policy,
        extract_features,
        feature_rng,
        initial_observation,
        diffusion_steps=args.diffusion_steps,
    )
    state_dim = int(initial_state.shape[-1])
    action_dim = int(initial_reference.shape[-1])
    replay = _replay_buffer.ReplayBuffer(args.replay_capacity, state_dim=state_dim, action_dim=action_dim)

    policy_state = _trainer.init_policy_state(policy.model, args.actor_lr)
    critic_model, critic_state = _trainer.init_critic_state(
        rng,
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=train_config.model.rlt_actor_hidden_dim,
        learning_rate=args.critic_lr,
    )

    start_step = 0
    if resuming:
        last_step = _checkpointing.latest_step(checkpoint_root)
        if last_step is not None:
            step_dir = checkpoint_root / str(last_step)
            policy_state = _trainer.restore_policy_state(
                policy.model,
                _checkpointing.restore_bundle(step_dir, "policy_state"),
                args.actor_lr,
            )
            critic_state = _trainer.restore_critic_state(
                critic_model,
                _checkpointing.restore_bundle(step_dir, "critic_state"),
                args.critic_lr,
            )
            start_step = last_step + 1
            logging.info("Resumed online RLT state from %s", step_dir)

    observation = initial_observation
    np_rng = np.random.default_rng(args.seed)
    metrics: list[dict[str, float]] = []
    reward_history: list[float] = [0.0] * start_step
    feature_history: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    action_history: dict[int, np.ndarray] = {}
    intervention_history: dict[int, bool] = {}
    pending_chunks: list[int] = []
    current_step = start_step

    try:
        while current_step < args.max_env_steps:
            if current_step in feature_history:
                state, reference_action = feature_history[current_step]
            else:
                rng, feature_rng = jax.random.split(rng)
                state, reference_action = _compute_features(
                    policy,
                    extract_features,
                    feature_rng,
                    observation,
                    diffusion_steps=args.diffusion_steps,
                )
                feature_history[current_step] = (state, reference_action)
                pending_chunks.append(current_step)

            if current_step < args.warmup_steps:
                action = reference_action
            else:
                rng, actor_sample_rng = jax.random.split(rng)
                action = _actor_action(policy_state, actor_sample_rng, state, reference_action)

            # action / reference_action are in normalized model space; denormalize at env boundary.
            executed_chunk_norm = action.reshape(online_action_horizon, online_env_action_dim)
            executed_chunk_real = _denormalize_action(norm_stats, data_config.use_quantile_norm, executed_chunk_norm)
            next_observation, reward, done, info = env.step(executed_chunk_real)
            step_rewards = _extract_step_rewards(reward, info, online_action_horizon)
            # executed_actions returned by env are in real joint-position space; normalize back before
            # storing in replay so that replay actions stay in the same space as reference_action.
            executed_actions_real = _extract_executed_actions(
                info,
                executed_steps=len(step_rewards),
                fallback_actions=executed_chunk_real,
            )
            intervened_mask = _extract_intervened_mask(info, executed_steps=len(step_rewards))
            executed_actions_norm = _normalize_action(
                norm_stats, data_config.use_quantile_norm, executed_actions_real
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
                if sample_step in feature_history:
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
                rng, feature_rng = jax.random.split(rng)
                feature_history[sample_step] = _compute_features(
                    policy,
                    extract_features,
                    feature_rng,
                    sample_observation,
                    diffusion_steps=args.diffusion_steps,
                )
                pending_chunks.append(sample_step)

            current_step += len(step_rewards)

            terminal_features: tuple[np.ndarray, np.ndarray] | None = None
            if done:
                if current_step in feature_history:
                    terminal_features = feature_history[current_step]
                else:
                    rng, terminal_rng = jax.random.split(rng)
                    terminal_features = _compute_features(
                        policy,
                        extract_features,
                        terminal_rng,
                        next_observation,
                        diffusion_steps=args.diffusion_steps,
                    )
                    feature_history[current_step] = terminal_features

            _flush_ready_chunks(
                pending_chunks,
                feature_history,
                action_history,
                intervention_history,
                reward_history,
                replay,
                current_step=current_step,
                action_horizon=online_action_horizon,
                env_action_dim=online_env_action_dim,
                discount=args.online.discount,
                terminal=done,
                terminal_features=terminal_features,
            )

            if len(replay) >= args.batch_size and current_step >= args.warmup_steps:
                for _ in range(args.utd_ratio):
                    batch = _to_jax_batch(replay.sample(args.batch_size, rng=np_rng))
                    rng, critic_rng = jax.random.split(rng)
                    critic_state, critic_info = _trainer.critic_step(
                        critic_model,
                        critic_state,
                        policy_state,
                        batch,
                        critic_rng,
                        args.online,
                    )
                    metrics.append({k: float(np.asarray(v)) for k, v in critic_info.items()})
                    if critic_state.step % args.online.actor_update_interval == 0:
                        rng, actor_rng = jax.random.split(rng)
                        policy_state, actor_info = _trainer.actor_step(
                            critic_model,
                            critic_state,
                            policy_state,
                            batch,
                            actor_rng,
                            args.online,
                        )
                        metrics.append({k: float(np.asarray(v)) for k, v in actor_info.items()})

            metrics.append(
                {
                    "reward": reward,
                    "replay_size": float(len(replay)),
                    "done": float(done),
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

            if current_step % args.save_interval == 0 and current_step > start_step:
                _checkpointing.save_checkpoint(
                    checkpoint_root,
                    current_step,
                    policy_state=_trainer.bundle_policy_state(policy_state),
                    critic_state=_trainer.bundle_critic_state(critic_state),
                    norm_stats=norm_stats,
                    asset_id=data_config.asset_id,
                )

            if done:
                observation = env.reset()
                action_history.clear()
                intervention_history.clear()
                feature_history.clear()
            else:
                observation = next_observation

        if pending_chunks:
            rng, final_rng = jax.random.split(rng)
            final_features = _compute_features(
                policy,
                extract_features,
                final_rng,
                observation,
                diffusion_steps=args.diffusion_steps,
            )
            feature_history[current_step] = final_features
            _flush_ready_chunks(
                pending_chunks,
                feature_history,
                action_history,
                intervention_history,
                reward_history,
                replay,
                current_step=current_step,
                action_horizon=online_action_horizon,
                env_action_dim=online_env_action_dim,
                discount=args.online.discount,
            )
            _flush_ready_chunks(
                pending_chunks,
                feature_history,
                action_history,
                intervention_history,
                reward_history,
                replay,
                current_step=current_step,
                action_horizon=online_action_horizon,
                env_action_dim=online_env_action_dim,
                discount=args.online.discount,
                terminal=True,
                terminal_features=final_features,
            )

        _checkpointing.save_checkpoint(
            checkpoint_root,
            max(current_step - 1, 0),
            policy_state=_trainer.bundle_policy_state(policy_state),
            critic_state=_trainer.bundle_critic_state(critic_state),
            norm_stats=norm_stats,
            asset_id=data_config.asset_id,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
