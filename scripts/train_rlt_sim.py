"""
离线模拟 Stage 2 在线 RL 训练。

用 LeRobot 数据集帧替代真实机器人环境，复用 train_rlt_online.py 的所有训练逻辑，
并在 trainer.py 的 actor_sample / actor_step / critic_step 中记录详细统计，
以定位训练过程中 actor mean 爆炸的根源。

运行示例：
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_rlt_sim.py \\
    --config pi05_airbot_rlt \\
    --init-checkpoint-dir checkpoints/pi05_airbot_rlt_token/stage1_airbot_rlt/9999 \\
    --checkpoint-dir checkpoints/pi05_airbot_rlt/sim_debug \\
    --warmup-steps 50 --max-env-steps 500 --batch-size 32 --utd-ratio 1 --overwrite
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")

import dataclasses
import logging
import pathlib
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
import tyro

from openpi.models import model as _model_module
from openpi.policies import policy_config as _policy_config
from openpi.rlt import checkpointing as _checkpointing
from openpi.rlt import replay_buffer as _replay_buffer
from openpi.rlt import trainer as _trainer
from openpi.shared import nnx_utils
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.training.data_loader as _data_loader


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Args:
    # Stage 2 model config (defines RLT topology and Airbot transforms).
    config: str = "pi05_airbot_rlt"
    # Stage 1 checkpoint (RLToken weights).
    init_checkpoint_dir: pathlib.Path = pathlib.Path(
        "checkpoints/pi05_airbot_rlt_token/stage1_airbot_rlt/9999"
    )
    # Output directory for Stage 2 checkpoints.
    checkpoint_dir: pathlib.Path = pathlib.Path("checkpoints/pi05_airbot_rlt/sim_debug")
    # Stage 1 config name – used to load the real dataset.
    stage1_config: str = "pi05_airbot_rlt_token"

    max_env_steps: int = 500
    warmup_steps: int = 50
    replay_capacity: int = 10_000
    batch_size: int = 32
    utd_ratio: int = 1
    diffusion_steps: int = 10
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    save_interval: int = 1_000
    log_interval: int = 10
    overwrite: bool = False
    resume: bool = False
    seed: int = 0

    online: _trainer.OnlineRLTConfig = dataclasses.field(
        default_factory=_trainer.OnlineRLTConfig
    )


# ---------------------------------------------------------------------------
# Helper functions (copied / adapted from train_rlt_online.py)
# ---------------------------------------------------------------------------


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


def _discounted_return(
    reward_history: list[float], start_step: int, horizon: int, discount: float
) -> float:
    return float(
        sum(
            (discount**offset) * reward_history[start_step + offset]
            for offset in range(horizon)
        )
    )


def _flush_ready_chunks(
    pending_chunks: list[int],
    feature_history: dict[int, tuple[np.ndarray, np.ndarray]],
    action_history: dict[int, np.ndarray],
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
    """Flush completed (or terminal) chunks from pending list into the replay buffer."""
    emitted = 0
    while pending_chunks:
        chunk_step = pending_chunks[0]
        available_steps = current_step - chunk_step
        if available_steps <= 0:
            break
        if available_steps < action_horizon and not terminal:
            break

        bootstrap_steps = min(action_horizon, available_steps)
        discounted_reward = _discounted_return(
            reward_history, chunk_step, bootstrap_steps, discount
        )
        state, reference_action = feature_history[chunk_step]
        reference_chunk = (
            np.asarray(reference_action, dtype=np.float32)
            .reshape(action_horizon, env_action_dim)
            .copy()
        )
        action_chunk = reference_chunk.copy()
        for offset in range(bootstrap_steps):
            step = chunk_step + offset
            if step in action_history:
                action_chunk[offset] = action_history[step]

        if bootstrap_steps == action_horizon and not terminal:
            next_state, next_reference_action = feature_history[chunk_step + action_horizon]
            done = False
        else:
            if terminal_features is None:
                raise ValueError(
                    "terminal_features required when flushing a truncated chunk."
                )
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


def _actor_action(
    actor_model, actor_state, rng: jax.Array, state: np.ndarray, reference_action: np.ndarray
) -> np.ndarray:
    action = _trainer.actor_sample_params(
        actor_model,
        actor_state.params,
        jnp.asarray(state)[None, ...],
        jnp.asarray(reference_action)[None, ...],
        rng,
    )
    return np.asarray(action[0], dtype=np.float32)


# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------


def _to_jax(v: Any) -> jax.Array:
    if isinstance(v, torch.Tensor):
        return jnp.asarray(v.numpy())
    return jnp.asarray(np.asarray(v))


def _compute_frame_features(
    transformed_dataset,
    extract_features,
    rng: jax.Array,
    frame_idx: int,
    *,
    diffusion_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract (rlt_state, reference_action) from a single dataset frame."""
    sample = transformed_dataset[frame_idx]
    proprio = sample.get("proprio")
    batched: dict[str, Any] = {}
    for k, v in sample.items():
        if isinstance(v, dict):
            batched[k] = {
                cam: _to_jax(img)[None]
                for cam, img in v.items()
            }
        elif isinstance(v, (torch.Tensor, np.ndarray)):
            batched[k] = _to_jax(v)[None]
        # skip non-array values (e.g. strings)
    obs = _model_module.Observation.from_dict(batched)
    _, rl_token, reference_actions = extract_features(
        rng, obs, num_steps=diffusion_steps
    )
    state = np.concatenate(
        [
            np.asarray(rl_token[0], dtype=np.float32),
            np.asarray(proprio, dtype=np.float32),
        ],
        axis=-1,
    )
    reference = np.asarray(reference_actions[0].reshape(-1), dtype=np.float32)
    return state, reference


def _build_episode_index(
    raw_dataset,
) -> list[tuple[int, int]]:
    """Return list of (frame_from, frame_to_exclusive) per episode."""
    # Walk the wrapper chain to find the underlying LeRobotDataset.
    ds = raw_dataset
    while hasattr(ds, "_dataset"):
        ds = ds._dataset
    if hasattr(ds, "episode_data_index"):
        frm = ds.episode_data_index["from"]
        to_ = ds.episode_data_index["to"]
        return [(int(f), int(t)) for f, t in zip(frm, to_)]
    # Fallback: scan episode_index field.
    logging.warning("episode_data_index not found; scanning dataset (may be slow)")
    episodes: list[tuple[int, int]] = []
    prev_ep = -1
    start = 0
    for i in range(len(raw_dataset)):
        ep_idx = int(raw_dataset[i].get("episode_index", i))
        if ep_idx != prev_ep:
            if prev_ep >= 0:
                episodes.append((start, i))
            start = i
            prev_ep = ep_idx
    episodes.append((start, len(raw_dataset)))
    return episodes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    _trainer.set_actor_sample_debug(True)

    # ── Stage 2 model config ──────────────────────────────────────────────
    train_config = _config.get_config(args.config)
    if not getattr(train_config.model, "use_rlt", False):
        raise ValueError(f"Config {args.config} must enable use_rlt=True.")

    action_horizon = getattr(
        train_config.model, "rlt_action_horizon", train_config.model.action_horizon
    )
    env_action_dim = getattr(
        train_config.model, "rlt_env_action_dim", train_config.model.action_dim
    )

    # ── Checkpoint dir ────────────────────────────────────────────────────
    checkpoint_root, resuming = _checkpointing.initialize_checkpoint_dir(
        args.checkpoint_dir,
        overwrite=args.overwrite,
        resume=args.resume,
    )

    # ── Load Stage 1 model ────────────────────────────────────────────────
    logging.info("Loading Stage 1 model from %s ...", args.init_checkpoint_dir)
    stage1_policy_config = train_config
    if getattr(train_config.model, "use_rlt", False) and getattr(train_config.model, "rlt_actor_enabled", False):
        # Stage 2 only reuses the Stage 1 backbone/reference policy features. The online actor
        # is always initialized separately below, so skip loading any historical rlt_actor params.
        stage1_policy_config = dataclasses.replace(
            train_config,
            model=dataclasses.replace(train_config.model, rlt_actor_enabled=False),
        )
    policy = _policy_config.create_trained_policy(stage1_policy_config, args.init_checkpoint_dir)
    extract_features = nnx_utils.module_jit(policy.model.extract_rlt_features)

    # ── Load dataset ──────────────────────────────────────────────────────
    logging.info("Loading dataset via Stage 1 config (%s) ...", args.stage1_config)
    s1_config = _config.get_config(args.stage1_config)
    data_config_s1 = s1_config.data.create(s1_config.assets_dirs, s1_config.model)
    raw_ds = _data_loader.create_torch_dataset(
        data_config_s1, s1_config.model.action_horizon, s1_config.model
    )
    transformed_ds = _data_loader.transform_dataset(raw_ds, data_config_s1)

    episodes = _build_episode_index(raw_ds)
    logging.info(
        "Dataset: %d episodes, %d total frames", len(episodes), len(raw_ds)
    )

    # ── Initialise RL components ──────────────────────────────────────────
    rng = jax.random.key(args.seed)
    rng, feat_rng = jax.random.split(rng)

    ep0_from, ep0_to = episodes[0]
    logging.info("Computing initial features from frame 0 ...")
    initial_state, initial_reference = _compute_frame_features(
        transformed_ds,
        extract_features,
        feat_rng,
        ep0_from,
        diffusion_steps=args.diffusion_steps,
    )
    state_dim = int(initial_state.shape[-1])
    action_dim = int(initial_reference.shape[-1])
    logging.info(
        "state_dim=%d  action_dim=%d (action_horizon=%d × env_action_dim=%d)",
        state_dim, action_dim, action_horizon, env_action_dim,
    )

    replay = _replay_buffer.ReplayBuffer(
        args.replay_capacity, state_dim=state_dim, action_dim=action_dim
    )
    rng, actor_rng = jax.random.split(rng)
    actor_model, actor_state = _trainer.init_actor_state(
        actor_rng,
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=train_config.model.rlt_actor_hidden_dim,
        learning_rate=args.actor_lr,
    )
    # Stage 2 starts from a freshly initialized RL actor. We intentionally do not
    # copy the Stage 1 actor params here because those checkpoints predate the
    # small-weight / zero-bias output-head init and would overwrite it.
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
            logging.info("Resumed from %s", step_dir)

    np_rng = np.random.default_rng(args.seed)
    metrics: list[dict[str, float]] = []

    # Per-episode transient histories (keyed by absolute env step).
    feature_history: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    action_history: dict[int, np.ndarray] = {}
    # reward_history is indexed by env step; pad to start_step.
    reward_history: list[float] = [0.0] * start_step
    pending_chunks: list[int] = []

    current_step = start_step
    ep_idx = 0  # cycles through episodes

    while current_step < args.max_env_steps:
        # ── Start a new episode ───────────────────────────────────────────
        ep_from, ep_to = episodes[ep_idx % len(episodes)]
        ep_len = ep_to - ep_from
        ep_idx += 1

        feature_history.clear()
        action_history.clear()
        pending_chunks.clear()
        # reward_history is cumulative; only clear within-episode values below.

        logging.info(
            "Episode %d: frames [%d, %d), len=%d  (env step %d)",
            ep_idx, ep_from, ep_to, ep_len, current_step,
        )

        frame_in_ep = 0
        while frame_in_ep < ep_len and current_step < args.max_env_steps:
            abs_frame = ep_from + frame_in_ep

            # ── Compute features for current frame ────────────────────────
            if current_step not in feature_history:
                rng, feat_rng = jax.random.split(rng)
                state, ref = _compute_frame_features(
                    transformed_ds,
                    extract_features,
                    feat_rng,
                    abs_frame,
                    diffusion_steps=args.diffusion_steps,
                )
                feature_history[current_step] = (state, ref)
                pending_chunks.append(current_step)
            else:
                state, ref = feature_history[current_step]

            # ── Choose action ─────────────────────────────────────────────
            if current_step < args.warmup_steps:
                # During warmup use the reference chunk; record only the first step.
                action_1step = ref.reshape(action_horizon, env_action_dim)[0].copy()
            else:
                rng, actor_rng = jax.random.split(rng)
                full_action = _actor_action(actor_model, actor_state, actor_rng, state, ref)
                action_1step = full_action.reshape(action_horizon, env_action_dim)[0]
            action_history[current_step] = action_1step

            # ── Env step (advance 1 dataset frame) ───────────────────────
            done = frame_in_ep == ep_len - 1
            reward = 1.0 if done else 0.0
            reward_history.append(reward)

            frame_in_ep += 1
            current_step += 1

            # Pre-compute features for the NEXT frame so _flush_ready_chunks
            # can find feature_history[chunk_step + action_horizon].
            next_abs = ep_from + frame_in_ep
            if frame_in_ep < ep_len and current_step not in feature_history:
                rng, feat_rng = jax.random.split(rng)
                feature_history[current_step] = _compute_frame_features(
                    transformed_ds,
                    extract_features,
                    feat_rng,
                    next_abs,
                    diffusion_steps=args.diffusion_steps,
                )
                pending_chunks.append(current_step)

            # Terminal features: reuse last frame's state for done=True transitions.
            terminal_features = (
                feature_history[current_step - 1] if done else None
            )

            _flush_ready_chunks(
                pending_chunks,
                feature_history,
                action_history,
                reward_history,
                replay,
                current_step=current_step,
                action_horizon=action_horizon,
                env_action_dim=env_action_dim,
                discount=args.online.discount,
                terminal=done,
                terminal_features=terminal_features,
            )

            # ── Training ──────────────────────────────────────────────────
            if len(replay) >= args.batch_size and current_step >= args.warmup_steps:
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
                    metrics.append(
                        {k: float(np.asarray(v)) for k, v in critic_info.items()}
                    )
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
                        metrics.append(
                            {k: float(np.asarray(v)) for k, v in actor_info.items()}
                        )

            metrics.append({"reward": reward, "replay_size": float(len(replay))})

            if current_step % args.log_interval == 0 and metrics:
                reduced = {
                    key: float(
                        np.mean([e[key] for e in metrics if key in e])
                    )
                    for key in sorted({k for e in metrics for k in e})
                }
                logging.info(
                    "Step %d: %s",
                    current_step,
                    ", ".join(f"{k}={v:.4f}" for k, v in reduced.items()),
                )
                metrics.clear()

            if (
                current_step % args.save_interval == 0
                and current_step > start_step
            ):
                _checkpointing.save_checkpoint(
                    checkpoint_root,
                    current_step,
                    policy_state=_trainer.bundle_actor_train_state(actor_state),
                    critic_state=_trainer.bundle_critic_state(critic_state),
                    norm_stats=None,
                    asset_id=None,
                )

    # Final checkpoint.
    _checkpointing.save_checkpoint(
        checkpoint_root,
        max(current_step - 1, 0),
        policy_state=_trainer.bundle_actor_train_state(actor_state),
        critic_state=_trainer.bundle_critic_state(critic_state),
        norm_stats=None,
        asset_id=None,
    )
    logging.info("Done. Total env steps: %d", current_step)


if __name__ == "__main__":
    main(tyro.cli(Args))
