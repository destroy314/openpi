from __future__ import annotations

import dataclasses
import logging
import pathlib
import signal
import traceback
import multiprocessing as mp

import jax
import jax.numpy as jnp
import numpy as np

from openpi.rlt import checkpointing as _checkpointing
from openpi.rlt import replay_buffer as _replay_buffer
from openpi.rlt import repro as _repro
from openpi.rlt import trainer as _trainer
from openpi.rlt.rtvla_contract import LearnerError
from openpi.rlt.rtvla_contract import LearnerInit
from openpi.rlt.rtvla_contract import LearnerStats
from openpi.rlt.rtvla_contract import ReplayItem
from openpi.rlt.rtvla_contract import StopSignal
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
from rt_vla.client.learner_runtime import publish_policy_update


@dataclasses.dataclass(frozen=True)
class OnlineRLTLearnerConfig:
    config: str
    init_checkpoint_dir: pathlib.Path
    replay_capacity: int
    batch_size: int
    utd_ratio: int
    actor_lr: float
    critic_lr: float
    save_interval: int
    log_interval: int
    warmup_steps: int
    resume: bool
    seed: int
    record_repro: bool
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


def run_online_rlt_learner(
    config: OnlineRLTLearnerConfig,
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

        train_config = _config.get_config(config.config)
        data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
        norm_stats = (
            _checkpoints.load_norm_stats(config.init_checkpoint_dir / "assets", data_config.asset_id)
            if data_config.asset_id is not None
            else None
        )

        rng = jax.random.key(config.seed)
        replay = _replay_buffer.ReplayBuffer(config.replay_capacity, state_dim=state_dim, action_dim=action_dim)
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
            learning_rate=config.actor_lr,
        )
        critic_model, critic_state = _trainer.init_critic_state(
            rng,
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=train_config.model.rlt_actor_hidden_dim,
            learning_rate=config.critic_lr,
        )

        start_step = 0
        last_saved_step = -1
        if config.resume:
            last_step = _checkpointing.latest_step(checkpoint_root)
            if last_step is not None:
                step_dir = checkpoint_root / str(last_step)
                actor_state = _trainer.restore_actor_state(
                    actor_model,
                    _checkpointing.restore_bundle(step_dir, "policy_state"),
                    config.actor_lr,
                )
                critic_state = _trainer.restore_critic_state(
                    critic_model,
                    _checkpointing.restore_bundle(step_dir, "critic_state"),
                    config.critic_lr,
                )
                start_step = last_step + 1
                last_saved_step = last_step
                logging.info("Resumed online RLT state from %s", step_dir)

        if config.record_repro:
            repro_dir = checkpoint_root / "repro"
            _repro.save_initial_state(repro_dir, "policy_state", _trainer.bundle_actor_train_state(actor_state))
            _repro.save_initial_state(repro_dir, "critic_state", _trainer.bundle_critic_state(critic_state))

        status_queue.put(LearnerInit(start_step=start_step, actor_params=_trainer.bundle_actor_params(actor_state)))

        np_rng = np.random.default_rng(config.seed)
        metrics: list[dict[str, float]] = []
        latest_env_step = start_step
        latest_checkpoint: pathlib.Path | None = None

        while True:
            message = sample_queue.get()
            if isinstance(message, StopSignal):
                final_step = max(message.final_step, 0)
                latest_checkpoint = _checkpointing.save_checkpoint(
                    checkpoint_root,
                    final_step,
                    policy_state=_trainer.bundle_actor_train_state(actor_state),
                    critic_state=_trainer.bundle_critic_state(critic_state),
                    norm_stats=norm_stats,
                    asset_id=data_config.asset_id,
                )
                status_queue.put(
                    LearnerStats(
                        learner_step=int(actor_state.step),
                        env_step=final_step,
                        replay_size=len(replay),
                        latest_checkpoint=str(latest_checkpoint),
                    )
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
            status_queue.put(
                LearnerStats(
                    learner_step=int(actor_state.step),
                    env_step=latest_env_step,
                    replay_size=len(replay),
                    latest_checkpoint=str(latest_checkpoint) if latest_checkpoint is not None else None,
                )
            )

            if len(replay) >= config.batch_size and latest_env_step >= config.warmup_steps:
                for _ in range(config.utd_ratio):
                    batch = _to_jax_batch(replay.sample(config.batch_size, rng=np_rng))
                    rng, critic_rng = jax.random.split(rng)
                    critic_state, critic_info = _trainer.critic_step_with_actor(
                        actor_model,
                        actor_state,
                        critic_model,
                        critic_state,
                        batch,
                        critic_rng,
                        config.online,
                    )
                    metrics.append({k: float(np.asarray(v)) for k, v in critic_info.items()})
                    if critic_state.step % config.online.actor_update_interval == 0:
                        rng, actor_rng = jax.random.split(rng)
                        actor_state, actor_info = _trainer.actor_step_with_actor(
                            actor_model,
                            critic_model,
                            critic_state,
                            actor_state,
                            batch,
                            actor_rng,
                            config.online,
                        )
                        metrics.append({k: float(np.asarray(v)) for k, v in actor_info.items()})
                        publish_policy_update(policy_queue, _trainer.bundle_actor_params(actor_state))

            if latest_env_step % config.log_interval == 0 and metrics:
                reduced = {
                    key: float(np.mean([entry[key] for entry in metrics if key in entry]))
                    for key in sorted({key for entry in metrics for key in entry})
                }
                logging.info("Learner step %d: %s", latest_env_step, ", ".join(f"{k}={v:.4f}" for k, v in reduced.items()))
                metrics.clear()

            if (
                latest_env_step % config.save_interval == 0
                and latest_env_step > start_step
                and latest_env_step != last_saved_step
            ):
                latest_checkpoint = _checkpointing.save_checkpoint(
                    checkpoint_root,
                    latest_env_step,
                    policy_state=_trainer.bundle_actor_train_state(actor_state),
                    critic_state=_trainer.bundle_critic_state(critic_state),
                    norm_stats=norm_stats,
                    asset_id=data_config.asset_id,
                )
                last_saved_step = latest_env_step
                status_queue.put(
                    LearnerStats(
                        learner_step=int(actor_state.step),
                        env_step=latest_env_step,
                        replay_size=len(replay),
                        latest_checkpoint=str(latest_checkpoint),
                    )
                )
    except Exception as exc:  # pragma: no cover - best effort propagation across processes.
        status_queue.put(LearnerError(message=str(exc), traceback=traceback.format_exc()))
        raise
