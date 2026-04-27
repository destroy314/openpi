from __future__ import annotations

import dataclasses
import logging
import pathlib

import jax
import numpy as np
import tyro

from openpi.rlt import checkpointing as _checkpointing
from openpi.rlt import replay_buffer as _replay_buffer
from openpi.rlt import repro as _repro
from openpi.rlt import trainer as _trainer
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
from scripts import train_rlt_online as _online


@dataclasses.dataclass
class Args:
    repro_dir: pathlib.Path
    checkpoint_dir: pathlib.Path
    overwrite: bool = False


def _restore_online_args(manifest: dict[str, object], checkpoint_dir: pathlib.Path) -> _online.Args:
    raw_args = dict(manifest["args"])
    raw_args["checkpoint_dir"] = checkpoint_dir
    raw_args["init_checkpoint_dir"] = pathlib.Path(raw_args["init_checkpoint_dir"])
    raw_args["online"] = _trainer.OnlineRLTConfig(**raw_args["online"])
    return _online.Args(**raw_args)


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)

    repro_dir = args.repro_dir.resolve()
    manifest = _repro.load_manifest(repro_dir)
    online_args = _restore_online_args(manifest, args.checkpoint_dir.resolve())

    checkpoint_root, _ = _checkpointing.initialize_checkpoint_dir(
        online_args.checkpoint_dir,
        overwrite=args.overwrite,
        resume=False,
    )
    _trainer.set_actor_sample_debug(True, log_path=checkpoint_root / "actor_sample.log")

    train_config = _config.get_config(online_args.config)
    state_dim = int(manifest["state_dim"])
    action_dim = int(manifest["action_dim"])
    actor_hidden_dim = int(manifest["actor_hidden_dim"])
    asset_id = manifest.get("asset_id")
    norm_stats = None
    if asset_id is not None:
        norm_stats = _checkpoints.load_norm_stats(repro_dir / "assets", asset_id)

    rng = jax.random.key(online_args.seed)
    replay = _replay_buffer.ReplayBuffer(online_args.replay_capacity, state_dim=state_dim, action_dim=action_dim)
    rng, actor_rng = jax.random.split(rng)
    actor_model, _ = _trainer.init_actor_state(
        actor_rng,
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=actor_hidden_dim,
        learning_rate=online_args.actor_lr,
    )
    critic_model, _ = _trainer.init_critic_state(
        rng,
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=train_config.model.rlt_actor_hidden_dim,
        learning_rate=online_args.critic_lr,
    )
    actor_state = _trainer.restore_actor_state(
        actor_model,
        _repro.load_initial_state(repro_dir, "policy_state"),
        online_args.actor_lr,
    )
    critic_state = _trainer.restore_critic_state(
        critic_model,
        _repro.load_initial_state(repro_dir, "critic_state"),
        online_args.critic_lr,
    )

    start_step = int(manifest.get("start_step", 0))
    last_saved_step = start_step - 1
    latest_env_step = start_step
    np_rng = np.random.default_rng(online_args.seed)
    metrics: list[dict[str, float]] = []

    for payload in _repro.iter_transition_dicts(repro_dir):
        message = _online.ReplayItem(**payload)
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

        if len(replay) >= online_args.batch_size and latest_env_step >= online_args.warmup_steps:
            for _ in range(online_args.utd_ratio):
                batch = _online._to_jax_batch(replay.sample(online_args.batch_size, rng=np_rng))
                rng, critic_rng = jax.random.split(rng)
                critic_state, critic_info = _trainer.critic_step_with_actor(
                    actor_model,
                    actor_state,
                    critic_model,
                    critic_state,
                    batch,
                    critic_rng,
                    online_args.online,
                )
                metrics.append({k: float(np.asarray(v)) for k, v in critic_info.items()})
                if critic_state.step % online_args.online.actor_update_interval == 0:
                    rng, actor_rng = jax.random.split(rng)
                    actor_state, actor_info = _trainer.actor_step_with_actor(
                        actor_model,
                        critic_model,
                        critic_state,
                        actor_state,
                        batch,
                        actor_rng,
                        online_args.online,
                    )
                    metrics.append({k: float(np.asarray(v)) for k, v in actor_info.items()})

        if latest_env_step % online_args.log_interval == 0 and metrics:
            reduced = {
                key: float(np.mean([entry[key] for entry in metrics if key in entry]))
                for key in sorted({key for entry in metrics for key in entry})
            }
            logging.info(
                "Replay learner step %d: %s",
                latest_env_step,
                ", ".join(f"{k}={v:.4f}" for k, v in reduced.items()),
            )
            metrics.clear()

        if (
            latest_env_step % online_args.save_interval == 0
            and latest_env_step > start_step
            and latest_env_step != last_saved_step
        ):
            _checkpointing.save_checkpoint(
                checkpoint_root,
                latest_env_step,
                policy_state=_trainer.bundle_actor_train_state(actor_state),
                critic_state=_trainer.bundle_critic_state(critic_state),
                norm_stats=norm_stats,
                asset_id=asset_id,
            )
            last_saved_step = latest_env_step

    stop_signal = _repro.load_stop_signal(repro_dir)
    final_step = int(stop_signal["final_step"])
    _checkpointing.save_checkpoint(
        checkpoint_root,
        final_step,
        policy_state=_trainer.bundle_actor_train_state(actor_state),
        critic_state=_trainer.bundle_critic_state(critic_state),
        norm_stats=norm_stats,
        asset_id=asset_id,
    )
    logging.info("Replay completed at step %d", final_step)


if __name__ == "__main__":
    main(tyro.cli(Args))
