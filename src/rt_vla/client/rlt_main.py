from __future__ import annotations

from collections.abc import Callable
import dataclasses
import pathlib
import pickle
from typing import Any

import jax
import numpy as np
import requests
import tyro

from openpi import transforms as _transforms
from openpi.rlt import RLT_ACTION_DIM
from openpi.rlt import RLT_PROPRIO_DIM
from openpi.rlt import RLT_RL_TOKEN_DIM
from openpi.rlt import checkpointing as _checkpointing
from openpi.rlt import repro as _repro
from openpi.rlt import trainer as _trainer
from openpi.shared.normalize import NormStats
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
from rt_vla.client import builders as _builders
from rt_vla.client import learner_runtime as _learner_runtime
from rt_vla.client import rlt_learner as _rlt_learner
from rt_vla.client.config import Config
from rt_vla.client.config import load_config
from rt_vla.client.rlt_local_client import build_feature_metadata

Transport = Callable[[str, dict[str, Any] | None], dict[str, Any]]
LearnerFactory = Callable[[Config, int, int], Any]


@dataclasses.dataclass(frozen=True)
class ActionRuntimeConfig:
    norm_stats: dict[str, NormStats]
    use_quantiles: bool
    use_delta_joint_actions: bool
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class Args:
    config_path: str
    num_iterations: int = 1
    episode_id: str = "episode-0"


class HttpTransport:
    def __init__(self, *, base_url: str, timeout_s: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = float(timeout_s)
        self._session = requests.Session()

    def __call__(self, path: str, payload: dict[str, Any] | None) -> dict[str, Any]:
        response = self._session.post(
            f"{self._base_url}{path}",
            data=pickle.dumps(payload),
            headers={"Content-Type": "application/octet-stream"},
            timeout=self._timeout_s,
        )
        response.raise_for_status()
        return dict(pickle.loads(response.content))


def make_identity_norm_stats(action_dim: int = RLT_ACTION_DIM) -> dict[str, NormStats]:
    return {
        "actions": NormStats(
            mean=np.zeros((action_dim,), dtype=np.float32),
            std=np.ones((action_dim,), dtype=np.float32),
            q01=np.full((action_dim,), -1.0, dtype=np.float32),
            q99=np.full((action_dim,), 1.0, dtype=np.float32),
        )
    }


def load_action_runtime_config(cfg: Config) -> ActionRuntimeConfig:
    learner = cfg.rlt.learner
    if not learner.config_name or not learner.init_checkpoint_dir:
        return ActionRuntimeConfig(
            norm_stats=make_identity_norm_stats(),
            use_quantiles=False,
            use_delta_joint_actions=False,
            asset_id=None,
        )

    train_config = _config.get_config(learner.config_name)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    asset_id = data_config.asset_id
    norm_stats = (
        _checkpoints.load_norm_stats(pathlib.Path(learner.init_checkpoint_dir) / "assets", asset_id)
        if asset_id is not None
        else make_identity_norm_stats()
    )
    data_transforms = getattr(data_config, "data_transforms", None)
    input_transforms = getattr(data_transforms, "inputs", ()) if data_transforms is not None else ()
    use_delta_joint_actions = any(isinstance(transform, _transforms.DeltaActions) for transform in input_transforms)
    return ActionRuntimeConfig(
        norm_stats=norm_stats,
        use_quantiles=bool(getattr(data_config, "use_quantile_norm", False)),
        use_delta_joint_actions=use_delta_joint_actions,
        asset_id=asset_id,
    )


def get_actor_hidden_dim(cfg: Config) -> int:
    config_name = cfg.rlt.learner.config_name
    if not config_name:
        return 256
    train_config = _config.get_config(config_name)
    return int(getattr(train_config.model, "rlt_actor_hidden_dim", 256))


def _build_learner_config(cfg: Config) -> _rlt_learner.OnlineRLTLearnerConfig:
    learner = cfg.rlt.learner
    if not learner.config_name:
        raise ValueError("cfg.rlt.learner.config_name must be set to spawn the real learner process")
    if not learner.init_checkpoint_dir:
        raise ValueError("cfg.rlt.learner.init_checkpoint_dir must be set to spawn the real learner process")
    return _rlt_learner.OnlineRLTLearnerConfig(
        config=learner.config_name,
        init_checkpoint_dir=pathlib.Path(learner.init_checkpoint_dir),
        replay_capacity=learner.replay_capacity,
        batch_size=learner.batch_size,
        utd_ratio=learner.utd_ratio,
        actor_lr=learner.actor_lr,
        critic_lr=learner.critic_lr,
        save_interval=learner.save_interval,
        log_interval=learner.log_interval,
        warmup_steps=learner.warmup_steps,
        resume=learner.resume,
        seed=learner.seed,
        record_repro=learner.record_repro,
    )


def spawn_real_learner_runtime(cfg: Config, state_dim: int, action_dim: int):
    learner_cfg = _build_learner_config(cfg)
    checkpoint_root = pathlib.Path(cfg.rlt.learner.checkpoint_dir)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    actor_hidden_dim = get_actor_hidden_dim(cfg)
    return _learner_runtime.LearnerProcessHandle.spawn(
        target=_rlt_learner.run_online_rlt_learner,
        target_args=(
            learner_cfg,
            checkpoint_root,
            state_dim,
            action_dim,
            actor_hidden_dim,
        ),
        config=_learner_runtime.LearnerProcessConfig(
            sample_queue_size=cfg.rlt.learner.sample_queue_size,
            policy_queue_size=cfg.rlt.learner.policy_queue_size,
            process_name="rlt-learner",
        ),
    )


def initialize_client_checkpoint_dir(cfg: Config) -> pathlib.Path:
    checkpoint_root, _ = _checkpointing.initialize_checkpoint_dir(
        cfg.rlt.learner.checkpoint_dir,
        overwrite=cfg.rlt.learner.overwrite,
        resume=cfg.rlt.learner.resume,
    )
    cfg.rlt.learner.checkpoint_dir = str(checkpoint_root)
    return checkpoint_root


def run(
    args: Args,
    *,
    config_loader: Callable[[str], Config] = load_config,
    feature_transport: Transport | None = None,
    learner_factory: LearnerFactory | None = None,
    norm_stats: dict[str, NormStats] | None = None,
) -> Any:
    cfg = config_loader(args.config_path)
    action_horizon = int(cfg.rlt.observation.action_horizon)
    state_dim = RLT_RL_TOKEN_DIM + RLT_PROPRIO_DIM
    action_dim = action_horizon * RLT_ACTION_DIM
    actor_hidden_dim = get_actor_hidden_dim(cfg)
    checkpoint_dir = initialize_client_checkpoint_dir(cfg)
    repro_dir = checkpoint_dir / "repro"

    if feature_transport is None:
        feature_transport = HttpTransport(
            base_url=cfg.rlt.service.feature_server_url,
            timeout_s=cfg.rlt.service.timeout_s,
        )
    if learner_factory is None:
        learner_factory = spawn_real_learner_runtime
    if norm_stats is None:
        action_runtime = load_action_runtime_config(cfg)
    else:
        action_runtime = ActionRuntimeConfig(
            norm_stats=norm_stats,
            use_quantiles=False,
            use_delta_joint_actions=False,
            asset_id=None,
        )

    rng = jax.random.key(int(cfg.rlt.learner.seed))
    rng, actor_rng = jax.random.split(rng)
    actor_model, actor_state = _trainer.init_actor_state(
        actor_rng,
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=actor_hidden_dim,
        learning_rate=cfg.rlt.learner.actor_lr,
    )

    learner_runtime = None
    transition_recorder = None
    execution_recorder = None
    runtime = None
    graceful_stop_sent = False
    final_step = 0
    try:
        learner_runtime = learner_factory(cfg, state_dim, action_dim)
        init_message = learner_runtime.wait_for_init()
        start_step = int(init_message.start_step)
        final_step = start_step
        actor_state = _trainer.apply_actor_params(actor_state, init_message.actor_params)
        transition_recorder = (
            _repro.TransitionRecorder(repro_dir, shard_size=cfg.rlt.learner.repro_shard_size)
            if cfg.rlt.learner.record_repro
            else None
        )
        execution_recorder = (
            _repro.ExecutionRecordRecorder(repro_dir)
            if cfg.rlt.learner.record_repro
            else None
        )
        if cfg.rlt.learner.record_repro:
            _repro.write_manifest(
                repro_dir,
                {
                    "schema_version": 1,
                    "args": dataclasses.asdict(args),
                    "config": _repro.to_jsonable(cfg),
                    "runtime": _repro.collect_runtime_metadata(),
                    "state_dim": state_dim,
                    "action_dim": action_dim,
                    "actor_hidden_dim": actor_hidden_dim,
                    "checkpoint_root": str(checkpoint_dir),
                    "init_checkpoint_dir": cfg.rlt.learner.init_checkpoint_dir,
                    "asset_id": action_runtime.asset_id,
                    "use_quantiles": action_runtime.use_quantiles,
                    "use_delta_joint_actions": action_runtime.use_delta_joint_actions,
                    "start_step": start_step,
                },
            )
            _repro.save_norm_stats_snapshot(repro_dir, action_runtime.norm_stats, action_runtime.asset_id)
        runtime = _builders.build_rlt_training_runtime(
            cfg,
            feature_transport=feature_transport,
            actor_model=actor_model,
            actor_state=actor_state,
            learner_runtime=learner_runtime,
            norm_stats=action_runtime.norm_stats,
            use_quantiles=action_runtime.use_quantiles,
            use_delta_joint_actions=action_runtime.use_delta_joint_actions,
            action_horizon=action_horizon,
            control_dt_s=cfg.executor.control_dt_s,
            apply_actor_params=_trainer.apply_actor_params,
        )
        runtime.episode_id = args.episode_id
        runtime.env_step = start_step
        result = runtime.run_with_artifacts(
            num_iterations=args.num_iterations,
            rng=rng,
            transition_recorder=transition_recorder,
        )
        if execution_recorder is not None:
            for iteration in result.iterations:
                execution_recorder.add_chunk(
                    episode_id=args.episode_id,
                    env_step=iteration.chunk.chunk_env_step,
                    action_sources=iteration.chunk.execution.action_sources,
                    records=iteration.chunk.execution.records,
                    feature_metadata=build_feature_metadata(iteration.chunk),
                )
        final_step = max(runtime.env_step, 0)
        graceful_stop_sent = learner_runtime.shutdown(final_step=final_step, graceful_stop_sent=graceful_stop_sent)
        learner_runtime.check_status()
        if getattr(learner_runtime, "exitcode", None) not in (0, None):
            raise RuntimeError(f"Learner process exited with code {learner_runtime.exitcode}")
        return result
    finally:
        if runtime is not None:
            final_step = max(runtime.env_step, final_step, 0)
        if transition_recorder is not None:
            transition_recorder.close()
        if execution_recorder is not None:
            execution_recorder.close()
        if cfg.rlt.learner.record_repro:
            _repro.save_stop_signal(repro_dir, final_step)
        if learner_runtime is not None:
            learner_runtime.shutdown(final_step=final_step, graceful_stop_sent=graceful_stop_sent)
        if runtime is not None:
            runtime.close()


def main(args: Args) -> None:
    run(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
