from __future__ import annotations

import dataclasses
import queue
import threading
import time
from types import SimpleNamespace

import jax
import numpy as np

from openpi.rlt import trainer
from openpi.rlt.rtvla_contract import LearnerError
from openpi.rlt.rtvla_contract import LearnerInit
from openpi.rlt.rtvla_contract import PolicyUpdate
from openpi.rlt.rtvla_contract import StopSignal
from openpi.shared.normalize import NormStats
from rt_vla.client.feedback import NoopFeedbackProvider
from rt_vla.client.rlt_actor_runtime import RLTFeatureClient
import rt_vla.client.rlt_learner as learner_module
from rt_vla.client.rlt_learner import OnlineRLTLearnerConfig
from rt_vla.client.rlt_learner import run_online_rlt_learner
from rt_vla.client.rlt_local_client import RLTLocalClient
from rt_vla.client.rlt_local_client_test import _FakeExecutor
from rt_vla.client.rlt_local_client_test import _FakeFeatureModel
from rt_vla.client.rlt_local_client_test import _FakeObserver
from rt_vla.client.rlt_local_client_test import _make_config
from rt_vla.client.rlt_training_runtime import RLTTrainingRuntime
from rt_vla.server.rlt_feature_server import RLTFeatureServer


class _QueueLearnerRuntime:
    def __init__(self, sample_queue: queue.Queue, policy_queue: queue.Queue, status_queue: queue.Queue) -> None:
        self._sample_queue = sample_queue
        self._policy_queue = policy_queue
        self._status_queue = status_queue
        self.latest_update: PolicyUpdate | None = None

    def queue_sample(self, item) -> None:
        self._sample_queue.put(item)

    def check_status(self) -> None:
        while True:
            try:
                message = self._status_queue.get_nowait()
            except queue.Empty:
                return
            if isinstance(message, LearnerError):
                raise RuntimeError(message.message)

    def sync_policy_state(self, actor_state, *, apply_actor_params, sync_logger=None):
        del sync_logger
        while True:
            try:
                message = self._policy_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(message, PolicyUpdate):
                self.latest_update = message
        if self.latest_update is None:
            return actor_state
        return apply_actor_params(actor_state, self.latest_update.actor_params)


def _fake_train_config() -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(rlt_actor_hidden_dim=8),
        data=SimpleNamespace(create=lambda *_args: SimpleNamespace(asset_id=None)),
        assets_dirs=(),
    )


def test_rlt_client_runtime_drives_real_learner_entrypoint_with_mock_server(
    tmp_path,
    monkeypatch,
) -> None:
    saved_checkpoints: list[tuple[int, dict, dict]] = []

    def fake_get_config(_name: str):
        return _fake_train_config()

    def fake_save_checkpoint(checkpoint_dir, step, *, policy_state, critic_state, norm_stats, asset_id):
        del checkpoint_dir, norm_stats, asset_id
        saved_checkpoints.append((int(step), policy_state, critic_state))
        return tmp_path / str(step)

    monkeypatch.setattr(learner_module._config, "get_config", fake_get_config)  # noqa: SLF001
    monkeypatch.setattr(learner_module._checkpointing, "save_checkpoint", fake_save_checkpoint)  # noqa: SLF001
    monkeypatch.setattr(learner_module._repro, "save_initial_state", lambda *_args, **_kwargs: tmp_path)  # noqa: SLF001
    monkeypatch.setattr(learner_module.signal, "signal", lambda *_args, **_kwargs: None)

    sample_queue: queue.Queue = queue.Queue()
    policy_queue: queue.Queue = queue.Queue(maxsize=1)
    status_queue: queue.Queue = queue.Queue()
    learner_cfg = OnlineRLTLearnerConfig(
        config="fake-rlt",
        init_checkpoint_dir=tmp_path,
        replay_capacity=8,
        batch_size=1,
        utd_ratio=1,
        actor_lr=1e-3,
        critic_lr=1e-3,
        save_interval=100,
        log_interval=100,
        warmup_steps=0,
        resume=False,
        seed=0,
        record_repro=True,
        online=trainer.OnlineRLTConfig(actor_update_interval=1),
    )
    learner_thread = threading.Thread(
        target=run_online_rlt_learner,
        args=(learner_cfg, tmp_path, 284, 28, 8, sample_queue, policy_queue, status_queue),
        daemon=True,
    )
    learner_thread.start()

    try:
        init_message = status_queue.get(timeout=10.0)
        if isinstance(init_message, LearnerError):
            raise RuntimeError(init_message.message)
        assert isinstance(init_message, LearnerInit)

        actor_model, actor_state = trainer.init_actor_state(
            jax.random.key(0),
            state_dim=284,
            action_dim=28,
            hidden_dim=8,
            learning_rate=1e-3,
        )
        actor_state = trainer.apply_actor_params(actor_state, init_message.actor_params)
        norm_stats = {
            "actions": NormStats(
                mean=np.zeros((14,), dtype=np.float32),
                std=np.ones((14,), dtype=np.float32),
                q01=np.full((14,), -1.0, dtype=np.float32),
                q99=np.full((14,), 1.0, dtype=np.float32),
            )
        }
        server = RLTFeatureServer(feature_model=_FakeFeatureModel(), feature_id_factory=lambda: "feature-e2e")
        runtime = RLTTrainingRuntime(
            config=_make_config(),
            local_client=RLTLocalClient(
                config=_make_config(),
                feature_client=RLTFeatureClient(transport=server.dispatch, request_id_factory=lambda: "req-e2e"),
                observer=_FakeObserver(),
            ),
            actor_model=actor_model,
            actor_state=actor_state,
            executor=_FakeExecutor(),
            learner_runtime=_QueueLearnerRuntime(sample_queue, policy_queue, status_queue),
            feedback_provider=NoopFeedbackProvider(),
            norm_stats=norm_stats,
            use_quantiles=False,
            use_delta_joint_actions=False,
            action_horizon=2,
            control_dt_s=0.02,
            apply_actor_params=trainer.apply_actor_params,
            prompt="pick up block",
        )

        result = runtime.run(num_iterations=1, rng=jax.random.key(8))
        deadline = time.time() + 20.0
        latest_update = runtime.learner_runtime.latest_update
        while latest_update is None and time.time() < deadline:
            runtime.learner_runtime.check_status()
            runtime.actor_state = runtime.learner_runtime.sync_policy_state(
                runtime.actor_state,
                apply_actor_params=trainer.apply_actor_params,
            )
            latest_update = runtime.learner_runtime.latest_update
            time.sleep(0.01)

        assert len(result.iterations) == 1
        assert result.final_env_step == 2
        assert latest_update is not None
        assert int(np.asarray(latest_update.actor_params["step"])) >= 1
    finally:
        sample_queue.put(StopSignal(final_step=2))
        learner_thread.join(timeout=10.0)

    assert not learner_thread.is_alive()
    assert saved_checkpoints[-1][0] == 2


def test_run_online_rlt_learner_resume_restores_latest_checkpoint(
    tmp_path,
    monkeypatch,
) -> None:
    saved_checkpoints: list[int] = []
    actor_model, actor_state = trainer.init_actor_state(
        jax.random.key(10),
        state_dim=284,
        action_dim=28,
        hidden_dim=8,
        learning_rate=1e-3,
    )
    critic_model, critic_state = trainer.init_critic_state(
        jax.random.key(11),
        state_dim=284,
        action_dim=28,
        hidden_dim=8,
        learning_rate=1e-3,
    )
    del actor_model, critic_model
    actor_state = dataclasses.replace(actor_state, step=4)
    critic_state = dataclasses.replace(critic_state, step=4)
    actor_bundle = trainer.bundle_actor_train_state(actor_state)
    critic_bundle = trainer.bundle_critic_state(critic_state)

    def fake_restore_bundle(_step_dir, item: str):
        if item == "policy_state":
            return actor_bundle
        if item == "critic_state":
            return critic_bundle
        raise AssertionError(f"unexpected bundle item {item}")

    def fake_save_checkpoint(_checkpoint_dir, step, *, policy_state, critic_state, norm_stats, asset_id):
        del policy_state, critic_state, norm_stats, asset_id
        saved_checkpoints.append(int(step))
        return tmp_path / str(step)

    monkeypatch.setattr(learner_module._config, "get_config", lambda _name: _fake_train_config())  # noqa: SLF001
    monkeypatch.setattr(learner_module._checkpointing, "latest_step", lambda _root: 4)  # noqa: SLF001
    monkeypatch.setattr(learner_module._checkpointing, "restore_bundle", fake_restore_bundle)  # noqa: SLF001
    monkeypatch.setattr(learner_module._checkpointing, "save_checkpoint", fake_save_checkpoint)  # noqa: SLF001
    monkeypatch.setattr(learner_module._repro, "save_initial_state", lambda *_args, **_kwargs: tmp_path)  # noqa: SLF001
    monkeypatch.setattr(learner_module.signal, "signal", lambda *_args, **_kwargs: None)

    sample_queue: queue.Queue = queue.Queue()
    policy_queue: queue.Queue = queue.Queue(maxsize=1)
    status_queue: queue.Queue = queue.Queue()
    learner_cfg = OnlineRLTLearnerConfig(
        config="fake-rlt",
        init_checkpoint_dir=tmp_path,
        replay_capacity=8,
        batch_size=4,
        utd_ratio=1,
        actor_lr=1e-3,
        critic_lr=1e-3,
        save_interval=100,
        log_interval=100,
        warmup_steps=0,
        resume=True,
        seed=0,
        record_repro=True,
    )
    learner_thread = threading.Thread(
        target=run_online_rlt_learner,
        args=(learner_cfg, tmp_path, 284, 28, 8, sample_queue, policy_queue, status_queue),
        daemon=True,
    )
    learner_thread.start()

    try:
        init_message = status_queue.get(timeout=10.0)
        if isinstance(init_message, LearnerError):
            raise RuntimeError(init_message.message)
        assert isinstance(init_message, LearnerInit)
        assert init_message.start_step == 5
        assert int(np.asarray(init_message.actor_params["step"])) == 4
    finally:
        sample_queue.put(StopSignal(final_step=5))
        learner_thread.join(timeout=10.0)

    assert not learner_thread.is_alive()
    assert saved_checkpoints[-1] == 5
