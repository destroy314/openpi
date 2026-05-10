from __future__ import annotations

from pathlib import Path
import pickle
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from openpi.rlt import repro as repro_module
from openpi.rlt import trainer
from openpi.rlt.rtvla_contract import LearnerError
from openpi.rlt.rtvla_contract import LearnerInit
from openpi.rlt.rtvla_contract import PolicyUpdate
from openpi.rlt.rtvla_contract import ReplayItem
from openpi.rlt.rtvla_contract import StopSignal
from rt_vla.client.config import ClientConfig
from rt_vla.client.config import Config
from rt_vla.client.config import ExecutorConfig
from rt_vla.client.config import FeedbackConfig
from rt_vla.client.config import ObserverConfig
from rt_vla.client.config import RLTConfig
from rt_vla.client.config import RLTLearnerConfig
from rt_vla.client.config import RLTObservationConfig
from rt_vla.client.config import RLTServiceConfig
from rt_vla.client.config import VisualizationConfig
from rt_vla.client.learner_runtime import LearnerProcessConfig
from rt_vla.client.learner_runtime import LearnerProcessHandle
from rt_vla.client.rlt_local_client_test import _FakeFeatureModel
from rt_vla.client.rlt_main import Args
from rt_vla.client.rlt_main import HttpTransport
from rt_vla.client.rlt_main import initialize_client_checkpoint_dir
from rt_vla.client.rlt_main import load_action_runtime_config
from rt_vla.client.rlt_main import make_identity_norm_stats
from rt_vla.client.rlt_main import run
import rt_vla.client.rlt_main as rlt_main_module
from rt_vla.server.rlt_feature_server import RLTFeatureServer


def _make_config() -> Config:
    return Config(
        client=ClientConfig(
            infer_url="http://127.0.0.1:8000",
            endpoint="/infer",
            timeout_s=1.0,
            run_duration_s=1.0,
        ),
        observer=ObserverConfig(
            name="mock",
            image_size=(224, 224),
            fps=30,
            state_dim=14,
            airbot_host="localhost",
            airbot_left_port=50051,
            airbot_right_port=50053,
            top_camera_id="top",
            left_camera_id="left",
            right_camera_id="right",
            enable_cameras=False,
        ),
        executor=ExecutorConfig(name="raw_action"),
        visualization=VisualizationConfig(output_dir="/tmp/rt-vla", enable_recording=False),
        feedback=FeedbackConfig(name="noop"),
        rlt=RLTConfig(
            service=RLTServiceConfig(),
            observation=RLTObservationConfig(
                prompt="pick up block",
                velocity_source="sdk",
                diffusion_steps=7,
                reference_horizon=5,
                action_horizon=2,
                chunk_stride=2,
                vla_replan_horizon_scale=2,
            ),
            learner=RLTLearnerConfig(seed=0, record_repro=True),
        ),
    )


def _fake_learner_worker(initial_actor_params, sample_queue, policy_queue, status_queue) -> None:
    status_queue.put(LearnerInit(start_step=0, actor_params=initial_actor_params))
    while True:
        message = sample_queue.get(timeout=5.0)
        if isinstance(message, ReplayItem):
            policy_queue.put(PolicyUpdate(actor_params=initial_actor_params))
            continue
        if isinstance(message, StopSignal):
            return
        status_queue.put(LearnerError(message=f"unexpected message: {message!r}", traceback=""))


class _InlineLearnerRuntime:
    def __init__(self, *, start_step: int, actor_params: dict) -> None:
        self._init = LearnerInit(start_step=start_step, actor_params=actor_params)
        self.samples: list[ReplayItem] = []
        self.shutdown_calls: list[int] = []
        self.exitcode = 0

    def wait_for_init(self):
        return self._init

    def queue_sample(self, item) -> None:
        self.samples.append(item)

    def check_status(self) -> None:
        return

    def sync_policy_state(self, actor_state, *, apply_actor_params, sync_logger=None):
        del apply_actor_params, sync_logger
        return actor_state

    def shutdown(self, *, final_step: int, graceful_stop_sent: bool, join_timeout_s: float = 60.0) -> bool:
        del graceful_stop_sent, join_timeout_s
        self.shutdown_calls.append(int(final_step))
        return True


def _make_inline_learner_runtime(cfg: Config, state_dim: int, action_dim: int, *, start_step: int = 0):
    rng = jax.random.key(int(cfg.rlt.learner.seed))
    _, actor_rng = jax.random.split(rng)
    _, initial_actor_state = trainer.init_actor_state(
        actor_rng,
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        learning_rate=cfg.rlt.learner.actor_lr,
    )
    actor_params = trainer.bundle_actor_params(initial_actor_state)
    actor_params["step"] = start_step
    return _InlineLearnerRuntime(start_step=start_step, actor_params=actor_params)


def test_http_transport_uses_pickle_payload_for_image_bytes() -> None:
    class _Response:
        content = pickle.dumps({"ok": True})

        def raise_for_status(self) -> None:
            return

    class _Session:
        def __init__(self) -> None:
            self.calls = []

        def post(self, url, *, data, headers, timeout):
            self.calls.append((url, data, headers, timeout))
            return _Response()

    transport = HttpTransport(base_url="http://feature-server:8000/", timeout_s=3.0)
    session = _Session()
    transport._session = session

    result = transport("/rlt/infer", {"images": {"high": b"\x89PNG"}})

    assert result == {"ok": True}
    url, data, headers, timeout = session.calls[0]
    assert url == "http://feature-server:8000/rlt/infer"
    assert pickle.loads(data)["images"]["high"] == b"\x89PNG"
    assert headers["Content-Type"] == "application/octet-stream"
    assert timeout == 3.0


def test_run_executes_mock_training_session_with_real_learner_handle(tmp_path: Path) -> None:
    server = RLTFeatureServer(feature_model=_FakeFeatureModel(), feature_id_factory=lambda: "feature-main")

    def config_loader(_path: str) -> Config:
        cfg = _make_config()
        cfg.rlt.learner.checkpoint_dir = str(tmp_path / "checkpoints")
        return cfg

    def learner_factory(_cfg: Config, _state_dim: int, _action_dim: int):
        rng = jax.random.key(int(_cfg.rlt.learner.seed))
        _, actor_rng = jax.random.split(rng)
        _, initial_actor_state = trainer.init_actor_state(
            actor_rng,
            state_dim=_state_dim,
            action_dim=_action_dim,
            hidden_dim=256,
            learning_rate=_cfg.rlt.learner.actor_lr,
        )
        return LearnerProcessHandle.spawn(
            target=_fake_learner_worker,
            target_args=(trainer.bundle_actor_params(initial_actor_state),),
            config=LearnerProcessConfig(sample_queue_size=8, policy_queue_size=1, process_name="rlt-main-test"),
        )

    result = run(
        Args(config_path=str(Path("/tmp/fake.yaml")), num_iterations=2, episode_id="episode-main"),
        config_loader=config_loader,
        feature_transport=server.dispatch,
        learner_factory=learner_factory,
        norm_stats=make_identity_norm_stats(),
    )

    assert len(result.iterations) == 2
    repro_dir = tmp_path / "checkpoints" / "repro"
    manifest = repro_module.load_manifest(repro_dir)
    stop_signal = repro_module.load_stop_signal(repro_dir)
    transitions = list(repro_module.iter_transition_dicts(repro_dir))
    execution_chunks = list(repro_module.iter_execution_record_chunks(repro_dir))

    assert manifest["args"]["episode_id"] == "episode-main"
    assert manifest["start_step"] == 0
    assert stop_signal["final_step"] == 4
    assert len(transitions) == 2
    assert len(execution_chunks) == 2
    assert execution_chunks[0]["episode_id"] == "episode-main"
    assert execution_chunks[0]["env_step"] == 0
    assert execution_chunks[0]["action_sources"] == ["command_action", "command_action"]
    assert execution_chunks[0]["feature_metadata"]["plan"]["feature_id"] == "feature-main"
    assert "reference_plan_norm_sha256" in execution_chunks[0]["feature_metadata"]["plan"]
    assert [record["source"] for record in execution_chunks[0]["records"]] == [
        "execute_training_chunk",
        "execute_training_chunk",
    ]


def test_load_action_runtime_config_uses_stage1_norm_stats_and_delta_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeDeltaActions:
        pass

    cfg = _make_config()
    cfg.rlt.learner.config_name = "fake-rlt"
    cfg.rlt.learner.init_checkpoint_dir = str(tmp_path / "stage1")
    fake_norm_stats = {
        "actions": rlt_main_module.NormStats(
            mean=np.full((14,), 5.0, dtype=np.float32),
            std=np.full((14,), 2.0, dtype=np.float32),
        )
    }
    fake_data_config = SimpleNamespace(
        asset_id="airbot",
        use_quantile_norm=True,
        data_transforms=SimpleNamespace(inputs=[_FakeDeltaActions()]),
    )
    fake_train_config = SimpleNamespace(
        assets_dirs=(),
        model=SimpleNamespace(),
        data=SimpleNamespace(create=lambda *_args: fake_data_config),
    )

    monkeypatch.setattr(rlt_main_module._transforms, "DeltaActions", _FakeDeltaActions)
    monkeypatch.setattr(rlt_main_module._config, "get_config", lambda _name: fake_train_config)
    monkeypatch.setattr(rlt_main_module._checkpoints, "load_norm_stats", lambda *_args: fake_norm_stats)

    runtime_config = load_action_runtime_config(cfg)

    assert runtime_config.norm_stats is fake_norm_stats
    assert runtime_config.asset_id == "airbot"
    assert runtime_config.use_quantiles is True
    assert runtime_config.use_delta_joint_actions is True


def test_initialize_client_checkpoint_dir_requires_overwrite_or_resume_for_existing_dir(tmp_path: Path) -> None:
    cfg = _make_config()
    cfg.rlt.learner.checkpoint_dir = str(tmp_path / "checkpoints")
    Path(cfg.rlt.learner.checkpoint_dir).mkdir()

    with pytest.raises(FileExistsError):
        initialize_client_checkpoint_dir(cfg)


def test_initialize_client_checkpoint_dir_overwrite_removes_stale_artifacts(tmp_path: Path) -> None:
    cfg = _make_config()
    cfg.rlt.learner.checkpoint_dir = str(tmp_path / "checkpoints")
    cfg.rlt.learner.overwrite = True
    checkpoint_dir = Path(cfg.rlt.learner.checkpoint_dir)
    checkpoint_dir.mkdir()
    stale_file = checkpoint_dir / "stale.txt"
    stale_file.write_text("old", encoding="utf-8")

    checkpoint_root = initialize_client_checkpoint_dir(cfg)

    assert checkpoint_root == checkpoint_dir.resolve()
    assert not stale_file.exists()
    assert Path(cfg.rlt.learner.checkpoint_dir) == checkpoint_root


def test_run_resume_start_step_drives_collector_env_step_and_manifest(tmp_path: Path) -> None:
    server = RLTFeatureServer(feature_model=_FakeFeatureModel(), feature_id_factory=lambda: "feature-resume")
    learner_handles: list[_InlineLearnerRuntime] = []

    def config_loader(_path: str) -> Config:
        cfg = _make_config()
        cfg.rlt.learner.checkpoint_dir = str(tmp_path / "checkpoints")
        cfg.rlt.learner.resume = True
        Path(cfg.rlt.learner.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        return cfg

    def learner_factory(cfg: Config, state_dim: int, action_dim: int):
        handle = _make_inline_learner_runtime(cfg, state_dim, action_dim, start_step=6)
        learner_handles.append(handle)
        return handle

    result = run(
        Args(config_path=str(Path("/tmp/fake.yaml")), num_iterations=1, episode_id="episode-resume"),
        config_loader=config_loader,
        feature_transport=server.dispatch,
        learner_factory=learner_factory,
        norm_stats=make_identity_norm_stats(),
    )

    handle = learner_handles[0]
    repro_dir = tmp_path / "checkpoints" / "repro"
    manifest = repro_module.load_manifest(repro_dir)
    stop_signal = repro_module.load_stop_signal(repro_dir)

    assert result.final_env_step == 8
    assert handle.samples[0].env_step == 6
    assert handle.shutdown_calls[-1] == 8
    assert manifest["start_step"] == 6
    assert stop_signal["final_step"] == 8


def test_run_shuts_down_learner_and_records_stop_signal_when_runtime_raises(tmp_path: Path) -> None:
    learner_handles: list[_InlineLearnerRuntime] = []

    def config_loader(_path: str) -> Config:
        cfg = _make_config()
        cfg.rlt.learner.checkpoint_dir = str(tmp_path / "checkpoints")
        return cfg

    def learner_factory(cfg: Config, state_dim: int, action_dim: int):
        handle = _make_inline_learner_runtime(cfg, state_dim, action_dim)
        learner_handles.append(handle)
        return handle

    def failing_transport(_path: str, _payload: dict | None):
        raise RuntimeError("feature server down")

    with pytest.raises(RuntimeError, match="feature server down"):
        run(
            Args(config_path=str(Path("/tmp/fake.yaml")), num_iterations=1, episode_id="episode-fail"),
            config_loader=config_loader,
            feature_transport=failing_transport,
            learner_factory=learner_factory,
            norm_stats=make_identity_norm_stats(),
        )

    handle = learner_handles[0]
    stop_signal = repro_module.load_stop_signal(tmp_path / "checkpoints" / "repro")
    assert handle.shutdown_calls
    assert handle.shutdown_calls[-1] == 0
    assert stop_signal["final_step"] == 0
