from __future__ import annotations

from pathlib import Path

import pytest

from rt_vla.client.config import load_config
from rt_vla.client.rlt_local_client import build_observation_adapter


def test_load_config_preserves_existing_shape_and_adds_default_rlt_section(tmp_path: Path) -> None:
    config_path = tmp_path / "client.yaml"
    config_path.write_text(
        """
client:
  infer_url: http://127.0.0.1:8000
  endpoint: /infer
  timeout_s: 2.0
  run_duration_s: 1.0
observer:
  name: mock
  image_size: [224, 224]
  fps: 30
  state_dim: 14
  airbot_host: localhost
  airbot_left_port: 50051
  airbot_right_port: 50053
  top_camera_id: top
  left_camera_id: left
  right_camera_id: right
  enable_cameras: false
executor:
  name: raw_action
visualization:
  output_dir: /tmp/rt-vla
  enable_recording: false
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.client.endpoint == "/infer"
    assert config.rlt.service.infer_endpoint == "/rlt/infer"
    assert config.rlt.observation.velocity_source is None
    assert config.rlt.learner.sample_queue_size == 1024
    assert config.rlt.learner.overwrite is False
    assert config.rlt.learner.repro_shard_size == 1024


def test_rlt_observation_adapter_requires_explicit_velocity_source(tmp_path: Path) -> None:
    config_path = tmp_path / "client.yaml"
    config_path.write_text(
        """
client:
  infer_url: http://127.0.0.1:8000
  endpoint: /infer
  timeout_s: 2.0
  run_duration_s: 1.0
observer:
  name: mock
  image_size: [224, 224]
  fps: 30
  state_dim: 14
  airbot_host: localhost
  airbot_left_port: 50051
  airbot_right_port: 50053
  top_camera_id: top
  left_camera_id: left
  right_camera_id: right
  enable_cameras: false
executor:
  name: raw_action
visualization:
  output_dir: /tmp/rt-vla
  enable_recording: false
""".strip(),
        encoding="utf-8",
    )
    config = load_config(config_path)

    with pytest.raises(ValueError, match="velocity_source must be configured explicitly"):
        build_observation_adapter(config)


def test_load_config_accepts_explicit_rlt_section(tmp_path: Path) -> None:
    config_path = tmp_path / "client.yaml"
    config_path.write_text(
        """
client:
  infer_url: http://127.0.0.1:8000
  endpoint: /infer
  timeout_s: 2.0
  run_duration_s: 1.0
observer:
  name: mock
  image_size: [224, 224]
  fps: 30
  state_dim: 14
  airbot_host: localhost
  airbot_left_port: 50051
  airbot_right_port: 50053
  top_camera_id: top
  left_camera_id: left
  right_camera_id: right
  enable_cameras: false
executor:
  name: raw_action
visualization:
  output_dir: /tmp/rt-vla
  enable_recording: false
rlt:
  service:
    feature_server_url: http://127.0.0.1:9000
    infer_endpoint: /rlt/infer
    token_endpoint: /rlt/token
    status_endpoint: /rlt/status
    timeout_s: 9.0
  observation:
    prompt: pick up block
    velocity_source: sdk
    diffusion_steps: 7
    reference_horizon: 40
    action_horizon: 10
    chunk_stride: 10
    vla_replan_horizon_scale: 2
  learner:
    checkpoint_dir: /tmp/checkpoints
    init_checkpoint_dir: /tmp/stage1
    overwrite: true
    resume: true
    record_repro: false
    repro_shard_size: 8
    batch_size: 8
    warmup_steps: 4
    utd_ratio: 2
    sample_queue_size: 2048
    policy_queue_size: 2
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.rlt.service.feature_server_url == "http://127.0.0.1:9000"
    assert config.rlt.service.timeout_s == 9.0
    assert config.rlt.observation.prompt == "pick up block"
    assert config.rlt.observation.velocity_source == "sdk"
    assert config.rlt.observation.chunk_stride == 10
    assert config.rlt.observation.vla_replan_horizon_scale == 2
    assert config.rlt.learner.resume is True
    assert config.rlt.learner.overwrite is True
    assert config.rlt.learner.repro_shard_size == 8
    assert config.rlt.learner.sample_queue_size == 2048


def test_load_config_expands_environment_variables_and_coerces_scalars(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RTVLA_FEATURE_SERVER_URL", "http://10.0.0.2:8000")
    monkeypatch.setenv("AIRBOT_LEFT_PORT", "50051")
    monkeypatch.setenv("AIRBOT_RIGHT_PORT", "50053")
    monkeypatch.setenv("RLT_TIMEOUT_S", "7.5")
    monkeypatch.setenv("RLT_RECORD_REPRO", "true")
    monkeypatch.setenv("RLT_PROMPT", "pick up cube")
    config_path = tmp_path / "client.yaml"
    config_path.write_text(
        """
client:
  infer_url: ${RTVLA_FEATURE_SERVER_URL}
  endpoint: /infer
  timeout_s: ${RLT_TIMEOUT_S}
  run_duration_s: 1.0
observer:
  name: mock
  image_size: [224, 224]
  fps: 30
  state_dim: 14
  airbot_host: localhost
  airbot_left_port: ${AIRBOT_LEFT_PORT}
  airbot_right_port: ${AIRBOT_RIGHT_PORT}
  top_camera_id: top
  left_camera_id: left
  right_camera_id: right
  enable_cameras: false
executor:
  name: raw_action
visualization:
  output_dir: /tmp/rt-vla
  enable_recording: false
rlt:
  service:
    feature_server_url: ${RTVLA_FEATURE_SERVER_URL}
    timeout_s: ${RLT_TIMEOUT_S}
  observation:
    prompt: ${RLT_PROMPT}
  learner:
    record_repro: ${RLT_RECORD_REPRO}
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.client.infer_url == "http://10.0.0.2:8000"
    assert config.client.timeout_s == 7.5
    assert config.observer.airbot_left_port == 50051
    assert config.observer.airbot_right_port == 50053
    assert config.rlt.service.feature_server_url == "http://10.0.0.2:8000"
    assert config.rlt.service.timeout_s == 7.5
    assert config.rlt.observation.prompt == "pick up cube"
    assert config.rlt.learner.record_repro is True


def test_acceptance_config_templates_load_with_environment(monkeypatch) -> None:
    root = Path(__file__).resolve().parents[3]
    config_dir = root / "docs" / "rlt_stage2_realtime_vla_v2_acceptance_configs"
    monkeypatch.setenv("RTVLA_FEATURE_SERVER_URL", "http://127.0.0.1:8000")
    monkeypatch.setenv("RLT_ACCEPTANCE_DIR", "/tmp/rlt-acceptance")
    monkeypatch.setenv("RLT_PROMPT", "pick up cube")
    monkeypatch.setenv("RLT_TRAIN_CONFIG", "pi05_airbot_rlt")
    monkeypatch.setenv("RLT_STAGE1_CHECKPOINT", "/tmp/stage1")
    monkeypatch.setenv("AIRBOT_HOST", "localhost")
    monkeypatch.setenv("AIRBOT_LEFT_PORT", "50051")
    monkeypatch.setenv("AIRBOT_RIGHT_PORT", "50053")

    mock_cfg = load_config(config_dir / "mock_client.yaml")
    mock_resume_cfg = load_config(config_dir / "mock_client_resume.yaml")
    raw_cfg = load_config(config_dir / "airbot_raw_client.yaml")
    mpc_cfg = load_config(config_dir / "airbot_mpc_client.yaml")

    assert mock_cfg.observer.name == "mock"
    assert mock_cfg.rlt.observation.velocity_source == "sdk"
    assert mock_resume_cfg.rlt.learner.resume is True
    assert mock_resume_cfg.rlt.learner.overwrite is False
    assert mock_resume_cfg.rlt.learner.checkpoint_dir == mock_cfg.rlt.learner.checkpoint_dir
    assert raw_cfg.observer.name == "airbot_real"
    assert raw_cfg.executor.name == "raw_action"
    assert raw_cfg.observer.airbot_left_port == 50051
    assert raw_cfg.observer.top_camera_id == "cam_high"
    assert raw_cfg.observer.left_camera_id == "cam_left_wrist"
    assert raw_cfg.observer.right_camera_id == "cam_right_wrist"
    assert raw_cfg.observer.airbot_rlt_config_path.endswith("airbot_rlt_env.toml")
    assert raw_cfg.executor.airbot_left_port == 50051
    assert raw_cfg.executor.control_dt_s == 0.04
    assert raw_cfg.executor.action_interval_ms == 40.0
    assert raw_cfg.executor.enable_servo_interpolation is False
    assert raw_cfg.executor.left_gripper_bias == 0.0
    assert raw_cfg.executor.infer_fixed_dims == []
    assert raw_cfg.executor.command_fixed_dims == []
    assert raw_cfg.feedback.enable_leader_override is True
    assert raw_cfg.feedback.left_leader_port == 50050
    assert raw_cfg.feedback.right_leader_port == 50052
    assert raw_cfg.rlt.observation.velocity_source == "sdk"
    assert raw_cfg.rlt.observation.chunk_stride == 2
    assert raw_cfg.rlt.learner.batch_size == 256
    assert raw_cfg.rlt.learner.warmup_steps == 1000
    assert raw_cfg.rlt.learner.utd_ratio == 5
    assert raw_cfg.rlt.learner.repro_shard_size == 64
    assert raw_cfg.rlt.learner.sample_queue_size == 1280
    assert mpc_cfg.executor.name == "ondevice_mpc"
    assert mpc_cfg.executor.planner_dims
    assert mpc_cfg.executor.infer_fixed_dims == []
    assert mpc_cfg.executor.command_fixed_dims == []
    assert mpc_cfg.rlt.observation.velocity_source == "sdk"
    assert mpc_cfg.rlt.learner.sample_queue_size == 1280
    assert mpc_cfg.rlt.learner.checkpoint_dir == "/tmp/rlt-acceptance/airbot_mpc_checkpoint"
