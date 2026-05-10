# RLT Stage 2 RT-VLA v2 整体迁移验收流程

本文档是一份可执行验收 SOP。验收者不需要改代码，也不需要手写 YAML；client 配置已经放在：

```text
docs/rlt_stage2_realtime_vla_v2_acceptance_configs/
```

只需要按本文设置现场变量、运行命令、操作硬件并记录观察结果。本文覆盖完整迁移目标，不只覆盖 Batch E：

- Batch A：协议、queue schema、checkpoint ownership。
- Batch B：`src/rt_vla` package、action helper、observation / proprio、feedback。
- Batch C：JAX feature server、RawActionExecutor training mode、learner process。
- Batch D：client actor / collector runtime、client 主入口、checkpoint / repro / resume、mock E2E。
- Batch E：真机 raw smoke、MPC / smoothing 归因、Triton / RTC 加速 gate。

## 0. 配置文件

已预置四份 client 配置：

```text
docs/rlt_stage2_realtime_vla_v2_acceptance_configs/mock_client.yaml
docs/rlt_stage2_realtime_vla_v2_acceptance_configs/mock_client_resume.yaml
docs/rlt_stage2_realtime_vla_v2_acceptance_configs/airbot_raw_client.yaml
docs/rlt_stage2_realtime_vla_v2_acceptance_configs/airbot_mpc_client.yaml
docs/rlt_stage2_realtime_vla_v2_acceptance_configs/airbot_rlt_env.toml
```

这些 YAML 使用环境变量展开。验收者只改下面这一段变量，不改 YAML：

```bash
export RLT_ACCEPTANCE_DIR=/tmp/rlt_stage2_rtvla_acceptance
export RLT_TRAIN_CONFIG=pi05_airbot_rlt
export RLT_STAGE1_CHECKPOINT=/home/icrlab02/airbot/openpi_rlt_rt/checkpoints/pi05_airbot_rlt_token/stage1_airbot_rlt/9999
export RTVLA_FEATURE_SERVER_URL=http://127.0.0.1:8000
export RLT_PROMPT="YOUR_TASK_PROMPT"

export AIRBOT_HOST=localhost
export AIRBOT_LEFT_PORT=50051
export AIRBOT_RIGHT_PORT=50053

mkdir -p "$RLT_ACCEPTANCE_DIR"
```

说明：

- `mock_client.yaml` 使用 mock observer / noop actuator，用来验收软件闭环。
- `mock_client_resume.yaml` 复用 mock checkpoint dir，专门验收 resume。
- `airbot_raw_client.yaml` 使用 Airbot 真机、keyboard feedback、RawActionExecutor。
- `airbot_mpc_client.yaml` 使用 Airbot 真机、keyboard feedback、OnDeviceMpcExecutor。
- `airbot_rlt_env.toml` 是 RLT Airbot env 配置，真机相机从这里读取，不使用 RT-VLA RealSense serial / camera id 配置。
- 真机配置使用 `velocity_source: sdk`；RT-VLA Airbot observer 会把 follower SDK 读到的 joint velocity 写入 `joint_velocity`。

真机配置与非 RT-VLA `scripts/train_rlt_online.py` / `src/openpi/rlt/airbot_env.py` 的默认值保持一致：

- follower ports：left `50051`，right `50053`。
- leader ports：left `50050`，right `50052`，`s` 键进入 leader-arm intervention。
- 初始复位姿态：left `[-0.3736, -0.8108, 0.6645, 1.4765, -0.8911, -1.3767, 0.0]`，right `[0.3809, -0.8669, 0.7577, -1.5047, 0.9221, 1.6538, 0.0]`。
- 控制频率：25 Hz，对应 `control_dt_s: 0.04`、`action_interval_ms: 40.0`、`servo_interval_ms: 40.0`。
- 相机配置：`airbot_rlt_env.toml` 中的 `cam_high=4`、`cam_left_wrist=2`、`cam_right_wrist=0`，以及 RLT 的 width / height / fps / crop 字段。
- proprio 速度：arm joint velocity 从 follower SDK `get_joint_vel()` 读取；gripper velocity 与非 RT-VLA env 一样由 eef position 时间差估算。
- 双臂 action 直通：`infer_fixed_dims: []`、`command_fixed_dims: []`，不固定右臂或 gripper。
- RLT 训练参数：`chunk_stride: 2`、`warmup_steps: 1000`、`batch_size: 256`、`utd_ratio: 5`、`replay_capacity: 100000`、`save_interval: 1000`、`log_interval: 100`、`repro_shard_size: 64`、`sample_queue_size: 1280`、`policy_queue_size: 1`。
- keyboard 热键：`s/y/n/esc`。

follower I/O 说明：

- 非 RT-VLA `AirbotRLTEnv` 直接在 env 内用 `AIRBOTArm` 管 follower reset / observe / action。
- RT-VLA RLT 迁移沿用 realtime-vla client 的 follower I/O：observer 和 executor 共享同一个 `AIRBOTPlay` follower 连接，由 executor 负责 14D joint/eef action 下发，由 observer 从同一 SDK 连接读取 state / joint velocity。
- 这是一处实现层差异；验收配置层的 follower ports、reset pose、control period、action 维度和 leader 接管端口都与非 RT-VLA 保持一致。

## 1. 本地单测 Gate

运行：

```bash
UV_CACHE_DIR=/tmp/uv-cache PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest \
  src/rt_vla/client \
  src/rt_vla/server/rlt_feature_server_test.py \
  src/openpi/rlt/action_space_test.py \
  src/openpi/rlt/rlt_test.py
```

通过标准：

- pytest 全部通过，允许 manual test 被 deselect。
- `executor_test.py` 覆盖 Raw 和 MPC training telemetry。
- `rlt_feature_server_test.py` 覆盖 `/rlt/infer`、`/rlt/token` 和 hybrid RTC/JAX feature model。
- `rlt_main_test.py` 覆盖 mock E2E、resume、stop signal、execution record repro。

## 2. 启动 Feature Server

打开终端 A，运行：

```bash
uv run python -m rt_vla.server.rlt_feature_server \
  --host 0.0.0.0 \
  --port 8000 \
  --config-name "$RLT_TRAIN_CONFIG" \
  --checkpoint-dir "$RLT_STAGE1_CHECKPOINT" \
  --default-prompt "$RLT_PROMPT"
```

预期表现：

- server 启动后保持运行。
- server 只加载 Stage 1 JAX VLA feature / reference path。
- server 不加载 Stage 2 actor / critic / learner。

打开终端 B，检查 `/rlt/status`：

```bash
uv run python - <<'PY'
import os
import pickle
import requests

base = os.environ["RTVLA_FEATURE_SERVER_URL"].rstrip("/")
resp = requests.post(
    base + "/rlt/status",
    data=pickle.dumps(None),
    headers={"Content-Type": "application/octet-stream"},
    timeout=30,
)
resp.raise_for_status()
payload = pickle.loads(resp.content)
print(payload)
server = payload["server"]
assert server["stage1_feature_backend"] == "jax"
assert server["stage2_checkpoint_owner"] == "client"
assert not server["server_runs_actor"]
assert not server["server_runs_critic"]
assert not server["server_runs_learner"]
assert not server["server_builds_replay"]
assert not server["server_saves_stage2_checkpoint"]
PY
```

通过标准：

- 命令无 assertion error。
- 输出显示 Stage 2 checkpoint owner 是 client。

## 3. Mock E2E 验收

终端 B 运行：

```bash
uv run python -m rt_vla.client.rlt_main \
  --config-path docs/rlt_stage2_realtime_vla_v2_acceptance_configs/mock_client.yaml \
  --num-iterations 2 \
  --episode-id mock-e2e-001
```

预期表现：

- 不需要真机。
- client 请求 `/rlt/infer` 获取完整 reference plan。
- client 本地 actor 生成 refined action。
- RawActionExecutor mock/noop 执行 atomic chunk。
- next observation 在同一 plan 覆盖范围内调用 `/rlt/token`；到达 replan boundary 时调用 `/rlt/infer` 生成下一段完整 plan。
- learner process 收到 ReplayItem 并发布 PolicyUpdate。
- client checkpoint / repro 写入 `$RLT_ACCEPTANCE_DIR/mock_client_checkpoint`。
- client action normalize / denormalize 使用 Stage 1 checkpoint assets 中的 norm stats、quantile 和 delta-action 配置。

检查产物：

```bash
uv run python - <<'PY'
import os
from pathlib import Path
from openpi.rlt import repro

root = Path(os.environ["RLT_ACCEPTANCE_DIR"]) / "mock_client_checkpoint"
repro_dir = root / "repro"
manifest = repro.load_manifest(repro_dir)
stop_signal = repro.load_stop_signal(repro_dir)
transitions = list(repro.iter_transition_dicts(repro_dir))
execution = list(repro.iter_execution_record_chunks(repro_dir))

print("checkpoint_root:", root)
print("state_dim:", manifest["state_dim"])
print("action_dim:", manifest["action_dim"])
print("stop_signal:", stop_signal)
print("transitions:", len(transitions))
print("execution_chunks:", len(execution))
first_feature_metadata = execution[0]["feature_metadata"]
print("first_feature_metadata:", first_feature_metadata)

assert manifest["state_dim"] == 284
assert manifest["action_dim"] == 140
assert len(transitions) == 2
assert len(execution) == 2
assert all(chunk["action_sources"] for chunk in execution)
assert "asset_id" in manifest
assert "use_quantiles" in manifest
assert "use_delta_joint_actions" in manifest
assert "plan" in first_feature_metadata
assert first_feature_metadata["plan"]["feature_id"]
assert first_feature_metadata["plan"]["reference_plan_shape"] == [50, 14]
assert first_feature_metadata["plan"]["reference_plan_norm_sha256"]
assert first_feature_metadata["plan"]["reference_plan_real_sha256"]
assert "next_token" in first_feature_metadata or "next_plan" in first_feature_metadata
asset_id = manifest.get("asset_id")
if asset_id is not None:
    assert (repro_dir / "assets" / asset_id / "norm_stats.json").exists()
PY
```

通过标准：

- `state_dim == 284`。
- `action_dim == 140`。
- transition 数量为 2。
- `execution_records.jsonl` 中每个 chunk 都有 action source。
- `execution_records.jsonl` 中每个 chunk 都有 feature metadata，包含 plan id、reference plan shape 和 reference plan hash。
- manifest 记录 `asset_id`、`use_quantiles`、`use_delta_joint_actions`；如果 `asset_id` 非空，`repro/assets/<asset_id>/norm_stats.json` 存在。

## 4. 真机前安全检查

验收者执行并口头确认：

1. 急停可用，操作者手在急停附近。
2. Airbot 两臂处于安全初始姿态，工作空间内无人员和障碍。
3. 相机画面正常，三路 OpenCV index 与 `airbot_rlt_env.toml` 一致。
4. leader / manual takeover 可用。
5. 先跑 1 个 iteration；如果动作异常，按 `esc` 或急停。
6. 键盘焦点在 client 终端，feedback 热键为：
   - `y`：当前 step reward `+1`。
   - `n`：结束 episode。
   - `s`：切换 leader-arm intervention；如果配置关闭 leader override，则保持当前 state 并标记 intervention。
   - `esc`：abort。

## 5. 真机 Raw Action Smoke

终端 B 运行：

```bash
uv run python -m rt_vla.client.rlt_main \
  --config-path docs/rlt_stage2_realtime_vla_v2_acceptance_configs/airbot_raw_client.yaml \
  --num-iterations 1 \
  --episode-id airbot-raw-001
```

验收者操作：

1. 观察机器人执行一个 10-step atomic chunk。
2. 期间按一次 `y`，确认 reward 被消费。
3. 按一次 `s`，移动 leader arms，观察 follower arms 按 leader 相对位移进入接管模式。
4. 如需提前结束，按 `n`；如有安全风险，按 `esc` 或急停。

预期表现：

- 动作连续，无 NaN、无突然大幅跳变。
- client terminal 不出现 feature schema、velocity、shape、queue 或 learner error。
- server terminal 能持续响应 `/rlt/infer` 和 `/rlt/token`。
- `$RLT_ACCEPTANCE_DIR/airbot_raw_checkpoint/repro` 写入 manifest、transition、execution records、stop signal。

检查产物：

```bash
uv run python - <<'PY'
import os
from pathlib import Path
from openpi.rlt import repro

root = Path(os.environ["RLT_ACCEPTANCE_DIR"]) / "airbot_raw_checkpoint"
repro_dir = root / "repro"
manifest = repro.load_manifest(repro_dir)
transitions = list(repro.iter_transition_dicts(repro_dir))
execution = list(repro.iter_execution_record_chunks(repro_dir))
print("manifest checkpoint_root:", manifest["checkpoint_root"])
print("transitions:", len(transitions))
print("execution chunks:", len(execution))
print("first action sources:", execution[0]["action_sources"] if execution else None)
print("first feature metadata:", execution[0].get("feature_metadata") if execution else None)
assert transitions
assert execution
assert execution[0]["feature_metadata"]["plan"]["reference_plan_norm_sha256"]
assert "next_token" in execution[0]["feature_metadata"] or "next_plan" in execution[0]["feature_metadata"]
assert transitions[0]["state"].shape == (284,)
assert transitions[0]["action"].shape == (140,)
assert transitions[0]["reference_action"].shape == (140,)
assert transitions[0]["next_reference_action"].shape == (140,)
PY
```

通过标准：

- 至少 1 条 transition。
- replay shape 正确。
- execution action source 为 `executed_action`。
- execution feature metadata 包含 plan hash，并记录 next token 或 next plan 来源。
- `actor_sample.log` 中 state / reference / delta 为有限值。

## 6. Checkpoint / Repro / Resume 验收

确认 checkpoint 目录：

```bash
find "$RLT_ACCEPTANCE_DIR/airbot_raw_checkpoint" -maxdepth 3 -type f | sort
```

必须看到：

```text
repro/manifest.json
repro/stop_signal.json
repro/execution_records.jsonl
repro/assets/<asset_id>/norm_stats.json  # asset_id 非空时
repro/transitions/transitions_*.npz
actor_sample.log
```

resume smoke 使用预置 `mock_client_resume.yaml`，不碰真机：

```bash
uv run python -m rt_vla.client.rlt_main \
  --config-path docs/rlt_stage2_realtime_vla_v2_acceptance_configs/mock_client_resume.yaml \
  --num-iterations 1 \
  --episode-id mock-resume-001
```

通过标准：

- 命令从 `$RLT_ACCEPTANCE_DIR/mock_client_checkpoint` 的 latest checkpoint 恢复。
- 新 manifest 的 `start_step` 大于 0。
- stop signal final step 大于 start step。

检查：

```bash
uv run python - <<'PY'
import os
from pathlib import Path
from openpi.rlt import repro

repro_dir = Path(os.environ["RLT_ACCEPTANCE_DIR"]) / "mock_client_checkpoint" / "repro"
manifest = repro.load_manifest(repro_dir)
stop_signal = repro.load_stop_signal(repro_dir)
print("start_step:", manifest["start_step"])
print("final_step:", stop_signal["final_step"])
print("asset_id:", manifest.get("asset_id"))
print("use_quantiles:", manifest.get("use_quantiles"))
print("use_delta_joint_actions:", manifest.get("use_delta_joint_actions"))
assert manifest["start_step"] > 0
assert stop_signal["final_step"] > manifest["start_step"]
assert "use_quantiles" in manifest
assert "use_delta_joint_actions" in manifest
PY
```

说明：

- client runtime 内部 `RLTTrainingRuntime.status()` 会暴露 `env_step`、`actor_step` 和 learner stats（learner step、replay size、latest checkpoint）。当前 SOP 不要求单独运行 status CLI；该行为由本地单测 gate 覆盖。

## 7. Client Deploy Actor 验收

使用 mock E2E 产出的 client Stage 2 checkpoint 验证 deploy actor 能恢复并输出 `(10, 14)` refined action：

```bash
uv run python - <<'PY'
import os
from pathlib import Path

import numpy as np

from rt_vla.client.rlt_deploy import DeployActorConfig
from rt_vla.client.rlt_deploy import load_deploy_actor
from rt_vla.client.rlt_deploy import sample_deploy_action

root = Path(os.environ["RLT_ACCEPTANCE_DIR"]) / "mock_client_checkpoint"
steps = sorted(int(path.name) for path in root.iterdir() if path.is_dir() and path.name.isdigit())
assert steps, f"no numeric checkpoint step found under {root}"
checkpoint_dir = root / str(steps[-1])
actor = load_deploy_actor(
    DeployActorConfig(
        config_name=os.environ["RLT_TRAIN_CONFIG"],
        checkpoint_dir=checkpoint_dir,
        action_horizon=10,
    )
)
action = sample_deploy_action(
    actor,
    rlt_state=np.zeros((284,), dtype=np.float32),
    reference_action=np.zeros((10, 14), dtype=np.float32),
    deterministic=True,
)
print("checkpoint_dir:", checkpoint_dir)
print("deploy_action_shape:", action.shape)
print("deploy_action_max_abs:", float(np.max(np.abs(action))))
assert action.shape == (10, 14)
assert np.isfinite(action).all()
PY
```

通过标准：

- deploy actor 从 client Stage 2 checkpoint 的 `policy_state` 恢复。
- refined action shape 为 `(10, 14)`。
- 输出为有限值。

## 8. MPC / Smoothing 归因验收

确认 raw smoke 通过后，再运行 MPC：

```bash
uv run python -m rt_vla.client.rlt_main \
  --config-path docs/rlt_stage2_realtime_vla_v2_acceptance_configs/airbot_mpc_client.yaml \
  --num-iterations 1 \
  --episode-id airbot-mpc-001
```

验收者观察：

- 机器人动作仍连续。
- MPC 被视为 client 侧环境 wrapper；replay action 应对齐 actor 的 policy command。
- 如果 MPC solver 不可用，executor fallback 仍应记录 post-MPC / executed 归因；不应改变 replay action 语义。
- 若动作异常，立即 `esc` 或急停。

检查 MPC telemetry：

```bash
uv run python - <<'PY'
import os
from pathlib import Path
from openpi.rlt import repro

repro_dir = Path(os.environ["RLT_ACCEPTANCE_DIR"]) / "airbot_mpc_checkpoint" / "repro"
chunks = list(repro.iter_execution_record_chunks(repro_dir))
print("execution chunks:", len(chunks))
assert chunks
sources = set()
for chunk in chunks:
    assert chunk["feature_metadata"]["plan"]["reference_plan_norm_sha256"]
    for record in chunk["records"]:
        sources.add(record["source"])
        telemetry_sources = {item["source"] for item in record.get("telemetry", [])}
        assert "post_mpc_action" in telemetry_sources
        assert "executed_action" in telemetry_sources
print("record sources:", sorted(sources))
PY
```

通过标准：

- `execution_records.jsonl` 每个 record 都有 `post_mpc_action` 和 `executed_action`。
- replay action 对齐 actor policy command；post-MPC / executed action 仅用于执行归因。
- 每个 execution chunk 都有 feature metadata。
- policy command、post-MPC action 和 executed action 可追溯并可视化对齐。

## 9. Triton / RTC 加速 Gate

Triton / RTC 不是首版训练闭环依赖。验收分两层：

1. 必跑单测 gate：`rlt_feature_server_test.py` 中 hybrid 用例通过，证明 JAX token + RTC reference 的 schema 和 normalize 逻辑成立。
2. 如果现场已有 RTC artifacts，再跑数值对齐。

现场有 RTC artifacts 时设置：

```bash
export RTC_CHECKPOINT=/path/to/rtc_checkpoint.pkl
export RTC_TOKENIZER=/path/to/tokenizer
export RTC_NORM_STATS_DIR=/path/to/norm_stats_dir
```

数值对齐要求：

- JAX `/rlt/infer` 和 hybrid `/rlt/infer` 使用同一 observation。
- hybrid `rl_token` 来自 JAX token path，shape 为 `(256,)`。
- hybrid `reference_plan_list` 来自 RTC / Triton adapter，shape 为 `(50, 14)`。
- 记录 reference action diff：max_abs、mean_abs、p95。
- 如果未来做 pure Triton RL token，必须新增 token max_abs / cosine similarity gate 和真机 rollout smoke。

通过标准：

- Triton 缺 RL token 时不阻塞 JAX 首版闭环。
- hybrid path 不把 Stage 2 actor / critic / learner 放到 server。
- 延迟、queue length、reference diff、post action 进入验收记录。

## 10. 最终验收报告

验收者按下面模板填写结果：

```text
date:
git_commit:
RLT_ACCEPTANCE_DIR:
RLT_STAGE1_CHECKPOINT:
RTVLA_FEATURE_SERVER_URL:
RLT_PROMPT:

local_tests:
  command:
  result:

server_status:
  stage1_feature_backend:
  stage2_checkpoint_owner:
  server_runs_actor:
  server_runs_critic:
  server_runs_learner:
  server_builds_replay:

mock_e2e:
  command:
  transitions:
  execution_chunks:
  checkpoint_dir:
  feature_metadata_ok:
  norm_stats_snapshot_ok:
  use_quantiles:
  use_delta_joint_actions:
  deploy_action_shape_ok:

airbot_raw:
  command:
  observed_motion_ok:
  reward_key_y_ok:
  terminate_key_n_ok:
  intervention_key_s_ok:
  abort_key_esc_ok:
  replay_shapes_ok:
  actor_sample_log_ok:
  feature_metadata_ok:
  norm_stats_snapshot_ok:

airbot_mpc:
  command:
  observed_motion_ok:
  post_mpc_action_recorded:
  executed_action_recorded:
  replay_action_source: policy_command
  feature_metadata_ok:

triton_rtc:
  unit_gate_passed:
  rtc_artifacts_available:
  reference_diff_max_abs:
  reference_diff_mean_abs:
  reference_diff_p95:
  token_source:
```
