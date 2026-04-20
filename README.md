# PI05-RLT Airbot

本仓库在 [openpi](https://github.com/Physical-Intelligence/openpi) 基础上实现了 `PI05 + RLT（RL Token）` 两阶段训练框架，并接入了 Airbot 双臂硬件。

**环境配置**

- 数据采集环境配置及采集说明：[Airbot Play 数据采集教程](https://docs.airbots.online/airbot-play/hardware-driver/tutorials/data-collection.html)
- 模型训练环境配置：[Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)

## 1. 前提

假设实验机已经满足下面条件：

- Ubuntu 22.04 或兼容环境
- NVIDIA GPU，且显存足以跑 `pi05` LoRA 微调和在线 RL
- 已安装 `uv`
- 已按 [openpi](https://github.com/Physical-Intelligence/openpi) 安装好 openpi 依赖
- 已安装可用的 `jax` / CUDA 运行时
- 如果要做真实在线 RL，实验机可以访问 Airbot 运行时或仿真环境

## 2. 先检查当前代码里的占位项和人工配置项

当前 Airbot-RLT 的训练配置已经接入，但 Stage 1 的数据集 `repo_id` 还是占位符。开始实验前，先把下面配置里的 `repo_id` 改成你的真实 LeRobot 数据集：

- `src/openpi/training/config.py` 中的 `pi05_airbot_rlt_token`

当前默认值是：

```python
repo_id="your_hf_username/my_airbot_dataset"
```

除此之外，下面这些项也需要人工确认或填写：

- `OPENPI_AIRBOT_RLT_CONFIG` 是可选的；默认值已对齐 TOK2 标准硬件（端口、相机 index、`max_episode_steps=500`）。只有在硬件端口或相机编号不是默认值、或需要修改 `prompt` 时，才需要准备这个文件（见第 6.1 节）
- 真机 Stage 2 依赖 `airbot_py`；键盘奖励和接管依赖 `pynput`
- Stage 1 数据里的任务文本分布，要和后面 rollout / Stage 2 用的 `prompt` 保持一致，不要训练时用任务名、上线时换成另一套自然语言描述
- 当前仓库并不是论文的逐项同构复现，主要差异见第 5.4 节

## 3. Stage 1 数据采集

`rl-token` 当前没有直接集成 `lerobot_play.record` 的采集入口，因此推荐工作流是：

1. 用官方 `lerobot_play.record` 先采集原始 LeRobot Play 数据
2. 在 `rl-token/` 下计算 norm stats 并启动 Stage 1 训练

数据采集环境配置及完整采集说明见 [Airbot Play 数据采集教程](https://docs.airbots.online/airbot-play/hardware-driver/tutorials/data-collection.html)。教程给出的 YAML 结构由 `robot / teleop / run / dataset` 四部分组成，采集命令为：

```bash
conda activate lerobot
python3 -m lerobot_play.record --yaml /path/to/openpi_ws/rl-token/examples/airbot/tok2_stage1_collection.yaml
```

其中：

- `dataset.repo_id` 决定数据默认写到 `~/.cache/huggingface/lerobot/<repo_id>/`
- `run.single_task` 会把任务文本作为数据集元数据写入数据集

当前仓库已经提供一个可直接改端口和相机编号的模板：

```text
examples/airbot/tok2_stage1_collection.yaml
```

这个模板按 Stage 2 Airbot 环境的默认硬件命名对齐了三路相机：

- `cam_high -> /dev/video0`
- `cam_left_wrist -> /dev/video2`
- `cam_right_wrist -> /dev/video4`

如果你的设备不是 TOK2，只需要把下面字段换成你实际硬件对应值：

- `robot.type`
- `teleop.type`
- `robot.port` / `teleop.port`
- 三路相机的 `index_or_path`
- `run.single_task`
- `dataset.repo_id`

### 3.1 采集质量建议

- `run.single_task` 直接决定 episode task 文本；同一任务不要混用多套完全不同的表述
- 采集阶段就固定三路相机命名
- 如果后续 Stage 2 运行时用的是双臂任务，Stage 1 不要只采单臂演示，否则 reference chunk 的分布会和在线 refinement 明显错位

### 3.2 从 MCAP 文件转换（可选路径）

如果采集工具直接输出的是 MCAP 格式（即每个 episode 对应一个 `.mcap` 文件），可以用本仓库提供的转换脚本生成 LeRobot 数据集：

```bash
uv run examples/airbot/convert_mcap_to_lerobot.py \
    --mcap-dir /path/to/mcap_dir \
    --repo-id <org>/<dataset-name>
```

脚本默认适配 PTK 双臂机器人，话题配置如下：

- **state**：`/left/follow/arm/joint_state/position`（6）+ `/left/follow/eef/joint_state/position`（1）+ 右臂同理，共 14 维
- **action**：`/left/lead/arm/joint_state/position` + eef + 右臂，共 14 维
- **camera**：MCAP 附件 `/env_camera/color/image_raw` → `cam_high`，`/left_camera/color/image_raw` → `cam_left_wrist`，`/right_camera/color/image_raw` → `cam_right_wrist`
- task 文本自动从 MCAP metadata 的 `task_info.task_description` 字段读取

## 4. Airbot 数据格式要求

训练和部署都复用同一套 Airbot 输入输出语义。

### 4.1 训练数据需要的关键字段

LeRobot 数据集中，样本应至少能提供：

```text
observation.images.cam_high
observation.images.cam_left_wrist
observation.images.cam_right_wrist
observation.state
action
```

约束如下：

- `observation.state`: `14` 维位置，用于 pi0.5 的离散 state prompt
- `observation.proprio`: 可选；如果提供，则应为 `28` 维状态，按 `14 维位置 + 14 维速度` 组织
- `action`: Airbot 动作序列，实际有效维度为 14
- 任务文本可以存在 episode task 里；当前 Airbot 训练配置已经会从 LeRobot task 自动恢复 `prompt`
- 3 路图像分别对应顶视角、左腕、右腕

当前实现里，模型内部动作维度仍是 32，但 `AirbotInputs/AirbotOutputs` 会把 14 维 Airbot 动作映射到模型输入输出，把 `observation.state` 保留为 prompt state；如果存在 `observation.proprio`，则优先把它 pad 到 32 维连续状态槽位。

### 4.2 在线推理时的 observation schema

Airbot 客户端给 policy server 发送的数据格式应为：

```python
observation = {
    "state": state_14d,  # shape (14,), position only, for VLA prompt tokenization
    "proprio": proprio_28d,  # shape (28,), positions first and velocities after
    "images": {
        "cam_high": cam_high_uint8_chw,
        "cam_left_wrist": cam_left_wrist_uint8_chw,
        "cam_right_wrist": cam_right_wrist_uint8_chw,
    },
    "prompt": task_instruction,
}
```

服务端返回：

```python
{
    "actions": np.ndarray,  # shape (10, 14)
}
```

当前默认设定与论文对齐为：

- Stage 1 VLA `action_horizon=50`
- Stage 2 RL actor 输出 `rlt_action_horizon=10`
- Airbot 环境默认 `control_hz=50`

## 5. Stage 1：训练 RL token

Stage 1 使用标准训练入口 `scripts/train.py`，训练配置名为 `pi05_airbot_rlt_token`。

### 5.1 先计算 norm stats

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_airbot_rlt_token
```

如果数据量很大，也可以先做一次抽样统计：

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_airbot_rlt_token --max-frames 100000
```

### 5.2 启动训练

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_airbot_rlt_token \
  --exp-name stage1_airbot_rlt \
  --overwrite
```

说明：

- 这里训练的是 `PI05 + RL token`
- 当前配置里 `rlt_actor_enabled=False`
- 这一步不会训练 online RL actor/critic
- 默认 `num_train_steps=20000`，所以最终 checkpoint 通常是：
  `checkpoints/pi05_airbot_rlt_token/stage1_airbot_rlt/19999`

建议先确认以下内容再进入 Stage 2：

- loss 正常下降
- checkpoint 能稳定保存
- `assets/` 下已经有 `airbot` 对应的 norm stats

### 5.3 Stage 1 完成后先 rollout 纯 VLA

这一步很重要。Stage 2 不是用来排查 Stage 1 是否把 VLA 训坏的；在进入在线 RL 前，先确认 Stage 1 checkpoint 仍然能给出正常的 VLA reference chunk。

先用 Stage 1 config 启动 policy server：

```bash
uv run scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config=pi05_airbot_rlt_token \
  --policy.dir=checkpoints/pi05_airbot_rlt_token/stage1_airbot_rlt/19999
```

然后至少做两层检查：

1. 用 `simple_client` 先检查接口和 schema：

```bash
uv run examples/simple_client/main.py --env AIRBOT --host 127.0.0.1 --port 8000
```

2. 如果已经有 Airbot runtime 或仿真 client，再用这个 Stage 1 checkpoint 做几次纯 VLA rollout，不要立刻进入 Stage 2

进入 Stage 2 前，至少确认这些现象成立：

- 服务端稳定返回 `(50, 14)` 的 action chunk，没有 `NaN` / 全零 / 明显爆幅
- `prompt`、状态维度和三路相机都没有串 schema
- 纯 VLA rollout 的动作是连续且合理的，不是明显抖动、长时间卡死或持续朝错误方向发散

原因很直接：当前 Stage 2 actor 只是在 Stage 1 产出的 reference action 周围做局部 refinement。如果 Stage 1 的 reference policy 已经不对，在线 RL 很难在短时间内把它救回来。

### 5.4 Stage 1 超参与论文默认值的差异

论文附录里提到，单任务 Stage 1 通常做 `2000` 到 `10000` 个 gradient steps。当前仓库的 Airbot 配置默认是 `num_train_steps=20000`，这是仓库里的工程默认值，不是论文里的精确同配超参。

所以：

- 如果目标是先跑通工程链路，可以先沿用当前 repo 默认值
- 如果目标是尽量贴近论文实验，需要手工回调 Stage 1 训练步数，而不是默认把 `20000` 当成论文设定

## 6. Stage 2：在线训练 RLT actor-critic

Stage 2 使用 `scripts/train_rlt_online.py`。这一步会：

- 从 Stage 1 checkpoint 初始化 `PI05 + RL token`
- 冻结 `PI05 + RL token`
- 只训练轻量 actor / twin critic
- actor / critic 输入使用 `RL token + proprio(position + velocity)`

### 6.1 内置 Airbot Stage 2 env

`rl-token` 里现在已经提供了一个可直接给 `train_rlt_online.py` 使用的双臂 Airbot 环境：

```text
openpi.rlt.airbot_env:create_env
```

它参考了 `hil-serl_airbot` 的 `dual_airbot` 真机接口，但按 RLT Stage 2 做了收敛，保留了以下能力：

- 双臂 AIRBOT follower 真机 joint position 控制
- leader 臂人工接管
- 操作员手动给 sparse `+1` reward
- 每个 chunk step 返回 `info["step_rewards"]`
- replay buffer 写入当前边界的实际执行动作；接管步会用人类执行动作替换对应 reference 前缀

默认设置：

- 控制频率默认 `50 Hz`
- 夹爪默认使用连续控制，不再沿用 `hil-serl_airbot` 的二值夹爪假设
- `y` 默认只打 reward，不终止 episode
- 如果你希望 `y` 同时结束 episode，可在配置里设 `terminate_on_success_reward=true`

默认键位：

- `s`：切换人工接管
- `y`：在当前低层控制步打一个 `+1` sparse reward
- `n`：结束当前 episode，reward 为 `0`
- `Esc`：中止当前 episode，reward 为 `0`

默认相机 / 端口配置：

- left follower: `50051`
- right follower: `50053`
- left leader: `50050`
- right leader: `50052`
- `cam_high=0`
- `cam_left_wrist=2`
- `cam_right_wrist=4`

如果你的硬件映射不同，可以通过环境变量 `OPENPI_AIRBOT_RLT_CONFIG` 指向一个 `.json` 或 `.toml` 配置文件，覆盖 prompt、端口、相机、episode 长度等参数。

最小 JSON 示意：

```json
{
  "prompt": "insert the ethernet connector",
  "terminate_on_success_reward": false,
  "max_episode_steps": 500,
  "left_arm": {
    "follower_port": 50051,
    "leader_port": 50050,
    "gripper_mode": "continuous"
  },
  "right_arm": {
    "follower_port": 50053,
    "leader_port": 50052,
    "gripper_mode": "continuous"
  },
  "cameras": {
    "cam_high": { "index": 0 },
    "cam_left_wrist": { "index": 2 },
    "cam_right_wrist": { "index": 4 }
  }
}
```

注意：

- `prompt` 最好与 Stage 1 训练时使用的任务提示保持一致
- 真机模式依赖 `airbot_py`
- 键盘奖励 / 接管依赖 `pynput`

### 6.2 启动在线 RL

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train_rlt_online.py \
  --config pi05_airbot_rlt \
  --init-checkpoint-dir checkpoints/pi05_airbot_rlt_token/stage1_airbot_rlt/9999 \
  --checkpoint-dir checkpoints/pi05_airbot_rlt/online_stage2 \
  --max-env-steps 50000 \
  --warmup-steps 1000 \
  --chunk-stride 2 \
  --save-interval 1000
```

如果硬件端口 / 相机 / prompt 与默认值不同，在命令前加：

```bash
OPENPI_AIRBOT_RLT_CONFIG=/abs/path/to/airbot_rlt.json \
```

默认关键设定：

- `VLA action_horizon=50`
- `RL actor horizon=10`
- `chunk_stride=2`
- `utd_ratio=5`
- `critic:actor=2:1`
- actor / critic 学习率默认 `3e-4`

如果中途断了，可以用：

```bash
uv run scripts/train_rlt_online.py \
  --config pi05_airbot_rlt \
  --init-checkpoint-dir checkpoints/pi05_airbot_rlt_token/stage1_airbot_rlt/19999 \
  --checkpoint-dir checkpoints/pi05_airbot_rlt/online_stage2 \
  --resume
```

## 7. 部署和验证

Stage 2 产出的 checkpoint 可以直接用于部署。

### 7.1 启动 policy server

```bash
uv run scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config=pi05_airbot_rlt \
  --policy.dir=checkpoints/pi05_airbot_rlt/online_stage2/49999
```

注意：

- 没有内置 `--env AIRBOT`
- Airbot 必须显式指定 `policy:checkpoint`

### 7.2 用 simple client 做通路检查

```bash
uv run examples/simple_client/main.py --env AIRBOT --host 127.0.0.1 --port 8000
```

这一步不会验证真实机器人行为，但可以先确认：

- server 能正常起
- Airbot observation schema 能被服务端接受
- 服务端能返回 `(10, 14)` 的 action chunk

### 7.3 用 Airbot rollout client 做真机纯 VLA 验证

如果已经接好 Airbot 双臂和三路相机，可以直接运行：

```bash
python -m examples.airbot.vla_rollout_client \
    --host 192.168.1.10 \
    --port 8000 \
    --prompt "pick up the cup"
```

常用补充参数：

- `--config-path /path/to/airbot_eval.toml`：从 `.toml` / `.json` 读取端口、相机、prompt 等环境配置
- `--action-horizon 25`：每次查询 policy server 后连续执行的动作步数
- `--control-hz 25`：低层控制频率
- `--normalize-gripper True`：Stage 1 数据若使用归一化夹爪（常见于 imitate-all / MCAP 转 LeRobot），保持 `True`

如果不使用配置文件，也可以直接通过命令行覆盖相机和端口，例如：

```bash
python -m examples.airbot.vla_rollout_client \
    --host 192.168.1.10 \
    --port 8000 \
    --prompt "pick up the cup" \
    --cam-high 6 \
    --cam-left-wrist 2 \
    --cam-right-wrist 0 \
    --left-follower-port 50051 \
    --right-follower-port 50053
```

运行时键位：

- `p`：暂停 / 继续
- `r`：结束当前 episode 并重置机械臂
- `q`：当前 episode 结束后退出

进入 Stage 2 前，建议至少确认：

- rollout 动作连续，没有明显抖动、卡死或发散
- `prompt`、状态维度和三路相机没有串 schema
- gripper 数值范围与训练数据一致；不确定时优先检查 `observation.state` 的第 `7`、`14` 维是否接近 `[0, 1]` 还是 `[0, 0.07]`

## 8. 建议的实验顺序

建议按下面顺序推进：

1. 只跑 `compute_norm_stats.py`，确认数据管线通
2. 跑 Stage 1，确认 `PI05 + RL token` 能正常收敛并保存 checkpoint
3. 用 Stage 1 checkpoint 先做一次纯 VLA rollout 检查
4. 启动 policy server，用 `simple_client` 验证 Airbot schema
5. 先把 `OPENPI_AIRBOT_FAKE_ENV=true` 跑通，确认 Stage 2 训练循环、checkpoint 和日志都正常
6. 再接真实 Airbot runtime 做在线 RL

## 9. 常见坑

- `pi05_airbot_rlt_token` 的 `repo_id` 是占位符，不改会直接跑不起来；`pi05_airbot_rlt` 已设为 `"not/needed"`，不需要修改
- `OPENPI_AIRBOT_RLT_CONFIG` 是可选的；默认值已对齐 TOK2 标准硬件，只有端口 / 相机 / prompt 与默认不同时才需要准备
- Stage 1 rollout 检查要用 `pi05_airbot_rlt_token`，不要一上来就拿 Stage 2 config 排查 VLA 是否正常
- Stage 2 需要传入的是 Stage 1 最终 checkpoint 目录，不是上一级实验目录
- Airbot 输出动作是 14 维，虽然模型内部动作维度仍是 32
- Stage 1 纯 VLA checkpoint 默认返回 `(50, 14)`，Stage 2 actor-critic checkpoint 默认返回 `(10, 14)`，不要把两者混用
- 真机部署前先用 `examples/simple_client/main.py --env AIRBOT` 做接口检查

## 10. 推荐记录项

每次实验至少记录：

- 训练配置名
- 数据集 `repo_id`
- Stage 1 checkpoint 路径
- Stage 2 checkpoint 路径
- `OPENPI_AIRBOT_RLT_CONFIG` 的版本
- `chunk_stride`、`max_env_steps`、`warmup_steps`
- 真实机器人上的 prompt 集合和成功率

## 11. 运行单元测试

> **注意**：当前代码在没有 `uv` / `jax` / `pytest` 的环境中编写，所有测试均未在开发机上实际运行过。在满足第 1 节前提条件的机器上按以下步骤执行测试。

### 11.1 前提

- 已按 [openpi](https://github.com/Physical-Intelligence/openpi) 安装好依赖（即执行过 `uv sync`）
- 已安装可用的 `jax`（CPU 即可运行大多数测试，GPU 可加速模型相关测试）

### 11.2 运行全部测试

在 `rl-token/` 目录下执行：

```bash
uv run pytest
```

pytest 会自动扫描 `src/`、`scripts/`、`packages/` 下所有 `*_test.py` 文件。

### 11.3 只运行部分测试

```bash
# 只跑 RLT 核心逻辑（replay buffer、trainer、checkpointing）
uv run pytest src/openpi/rlt/rlt_test.py

# 只跑 Airbot 环境（无需真机，使用 fake arm）
uv run pytest src/openpi/rlt/airbot_env_test.py

# 只跑 Airbot policy 的输入输出 transform
uv run pytest src/openpi/policies/airbot_policy_test.py

# 只跑训练循环冒烟测试（CPU，跑 debug config）
uv run pytest scripts/train_test.py

# 只跑 openpi-client 子包测试
uv run pytest packages/openpi-client/
```

### 11.4 强制使用 CPU（避免占用 GPU 显存）

```bash
JAX_PLATFORMS=cpu uv run pytest
```

### 11.5 跳过需要手动执行的测试

部分测试用 `@pytest.mark.manual` 标注，需要真实硬件或网络资源，默认不会被自动收集。如需显式排除：

```bash
uv run pytest -m "not manual"
```
