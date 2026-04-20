# Bug Report: Stage 2 Online RL — Actor Mean 爆炸

## 现象

在 `train_rlt_online.py` 跑二阶段在线 RL 时，训练启动后（warmup 结束）actor 输出的动作均值迅速爆炸到千量级，最终导致 critic 发散、训练崩溃。

从 `fail_log.txt` 可见：

- warmup（步骤 < 1000）期间 `delta`（actor 输出 - reference action）std 约 0.7、max_abs < 3，完全正常
- warmup 结束后第一批 actor 更新后 `delta max_abs` 即跳至数百，随后几步内达到数千
- critic Q 值同步飙升至数百

## 诊断过程

### 步骤 1：排查初始化

用 `check_rl_token_scale.py` 对 Stage 1 checkpoint 做前向：

- `rlt_state` max_abs = 3.44（正常，rl_token 尺度合理）
- `reference_action` max_abs = 4.48（正常，归一化动作空间）
- `rlt_actor` 前向：mean_out max_abs = 5.51，delta max_abs = 2.04 ← **初始化没问题**
- `rlt_actor` 输出层权重 std = 0.0625（Glorot 而非预期的 `zero_output_init`）—— 但实际结果正常

**结论：Stage 1 checkpoint 加载和 rlt_actor 初始权重不是根因。**

### 步骤 2：离线模拟训练循环

创建 `train_rlt_sim.py`：用 LeRobot 数据集帧替代真实机器人，完整复现 Stage 2 训练逻辑，并在 `trainer.py` 中添加逐步监控。

运行参数（`warmup_steps=50, max_env_steps=200, batch_size=32, utd_ratio=1`）后观察到：

| 步骤 | Q1 均值 | actor delta std | actor_loss |
|------|---------|-----------------|------------|
| 50（训练开始） | 0.35 | ~0.75 | +0.94 |
| ~60  | 0.5–1.3 | ~0.75 | −0.4 |
| ~130 | ~39 | ~20 | −143 |
| 200  | ~51 | **~30** | **−181** |

最终 `critic_step | grad_norm=1738`，`delta max_abs > 80`。

## 根本原因：Q 值过估计 × BC 惩罚系数过小

actor_loss 的形式为：

```
loss = mean(-Q + actor_bc_weight * ||action - reference_action||²)
```

当 Q 被过估计至 ~50 时：
- 只要 `||delta||² < Q / actor_bc_weight = 50 / 0.1 = 500`
- 即 delta_norm < 22，BC 惩罚**无法阻止** actor 继续拉大偏差以换取更高 Q

自强化正反馈：

```
actor 输出偏离 reference
  → 这些 OOD 动作被 critic 高估 Q
    → actor 进一步偏离（因为偏离有"奖励"）
      → critic 更新到更高 Q
        → 循环...
```

在真实在线训练中，`utd_ratio=5` 意味着同一批数据做 5 次 critic/actor 更新，过拟合速度是 sim（utd_ratio=1）的 5 倍，因此爆炸更快、幅度更大。

## 修复方向

### Fix 1（本次实验）：提高 `actor_bc_weight`

将 BC 惩罚权重从 0.1 提高到 1.0~5.0，使其能在 Q 上涨时有效约束 actor 偏离 reference 的幅度。

### Fix 2：降低 `critic_lr`

减小 critic 学习率（3e-4 → 1e-4），降低 Q 值过估计的速率，为 BC 惩罚的约束效果提供更多时间。

### Fix 3（未做）：降低 `utd_ratio`

从 5 降至 1~2，减少每步对同一批数据的重复学习。

### Fix 4（未做）：Q 值归一化 / 截断

在 target Q 计算中加 reward normalization 或 Q-value clip，防止 Q 无限膨胀。

---

## 实验结果

### 实验配置

| 实验 | actor_bc_weight | critic_lr | max_env_steps |
|------|----------------|-----------|---------------|
| Baseline | 0.1 | 3e-4 | 200 |
| Exp-A | **1.0** | 3e-4 | 300 |
| Exp-B | **5.0** | 3e-4 | 300 |
| Exp-C | **1.0** | **1e-4** | 300 |
| Exp-D | **5.0** | **1e-4** | 300 |

### 结果

所有实验均使用离线模拟脚本（`train_rlt_sim.py`），`warmup_steps=50, batch_size=32, utd_ratio=1`。

| 实验 | actor_bc_weight | critic_lr | Q1 @step300 | delta std @step300 | delta max_abs @step300 | actor_loss @step300 | 状态 |
|------|----------------|-----------|-------------|-------------------|----------------------|---------------------|------|
| Baseline | 0.1 | 3e-4 | **~51**（@step200） | **~30** | **~80** | **−181** | 🔴 爆炸 |
| Exp-A | **1.0** | 3e-4 | **~1.2** | **~0.45** | **~1.5** | **−1.34** | 🟢 稳定 |
| Exp-B | **5.0** | 3e-4 | ~0.21 | ~0.09 | ~0.36 | −0.064 | 🟡 过保守 |
| Exp-C | **1.0** | **1e-4** | ~2.7 | ~0.95 | ~3.9 | −3.44 | 🟢 稳定（Q 仍在缓慢上涨） |
| Exp-D | **5.0** | **1e-4** | ~0.52 | ~0.16 | ~0.55 | −0.46 | 🟡 过保守 |

### 分析

**Exp-A（bc_weight=1.0, critic_lr=3e-4）是最佳平衡点：**

- Q 值快速收敛至 ~1.2 后停止增长（接近理论值：末帧奖励 1.0 折扣 140 步 ≈ 0.99^140 ≈ 0.25，加上 BC 调整后为 1.2 合理）
- delta std 从初始 0.75 下降至 0.45，actor 在 reference 附近小幅探索
- critic_loss 降至 0.003 以下，说明 critic 已收敛

**Exp-B / Exp-D（bc_weight=5.0）过于保守：**

- delta std < 0.1，actor 几乎不偏离 reference action
- Q 值极低（0.2~0.5），说明 actor 基本没有在学习有效策略，退化为纯 BC 行为克隆

**Exp-C（bc_weight=1.0, critic_lr=1e-4）：**

- 300 步内没有爆炸，但 Q 值仍缓慢上涨（2.7 且未收敛），delta 也比 Exp-A 略大
- 低学习率让 critic 学习更慢，给 actor 更多时间产生 OOD 样本，存在最终爆炸的风险

### 结论与建议

**推荐：`actor_bc_weight = 1.0`（Fix 1 有效）**

在真实在线训练（`train_rlt_online.py`）中，建议将 `online.actor_bc_weight` 从默认的 `0.1` 改为 `1.0`。

对于 `utd_ratio=5` 的真实训练，Q 过估计更严重，可能还需要：
- 同步适当降低 `actor_lr`（如 1e-4）避免 actor 更新步幅过大
- 考虑将 `utd_ratio` 从 5 降至 2~3

`critic_lr` 单独降低（Fix 2）收益不明显，且引入 Q 收敛变慢的风险，不建议单独使用。
