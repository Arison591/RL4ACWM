# AWM × TempFlow 官方源码对齐说明（2026-08-17）

## 结论

这次对齐的是 TempFlow 的核心训练数据流，不是把 SD3/FLUX 专用常数机械搬到 AWM：

```text
冻结 old policy
-> 对一个 condition 只生成一次 deterministic ODE base trajectory
-> 在同一 old policy 下收集全部合法分叉时间步 k
-> 每个 k 从同一个 latent 采 6 个 SDE 分支并走 deterministic suffix
-> 各 k 独立计算 6 分支 group-relative advantage
-> 把所有 k 放进 rollout buffer
-> 打乱为 2 个 optimizer minibatch
-> 用保存的 old_log_prob 做 PPO clipped update
-> 丢弃整个 buffer，再用新 policy 重新采样
```

原实现是“采一个 k，立刻更新一次，再采下一个 k”。因此每个 group 更新时 current policy 仍等于 old policy，ratio 始终为 1，clip fraction 始终为 0。新实现中，第一个 minibatch 的 ratio 仍应为 1；它更新参数后，第二个 minibatch 仍使用同一冻结 old policy 保存的 log-prob，因此 ratio 可以偏离 1，PPO clipping 才真正有机会生效。

## 1. 采样噪声到底来自 AWM 还是 TempFlow

采样噪声强度来自 TempFlow，不使用旧 AWM/CoCA 的 P、Q、proposal probability 或 importance weight。

TempFlow 在 Rectified-Flow 坐标中的探索强度为：

```text
exploration_scale(t) = eta * sqrt(t / (1 - t))

rf_step_noise_std(t -> next_t)
    = exploration_scale(t) * sqrt(t - next_t)
```

当前 eta = 0.7，与官方 SD3.5-M 源码采用的数值一致。每个分支采样：

```text
epsilon_i ~ Normal(0, I)

x_next_i = mu_theta(x_t, t) + rf_step_noise_std * epsilon_i
```

AWM/GE-Sim 的不同之处只在 latent 坐标。它保存 EDM 坐标：

```text
y_t = x_t / (1 - t)
```

所以代码先按 TempFlow 在 RF 坐标构造 Gaussian mean/std，再映射回 AWM 的 EDM 坐标：

```text
edm_mean = rf_mean / (1 - next_t)
edm_std  = rf_step_noise_std / (1 - next_t)
```

这不是换一种探索噪声，只是同一个 TempFlow Gaussian transition 的坐标变换。该 Jacobian 与 policy 参数无关，所以 new/old PPO ratio 不变。

需要区分：

- `transition_std` 决定 rollout 实际采多少噪声，严格采用 TempFlow SDE 公式；
- `noise_weight` 是 policy loss 对不同时间步的乘法权重，不参与采样。

官方 SD3 源码使用 `2.25 * transition_std` 作为 loss 权重，其中 2.25 是模型/调度器专用归一化常数；FLUX、QwenImage 使用的常数也不同。AWM 没有官方给定常数，因此当前保留：

```text
noise_weight[k]
    = rf_step_noise_std[k]
    / mean(rf_step_noise_std over legal AWM steps)
```

它与官方使用相同的相对噪声尺度，均值归一化只适配 AWM scheduler。把 SD3 的 2.25 生搬过来并不算更忠实。

## 2. 时间节点怎么选

官方训练代码使用：

```text
num_train_timesteps = int(num_steps * timestep_fraction)
train_timesteps = [0, ..., num_train_timesteps - 1]
```

官方常用配置中 `num_steps = 10`、`timestep_fraction = 0.99`，因此训练前 9 个节点；per-step sampler 本身也不在最后一个 scheduler timestep 上分叉。

AWM 当前是 15 个反向 transition，正式配置改为：

```text
num_steps = 15
timestep_fraction = 0.99
num_train_timesteps = int(15 * 0.99) = 14
selected k = 0, 1, ..., 13
```

此外，GE-Sim scheduler 可能附带重复 terminal sigma。任何满足 `next_t == t` 的零长度 transition 都会被显式拒绝，不能当作 SDE action。

因此节点集合与此前仍然相同，都是 0–13；真正改变的是选择和收集的时序：

- 原来：硬编码 0–13，每采一个 k 就更新 policy，然后轮转到下一个 k；
- 现在：按官方 `timestep_fraction` 从实际 scheduler 自动解析 0–13，并在同一 old policy 下全部采完，再训练。

## 3. 具体代码改动

### Rollout sampler

`legacy_source/experiments/tempflow_video/sampler.py`

- 新增 `resolve_branch_timesteps`，实现官方 prefix fraction 语义并拒绝零长度节点；
- 新增可复用的 `sample_base`；
- `sample_group` 可接收已经生成的 base artifact；
- 同一 condition 的 14 个 k 不再各自重跑一遍 15-step base trajectory。

### Trainer

`legacy_source/experiments/tempflow_video/trainer.py`

- 新增 `update_rollout_buffer`；
- 在任何参数更新前验证整批数据来自同一个 collection policy version；
- 每个 k 的六个 sibling branch 保持在一起，不拆散其 advantage group；
- 多个 k 合并进 optimizer minibatch；
- 后续 minibatch 使用采样时保存的 old log-prob，因此 ratio/clip 可生效；
- buffer 一旦开始更新就整体标记 consumed，防止 stale rollout 被误用；
- checkpoint 状态新增 `rollout_epoch`。

### Training loop

`legacy_source/experiments/tempflow_video/run.py`

- 从“sample one k -> update”改为“sample all k -> buffered updates”；
- 一个 rollout epoch 对应一个 condition 和一个 frozen old policy；
- 14 个 timestep group 共用一次 base trajectory；
- reward/advantage 仍按每个 k 的六分支独立计算；
- 新增 `rollout_groups.jsonl` 和 `optimizer_steps.jsonl`，分别记录采样组与 optimizer ratio/clip 指标。

### Formal config

`configs/psnr_only_overfit224.yaml`

```text
timestep_fraction                         0.99
collection mode                           frozen_policy_timestep_buffer
14 timestep groups / rollout epoch
optimizer minibatches / rollout epoch     2
inner epochs                              1
16 conditions x 14 groups                224 branch groups
16 rollout epochs x 2 updates             32 optimizer steps
advantage clip                            [-5, 5]
learning rate                             1e-5
reference KL beta                         0.5
```

文件名中的 224 现在明确表示 16 conditions × 14 timestep groups，而不是 224 次互相使用不同 policy 的即时更新。

## 4. 弥补了哪些原差距

已经弥补：

1. old policy 现在在完整 rollout collection phase 内真正冻结；
2. PPO ratio 不再被结构性锁死为 1，clipping 可以在后续 minibatch 生效；
3. 所有合法 k 在同一 policy snapshot 下采集，时间 credit 不再混入 k 之间的 policy drift；
4. base ODE trajectory 在同一 condition 内复用，减少重复推理；
5. 时间节点从硬编码改为官方 fraction 语义，并绑定实际 AWM scheduler；
6. advantage clip 从原来的 1 对齐到官方默认 5。

仍然保留的任务适配/差异：

1. backbone 是 AWM/Cosmos 视频模型，不是 SD3/FLUX 图像模型；
2. reward 是三视角 future-only raw PSNR，不是 PickScore/HPS/OCR；
3. 当前仍是一个固定 initial seed × 6 个分支；官方典型大配置会跨多个 initial seed 形成更大 prompt group；
4. suffix 分支当前仍逐个执行，尚未像官方一样完全 batch-vectorize；
5. loss noise weight 使用 AWM schedule mean normalization，没有冒用 SD3 专用 2.25；
6. reference KL beta = 0.5 是当前实验指定值，明显强于官方不同任务常用的 0.001/0.004；
7. 尚未加入官方可选 EMA。

因此现在可以称为：

> AWM 上按官方 TempFlow 两阶段 rollout-buffer/PPO 数据流实现的单步分叉 GRPO，并带有必要的 EDM 坐标与视频 reward 适配。

还不应称为 SD3 官方配置的逐项复刻。

## 5. 验证结果

新增测试验证：

```text
minibatch 1: ratio_mean = 1, clip_fraction = 0
optimizer changes policy
minibatch 2: ratio_mean != 1, clip_fraction > 0
```

自动化测试：

```text
legacy_source/tests    24 passed
tests                  16 passed, 2 skipped
python syntax check    passed
git diff --check       passed
```

这里验证的是算法数据流和数值行为。尚未启动新的正式 GPU 训练，因此还没有新版 32-step run 的 PSNR 曲线，也不能提前宣称效果提升。

## 6. 官方源码依据

- TempFlow per-step sampler：<https://github.com/Shredded-Pork/TempFlow-GRPO/blob/main/flow_grpo/diffusers_patch/sd3_pipeline_with_logprob_perstep.py>
- TempFlow SDE transition：<https://github.com/Shredded-Pork/TempFlow-GRPO/blob/main/flow_grpo/diffusers_patch/sd3_sde_with_logprob_perstep.py>
- TempFlow training loop：<https://github.com/Shredded-Pork/TempFlow-GRPO/blob/main/scripts/train_sd3_pr.py>
- TempFlow released configs：<https://github.com/Shredded-Pork/TempFlow-GRPO/blob/main/config/dgx.py>
