# AWM-CoCA 训练说明

本目录实现 GE-Sim 单 chunk 视频世界模型的 fresh on-policy LoRA 训练。
正式训练默认使用 Action 与三视角 PSNR geometry 各占 50% 的联合奖励。

`configs/awm_coca_train.yaml` 中的相对路径统一从仓库根目录解析，因此仓库迁移到
其他机器后不需要修改代码中的绝对路径。同时支持 `${ENV_VAR}` 环境变量展开和命令行
路径覆盖。

## 快速开始

请先进入包含项目依赖的 Python 环境。启动脚本不绑定具体 Conda 安装目录，可通过
`PYTHON_BIN` 指定解释器。

```bash
# 只检查数据、三视角 GT、模型配置和 checkpoint，不加载大模型。
CUDA_VISIBLE_DEVICES=0 scripts/run_awm_coca.sh preflight

# 单卡验证：一个 condition、group=2、一个 optimizer step，不保存 checkpoint。
# 输出到带时间戳的 outputs/awm_coca_smoke/<time>/。
CUDA_VISIBLE_DEVICES=0 scripts/run_awm_coca.sh smoke

# 单卡正式训练；配置默认全局 group=16、1000 个 optimizer step。
CUDA_VISIBLE_DEVICES=0 scripts/run_awm_coca.sh train

# 单机四卡训练；全局 group=16，每张 GPU 负责 4 条 rollout。
CUDA_VISIBLE_DEVICES=0,1,2,3 scripts/run_awm_coca.sh train4
```

迁移机器时建议使用显式路径覆盖：

```bash
scripts/run_awm_coca.sh preflight \
  --prep-root /data/awm/prep \
  --gt-root /data/awm/selected_samples/samples \
  --checkpoint-root /models/genie-envisioner \
  --output-dir /output/awm_coca
```

这些参数同样适用于 `smoke`、`train` 和 `train4`。其他常用参数包括：

- `--print-effective-config`：打印路径解析和覆盖后的完整配置，然后退出。
- `--group-size`：覆盖全局 group 大小。
- `--max-optimizer-steps`：设置 optimizer step 总数。
- `--dataset-limit`：只使用前 N 个合法 condition。
- `--num-workers`：每个 rank 的 DataLoader worker 数。
- `--reward-workers`：每个 rank 的 reward 线程数。
- `--checkpoint-every`：checkpoint 间隔。
- `--rollout-retention videos`：仅保留视频、reward、credit 和元数据（正式训练默认）。
- `--rollout-retention all`：保留视频和全部中间张量，仅用于调试。
- `--rollout-retention none`：指标写入 JSONL 后删除整个 group。

`--checkpoint-root` 是统一模型资产根目录，至少需要包含：

```text
<checkpoint-root>/
  Cosmos-Predict2-2B-Video2World/
  gesim/ge_sim_cosmos_v0.1.safetensors
  sam3.pt
  cowtracker/cowtracker_model.pth
  yoloworld-EWMBench-v0.1.pt
```

`preflight` 会在加载模型前检查这些路径、预处理样本和全部奖励 GT 视频。

面向 4×A100 训练机的环境创建、固定 revision 模型下载、一键训练和结果回传流程见
仓库根目录的 [`TRAINING_REMOTE_README.md`](../../TRAINING_REMOTE_README.md)。

## 数据目录结构

预处理 condition 的结构：

```text
output/prep/<condition_id>/
  actions.npy                         # [29, 16]
  extrinsic_{head,hand_left,hand_right}.npy
  intrinsic_{head,hand_left,hand_right}.npy
  {head,hand_left,hand_right}_color/{0,1,2,3}.png
```

默认 GT 视频结构：

```text
data/agibotworld_beta/selected_samples/samples/<condition_id>/
  head_29_frames.mp4
  hand_left_29_frames.mp4
  hand_right_29_frames.mp4
```

联合奖励要求三个视角的 GT 都存在；缺少任意一个视角时，`preflight` 会直接报错，
而不是在训练过程中静默降低奖励维度。

## Rollout 和训练流程

每个 condition 包含 29 步双臂动作、三个相机的内外参和 4 帧历史图像。运行时执行：

1. 将动作通过三个相机投影成 trajectory map。
2. 计算每个相机的六通道 ray map。
3. 将 trajectory map、ray map、4 帧历史和文本条件送入 GE-Sim。
4. 使用同一个 LoRA policy version 生成一个 group 的独立随机 seed。
5. 每条 rollout 生成 25 帧未来视频，并记录 15 步 Flow Matching predicted-x0 与 post-step latent。
6. 计算 Action、PSNR geometry 和 50:50 joint reward。
7. 根据 reverse trajectory 计算 CoCA credit 和噪声 proposal。
8. 使用全局 group reward 计算 leave-one-out advantage，逐条 rollout 反传 LoRA。

四卡训练时，每个 rank 在自己的 GPU 上独立计算本地 rollout 的 SAM3/CoWTracker/YOLO
奖励。SAM3 会被显式设为 rank-local 单进程推理（`world_size=1`），不会复用训练进程组
执行其内部 `all_gather`；奖励完成后才进入训练侧的跨 rank gather/all-reduce。
所有 rank gather 完有效 seed 数后统一决定训练或跳过：只要任一 rank 的本地有效 seed 为
0，即使全局有效数已经达到阈值，也会四卡同步跳过该 task，避免空 rank 与其他 rank 的
反向传播路径不一致而发生 NCCL 超时。

Loader 会拒绝以下数据：

- policy version 已过期；
- group ID 已经训练过；
- 单 rank 少于两条 rollout；
- condition ID 或 policy version 不一致；
- trajectory 不是单 chunk；
- reward 无效或必要 latent 缺失。

Group loss 逐条反传，因此 `group=16` 不会同时保留 16 个完整 transformer 计算图。
1.97B transformer 开启 gradient checkpointing，只更新 attention LoRA 参数。

## 默认联合奖励

当前训练配置为：

```text
reward.mode = joint
geometry_enabled = true
action_weight = 0.5
geometry_weight = 0.5

R_total = 0.5 * R_action + 0.5 * R_geometry
```

`combine_rewards` 会按权重和归一化；当前配置显式写成 `0.5:0.5`。Action 或 geometry
任意一项无效时，joint reward 整体无效，训练会报错停止。

### Action reward

Action reward 只使用 head 视角完成动作跟随评价，但会对左右臂分别计算指标，然后取
两臂均值。GT 和生成视频按相同的前 29 个 frame index 对齐。

三个分量默认各占三分之一：

1. `mean_iou`

   SAM3 使用 `robot arm` 文本提示分别跟踪 GT 和生成视频中的机械臂前景。排除第 0 帧
   后计算逐帧 IoU 均值：

   ```text
   R_iou = clip(mean_iou, 0, 1)
   ```

2. `af_fdce_ate_norm`

   将 `actions.npy` 和 head 相机标定投影成二维命令轨迹，再将 CoWTracker 得到的生成
   视频前景质心与命令轨迹比较。ATE 按画面对角线归一化，越小越好：

   ```text
   R_command = 1 - clip(af_fdce_ate_norm / 0.2, 0, 1)
   ```

3. `fdce`

   Foreground displacement Chamfer error 比较生成视频与 GT 的前景运动，越小越好：

   ```text
   R_fdce = 1 - clip(fdce / 10.0, 0, 1)
   ```

最终：

```text
R_action = (R_iou + R_command + R_fdce) / 3
```

如果某个具有非零权重的 Action 指标无法计算，代码不会忽略该指标后重新归一化，而是
将 Action reward 标记为无效。`configs/awm_coca_eval.yaml` 中另有一套实验性
`0.2/0.1/0.7` 权重，训练不会自动采用它。

### 三视角 PSNR geometry reward

Geometry reward 使用 head、hand-left、hand-right 三个视角的 RGB PSNR。只评价未来
帧 4～28，共 25 帧；第 0～3 帧是历史条件，不参与 geometry reward。

每个视角先对 25 个 frame PSNR 求均值，得到：

```text
PSNR_head, PSNR_hand_left, PSNR_hand_right
```

然后计算三视角平均值和最差视角：

```text
mean_psnr  = mean(三个视角 PSNR)
worst_psnr = min(三个视角 PSNR)
balanced_psnr = 0.6 * mean_psnr + 0.4 * worst_psnr
```

引入最差视角可以防止模型只优化容易的 head 视角。最后将 dB 映射到 `[0,1]`：

```text
R_geometry = sigmoid((balanced_psnr - 20.4) / 1.8)
```

相关参数：

- `geometry_cameras`：默认三个视角。
- `geometry_future_start/end`：默认 4 和 28，包含首尾。
- `geometry_mean_weight/worst_weight`：默认 0.6/0.4，内部自动归一化。
- `geometry_psnr_center_db`：reward 等于 0.5 时对应的 PSNR，默认 20.4 dB。
- `geometry_psnr_temperature_db`：sigmoid 斜率，越小越敏感，默认 1.8 dB。

## CoCA credit 分配

一条 rollout 有 15 个 Flow Matching clean-latent prediction，并有最终 post-step latent：

```text
x0_hat_1, x0_hat_2, ..., x0_hat_15, z_final
```

其中 pipeline 在每步按 `x0_hat = c_skip * x_t + c_out * velocity` 得到 predicted-x0。
Credit 使用 predicted-x0 与最终 latent 的余弦相似度，不再使用仍带噪的中间 `x_t`：

```text
s_t  = cosine(x0_hat_t, z_final),  t=1..15
s_16 = cosine(z_final, z_final) = 1
```

这个末尾的 1 是最后一次反向更新的明确终点。默认 `coca_window_size=3`，15 个相邻转移
被分成 5 个连续窗口。每个窗口的贡献是当前窗口平均
相似度相对于上一个参考相似度的变化。窗口内各 reverse step 获得相同原始贡献，随后
将全部 15 个 step weight 归一化到和为 1。

诊断用的 step reward 为：

```text
step_reward_t = total_reward * step_weight_t
```

其和严格等于 total reward。Step weight 可能为负数，因为 reverse trajectory 的相似度
不保证单调；如果总贡献接近 0，则回退成均匀权重。

15 个 reverse step 再按顺序映射到 7 个训练噪声档位。每个档位累计原始 step
contribution 得到 `noise_score_k`：

```text
q_coca(k) = softmax(noise_score_k / temperature)
q(k)      = (1 - eta) * p(k) + eta * q_coca(k)
```

其中：

- `p(k)` 是基础 proposal，当前为 7 档均匀分布；
- `eta=0.9`，表示训练采样主要跟随 CoCA；
- `temperature=1.0`；
- 每条 rollout 从 `q(k)` 抽一个训练噪声级别；
- importance weight 使用 `p(k)/q(k)`。

Reward 不直接乘到 `noise_score` 上。CoCA credit 决定“更应该在哪些噪声阶段训练”，
reward 数值则通过 group advantage 决定更新方向和强度。

## Group credit 和训练目标

全局 group 使用 leave-one-out baseline：

```text
A_i = R_i - mean(R_j for j != i)
```

`group=2` 时两个 advantage 大小相同、符号相反；`group=16` 能提供方差更低的 baseline。

每条 rollout 的 loss：

```text
L_i = p(k)/q(k) * [
    A_i * MSE(v_policy, noise - clean)
    + beta * MSE(v_policy, v_base)
]
```

`v_base` 通过临时关闭 LoRA adapter 得到，默认 `beta=0.01`。正 advantage 让 policy
更贴近较好的 rollout，负 advantage 将 policy 推离较差 rollout。

训练使用 AdamW、梯度裁剪、学习率 warmup、EMA、policy-version 检查，以及可恢复的
sampler 和随机数状态。`fresh_on_policy=true` 被强制执行：每个 group 只消费一次，并且
必须来自当前 policy version。EMA 只更新和保存，rollout 与训练仍使用在线 policy。

## 需要重点调节的参数

- `rollout.group_size`：正式训练默认全局 16。单卡最小 2；四卡每个 rank 至少 2，
  因而四卡最小为 8。它影响 baseline 方差和 rollout 成本，不是 transformer batch size。
- `rollout.reverse_denoise_steps`：默认 15，必须与 GE-Sim 配置一致，不能单独修改。
- `proposal.coca_window_size`：越大越平滑，但 reverse-step 定位会变粗。
- `proposal.credit_source`：正式训练固定为 `predicted_x0`；`raw_state` 只用于旧版消融。
- `proposal.num_training_noise_levels`：噪声档位数；必须与基础概率列表长度一致。
- `proposal.eta`：越高越依赖 CoCA；不稳定时可先从 0.9 降到 0.5～0.7。
- `proposal.temperature`：越低 proposal 越尖锐，越高越接近均匀。
- `proposal.importance_clipping`：限制 `p/q`，可减小极端方差但会引入偏差；默认关闭。
- `optimizer.learning_rate`：默认 `1e-6`。
- `optimizer.warmup_steps`：默认 100，控制训练初期步长。
- `optimizer.reference_kl_beta`：默认 0.01；生成质量漂移过快时可提高。
- `optimizer.max_grad_norm`：默认 1.0。
- `optimizer.gradient_accumulation_steps`：累计完整全局 group 的数量，调大后计算代价很高。
- `model.lora_rank/lora_alpha`：默认 32/64；修改后旧 LoRA checkpoint 不再兼容。
- `reward.action_metric_weights`：定义 Action 内部三项比例。
- `reward.action_weight/geometry_weight`：定义 Action 与 geometry 的比例，当前均为 0.5。
- `geometry_psnr_center_db/temperature_db`：必须根据真实 PSNR 分布校准，决定 geometry
  reward 是否饱和。
- `reward.workers`：建议保持每卡 1。SAM3 和 CoWTracker 是 GPU singleton，不适合线程并发。
- `DIST_TIMEOUT_SECONDS`：不同 rank 的 rollout/reward 耗时可能相差较大；远端脚本默认
  7200 秒，避免先完成的 rank 在汇总点被 PyTorch 默认 10 分钟 NCCL 超时误杀。
- `dataset.num_workers`：每个 rank 的 worker 数；四卡默认合计 16 个。
- `max_optimizer_steps`、checkpoint 间隔和 rollout 保留策略只控制运行长度、恢复粒度和磁盘。

## 四卡执行

`train4` 使用单机 `torchrun`/NCCL，每张 GPU 一个进程。默认全局 group=16，因此每卡
顺序生成和训练 4 条 rollout。

四卡逻辑包括：

- 使用相同 condition 和 policy version，按 rank 分配不重复 seed；
- 汇总全局 16 条 reward 后计算 leave-one-out advantage；
- optimizer step 前对 LoRA 梯度求和并除以 world size；
- 初始化或 resume 后由 rank 0 广播 LoRA 参数；
- 只由 rank 0 写总指标和 checkpoint；
- 每个 rank 独立写入和清理临时 rollout group。

不要启动四个独立的 `train` 命令，否则会得到四个彼此分叉的 policy。

目标四卡机器首次建议执行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 scripts/run_awm_coca.sh train4 \
  --max-optimizer-steps 1 \
  --dataset-limit 1 \
  --rollout-retention videos
```

当前开发机只有一张 A100，无法做真实四卡 NCCL 压测；分布式 reward 聚合、指标聚合和
梯度同步已经使用独立双进程测试验证。

## 磁盘和 checkpoint

同一个本地 group 的 condition 只保存一份，不再为每个 seed 重复保存。正式训练会先把
完整 reward 和精简 credit 写入 `metrics/rollouts.jsonl`，完成梯度更新后保留 MP4、
`reward.json`、`credit.json` 和元数据，同时删除 `condition.pt`、`trajectory.pt` 和
`final_future_latent.pt`。

Smoke 会自动保留原始 rollout。正式训练默认设置：

```yaml
storage:
  rollout_retention: videos
```

临时调试可传入 `--rollout-retention all`；长期保留 group=16 的全部 latent 可能占用 TB
级磁盘，不建议在正式长跑中使用。

## 已验证项目

- 当前单张 A100 80GB 上，真实 `group=2` 已完成 rollout、Action reward、CoCA credit、
  LoRA 反传和一个 optimizer step，峰值显存约 39.7GB。
- 三视角 PSNR 已用真实生成视频检查，三个视角均读取未来 25 帧。
- 50:50 joint reward 与 `0.5 * (R_action + R_geometry)` 数值完全一致。
- `preflight` 已检查当前 203 个 condition 及其三视角 GT。
- Python 编译、Shell 语法、Git whitespace 检查通过。
