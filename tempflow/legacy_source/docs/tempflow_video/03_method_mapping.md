# TempFlow-GRPO 到 GE-Sim 视频的映射

## 依据

- TempFlow-GRPO paper：<https://arxiv.org/abs/2508.04324>
- 官方实现：<https://github.com/Shredded-Pork/TempFlow-GRPO>，审计 commit `63e4def7159940ba7d60e4e6250eee868342388c`

不整段复制图像 pipeline；新代码只复用公式、group 语义和 PPO/KL 目标，再适配现有 GE-Sim latent/scheduler/condition。

## 坐标与 transition

论文 RF reverse SDE（`dt=next_t-t<0`）为：

```text
x_next = x_t + [v_theta + sigma_t^2/(2t)*(x_t+(1-t)v_theta)]*dt
         + sigma_t*sqrt(-dt)*epsilon
sigma_t = eta*sqrt(t/(1-t))
```

GE-Sim 保存 `y_t=x_t/(1-t)=x0+s*epsilon`，transformer 仍输出 `v=epsilon-x0`。实现先用 `x_t=(1-t)y_t` 计算论文 Gaussian mean，再把 mean 与 std 同除 `1-next_t` 映射回 EDM coordinate。坐标 Jacobian 不依赖 policy，因此 PPO new/old ratio 不变。

采样时保存实际 `y_next` 与 old log-probability；训练时用当前 policy 对同一个 collected transition 重新计算 Gaussian log-probability。目标是真正的 PPO-clipped GRPO transition objective，加同方差 Gaussian 的 closed-form reference KL；没有把 flow-matching MSE 改名为 policy loss。

## Branch group

```text
same condition/action/prompt + same initial noise seed
                         |
                  deterministic ODE prefix
                         |
             identical latent at generation step k
                  /      |       \
          independent SDE branch noise
                  \      |       /
                  deterministic ODE suffix
                         |
              complete 29-frame, 3-view video
                         |
                 unchanged terminal reward
```

Group key 包含 condition ID、prompt ID、initial seed、branch timestep、video length、reward-config hash 和 policy version。只有 branch ID/noise seed 不同。训练拒绝混组、旧 policy rollout 和二次消费同一 group。

## Advantage、loss 与 KL

同一组使用 population standard deviation：

```text
A_i = (R_i - mean(R)) / (std(R) + epsilon)
```

近零方差组得到全零 advantage、记录并跳过 update；绝不向 reward 加噪声。

对 collected transition：

```text
ratio = exp(logp_policy - logp_old)
L_policy = -mean(min(ratio*A, clip(ratio,1-eps,1+eps)*A) * w_noise)
KL = mean((mu_policy-mu_reference)^2 / (2*transition_variance))
L_total = L_policy + beta*KL
```

初始化 LoRA 对输出为零，故 policy/reference mean 相同且 KL≈0；更新后 policy 可变而禁用 adapter 的 reference base 必须不变。

## Noise-aware weighting

论文写作 `Norm(sigma_t*sqrt(Delta t))`，但未定义 `Norm`。官方代码针对 SD3/Flux/Qwen 使用模型常数（分别 2.25/1.73/1.53）乘 transition std。视频 scheduler 没有可信常数，因此采用可审计的 schedule normalization：只在 14 个非零、可分叉 transition 上除以均值，保留精确相对比例且均值为 1。terminal duplicate no-op 的权重为 0 且不能分叉。

## 模式语义

- `base_eval`：只评估初始 policy。
- `tempflow_branch_only`：branching 开，noise-aware 权重关。
- `tempflow_full`：branching、noise-aware weighting、reference KL 全开。
- `video_grpo`：ordinary rollout；各 group member 使用不同 initial seed，在每个合法 reverse step 执行 SDE action，不使用 noise-aware weighting。
- `tempflow_noise_weight_only`：同一 all-SDE-steps ordinary collector，但启用 schedule-derived noise weights。

ordinary rollout 会把同一个 terminal video reward/advantage赋给该路径的全部 collected transitions，再对每条路径内 transition 求均值；不会用 deterministic rollout 冒充 GRPO。单卡只要求代码/config 支持，尚未将这两个对照模式做长运行。

## 内存策略

rollout 在 inference mode 下保存 CPU latent/condition 元数据。update 对每条 branch 重新 forward 并立即 backward，不跨整个 group 保留计算图。optimizer step 后 policy version 加一；后续 rollout 必须重新采样。checkpoint 保存 LoRA、optimizer、scheduler、Python/NumPy/Torch/CUDA RNG 与 consumed group IDs。
