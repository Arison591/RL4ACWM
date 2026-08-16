# AWM × TempFlow：面向动作条件视频世界模型的单步分叉强化学习

> 2026-08-17 更新：训练数据流已经改为 frozen-old-policy rollout buffer，PPO clipping 不再被结构性锁死。完整变更见 `11_tempflow_source_alignment_zh.md`。下文第 13 节的 80-step 数字是旧实现的历史结果，不能代表新版。

## 1. TL;DR

我们没有更换视频世界模型 backbone，仍使用原来的 action-conditioned AWM（GE-Sim/Cosmos）。变化发生在 AWM 外面的强化学习方式：

- 上一版 CoCA 根据去噪过程中 predicted-x̂₀ 与最终 latent 的相似度，为训练噪声时间分配 proposal，再做 reward-weighted Flow-Matching MSE。
- 新版 TempFlow 在一条 AWM 去噪轨迹的某个时间步复制相同 latent，注入不同 SDE 噪声得到多个分支；用分支最终视频的 reward 构造组内 advantage，并对这个真实发生的 latent transition 做 policy-gradient/GRPO 更新。
- 不使用 critic。当前先用冻结 old policy 收集同一 condition 的全部合法时间步，再通过两个 minibatch 做 PPO clipped update；第一个 minibatch 的 ratio 为 1，第二个开始 clipping 可以实际生效。
- 当前正式实验还同时把 reward 改成了三视角 future-only raw PSNR，因此现阶段不能把所有变化都归因给 TempFlow；需要补同 reward 的 CoCA 对照。

## 2. 我们解决的问题

AWM 根据四帧历史图像、未来机器人动作、三相机标定和文本条件，预测三个视角的未来 25 帧视频：

```text
4 帧历史 + 25 步 action + camera geometry
                    |
                    v
          AWM / GE-Sim / Cosmos
                    |
                    v
    head / hand-left / hand-right 未来视频
```

模型通过 15 步 Flow-Matching 反向过程，从噪声 latent 生成最终视频 latent。监督微调只能拟合数据分布，但我们还希望它生成的未来视频在任务指标上更好，例如：

- 三视角场景和机器人状态更接近真实未来；
- 机械臂运动更符合给定 action；
- 多视角之间不出现明显退化。

所以问题可以写成：如何利用不可微的终局视频 reward，稳定地更新这个高维连续生成策略？

## 3. 哪些东西没换

以下部分仍然沿用原来的 AWM：

- GE-Sim/Cosmos transformer backbone；
- VAE、text encoder 和 Flow-Matching scheduler；
- 4 帧历史、25 帧未来、三视角视频定义；
- action trajectory map 和 camera ray map 条件；
- 原始预训练权重；
- attention LoRA 微调方式，base model 冻结。

因此这不是换 backbone，而是换了 **rollout、credit assignment 和 policy objective**。

## 4. 上一版 CoCA 在做什么

上一版对同一 condition 生成多条独立完整 rollout。对每条 rollout，记录 15 个去噪步的 predicted clean latent：

~~~text
x0_hat[1], x0_hat[2], ..., x0_hat[15]
~~~

再计算每一步与最终 latent z_final 的余弦相似度：

~~~text
s[t] = cosine(x0_hat[t], z_final)
~~~

相似度沿时间的变化经过窗口聚合后，形成不同噪声档位的 CoCA score。随后构造：

~~~text
q_coca(k) = softmax(score[k] / T)

q(k) = (1 - eta) * p(k) + eta * q_coca(k)
~~~

从 q(k) 抽取一个训练噪声档位，再用 p(k) / q(k) 做 importance correction。最终目标是：

~~~text
L_coca = p(k) / q(k) * [
    A[i] * MSE(v_policy, noise - x[i])
    + beta * MSE(v_policy, v_base)
]
~~~

这里 x[i] 是 rollout 最终 latent。它本质上是 reward-weighted self-training：reward 决定向哪个样本靠近，CoCA 决定在哪个噪声阶段训练。

它的局限是：多个 rollout 从初始噪声开始就彼此不同，最终 reward 差异混合了整条 15 步路径的随机性；predicted-x̂₀ 相似度是启发式 credit，不是某个真实随机 action 的回报。

## 5. 新版 TempFlow 的核心做法

### 5.1 把去噪过程看成策略轨迹

在选定的时间步 k：

- 状态：当前 noisy latent y_t；
- 动作：采样得到下一个 latent y_next；
- 策略：由 AWM velocity 决定的 Gaussian transition；
- 回报：该分支最终解码视频的 terminal reward。

### 5.2 共享前缀、单步分叉

同一个 group 使用相同 condition、initial seed、policy version 和 branch timestep：

```text
同一 condition + 同一 initial seed
                |
                v
        deterministic ODE prefix
                |
                v
          完全相同的 y_tk
       /        |        \
  SDE noise 1  ...   SDE noise 6
       \        |        /
        各自 deterministic suffix
                |
                v
       6 组三视角完整视频与 reward
```

组内唯一主动改变的是分叉时的 SDE noise。相比独立完整 rollout，这种设计更接近 controlled comparison：reward 差异可以更直接地归因给第 k 步 transition。

### 5.3 SDE transition

GE-Sim 使用 EDM latent 坐标：

~~~text
y_t = x0 + sigma_t * noise

sigma_t = t / (1 - t)
~~~

transformer 输出原生 Flow-Matching velocity：

~~~text
v_theta = noise - x0
~~~

TempFlow adapter 将它映射成 Gaussian reverse-SDE transition：

~~~text
y_next ~ Normal(mu_theta(y_t, t), transition_std^2 * I)

其中 next_t < t
~~~

探索尺度由下式控制：

~~~text
g(t) = eta_sde * sqrt(t / (1 - t))

rf_noise_std = g(t) * sqrt(t - next_t)
~~~

当前使用 eta_sde = 0.7。每个分支采样一次：

~~~text
y_next[i] = mu_theta + transition_std * noise[i]

noise[i] ~ Normal(0, I)
~~~

采样时保存真实 transition 和 rollout policy 的 log-density：

~~~text
old_log_prob[i] = log pi_old(y_next[i] | y_t)
~~~

只有这一跳是随机 SDE；之后走确定性 ODE suffix，最后解码完整视频。

## 6. Reward 公式

当前正式实验先关闭昂贵且方差较复杂的 Action reward，只验证三视角视觉几何信号。训练只评价未来 25 帧，不评价作为条件输入的四帧历史。

对每个视角分别计算 future-only RGB PSNR：

~~~text
P_head, P_left, P_right
~~~

然后计算三视角平均值和最差视角：

~~~text
P_mean  = (P_head + P_left + P_right) / 3

P_worst = min(P_head, P_left, P_right)

R_psnr  = 0.6 * P_mean + 0.4 * P_worst
~~~

加入 worst-view 是为了避免模型只优化容易的 head view，而牺牲腕部相机。当前训练直接使用 raw dB 值构造 advantage，不先通过 sigmoid 压到 [0, 1]。

需要强调：上一版默认是 Action 与 sigmoid-PSNR 的 joint reward；当前 TempFlow 信号实验是 raw PSNR-only。因此“算法变化”和“reward 变化”目前同时存在。

## 7. Advantage 从哪里来

不使用 critic，也不训练 value network。一个 group 当前有 G = 6 个共享前缀的分支，对应 reward：

~~~text
R[1], R[2], ..., R[G]
~~~

使用组内 population mean 和 population standard deviation：

~~~text
R_mean = sum(R[j] for j = 1..G) / G

R_std  = sqrt(
    sum((R[j] - R_mean)^2 for j = 1..G) / G
)

A[i] = (R[i] - R_mean) / (R_std + 1e-6)
~~~

当前再将 advantage clip 到官方默认的 [-5, 5]。如果组内 PSNR 标准差低于 2e-4 dB，说明这些分支不可区分，整个 group 跳过，不向 reward 人为加噪声。

因此这里的 baseline 就是同一状态、同一分叉点下其他分支的组均值；不是 learned critic，也不再使用 CoCA 相似度。

## 8. Policy objective

训练时，将保存的 (y_t, y_next) 重新送入当前 LoRA policy，得到新的 transition mean 和 log-density：

~~~text
new_log_prob[i] = log pi_theta(y_next[i] | y_t)
~~~

定义 policy ratio：

~~~text
ratio[i] = exp(new_log_prob[i] - old_log_prob[i])
~~~

当前代码使用 PPO clipped surrogate：

~~~text
ratio_clipped[i] = clip(
    ratio[i],
    1 - clip_range,
    1 + clip_range
)

surrogate[i] = min(
    ratio[i]         * A[i],
    ratio_clipped[i] * A[i]
)

L_policy = -mean(noise_weight[k] * surrogate[i])
~~~

其中当前 clip_range = 1e-4。noise_weight[k] 是由该时间步 SDE 标准差得到的 schedule-aware 权重，并在合法时间步上归一化到均值约为 1；它不是采样概率，也不是 CoCA 的 p/q。

关闭 LoRA 后得到冻结 reference AWM 的 transition mean mu_ref。policy/reference 方差相同，因此使用 closed-form KL：

~~~text
L_kl = mean((mu_policy - mu_ref)^2) / (2 * transition_std^2)
~~~

最终目标：

~~~text
L_total = L_policy + beta * L_kl

beta = 0.5
~~~

只更新 attention LoRA，reference base 始终冻结。

## 9. 它究竟是 PPO、GRPO，还是别的？

最准确的描述是：

> **TempFlow-style single-transition branching + group-relative advantage + PPO clipped surrogate + reference KL。**

它像 GRPO 的地方：

- 同一 condition/state 采样多个分支；
- 用组内 reward 构造 advantage；
- 不需要 critic。

它像 PPO 的地方：

- 保存 old_log_prob；
- 重算 new_log_prob；
- 使用 importance ratio 和 clipped surrogate。

当前实现先冻结 rollout policy，收集 14 个时间步 group，再打乱成两个 optimizer minibatch。第一个 minibatch 更新前有：

~~~text
ratio[i] = 1
clip_fraction = 0
~~~

且梯度并不为零：

~~~text
grad_theta(ratio[i])
    = ratio[i] * grad_theta(log pi_theta)
~~~

第一个 minibatch 更新参数后，第二个 minibatch 仍使用采样时保存的 old_log_prob，因此通常有：

~~~text
ratio != 1
clip_fraction can be > 0
~~~

所以当前 PPO clipping 已经具有实际作用，而不是只保留目标形式。

~~~text
policy-gradient term:
    -noise_weight[k] * A[i] * grad_theta(log pi_theta)

plus reference regularization:
    +beta * grad_theta(L_kl)
~~~

## 10. 为什么不直接使用标准 PPO 或普通 GRPO？

### 不直接使用标准 PPO

标准 PPO 通常需要逐步 action log-probability、return/GAE 和 critic。这里的 action 是一个极高维视频 latent transition，reward 又只在完整视频解码并评价后得到。为整条 15 步轨迹建立 critic 会带来额外模型、显存、训练稳定性和 value-target 设计问题。

我们先用组内相对回报去掉 critic，再只随机化一个 transition，使 terminal reward 能够较清楚地归因到这个动作。

### 不直接使用“独立完整 rollout”的普通 GRPO

普通做法可以从不同初始噪声生成多个完整视频，再做组内相对 reward。但各 rollout 从第一步起就不同，reward 方差混合了整条去噪路径的随机性。对于视频模型，这种方差和 rollout 成本都很高。

TempFlow 分叉让所有分支共享前缀，只比较同一 y_t 下不同 y_next 的后果。它牺牲了“一次更新整条随机轨迹”，换取更局部、更接近因果对照的 credit signal。

### 为什么不是继续用 CoCA

CoCA 的时间 credit 来自 predicted-x̂₀ 与最终 latent 的相似度变化，它适合做训练时间 proposal，但不是 transition likelihood objective。TempFlow 直接对真实采样的 y_t → y_next 计算 log-density，并由最终 reward 决定提高或降低其概率，RL 语义更直接。

## 11. CoCA → TempFlow 到底换了什么

| 模块 | CoCA | 当前 TempFlow |
|---|---|---|
| Backbone | AWM | 同一个 AWM |
| Rollout group | 不同 initial seed 的完整 rollout | 相同 initial seed 和 prefix 的单步分叉 |
| Credit | predicted-x̂₀ / final-latent 相似度 | 分支 terminal reward |
| Advantage | leave-one-out reward difference | 组内标准化 reward，clip 到 [-5, 5] |
| 时间步选择 | CoCA proposal q(k) | 按 timestep_fraction=0.99 得到 0～13，同一 old policy 下全部收集 |
| 概率比 | 噪声档位 p(k)/q(k) | transition policy pi_theta/pi_old |
| 主目标 | reward-weighted FM-MSE | transition policy gradient / clipped surrogate |
| Reference 约束 | velocity MSE | equal-variance Gaussian KL |
| Critic | 无 | 无 |
| 更新参数 | LoRA | LoRA |

旧 CoCA 的 `proposal/base_probabilities/importance_clipping` 字段会因为配置继承出现在部分 TempFlow effective config 中，但运行时代码不读取它们；它们不是当前算法的一部分。

## 12. 当前实验设置

当前 PSNR-only 正式配置：

```text
数据规模                 16 conditions
反向去噪                 15 steps
合法分叉时间步           0..13，共 14 个
计划采样                 16 × 14 = 224 timestep groups
计划训练                 16 × 2 = 32 optimizer updates
每组分支                 6
固定 initial seed        123456
SDE eta                  0.7
学习率                   1e-5
warmup                   10 updates
PPO clip range           1e-4
reference KL beta        0.5
reward                    三视角 future-only raw balanced PSNR
计划固定集评估           Base / update 16 / update 32
```

## 13. 当前结果：什么已经成立，什么还没有

以下是 2026-08-16 旧版即时更新实现的历史训练记录：

```text
已完成 optimizer updates                 80 / 224
已覆盖 condition                         6 / 16
跳过的低方差 group                       0 / 80
具有非零 policy gradient 的 group        80 / 80
每次发生改变的 LoRA 参数张量             448 / 448
平均组内 raw-PSNR 标准差                 0.12135 dB
最小组内 raw-PSNR 标准差                 0.00148 dB
平均 policy gradient norm                4.12e-4
观测到的 policy gradient norm 范围       1.94e-5 ～ 1.26e-3
ratio mean                               始终为 1
clip fraction                            始终为 0
```

这些结果说明：

1. SDE 分支确实能产生可区分的最终视频 reward；
2. group-relative advantage 不是退化的全零信号；
3. PSNR policy term 在真实 AWM 上能够稳定地产生非零 LoRA 梯度；
4. reference 保持冻结，KL 有限；
5. 完整 rollout → reward → transition update 的工程闭环已经跑通。

但它们还不能证明：

- 固定测试集 PSNR 已经提高；
- 视频感知质量或 action-following 已经改善；
- TempFlow 优于 CoCA；
- 提升来自算法而不是 raw-PSNR reward 变化。

原因是最新运行停在 update 80，尚未到计划的 update-112 固定集评估。已知的训练前 fixed-16 基准是 future-only balanced PSNR 22.1865 dB；目前没有与 update 80 对应的同 seed 固定集结果。训练 group reward 跨 condition 和 timestep 不可直接连成性能曲线。

因此当前最稳妥的结论是：

> **新版比上一版更容易观察到稳定、非零且可解释的训练信号；泛化效果是否提升仍待固定集评估和同 reward 消融确认。**

## 14. 下一步验证

为了把“有训练信号”升级为“算法有效”，计划按以下顺序验证：

1. 用新版数据流从头训练，在 update 16 跑固定 16 条、固定 seed 的中期评估；
2. 完成 update 32，比较 Base / 16 / 32 的 future-only PSNR；
3. 在完全相同 raw-PSNR reward 下比较 CoCA 与 TempFlow；
4. 恢复 Action reward，检查 PSNR 提升是否损害动作一致性；
5. 做 branch-only、noise-weight-only、full TempFlow 消融；
6. 用新版 rollout buffer 正式重跑，并监控两个 minibatch 各自的 ratio、clip fraction 和 KL。

最关键的公平对照是：

```text
AWM + CoCA     + raw PSNR-only
AWM + TempFlow + raw PSNR-only
```

两组保持相同数据、seed、LoRA、学习率、训练更新数和固定评估集。
