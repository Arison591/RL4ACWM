# AWM × TempFlow：单步分叉强化学习技术简报

> 面向了解强化学习、但没有参与本项目的读者。感谢学长提供 GPU 支持。

> 2026-08-17 更新：当前代码已经改成 frozen-old-policy rollout buffer。先在同一 policy 下收集全部 k，再做两个 minibatch update，第二批的 PPO ratio/clip 可以实际生效。详细审计见 `11_tempflow_source_alignment_zh.md`。文末 80/224 是旧实现的历史记录。

## 一句话版本

视频生成 backbone 没换，仍然是原来的 action-conditioned AWM。

我们把外层训练算法从 CoCA 的“相似度 credit + 噪声 proposal + Flow-Matching MSE”，换成了 TempFlow 的“共享前缀 + 单步 SDE 分叉 + 组内 advantage + transition policy gradient”。

当前不使用 critic。先冻结 old policy 收集同一 condition 下全部合法 k，再用两个 minibatch 做 PPO clipped update；第一批 ratio 为 1，第二批开始 clipping 可以实际生效。

## 1. 任务是什么

AWM 的输入：

~~~text
4 帧三视角历史图像
+ 未来 25 步机器人 action
+ 三相机内外参
+ 文本条件
~~~

AWM 的输出：

~~~text
head / hand-left / hand-right
三个视角的未来 25 帧视频
~~~

模型通过 15 步 Flow-Matching 反向过程，从噪声 latent 生成最终视频。

我们的目标是利用完整视频上的 reward 微调 AWM，使生成结果更接近真实未来，并保持三视角稳定。

## 2. 哪些东西没换

- GE-Sim/Cosmos AWM backbone 没换。
- VAE、text encoder、scheduler 没换。
- action trajectory map 和 camera ray map 条件没换。
- 仍然冻结 base model。
- 仍然只训练 attention LoRA。

变化的是 rollout 方式、credit assignment、advantage 和 policy objective。

## 3. 上一版 CoCA

CoCA 对同一 condition 生成多条独立完整 rollout。

每条 rollout 保存 15 个 predicted clean latent：

~~~text
x0_hat[1], x0_hat[2], ..., x0_hat[15]
~~~

计算它们与最终 latent 的相似度：

~~~text
similarity[t] = cosine(x0_hat[t], final_latent)
~~~

再把相似度变化转换成噪声档位 proposal：

~~~text
q_coca(k) = softmax(score[k] / temperature)

q(k) = (1 - eta) * p(k) + eta * q_coca(k)
~~~

从 q(k) 抽一个噪声档位，并用 p(k) / q(k) 修正：

~~~text
L_coca = p(k) / q(k) * [
    advantage[i] * FM_MSE[i]
    + beta * reference_MSE[i]
]
~~~

它本质上是 reward-weighted self-training。

主要问题是不同 rollout 从初始噪声开始就不一样，最终 reward 混合了整条 15 步路径的随机性；相似度 credit 是启发式信号，不是某个真实随机 transition 的回报。

## 4. 新版 TempFlow

### 4.1 强化学习映射

在去噪时间步 k：

~~~text
state   = 当前 noisy latent y_t
action  = 采样得到下一个 latent y_next
policy  = AWM velocity 定义的 Gaussian transition
reward  = 最终完整视频的 terminal reward
~~~

### 4.2 共享前缀、单步分叉

~~~text
同一 condition + 同一 initial seed
                |
                v
        deterministic ODE prefix
                |
                v
          完全相同的 y_t
      ↙         ↓         ↘
  SDE noise 1  ...   SDE noise 6
      ↘         ↓         ↙
       6 个不同的 y_next
                |
                v
        各自 deterministic suffix
                |
                v
       6 组三视角完整视频与 reward
~~~

同一 group 中，condition、initial seed、prefix、branch timestep 和 policy version 全部相同，只有分叉噪声不同。

这样最终 reward 的差异可以更直接地归因给这一跳 transition。

### 4.3 SDE transition

GE-Sim 使用 EDM latent：

~~~text
y_t = x0 + sigma_t * noise

sigma_t = t / (1 - t)
~~~

AWM transformer 输出：

~~~text
velocity = noise - x0
~~~

TempFlow 把 velocity 转成 Gaussian transition：

~~~text
y_next ~ Normal(
    mean = mu_policy(y_t, t),
    variance = transition_std^2
)
~~~

探索尺度：

~~~text
exploration_scale(t)
    = eta_sde * sqrt(t / (1 - t))

rf_noise_std
    = exploration_scale(t) * sqrt(t - next_t)
~~~

当前 eta_sde = 0.7。

这两个式子可以分两层理解。

第一层：当前时间点允许多强的探索。

~~~text
exploration_scale(t)
    = eta_sde * sqrt(t / (1 - t))
~~~

- t 是当前去噪时刻，反向生成从接近 1 的高噪声区逐渐走向 0 的干净区。
- t 越接近 1，当前 latent 越像噪声，可以允许更强的随机探索。
- t 越接近 0，视频结构已经基本形成，随机扰动应该变小。
- eta_sde 是全局旋钮：设为 0 就没有分叉随机性；数值越大，六个分支差异越大。

第二层：把“单位时间噪声强度”换算成这一小步真正加入的噪声。

~~~text
step_length = t - next_t

rf_noise_std
    = exploration_scale(t) * sqrt(step_length)
~~~

这里出现 sqrt(step_length)，是因为 Brownian/SDE 增量的方差与时间长度成正比，因此标准差与时间长度的平方根成正比：

~~~text
variance  ∝ step_length
std       ∝ sqrt(step_length)
~~~

最后每个分支执行：

~~~text
branch_noise[i] ~ Normal(0, I)

next_latent[i]
    = policy_mean
    + transition_std * branch_noise[i]
~~~

所以它不是一个额外 reward，也不是选择时间步的概率。它只控制：在选定的分叉时间步上，六个分支围绕 AWM 给出的 policy mean 散开多远。

需要区分两个量：

- rf_noise_std 是 Rectified-Flow 坐标中的单步噪声标准差。
- GE-Sim 实际保存 EDM latent，因此代码还会除以 1 - next_t，得到 EDM 坐标下真正用于采样的 transition_std。

最直观地说：

~~~text
AWM velocity 决定分支中心 policy_mean
TempFlow SDE 决定分支围绕中心散开多少
六个独立 Gaussian noise 决定各分支往哪个方向散开
~~~

每个分支实际采样：

~~~text
y_next[i]
    = mu_policy
    + transition_std * standard_normal_noise[i]
~~~

采样时保存：

~~~text
old_log_prob[i]
    = log pi_old(y_next[i] | y_t)
~~~

## 5. 当前 reward

当前信号实验先关闭 Action reward，只训练三视角 future-only raw PSNR。

三个视角分别得到：

~~~text
P_head
P_left
P_right
~~~

聚合方式：

~~~text
P_mean
    = (P_head + P_left + P_right) / 3

P_worst
    = min(P_head, P_left, P_right)

reward
    = 0.6 * P_mean
    + 0.4 * P_worst
~~~

加入 worst-view 是为了防止模型只优化容易的 head view，而牺牲腕部相机。

训练直接使用 raw PSNR dB，不先通过 sigmoid 压到 [0, 1]。

## 6. Advantage 怎么来

不使用 critic，也不训练 value network。

同一个状态 y_t 产生 6 个分支：

~~~text
reward[1], reward[2], ..., reward[6]
~~~

组均值：

~~~text
reward_mean
    = sum(reward[j]) / group_size
~~~

population standard deviation：

~~~text
reward_std
    = sqrt(
        sum((reward[j] - reward_mean)^2)
        / group_size
    )
~~~

advantage：

~~~text
advantage[i]
    = (reward[i] - reward_mean)
    / (reward_std + 1e-6)
~~~

最后：

~~~text
advantage[i] = clip(advantage[i], -5, 5)
~~~

所以这里的 baseline 就是同状态、同分叉点下其他分支的组均值，不是 learned critic，也不是 CoCA 相似度。

如果组内 PSNR 标准差低于 2e-4 dB，说明分支不可区分，整个 group 跳过。

## 7. Policy loss

训练时对保存的 transition 重新打分：

~~~text
new_log_prob[i]
    = log pi_policy(y_next[i] | y_t)
~~~

新旧策略概率比：

~~~text
ratio[i]
    = exp(new_log_prob[i] - old_log_prob[i])
~~~

PPO clipping：

~~~text
ratio_clipped[i]
    = clip(
        ratio[i],
        1 - clip_range,
        1 + clip_range
    )
~~~

clipped surrogate：

~~~text
surrogate[i]
    = min(
        ratio[i]         * advantage[i],
        ratio_clipped[i] * advantage[i]
    )
~~~

policy loss：

~~~text
L_policy
    = -mean(
        noise_weight[k] * surrogate[i]
    )
~~~

当前 clip_range = 1e-4。

noise_weight[k] 来自该时间步的 SDE 噪声尺度，并在合法时间步上归一化到均值约为 1。它不是采样概率，也不是 CoCA 的 p(k) / q(k)。

## 8. Reference KL

临时关闭 LoRA，得到冻结原始 AWM 的 transition mean：

~~~text
mu_reference
~~~

policy 和 reference 使用相同 transition variance，因此：

~~~text
L_kl
    = mean((mu_policy - mu_reference)^2)
    / (2 * transition_std^2)
~~~

总目标：

~~~text
L_total
    = L_policy + beta * L_kl

beta = 0.5
~~~

只更新 LoRA，reference base 始终冻结。

## 9. 它是 PPO 还是 GRPO

最准确的说法：

> TempFlow-style single-transition branching + group-relative advantage + PPO clipped surrogate + reference KL。

像 GRPO 的部分：

- 同一状态采样多个分支。
- 用组内相对 reward 构造 advantage。
- 不需要 critic。

像 PPO 的部分：

- 保存 old_log_prob。
- 重新计算 new_log_prob。
- 使用 new/old ratio。
- 保留 clipped surrogate。

当前先冻结 policy，收集 14 个 k 的 rollout buffer，再分成两个 optimizer minibatch。第一批更新前数值上：

~~~text
ratio = 1
clip_fraction = 0
~~~

但是梯度不为零：

~~~text
gradient of ratio
    = ratio * gradient of log pi_policy
~~~

第一批更新后，第二批仍使用同一个 old policy 保存的 log probability，所以 ratio 可以偏离 1，clip fraction 可以大于 0。当前真正生效的主梯度可以理解为：

~~~text
-noise_weight[k]
* advantage[i]
* gradient(log pi_policy)

+ beta * gradient(L_kl)
~~~

因此当前不再只是 PPO-style 的外形：buffer 中后续 minibatch 已经具有实际 PPO clipping 行为。

## 10. 为什么不直接跑标准 PPO

标准 PPO 通常还需要：

- 整条轨迹的逐步 action log-probability；
- return 或 GAE；
- critic/value network；
- 对同一批 rollout 做多个 policy epoch。

这里一个 action 是高维视频 latent transition，reward 只能在完整视频解码后得到。给整条 15 步去噪轨迹训练 critic，会增加模型、显存、value target 和稳定性问题。

当前先用组内相对 reward 去掉 critic，再只随机化一个 transition，使 terminal reward 更容易归因。

## 11. 为什么不直接跑普通 GRPO

普通 full-rollout GRPO 可以从多个不同初始噪声生成完整视频，再做组内相对 reward。

问题是这些 rollout 从第一步开始就不同，最终 reward 差异混合了整条路径的随机性。

TempFlow 让所有分支共享前缀，只比较同一个 y_t 下不同 y_next 的最终后果，得到更局部的 credit signal。

## 12. CoCA 到 TempFlow 换了什么

| 模块 | CoCA | 当前 TempFlow |
|---|---|---|
| Backbone | AWM | 同一个 AWM |
| Rollout | 不同初始 seed 的完整轨迹 | 相同前缀的单步 SDE 分叉 |
| Credit | predicted-x0 相似度 | 分支 terminal reward |
| Baseline | leave-one-out reward | 组内均值和标准差 |
| 时间步 | 从 CoCA proposal q(k) 抽样 | timestep_fraction=0.99 得到 0～13，同一 old policy 下全部收集 |
| 概率比 | p(k) / q(k) | pi_policy / pi_old |
| 主目标 | reward-weighted FM-MSE | transition policy gradient |
| Critic | 无 | 无 |
| 更新参数 | LoRA | LoRA |

## 13. 当前实验状态

计划配置：

~~~text
16 conditions
14 个合法 branch timestep
总计 224 timestep groups
总计 32 optimizer updates
每组 6 个 SDE 分支
~~~

以下是截至 2026-08-16 的旧版即时更新历史记录：

~~~text
已完成 update                  80 / 224
已覆盖 condition               6 / 16
低方差跳组                     0 / 80
非零 policy gradient           80 / 80
每次改变的 LoRA 张量           448 / 448
平均组内 PSNR std              0.12135 dB
平均 policy grad norm          4.12e-4
ratio mean                     始终为 1
clip fraction                  始终为 0
~~~

已经能够确认：

- SDE 分支会产生可区分的 reward。
- group advantage 不是全零。
- 真实 AWM 上能得到稳定、非零的 LoRA 梯度。
- reference 保持冻结。
- rollout、reward、update 的完整工程链路跑通。

暂时不能确认：

- 固定测试集 PSNR 已经提升。
- TempFlow 已经优于 CoCA。
- 视觉提升不会损害 action-following。
- 当前变化来自算法而不是 reward 口径变化。

旧版运行停在 update 80，尚未到当时计划的 update 112 固定集评估。训练前 fixed-16 基准为 22.1865 dB，目前没有对应的 update-80 固定 seed 结果；新版必须从头训练，不能拿旧 checkpoint 续跑。

所以当前最稳妥的结论是：

> 新版已经表现出稳定、非零、可解释的训练信号；泛化效果是否提升，还需要固定集评估和同 reward 消融。

## 14. 下一步

1. 用新版数据流从头训练，到 update 16 完成固定 16 条、固定 seed 的中期评估。
2. 跑到 update 32，比较 Base、16、32。
3. 在完全相同 raw-PSNR reward 下比较 CoCA 和 TempFlow。
4. 恢复 Action reward，检查动作一致性。
5. 做 branch-only、noise-weight-only 和 full TempFlow 消融。
6. 用新版 rollout buffer 重跑，检查第二 minibatch 的 ratio、clip fraction 和 KL。

最关键的公平对照：

~~~text
AWM + CoCA     + raw PSNR-only
AWM + TempFlow + raw PSNR-only
~~~
