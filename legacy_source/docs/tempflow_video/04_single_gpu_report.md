# 单卡验证报告

更新时间：2026-08-16（Asia/Shanghai）。此文件区分“已测量”“未运行”，不以计划值冒充结果。

## 环境与已通过证据

- GPU：1× NVIDIA A100-SXM4-80GB；PyTorch 2.7.1+cu126，CUDA runtime 12.6。
- 数据 gate：16/16 prep valid；48/48 GT videos valid；manifest SHA256 `eb7615491d8c5e8acefea85355b25d45e1cf8ebeb53d2531170171c6b53e9d17`。
- model gate：1.969B base、27,525,120 trainable LoRA params、224 targets、BF16；reference frozen；load peak CUDA allocation 14,033,699,840 bytes。
- scheduler gate：15 reverse steps，14 legal branch steps，terminal duplicate no-op 被拒绝。
- unit/regression gate：35 tests passed；覆盖 advantage、branch invariants、真实 Gaussian log-prob/PPO loss、adapter forwarding、reference freeze、initial KL、optimizer update、checkpoint/resume、普通 video GRPO 更新、rollout RNG 隔离与既有 AWM tests。
- real reward parity：PASS；同一真实 branch MP4 的 adapter saved result 与 direct legacy output 逐叶比较，`1e-8` tolerance 下 0 mismatch。

证据文件位于外部运行目录：

- `tempflow_runtime/outputs/preflight_metadata.json`
- `tempflow_runtime/outputs/preflight_model.json`

## Gate 状态

| Gate | 状态 | 证据/结论 |
|---|---|---|
| 1 Preflight | PASS | 数据、资产、GPU、scheduler、LoRA/reference 实机通过 |
| 2 基础推理+reward+重复确定性 | PASS（含诊断风险） | fresh-condition back-to-back trajectory/三视角 MP4 完全相同；所有 reward-driving leaves 差 0 |
| 3 2-branch | PASS | 同 prefix hash、不同 noise hash，三视角 MP4 均不同，reward 0.38831/0.40498，advantage ±0.99988 |
| 4 KL | PASS | 初始 0；resume 后第二组 pre-update raw KL `1.31e-8`、KL grad `4.60e-7`，reference 不变 |
| 5 一次真实 update | PASS | 两次真实 update 有有限/nonzero policy grad；448 LoRA tensors 变化；checkpoint 真实恢复成功 |
| 6 10 updates | PASS | checkpoint 2 恢复后连续完成 step 3–10；10/10 组有非零方差/梯度，核心量有限，step 2–10 每步 448 个 LoRA tensor 变化；checkpoint 10 纯恢复成功 |
| 7 overfit16 ≤100 updates | RUNNING | base evaluation 后启动；截至本次写入已完成 40/100 个真实 6-branch updates、checkpoint 10/20/30/40 与固定 16 条 policy-10/20/30/40 evaluation |

## 真实 smoke 数值

Step 1：reward mean/std `0.396645 / 0.008332`，policy grad norm `1.46368e-4`，initial raw KL `0`，端到端 212.8 s。Step 2 从完整 checkpoint resume：reward mean/std `0.398381 / 0.016541`，policy grad norm `1.53090e-4`，raw KL `1.30958e-8`，端到端 216.7 s，peak CUDA allocated/reserved `37.316 / 41.238 GB`。两个点不能作为 reward 上升趋势结论。

从 checkpoint 2 恢复后，step 3–10 连续完成并正常退出。完整 10 步 reward mean 范围为 `[0.365699, 0.406891]`，组内 reward std 范围为 `[9.998e-5, 0.022292]`，policy grad norm 范围为 `[1.43088e-4, 1.81579e-4]`，raw reference KL 范围为 `[0, 1.38810e-8]`；所有核心标量有限，10/10 组均为非零方差与非零策略梯度。step 2–10 的可比 peak reserved memory 从 `38.40625 GiB` 到 `38.40820 GiB` 后持平；step 1 没有该早期字段，未补造。step 2–10 每步均记录 448 个 LoRA tensor 改变；step 1 当时也未记录该字段。

checkpoint 10 的 compact adapter policy 为 55,197,881 bytes、optimizer 为 110,478,947 bytes，并包含 scheduler、RNG、trainer/config state 与原子 `COMPLETE` 标记。以 `--resume checkpoint_10 --max-optimizer-steps 10` 做只加载不更新验证，进程 exit 0 且没有生成 step 11。

## 固定 16 条 base evaluation

已按 seed `12345678` 先生成全部 16 条、48 个三视角 MP4，随后才初始化并顺序运行既有 terminal reward，避免 reward-side CUDA 执行状态影响同一评估集后续生成。运行目录为 `tempflow_runtime/outputs/eval_overfit16/runs/20260815T184108Z/`，`rewards.jsonl` 有 16 行，`summary.json` 保留所有 numeric leaves 的 mean/std/min/max。

- total reward：`0.523525 ± 0.108714`，范围 `[0.354422, 0.715539]`。
- action reward：`0.384882 ± 0.128883`，范围 `[0.195468, 0.598987]`。
- geometry reward：`0.662168 ± 0.236572`，范围 `[0.174570, 0.987942]`。
- balanced / worst-view PSNR：`22.1865 ± 2.7150 dB` / `20.5345 ± 2.9588 dB`。
- per-view PSNR mean：head `23.7368 dB`、hand-left `24.3020 dB`、hand-right `21.8245 dB`。

这是训练前基线，不是 overfit 改善结论。正式训练后的固定 16 条必须使用同一 seed、同一先生成后评分协议逐 condition 对比。

## 正式 overfit16 运行中

运行目录：`tempflow_runtime/outputs/overfit16_single_gpu/runs/20260815T185747Z/`。step 1（condition `001...`、branch timestep 0）reward `0.393164 ± 0.021821`，policy grad norm `3.19218e-5`，初始 raw KL `0`；step 2（condition `014...`、branch timestep 1）reward `0.421780 ± 0.054334`，policy grad norm `4.52031e-5`，raw KL `4.79189e-9`。截至 step 10，10/10 组 reward 均有非零方差、每步 448 个 LoRA tensor 改变、peak reserved memory 均为 `38.40625 GiB`。step 10 policy grad norm `1.13875e-3`，raw KL `1.20493e-5`，仍有限。checkpoint 10 含 compact policy/optimizer/scheduler/RNG/trainer/config state，且原子 `COMPLETE=ok`。这些训练组统计不能和固定 16 条 base mean 直接作前后比较。

policy 10 按同一 seed `12345678` 和同一“先生成 16 条/48 个视角视频，再评分”协议完成。固定集 total reward 从 base `0.523525 ± 0.108714` 降为 `0.502195 ± 0.129646`，mean delta `-0.021329`；4/16 条改善、12/16 条下降。action reward mean delta `-0.042580`（5/16 改善），geometry reward mean delta `-7.84e-5`（8/16 改善），balanced/worst-view PSNR mean delta 分别为 `+3.57e-5/-2.35e-4 dB`。因此 step 10 没有出现固定 16 条 overfit 改善；下降主要来自 action，geometry 基本不变。目标没有规定单个评估点下降即早停，且未触发 P0，故继续运行并在后续固定评估点重新判断。

policy 20 也完成了严格同协议评估，16/16 reward 均 valid。total reward 为 `0.505284 ± 0.105466`：相对 base mean delta `-0.018241`（8/16 改善），相对 policy 10 回升 `+0.003089`。action reward 为 `0.348423 ± 0.143680`，相对 base `-0.036459`（7/16 改善）、相对 policy 10 `+0.006121`；geometry reward 为 `0.662145 ± 0.236553`，相对 base `-2.24e-5`、相对 policy 10 `+5.59e-5`。balanced/worst-view PSNR 相对 base 分别为 `+2.70e-4/+5.45e-5 dB`，量级仍近似不变。step 20 比 step 10 有小幅回升，但仍未超过 base，不能据此声称完成 overfit；训练继续到后续预定评估点。

policy 30 继续使用同一 fixed16 协议，16/16 reward 均 valid。total reward 为 `0.508397 ± 0.119558`：相对 base mean delta `-0.015128`（6/16 改善），相对 policy 20 回升 `+0.003113`。action reward 为 `0.354632 ± 0.127430`，相对 base `-0.030250`、相对 policy 20 `+0.006209`；geometry reward 为 `0.662162 ± 0.236632`，相对 base `-5.69e-6`、相对 policy 20 `+1.67e-5`。balanced/worst-view PSNR 相对 base 分别为 `+5.49e-4/-2.71e-4 dB`，仍可视为近似不变。policy 10→20→30 的 fixed16 mean 连续小幅回升，但 policy 30 仍低于 base，故继续运行且不宣称已 overfit。

policy 40 的同协议 total reward 为 `0.503668 ± 0.123174`：相对 base mean delta `-0.019857`（5/16 改善），相对 policy 30 回落 `-0.004729`。action reward 为 `0.345139 ± 0.155937`，相对 base `-0.039742`、相对 policy 30 `-0.009492`；geometry reward 为 `0.662196 ± 0.236336`，相对 base `+2.82e-5`、相对 policy 30 `+3.39e-5`。因此回落仍由 action 主导，geometry 近似不变；policy 10→30 的小幅回升不是单调趋势。最终报告必须同时给出预定 final checkpoint 和 fixed16 最佳 checkpoint，不能只挑有利点。

KL 日志是当前 branch timestep 上的局部 transition KL，不能把 timestep 0–13 的连续数值当成同一分布上的全局 KL 增长曲线。第一轮末端 timestep 13 的 raw KL 为 `2.59029e-4`，但 timestep 回绕后的 step 15 / timestep 0 raw KL 为 `2.23001e-9`；后续稳定性应按相同 timestep 跨轮次比较，同时保留 timestep/noise weight。该回绕不是 reference 重置，reference sentinel 始终未变。

首次 checkpoint 暴露 PEFT 检测错误，保存了约 4 GB full state；修复后 checkpoint 2 的 adapter policy 为 55,197,881 bytes，optimizer 为 110,478,947 bytes，且 resume 已真实通过。首次 resume 还暴露 sentinel 建立早于 restore 的版本计数问题；该尝试在 backward 前安全失败，artifact 保存在 `failed_resume_20260815T174400Z/`，修复后重试通过。

Gate 2 前两次失败分别定位了 conditioning cache 的随机数消费差异，以及 reward 初始化插入两次生成之间造成的 CUDA 执行状态变化；失败报告均保留。最终协议使用 fresh condition，并先 back-to-back 生成两个视频、再按固定顺序打 reward。最终 16 个 trajectory latent hashes 和三视角 MP4 hashes 均逐项相同，`total/action/geometry` 及所有 reward-driving leaves 差为 0。旧 YOLO tracker 输出中不参与 reward 公式的 plain `ate/ate_norm` diagnostics 仍有 `3.18e-4/4.97e-7` 变化，已记录为诊断 nondeterminism，不能混称完整 dict bitwise parity。

## 尚不能给出的结论

目前不能声称 overfit reward 超过 base 或 8 卡可运行。40-step gate 只证明当前单卡短程训练的数值、显存、checkpoint 与定期 fixed16 评估稳定；截至 policy 40 所有 checkpoint mean 均低于 base，且 policy 40 较 policy 30 回落。后续结果必须包含逐 condition 前后 reward、所有 component/per-view/worst-view、KL/gradient/memory 曲线，并检查单分量主导与 reward hacking。

GT 为 30 fps、prediction 为 16 fps，而旧 reward 按 frame index 配对。即使 total reward 上升，也只能解释为该固定旧 reward 的改善，不代表动作、真实 3D 一致性或泛化提升。
