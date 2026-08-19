# 现有视频 Reward 接入与 Parity

## 唯一入口

`VideoRewardAdapter` 不复制任何 reward 数学，仅将 condition、prep 目录和三视角 prediction 路径转交给：

`experiments.awm_coca.reward_runner.compute_head_reward`

调用在 `torch.no_grad()` 下执行。输入是完整 29-frame 三视角 MP4、head GT、三视角 GT 和 condition 的动作/相机数组；输出是 Python dict，其中 `total_reward` 是越大越好的一维有限标量，并保留 `action_metrics`、`action_reward_components`、完整 `geometry`/per-frame/per-view 指标及预处理协议。

## 实际公式

Action 只在 head view 计算，左右臂结果先做 finite mean：

```text
r_iou  = clip(mean_iou, 0, 1)
r_ate  = 1 - clip(af_fdce_ate_norm / 0.2, 0, 1)
r_fdce = 1 - clip(fdce / 10.0, 0, 1)
R_action = (0.1*r_iou + 0.7*r_ate + 0.2*r_fdce) / 1.0
```

Action 依赖 SAM3 mask、YOLO/动作命令轨迹以及 CoWTracker FDCE；任一正权重 required component 缺失会令 reward invalid，不做伪造或补零。

所谓 geometry 实际为三视角 RGB PSNR。对 frame indices 4..28 逐帧计算 PSNR，各视角先求均值，再计算：

```text
P_mean = mean(P_head, P_hand_left, P_hand_right)
P_worst = min(P_head, P_hand_left, P_hand_right)
P_balanced = 0.6*P_mean + 0.4*P_worst
R_geometry = sigmoid((P_balanced - 20.4) / 1.8)
R_total = 0.5*R_action + 0.5*R_geometry
```

代码中没有独立 DA3/depth、temporal、motion-region 或 quality reward；不得在日志中虚构这些项。per-frame PSNR 与 worst-view 已原样保存。

## 时间与空间对齐

Action 路径读取 head GT/pred，最多取 29 帧，截断到共同长度，若分辨率不同则 resize GT 到 pred，按 frame index 配对。Geometry 对三视角 indices 4..28 使用相同 resize-to-pred 规则。当前 GT 是 30 fps，prediction 是 16 fps；旧实现记录两者 FPS 但不 resample。首轮 parity 必须保留此行为。

## 权重交叉核对

| 来源 | Action 内部 `(IoU, ATE, FDCE)` | Joint `(Action, geometry)` | geometry 聚合 `(mean,worst)` |
|---|---:|---:|---:|
| 当前训练 config / TempFlow effective config | `(0.1,0.7,0.2)` | `(0.5,0.5)` | `(0.6,0.4)` |
| 当前 eval config | `(0.2,0.1,0.7)` | 以文件为准 | `(0.6,0.4)` |
| reward 函数无 config 默认 | `(1/3,1/3,1/3)` | `(1,1)` 归一 | `(0.6,0.4)` |
| 旧 `checkpoint_100/train_config.json` | `(1/3,1/3,1/3)`，action-only | 不适用 | 不适用 |

首轮采用当前训练代码实际读取的 `(0.1,0.7,0.2)` 与 joint 50/50，并通过 effective config 和 reward-config SHA256 固化，不静默使用 eval 或旧 checkpoint 权重。

## Parity 证据

- `tests/test_legacy_reward_parity.py` 验证 adapter 对所有路径、参数和旧函数返回 dict 不作变换。
- adapter 与旧入口是同一函数对象调用，因此数值容差目标为 `1e-8`；真实 MP4 gate 的两条路径逐层比较全部 dict/list 数值叶子，而不只比较 total。
- 真实生成 branch MP4 已分别经过 adapter 与 direct legacy 入口：全部 keys/leaves 在 `1e-8` 容差内一致，0 个 mismatch，报告为 `tempflow_runtime/outputs/reward_parity_real.json`。
