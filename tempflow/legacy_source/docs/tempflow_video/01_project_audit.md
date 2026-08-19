# TempFlow 视频项目审计

## 审计范围与基线

- 独立分支：`exp/tempflow-video-overfit16`
- 起点：`dce69e48a952449e873a791812e506df878bc8a9`
- 模型：GE-Sim Cosmos2 multi-view video，约 1.969B base parameters；3 视角，4 history + 25 future，共 29 帧，384×512，15 个 reverse denoise steps。
- 训练方式：仅 attention LoRA；实测 27,525,120 个可训练参数、224 个 target projections。base parameters 冻结，并通过禁用 adapter 构成 reference policy。
- 生成目标：transformer 原生 rectified-flow velocity `v = epsilon - x0`。GE-Sim pipeline 保存的是 EDM 坐标 `y_t=x_t/(1-t)=x0+sigma_edm*epsilon`。
- 数据：固定 16 条 condition，`actions.npy` 均为 `[29,16]`，相机顺序固定为 `head, hand_left, hand_right`。manifest SHA256 为 `eb7615491d8c5e8acefea85355b25d45e1cf8ebeb53d2531170171c6b53e9d17`。

视频时间与生成时间严格分开：视频时间是 29 个 frame index、动作序列和 FPS；生成时间是 15 个 flow/EDM scheduler transition。TempFlow 在后者分叉，不在视频帧之间分叉。

## 既有训练链路

```text
prep condition + actions + cameras
                |
                v
PrepConditionDataset -> PersistentGeSimRuntime.prepare_condition
                |
                v
GE-Sim ODE rollout -> full 3-view MP4 -> legacy terminal reward
                |                              |
                v                              v
trajectory tensors                    reward components
                |                              |
                +------ AWM-CoCA proposal -----+
                             |
                    flow-matching MSE + ref MSE
                             |
                       AdamW / checkpoint
```

既有 AWM-CoCA 的 loss 是 advantage-weighted flow-matching MSE 加 reference MSE，不是基于随机 transition log-probability 的 TempFlow/GRPO。新 baseline 放在 `experiments/tempflow_video/`，不修改或冒充该主线。

新插入点：GE-Sim 先产生同 seed 的确定性 prefix；`TempFlowBranchSampler` 在一个合法生成 timestep 复制 latent、执行真实 SDE Gaussian transition，再以 ODE 完成 suffix。完整视频只进入既有 terminal reward。`TempFlowVideoTrainer` 对采样 transition 重新打分并计算 PPO ratio 与 closed-form reference KL。

## Scheduler 审计

15-step Karras EDM sigma 为约 `80 -> ... -> 0.002`。pipeline 因 `final_sigmas_type=sigma_min` 把附加的最后 `0` 替换为前一项 `0.002`，因此最后一对是 no-op。合法 branch indices 只有 0–13；配置和 preflight 均拒绝 index 14。

实测 paper transition noise `eta*sqrt(t/(1-t))*sqrt(t-next_t)` 范围为 `0.00601855` 到 `0.52710130`（终端 no-op 为 0）。按 14 个可分叉 transition 的均值归一后，noise-aware weight 范围为 `0.01769754` 到 `1.54994023`，均值为 1。

## 数据、视频与依赖

- 16 个 prep 目录与 ID 文件精确一致；48 个 GT MP4 全部可解码，均为 640×480、29 frames、30 fps。
- 模型输出 MP4 为 16 fps。旧 reward 不按 FPS 重采样，而是截断到共同帧数、将 GT resize 到 prediction 大小并按 frame index 对齐。这是已知时序风险，首轮不改变协议。
- reward 完全本地运行，需要 SAM3、YOLO-World、CoWTracker 及其源码/权重；不依赖远程 reward 服务。
- `AWM_ASSET_ROOT` 是这些资源的单一外部根路径；W&B online 只读取 `WANDB_API_KEY` 环境变量。

## 现有能力位置

- 数据/manifest：`experiments/awm_coca/condition_dataset.py`
- GE-Sim persistent rollout：`experiments/awm_coca/gesim_runtime.py`
- 原模型适配：`experiments/awm_coca/gesim_adapter.py`
- reward 入口：`experiments/awm_coca/reward_runner.py::compute_head_reward`
- optimizer/checkpoint/resume/distributed/eval/log：`experiments/awm_coca/run_train.py` 及相邻模块
- 新 baseline：`experiments/tempflow_video/`

## 未改变的已知风险

- RGB full-frame PSNR 容易由静态背景主导，不能称作真实 3D geometry。
- Action 与 PSNR 可能弱相关；total 提升不能自动推断动作或几何能力提升。
- worst-view 可能显著差于平均视角。
- GT 30 fps 与 prediction 16 fps 仅按 frame index 对齐，可能存在约一帧的语义偏移。
- 同一 `PreparedGeSimCondition` 的首轮与 cache-hit 轮次会改变 pipeline 隐式随机数消费；固定评估必须每组 fresh prepare。生成集必须先完成再启动 reward stack，且 rollout 内部 fork/seed 全局 Torch RNG。未加权的 YOLO plain ATE diagnostics 有微小状态性变化，尽管 reward-driving leaves 稳定。
- 首轮保留实际训练配置权重，不借本次迁移调 reward。
