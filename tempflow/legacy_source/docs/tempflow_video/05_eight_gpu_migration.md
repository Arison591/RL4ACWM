# 单节点 8 卡迁移准备

当前机器只有 1 张 A100，因此没有也不会伪装 8 卡验证。`single_node_8gpu.yaml` 与启动脚本是迁移规格；启动脚本会在 GPU 数量不为 8 时拒绝运行，并在 distributed collector 未通过 2-card gate 前保持实验开关关闭。

## 目标拓扑

- world size 8；每 GPU 同一 group 的 1 个 branch，global branch factor 8。
- rank 0 决定 condition、initial seed、branch timestep 和 policy version，并广播。每 rank 以相同 initial seed 重放完全相同 ODE prefix；`branch_id=rank`，noise seed 由 `(base, policy_version, global branch_id)` 唯一派生。
- 每 rank 本地完成一个 suffix 与 terminal reward。只 all-gather 小型 reward/status/branch metadata，不 gather MP4。
- 所有 8 个有效 reward 到齐后，按同一完整 group 计算 mean/std/advantages；绝不能在每卡各自标准化，也不能用 gradient accumulation 假装 group 扩大。
- advantage 广播回各 rank，各 rank只重算本地 collected transition；DDP all-reduce 梯度。因此每次 optimizer update 的 effective policy batch 为一个 8-branch global group，gradient accumulation 初始为 1。
- reference 是每 rank 本地同一冻结 base/disabled adapter，增加每卡约一份 base storage 但不复制第二套参数；KL 每 rank计算本地 action。
- rank 0 独占 checkpoint、effective config、global metrics 和 W&B；每 rank保存自己的 branch MP4 到 rank-specific 目录，避免碰撞。
- checkpoint 必须包含 LoRA、optimizer/scheduler、global step/policy version、consumed group、每 rank RNG 派生规则；所有 rank resume 后 barrier，再生成新 fresh-policy group。

## NCCL 与容错

建议初始环境：`NCCL_ASYNC_ERROR_HANDLING=1`、`TORCH_NCCL_BLOCKING_WAIT=1`，timeout 7200 s。接口名、IB 开关必须按目标节点实际网络审计，不硬编码。任一 rank reward invalid 时先 all-gather status；若全局有效数低于阈值，所有 rank一致跳过 group，不能只有部分 rank backward。

## 尚缺实现

当前单卡 runner 尚未实现 DDP wrapper、global reward/status gather、global advantage broadcast、rank-local artifact 命名和 distributed resume barrier。因此启动脚本无条件拒绝训练并只打印未来的 `torchrun` 命令；在这些项目完成之前，8 卡脚本不是训练成功声明。

## 强制迁移顺序

```text
single-GPU smoke
-> single-GPU overfit16
-> freeze same commit and dataset/reward hashes
-> 2-GPU distributed smoke (branch IDs/reward gather/grad parity)
-> 8-GPU distributed smoke (memory/NCCL/checkpoint-resume)
-> bounded 8-GPU experiment
```

2 卡 gate 需与单卡相同 global group 的 reward ordering、advantages 和一次 update 数值做容差内对照。8 卡 gate 还需验证 rank 0 单写、无文件碰撞、无 collective hang、显存无持续泄漏和 resume 后 seed 不重复。
