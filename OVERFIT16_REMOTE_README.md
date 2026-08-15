# AWM-CoCA 16-condition overfit（4/8 卡）

该流程用于先验证 AWM-CoCA 能否在固定的 16 个 condition 上明显过拟合。训练集和周期评估集
是同一批 condition；评估使用固定 seed，因此曲线变化主要反映 policy 更新。

## 1. 拉取固定分支

已有仓库：

```bash
git fetch origin \
  '+refs/heads/agent/awm-coca-psnr-4a100:refs/remotes/origin/agent/awm-coca-psnr-4a100'
git switch agent/awm-coca-psnr-4a100
git merge --ff-only origin/agent/awm-coca-psnr-4a100
```

## 2. 放置 16 条数据

下载 `awm_coca_overfit16.tar.gz` 和同名 `.sha256` 后：

```bash
sha256sum -c awm_coca_overfit16.tar.gz.sha256
tar -xzf awm_coca_overfit16.tar.gz \
  -C /hpc2hdd/home/bohantan/jhupload/hr_data
```

最终目录必须是：

```text
/hpc2hdd/home/bohantan/jhupload/hr_data/
  awm_coca_overfit16/
    condition_ids.txt
    prep/<16 个 condition>/...
    selected_samples/samples/<16 个 condition>/...
  awm_coca_models/
    Cosmos-Predict2-2B-Video2World/...
    gesim/ge_sim_cosmos_v0.1.safetensors
    sam3.pt
    cowtracker/cowtracker_model.pth
    yoloworld-EWMBench-v0.1.pt
```

如果模型还没有下载：

```bash
MODEL_DIR=/hpc2hdd/home/bohantan/jhupload/hr_data/awm_coca_models \
  bash scripts/download_models.sh
```

## 3. 环境

```bash
conda env update -n genie-psnr -f environment.yml --prune
conda activate genie-psnr
```

首次创建使用 `conda env create -f environment.yml`。

## 4. 检查最终命令

四卡：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
OVERFIT_DRY_RUN=1 \
bash scripts/train_overfit16_remote.sh
```

八卡：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
OVERFIT_DRY_RUN=1 \
bash scripts/train_overfit16_remote.sh
```

全局 group 固定为 16：四卡时每卡 4 条 rollout，八卡时每卡 2 条。评估在 group 0
先记录基线，之后每 10 个 group 运行一次；每个 condition 使用 8 个固定 seed，按 batch=2
推理，避免 8 条同时生成造成显存峰值。

## 5. 先跑 1-step 联调

四卡：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
MAX_OPTIMIZER_STEPS=1 \
OUTPUT_DIR=/hpc2hdd/home/bohantan/jhupload/hr_data/awm_coca_overfit16_outputs/check_1step \
bash scripts/train_overfit16_remote.sh
```

八卡只需把 GPU 列表和 `NPROC_PER_NODE` 改成 8。该模式不同于旧 `smoke`：它会真的执行
group-0 全量评估，从而覆盖此次新增的 train/eval 同集机制。

成功后检查：

```text
<OUTPUT_DIR>/metrics/eval.jsonl
<OUTPUT_DIR>/metrics/train.jsonl
<OUTPUT_DIR>/logs/effective_config_*.yaml
```

## 6. 正式 overfit

1-step 通过后必须换一个新的输出目录：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
MAX_OPTIMIZER_STEPS=1000 \
OUTPUT_DIR=/hpc2hdd/home/bohantan/jhupload/hr_data/awm_coca_overfit16_outputs/overfit_1000 \
WANDB_MODE=online \
bash scripts/train_overfit16_remote.sh
```

重点观察 W&B：

- `eval/reward_total`、`eval/reward_action`、`eval/reward_geometry`；
- `eval/valid_fraction`；
- `trainer/fm_grad_norm`、`trainer/kl_grad_norm`；
- `trainer/loss`、`trainer/grad_norm`。

## 可配置项

均通过环境变量覆盖：

- `DATA_ROOT`：默认 `/hpc2hdd/home/bohantan/jhupload/hr_data`；
- `OVERFIT16_DATA_DIR`：默认 `$DATA_ROOT/awm_coca_overfit16`；
- `MODEL_DIR`：默认 `$DATA_ROOT/awm_coca_models`；
- `OUTPUT_DIR`：默认自动创建带时间戳的独立 run；
- `NPROC_PER_NODE`：4 或 8；未设置时按可见 GPU 数自动检测；
- `GROUP_SIZE`：默认 16；
- `MAX_OPTIMIZER_STEPS`：默认 1000；
- `EVAL_EVERY_GROUP_STEPS`：默认 10；
- `EVAL_SEEDS_PER_CONDITION`：默认 8；
- `EVAL_ROLLOUT_BATCH_SIZE`：默认 2；
- `EVAL_SEED`：默认 12345678；
- `ROLLOUT_RETENTION`：默认 `videos`。

数据打包维护者可执行：

```bash
PREP_SOURCE=/path/to/full/prep \
GT_SOURCE=/path/to/full/selected_samples/samples \
bash scripts/package_awm_coca_overfit16.sh /path/to/awm_coca_overfit16.tar.gz
```
