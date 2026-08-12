# AWM-CoCA PSNR 远端训练交付说明

这份说明面向拿到代码后负责在单机 4×A100 上训练的同学。交付不包含 Docker、Conda
环境压缩包、模型权重或数据；环境由 `environment.yml` 创建，模型由脚本按固定 revision
下载。

## 最短启动流程

```bash
git clone https://github.com/Arison591/RL4ACWM.git
cd RL4ACWM
git checkout agent/awm-coca-psnr-4a100

conda env create -f environment.yml
conda activate genie-psnr

bash scripts/download_models.sh

bash scripts/train_remote.sh
```

默认约定：

- 数据目录：目标训练机默认自动识别 `/hpc2hdd/home/bohantan/jhupload/hr_data`，其他机器回退到仓库根目录的 `dataset/`；
- 模型目录：仓库根目录下的 `checkpoints/`；
- 输出目录：每次新训练自动创建独立的 `awm_coca_outputs/runs/<timestamp_pid>/`；其他机器没有外部数据盘时使用 `outputs/awm_coca_remote/runs/<timestamp_pid>/`；
- GPU：`CUDA_VISIBLE_DEVICES=0,1,2,3`；
- 全局 `group_size=16`，每张 GPU 负责 4 条 rollout；
- reward：50% Action + 50% 三视角 PSNR geometry；
- 默认保留三视角 MP4、reward、credit 和元数据，训练消费后删除大体积中间张量。
- 默认尝试把训练曲线上传到 W&B 项目 `awm-coca`，失败时自动继续使用本地日志。

## 1. 机器要求

- Linux x86_64；
- 4 张 NVIDIA A100，建议每张 80GB；
- NVIDIA 驱动能够运行 CUDA 12.6 PyTorch wheel；
- 建议至少预留 30GB 模型下载空间；
- 若保留全部 rollout，训练输出可能达到 TB 级，请预留足够数据盘；
- Conda、Git、Git LFS 和可访问 ModelScope、GitHub 的网络。

环境基于当前真实跑通版本整理，没有直接导出开发机上的整个 Conda 环境。关键版本为：

```text
Python        3.10
PyTorch       2.7.1 + CUDA 12.6
torchvision   0.22.1
xformers      0.0.31.post1
transformers  4.51.3
diffusers     0.32.0
accelerate    1.0.0
deepspeed     0.15.3
peft          0.10.0
```

创建环境：

```bash
conda env create -f environment.yml
conda activate genie-psnr
```

如果之前已经创建过 `genie-psnr` 环境，拉取本次更新后执行：

```bash
conda env update -n genie-psnr -f environment.yml --prune
conda activate genie-psnr
```

依赖统一由 `environment.yml` 提供（仓库已移除 `requirements.txt`），请勿再单独执行 `pip install -r requirements.txt` 或 `pip install` 其他依赖文件，以免覆盖已锁定的版本。

## 2. W&B 实时训练日志

W&B 默认以 online 模式启用。交付代码已按要求内置登录凭据，并固定上传到
`hrqian06-huazhong-university-of-science-and-technology/awm-coca`。训练执行者不需要运行
`wandb login`，也不需要手动设置 `WANDB_API_KEY`、`WANDB_ENTITY` 或 `WANDB_PROJECT`。
如需轮换凭据或切换项目，仍可使用同名环境变量覆盖。

训练启动后，控制台会打印 run URL，并写入：

```text
/hpc2hdd/home/bohantan/jhupload/hr_data/awm_coca_outputs/runs/<timestamp_pid>/logs/wandb_run.txt
```

把其中的 `url=` 发回来，即可在网页实时查看：loss、learning rate、gradient norm、
Action/Geometry/joint reward、三视角 PSNR、advantage、CoCA 采样指标和 GPU 系统状态。
项目若是私有的，查看者仍需加入对应 W&B team；只有 URL 并不会绕过项目权限。

默认第 1 step 以及之后每 50 step 上传 1 条 rollout 的三视角视频。可调整：

```bash
WANDB_VIDEO_EVERY=100 WANDB_VIDEO_SAMPLES=1 bash scripts/train_remote.sh
```

W&B 初始化或上传异常会被捕获，训练继续写本地日志，不会因监控服务失败而停训。完全
关闭远端监控时：

```bash
WANDB_MODE=disabled bash scripts/train_remote.sh
```

断网环境也可设置 `WANDB_MODE=offline`，之后在训练机器上手动同步。checkpoint 不上传
W&B，训练完成后仍按第 9 节打包回传。

## 3. 数据目录

目标训练机的数据根目录为：

```text
/hpc2hdd/home/bohantan/jhupload/hr_data
```

代码位于空间较小的 `workspace`，数据保留在 `jhupload/hr_data`，不需要复制或软链接到
代码仓库。直接执行 `bash scripts/train_remote.sh` 时，脚本会优先自动识别该目录，并将
checkpoint、rollout、日志等训练输出写入 `hr_data/awm_coca_outputs/runs/<timestamp_pid>/`，
避免占满 workspace，也避免失败后重启时与旧 rollout 的 policy/seed 目录冲突。

数据集内部推荐结构：

```text
dataset/
  prep/
    <condition_id>/
      actions.npy
      extrinsic_head.npy
      extrinsic_hand_left.npy
      extrinsic_hand_right.npy
      intrinsic_head.npy
      intrinsic_hand_left.npy
      intrinsic_hand_right.npy
      head_color/{0,1,2,3}.png
      hand_left_color/{0,1,2,3}.png
      hand_right_color/{0,1,2,3}.png

  selected_samples/samples/
    <condition_id>/
      head_29_frames.mp4
      hand_left_29_frames.mp4
      hand_right_29_frames.mp4
```

`train_remote.sh` 也会自动识别以下兼容结构：

- `dataset/output/prep/`；
- `dataset/data/agibotworld_beta/selected_samples/samples/`；
- condition 同时直接放在 `dataset/<condition_id>/`。
- 数据根目录外面或里面额外套一层 `dataset/`，例如
  `/hpc2hdd/home/bohantan/jhupload/hr_data/dataset/prep/`。

目标训练机直接运行即可，不用设置 `DATA_DIR`：

```bash
bash scripts/train_remote.sh
```

如果迁移到其他机器且实际路径不同，也不用改代码：

```bash
DATA_DIR=/data/dataset bash scripts/train_remote.sh
```

也可以分别覆盖：

```bash
PREP_DIR=/data/prep \
GT_DIR=/data/selected_samples/samples \
bash scripts/train_remote.sh
```

## 4. 下载模型

执行：

```bash
bash scripts/download_models.sh
```

脚本会下载并校验：

| 资产 | 固定仓库/revision |
| --- | --- |
| Cosmos text encoder、tokenizer、VAE、scheduler | ModelScope `nv-community/Cosmos-Predict2-2B-Video2World@efdf2314d7edb0f9bee14e9753462f8e2e0ff075` |
| GE-Sim Cosmos | ModelScope `agibot_world/Genie-Envisioner@1422f1783e5eed8e00925d3ce9ba3a0ba59e84df` |
| SAM3 权重 | ModelScope `facebook/sam3@96f3e1b404ba14f2cfac60ee6ae87c269a7b7923` |
| YOLO-World EEF 权重 | ModelScope `agibot-world/EWMBench-model@6ec404f1f68d362a9b625f570040c200370077f6` |
| CoWTracker 权重 | ModelScope `facebook/cowtracker@f4633e5671c5f19ea8500943869f4c975b605fde` |
| SAM3 源码 | GitHub commit `6dbb02bd38288df755dfa1378000a861e65b84f6` |
| CoWTracker 源码 | GitHub commit `1454f20045d3b514e5b8417907152677f3dba621` |

大文件会执行 SHA256 校验，重复运行会跳过已正确下载的文件。脚本还会自动应用当前
PyTorch/timm/xformers 所需的 SAM3 和 CoWTracker 兼容补丁。
所有模型权重均从 ModelScope 的固定 revision 匿名下载，不需要 Hugging Face 登录或 token；
SAM3 和 CoWTracker 的第三方源码仍从 GitHub 固定 commit 拉取。

如果模型希望放在数据盘：

```bash
MODEL_DIR=/data/models/genie bash scripts/download_models.sh
```

训练时使用同一个目录：

```bash
MODEL_DIR=/data/models/genie bash scripts/train_remote.sh
```

CoWTracker 权重采用 CC-BY-NC-4.0，运行下载脚本前请确认训练用途符合其许可证。

## 5. 训练前检查

`train_remote.sh` 会自动完成以下检查：

1. Python 包及关键版本；
2. 4 张可见 GPU 和 CUDA runtime；
3. 全部模型文件与第三方源码；
4. 每个 condition 的动作、相机参数和 4 帧历史图像；
5. Action/geometry reward 所需的三个 GT 视频；
6. GE-Sim 配置与 15 步 reverse denoise 设置。

多卡训练中的 reward 是每个 rank 独立推理：SAM3 在本 rank 当前 GPU 上强制使用单进程
模式，不参与训练进程组的 collective，避免异步 reward 与训练 all-reduce 次序冲突。

也可以手动执行：

```bash
python scripts/check_remote_env.py --model-dir checkpoints --require-gpus 4

CUDA_VISIBLE_DEVICES=0 scripts/run_awm_coca.sh preflight \
  --prep-root dataset/prep \
  --gt-root dataset/selected_samples/samples \
  --checkpoint-root checkpoints
```

## 6. 启动四卡训练

默认一条命令：

```bash
bash scripts/train_remote.sh
```

指定三个主要目录时：

```bash
DATA_DIR=/data/dataset \
MODEL_DIR=/data/models/genie \
OUTPUT_DIR=/data/train_outputs/awm_coca \
bash scripts/train_remote.sh
```

这里显式设置 `OUTPUT_DIR` 时，该目录本身就是本次 run 的目录；新训练不要复用已有
rollout 的目录。一般直接使用默认自动创建的时间戳目录最省事。

训练脚本默认使用 `ROLLOUT_RETENTION=videos`。每条 seed 保留 head、left-hand、right-hand
三个 MP4，以及 `reward.json`、`credit.json`、`rollout.json`；完成梯度更新后自动删除：

- `condition.pt`；
- `trajectory.pt`；
- `final_future_latent.pt`。

如需排查训练中间张量，可以临时全保留：

```bash
ROLLOUT_RETENTION=all bash scripts/train_remote.sh
```

若完全不需要逐条视频，在完整 reward 和精简 credit 写进 `metrics/rollouts.jsonl` 后删除
整个 group：

```bash
ROLLOUT_RETENTION=none bash scripts/train_remote.sh
```

## 7. 输出、日志和 checkpoint

默认训练长度是 `1000` 次 optimizer update，不是整数 epoch。当前数据预检得到 203 个
condition，且梯度累积为 1，所以 1 epoch = 203 次 update，全部训练约为 4.93 epochs。
命令行会显示仅 rank 0 输出的总进度条，其中包含 `step/1000`、epoch、condition、当前
阶段（rollout/reward/update）、loss、reward、单步耗时以及动态 ETA，不会由四张卡重复刷屏。

目标训练机默认输出（其他机器以启动时打印的 `output_dir` 为准）：

```text
/hpc2hdd/home/bohantan/jhupload/hr_data/awm_coca_outputs/
  runs/
    <timestamp_pid>/                   # 每次启动的新训练互不覆盖
      logs/
        train_<timestamp_pid>.log      # 完整 stdout/stderr
        run_<timestamp_pid>.txt        # Git、GPU、数据和路径信息
        effective_config_<timestamp_pid>.yaml
        wandb_run.txt                  # W&B 状态、run ID 和网页 URL
      metrics/
        train.jsonl                    # 每个 optimizer step 的 loss/grad/sample 指标
        rollouts.jsonl                 # 每条 rollout 的 reward 和 CoCA 摘要
      checkpoints/
        checkpoint_100/
        checkpoint_200/
        ...
      rollouts/                        # 默认仅保留 MP4、reward、credit 和元数据
```

实时查看：

```bash
latest_run="$(find /hpc2hdd/home/bohantan/jhupload/hr_data/awm_coca_outputs/runs -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
tail -f "${latest_run}"/logs/train_*.log
tail -f "${latest_run}"/metrics/train.jsonl
nvidia-smi
watch -n 30 df -h
```

Checkpoint 默认每 100 个 optimizer step 保存一次，包含：

- LoRA policy；
- optimizer 和 LR scheduler；
- EMA；
- sampler、policy version 和 RNG 状态；
- 完整训练配置。

## 8. 断点续训

使用同一个 run 的 `OUTPUT_DIR`，指定完整 checkpoint：

```bash
OUTPUT_DIR=/data/train_outputs/awm_coca/runs/20260812_205500_12345 \
RESUME_CHECKPOINT=/data/train_outputs/awm_coca/runs/20260812_205500_12345/checkpoints/checkpoint_500 \
bash scripts/train_remote.sh
```

脚本会恢复 LoRA、optimizer、scheduler、EMA、采样位置和随机数状态。不要只复制
`policy_lora/` 后声称是完整断点续训。

## 9. 训练完成后回传

不需要回传 TB 级 rollout。执行：

```bash
latest_run="$(find /hpc2hdd/home/bohantan/jhupload/hr_data/awm_coca_outputs/runs -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
bash scripts/package_training_results.sh "${latest_run}"
```

脚本选择最新带 `COMPLETE` 标记的 checkpoint，并将以下内容打包：

- 最新完整 checkpoint；
- `logs/`；
- `metrics/`；
- 数据 manifest。

生成：

```text
<latest_run>/transfer/awm_coca_checkpoint_<step>_<timestamp>.tar.gz
<latest_run>/transfer/awm_coca_checkpoint_<step>_<timestamp>.tar.gz.sha256
```

请将这两个文件一起传回。收到后先运行：

```bash
sha256sum -c awm_coca_checkpoint_<step>_<timestamp>.tar.gz.sha256
```

## 10. 常见问题

### `hf` 命令不存在

说明没有激活正确环境：

```bash
conda activate genie-psnr
```

### 只识别到一张 GPU

检查：

```bash
echo "$CUDA_VISIBLE_DEVICES"
nvidia-smi -L
```

训练必须看到四张卡，默认值为 `0,1,2,3`。

### 数据预检提示缺少 GT

Joint reward 强制要求同一个 condition 同时存在：

```text
head_29_frames.mp4
hand_left_29_frames.mp4
hand_right_29_frames.mp4
```

### 磁盘增长过快

默认不会保留大体积中间张量。如果仍然增长过快，可以连逐条 MP4 一起关闭：

```bash
ROLLOUT_RETENTION=none bash scripts/train_remote.sh
```

不要在训练运行时手工删除正在写入的 group。
