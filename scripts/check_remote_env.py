#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


EXPECTED_VERSIONS = {
    "torch": "2.7.1",
    "torchvision": "0.22.1",
    "transformers": "4.51.3",
    "diffusers": "0.32.0",
    "accelerate": "1.0.0",
    "deepspeed": "0.15.3",
    "peft": "0.10.0",
    "xformers": "0.0.31.post1",
    "timm": "1.0.28",
    "ultralytics": "8.4.115",
    "numpy": "2.2.6",
    "opencv-python": "4.10.0.84",
    "huggingface-hub": "0.36.2",
    "modelscope": "1.39.1",
    "modelscope-hub": "0.2.0",
    "wandb": "0.28.1",
}

IMPORT_MODULES = (
    "torch",
    "torchvision",
    "transformers",
    "diffusers",
    "accelerate",
    "deepspeed",
    "peft",
    "xformers",
    "timm",
    "ultralytics",
    "cv2",
    "numpy",
    "scipy",
    "einops",
    "yaml",
    "PIL",
    "wandb",
    "modelscope",
)


def versions_match(actual: str, expected: str) -> bool:
    """接受 PyTorch wheel 的 +cu126 等 PEP 440 local version 后缀。"""
    return actual.split("+", maxsplit=1)[0] == expected


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 AWM-CoCA 远端训练环境")
    parser.add_argument("--model-dir", default=str(REPO_ROOT / "checkpoints"))
    parser.add_argument("--require-gpus", type=int, default=0)
    parser.add_argument("--skip-model-files", action="store_true")
    args = parser.parse_args()

    errors: list[str] = []
    versions = {}
    for distribution, expected in EXPECTED_VERSIONS.items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"缺少 Python 包: {distribution}=={expected}")
            continue
        versions[distribution] = actual
        if not versions_match(actual, expected):
            errors.append(f"版本不一致: {distribution}={actual}, 期望 {expected}")

    for module in IMPORT_MODULES:
        try:
            importlib.import_module(module)
        except Exception as exc:
            errors.append(f"导入失败: {module}: {exc}")

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if cuda_available else 0
        cuda_version = torch.version.cuda
        gpu_names = [torch.cuda.get_device_name(index) for index in range(gpu_count)]
        if args.require_gpus and gpu_count < args.require_gpus:
            errors.append(f"可见 GPU 数量为 {gpu_count}，至少需要 {args.require_gpus}")
        if cuda_available and cuda_version != "12.6":
            errors.append(f"PyTorch CUDA runtime={cuda_version}，期望 12.6")
    except Exception as exc:
        cuda_available = False
        gpu_count = 0
        cuda_version = None
        gpu_names = []
        errors.append(f"CUDA 检查失败: {exc}")

    if shutil.which("ffmpeg") is None:
        errors.append("找不到 ffmpeg")
    if shutil.which("modelscope") is None:
        errors.append("找不到 modelscope CLI")

    source_paths = (
        REPO_ROOT / "third_party" / "sam3" / "sam3" / "model_builder.py",
        REPO_ROOT / "third_party" / "cowtracker" / "cowtracker" / "__init__.py",
    )
    for path in source_paths:
        if not path.is_file():
            errors.append(f"第三方源码缺失: {path}")

    model_dir = Path(args.model_dir).expanduser().resolve()
    if not args.skip_model_files:
        model_files = (
            model_dir / "Cosmos-Predict2-2B-Video2World" / "model_index.json",
            model_dir / "Cosmos-Predict2-2B-Video2World" / "scheduler" / "scheduler_config.json",
            model_dir / "Cosmos-Predict2-2B-Video2World" / "text_encoder" / "model.safetensors.index.json",
            model_dir / "Cosmos-Predict2-2B-Video2World" / "tokenizer" / "tokenizer.json",
            model_dir / "Cosmos-Predict2-2B-Video2World" / "vae" / "diffusion_pytorch_model.safetensors",
            model_dir / "gesim" / "ge_sim_cosmos_v0.1.safetensors",
            model_dir / "sam3.pt",
            model_dir / "cowtracker" / "cowtracker_model.pth",
            model_dir / "yoloworld-EWMBench-v0.1.pt",
        )
        for path in model_files:
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"模型文件缺失或为空: {path}")

    try:
        importlib.import_module("experiments.awm_coca.run_train")
    except Exception as exc:
        errors.append(f"训练入口导入失败: {exc}")

    report = {
        "ok": not errors,
        "python": sys.version.split()[0],
        "cuda_available": cuda_available,
        "cuda_runtime": cuda_version,
        "gpu_count": gpu_count,
        "gpu_names": gpu_names,
        "model_dir": str(model_dir),
        "versions": versions,
        "errors": errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
