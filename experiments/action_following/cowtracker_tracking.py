"""
CoWTracker 前景内密集点跟踪封装（对齐 sam_tracking.py 的懒加载单例模式）。

依赖: third_party/cowtracker 源码（pinned rev）+ checkpoints/cowtracker/cowtracker_model.pth 权重。
安装见 scripts/setup_cowtracker.sh（独立脚本，不依赖 CD-LAM 的 fetch_optional_deps.sh）。

API:
    get_cowtracker() -> model              # 懒加载单例；显式本地权重路径（仿 SAM3_CKPT 模式）
    track_dense(video, max_frames=None)    # 全帧密集轨迹 -> DenseTracks
    track_points(video, mask, k, seed)     # 前景 mask 锚点采样 -> (T,N,2) + (T,N) bool

与 sam_tracking.py 相同：视频为 (T,H,W,3) uint8 BGR；本模块在送入模型前 BGR→RGB。
CoWTracker forward 接受 [B,S,3,H,W] 或 [S,3,H,W] 的 float 张量，值域 [0,255]（模型内部 /255）；
输出 track[t,y,x]=(x,y) 为输入分辨率下的绝对像素坐标。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)   # 允许以脚本运行时 import experiments.*
_ASSET_ROOT = os.environ.get("AWM_ASSET_ROOT", _PROJECT_ROOT)
_MODEL_ROOT = os.environ.get(
    "AWM_MODEL_ROOT",
    os.environ.get("TEMPFLOW_MODEL_ROOT", _ASSET_ROOT),
)
_THIRD_PARTY = os.path.join(_ASSET_ROOT, "third_party")
# Assets are normally mounted outside the source checkout on the training host.
COWTRACKER_CKPT = os.path.join(_MODEL_ROOT, "cowtracker", "cowtracker_model.pth")
COWTRACKER_SRC = os.path.join(_THIRD_PARTY, "cowtracker")

MAX_FRAMES = 256        # CoWTracker 单次推理的帧数上限
VISIBILITY_THRESHOLD = 0.5   # 置信度型 visibility 的阈值（协议默认，见 metrics_fdce._as_visibility）
PATCH_ALIGN = 112        # VGGT backbone 的 DINOv2 patch 尺寸，输入 H/W需为其倍数
_cowtracker = None


@dataclass
class DenseTracks:
    """全帧密集点跟踪结果。

    tracks:     (T,H,W,2) float32，track[t,y,x]=(x,y) 绝对像素坐标
    visibility: (T,H,W) bool，vis >= VISIBILITY_THRESHOLD
    confidence: (T,H,W) float32，原始置信度（诊断用）
    """

    tracks: np.ndarray
    visibility: np.ndarray
    confidence: np.ndarray


def get_cowtracker():
    """CoWTracker 懒加载单例（显式本地 checkpoint 路径，避免隐式联网下载）。"""
    global _cowtracker
    if _cowtracker is not None:
        return _cowtracker
    if not os.path.isdir(COWTRACKER_SRC):
        raise FileNotFoundError(
            f"CoWTracker 源码不存在: {COWTRACKER_SRC}。请先运行 scripts/setup_cowtracker.sh"
        )
    if not os.path.exists(COWTRACKER_CKPT):
        raise FileNotFoundError(
            f"CoWTracker 权重不存在: {COWTRACKER_CKPT}。请先运行 scripts/setup_cowtracker.sh"
        )

    sys.path.insert(0, COWTRACKER_SRC)
    from cowtracker import CoWTracker   # noqa: PLC0415  (懒加载，仅 GPU 环境)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    _cowtracker = CoWTracker.from_checkpoint(
        checkpoint_path=COWTRACKER_CKPT, device=device, dtype=dtype
    )
    return _cowtracker


def cowtracker_ready() -> bool:
    """轻量就绪检查（只查文件存在性，不加载模型）。"""
    return os.path.isdir(COWTRACKER_SRC) and os.path.exists(COWTRACKER_CKPT)


def track_dense(
    video_frames,
    max_frames: Optional[int] = None,
) -> DenseTracks:
    """在整段视频上运行 CoWTracker 密集点跟踪。

    Args:
        video_frames: (T,H,W,3) uint8 BGR（cv2 读取结果），或可 np.asarray 的等价数组
        max_frames: 只取前 N 帧；None = 全序列（上限 MAX_FRAMES）

    Returns:
        DenseTracks：全帧密集轨迹 + 可见性 + 置信度
    """
    frames = np.asarray(video_frames)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"video_frames 需为 (T,H,W,3)，收到 {frames.shape}")
    if not np.issubdtype(frames.dtype, np.integer) and frames.max() <= 1.0:
        raise ValueError("video_frames 需为 uint8（值域 [0,255]），收到归一化浮点")
    if max_frames is not None:
        frames = frames[:max_frames]
    T, H, W = frames.shape[:3]
    if T > MAX_FRAMES:
        raise ValueError(f"CoWTracker 单次最多 {MAX_FRAMES} 帧，收到 {T} 帧；请截断")
    pad_h = (PATCH_ALIGN - H % PATCH_ALIGN) % PATCH_ALIGN
    pad_w = (PATCH_ALIGN - W % PATCH_ALIGN) % PATCH_ALIGN
    model = get_cowtracker()
    device = next(model.parameters()).device

    rgb = np.ascontiguousarray(frames[..., ::-1])          # BGR→RGB
    tensor = torch.from_numpy(rgb).permute(0, 3, 1, 2).float().to(device)
    if pad_h or pad_w:
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h))

    with torch.no_grad():
        if device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                pred = model.forward(video=tensor, queries=None)
        else:
            pred = model.forward(video=tensor, queries=None)

    track = pred["track"][0].detach().cpu().numpy()   # (T,H+pad_h,W+pad_w,2)
    vis = pred["vis"][0].detach().cpu().numpy()       # (T,H+pad_h,W+pad_w)
    conf = pred["conf"][0].detach().cpu().numpy()     # (T,H+pad_h,W+pad_w)
    if pad_h or pad_w:
        track = track[:, :H, :W]
        vis = vis[:, :H, :W]
        conf = conf[:, :H, :W]
    return DenseTracks(
        tracks=track,
        visibility=vis >= VISIBILITY_THRESHOLD,
        confidence=conf,
    )


def track_points(
    video_frames,
    mask,
    k: int = 16,
    seed: int = 0,
    max_frames: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """前景内锚点点跟踪：mask 腐蚀 → 采样 ≤k 个锚点 → 从密集轨迹抽取 (T,N,2)。

    Args:
        video_frames: (T,H,W,3) uint8 BGR
        mask: (T,H,W) bool 前景 mask（来自 sam_tracking，仅有效区域布点）
        k: 期望锚点数（协议上限 16）
        seed: 锚点采样随机种子（同视频 GT/Pred 使用相同种子策略）

    Returns:
        (tracks (T,N,2) float32, visibility (T,N) bool)
    """
    from .fdce_tracks import sample_tracks   # noqa: PLC0415  (纯 numpy，懒加载)

    dense = track_dense(video_frames, max_frames=max_frames)
    return sample_tracks(dense, mask, k=k, seed=seed)


# ---------------------------------------------------------------------------
# 冒烟测试：GPU 目标环境用（本地无 torch 时只打印环境状态）
#   python experiments/action_following/cowtracker_tracking.py --video in.mp4 --out tracks.npz
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CoWTracker 密集点跟踪冒烟测试")
    parser.add_argument("--video", help="输入视频路径（(T,H,W,3) uint8）")
    parser.add_argument("--out", default=None, help="输出 NPZ 路径（dense tracks）")
    parser.add_argument("--max-frames", type=int, default=None, help="只取前 N 帧")
    args = parser.parse_args()

    print(f"[INFO] torch={torch.__version__} cuda={torch.cuda.is_available()}")
    print(f"[INFO] 源码存在={os.path.isdir(COWTRACKER_SRC)} 权重存在={os.path.exists(COWTRACKER_CKPT)}")

    if not args.video:
        print("[INFO] 未提供 --video，跳过实际推理（仅环境检查）")
        sys.exit(0)
    if not cowtracker_ready():
        print(f"[ERROR] CoWTracker 未就绪。请先运行 scripts/setup_cowtracker.sh")
        sys.exit(1)

    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(args.video)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        print(f"[ERROR] 读不到帧: {args.video}")
        sys.exit(1)
    video = np.stack(frames)
    print(f"[INFO] 输入: {video.shape}, dtype={video.dtype}")

    dense = track_dense(video, max_frames=args.max_frames)
    print(f"[INFO] dense tracks: {dense.tracks.shape}, visibility: {dense.visibility.shape}")
    print(f"[INFO] 可见像素比例: {dense.visibility.mean():.3f}")

    if args.out:
        import os as _os

        os.makedirs(_os.path.dirname(_os.path.abspath(args.out)), exist_ok=True)
        np.savez(args.out, tracks=dense.tracks, visibility=dense.visibility, confidence=dense.confidence)
        print(f"[DONE] 已写出: {args.out}")
