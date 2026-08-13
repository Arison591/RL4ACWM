"""
SAM3 video 纯文本 prompt 分割：无首帧 mask，直接以 text prompt 驱动全序列 EEF 分割（GT / Pred 独立推理）。

依赖: SAM3 (third_party/sam3) 的 build_sam3_video_model。
流程: init_state(video) → add_prompt(text_str=prompt) → propagate_in_video → 逐帧 mask
confidence=0: 关闭文本检测框/新目标的置信度过滤（score_threshold_detection、new_det_thresh），
              避免候选被 0.5/0.7 阈值全部滤掉。

用法（在 eval_action_following.py 中由 compute_all 调用）:
    M_gt   = track_gt(gt_video_frames, prompt)   # (T,H,W) bool
    M_pred = track_pred(pred_video_frames, prompt)
    prompt = 文本字符串，如 "left gripper" / "right gripper"
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from threading import Lock

import numpy as np
import torch
from PIL import Image

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)   # 允许以脚本运行时 import experiments.*
_THIRD_PARTY = f"{_PROJECT_ROOT}/third_party"
SAM3_SRC = f"{_THIRD_PARTY}/sam3"
SAM3_CKPT = f"{_PROJECT_ROOT}/checkpoints/sam3.pt"

_sam3_video = None
_sam3_lock = Lock()


@contextmanager
def _rank_local_sam3_environment():
    """Hide the trainer process group while constructing rank-local SAM3.

    SAM3 snapshots ``RANK``/``WORLD_SIZE`` in several constructors (and one
    import-time constant).  Changing model attributes only after construction
    is therefore too late for all internal sizing decisions.  Reward inference
    is deliberately local to each trainer rank, so construct it as rank 0 of a
    private world of size 1, then restore torchrun's environment unchanged.
    """
    overrides = {"RANK": "0", "WORLD_SIZE": "1"}
    previous = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _force_rank_local_inference(model):
    """Keep reward-side SAM3 independent from the trainer process group.

    ``torchrun`` exports RANK/WORLD_SIZE for AWM training.  SAM3 reads those
    variables in its constructors and otherwise assumes that all trainer ranks
    jointly execute one SAM3 inference, issuing collectives from reward worker
    threads.  In this project every rank evaluates its own local rollouts, so
    SAM3 must remain a single-process model on that rank's current CUDA device.
    """
    detector = getattr(model, "detector", None)
    if detector is None or not hasattr(model, "rank") or not hasattr(model, "world_size"):
        raise TypeError("unexpected SAM3 video model: missing distributed inference attributes")
    if not hasattr(detector, "rank") or not hasattr(detector, "world_size"):
        raise TypeError("unexpected SAM3 detector: missing distributed inference attributes")
    model.rank = 0
    model.world_size = 1
    detector.rank = 0
    detector.world_size = 1
    if hasattr(model, "_dist_pg_cpu"):
        model._dist_pg_cpu = None
    return model


def get_sam3_video_model():
    """SAM3 video model 懒加载单例（优先本地 checkpoint）。"""
    global _sam3_video
    if _sam3_video is not None:
        return _sam3_video
    with _sam3_lock:
        if _sam3_video is not None:
            return _sam3_video
        device = "cuda" if torch.cuda.is_available() else "cpu"
        with _rank_local_sam3_environment():
            sys.path.insert(0, SAM3_SRC)
            from sam3.model_builder import build_sam3_video_model

            _sam3_video = build_sam3_video_model(
                device=device, checkpoint_path=SAM3_CKPT, load_from_HF=False
            )
        _force_rank_local_inference(_sam3_video)
        _sam3_video.eval()
    return _sam3_video


DEFAULT_EEF_PROMPT = "robot arm"  # 默认 EEF 文本 prompt


def track_masks(
     video_frames,
    prompt: str = DEFAULT_EEF_PROMPT,
    confidence: float = 0.5,
    max_frames: int | None = None,
) -> np.ndarray:
    """
    纯文本 prompt 的 SAM3 video 分割：不产生首帧 mask，直接以 prompt 驱动全序列分割。

    Args:
        video_frames: (T,H,W,3) uint8 BGR，或视频文件路径 str
        prompt: 文本 prompt（如 "left gripper" / "right gripper"）
        confidence: 文本检测置信度阈值；0 = 不过滤任何候选（sigmoid>0 全保留）

    Returns:
        M: (T,H,W) bool  逐帧 EEF mask
    """
    if isinstance(video_frames, str):
        if max_frames is not None:
            raise ValueError("max_frames 仅支持帧数组；字符串路径请先在调用方截断")
        init_input = video_frames
    else:
        if max_frames is not None:
            video_frames = video_frames[:max_frames]
        init_input = [Image.fromarray(f[..., ::-1]) for f in video_frames]  # BGR→RGB

    model = get_sam3_video_model()
    # confidence=0：关闭文本检测框/新目标的置信度过滤，避免候选被 0.5/0.7 阈值全部滤掉
    model.score_threshold_detection = float(confidence)
    model.new_det_thresh = float(confidence)

    # 1. init_state：接受视频路径或 PIL 帧列表（见 sam3/model/io_utils.py）
    inference_state = model.init_state(init_input)

    # 2. 纯文本 prompt（无首帧 mask / box；add_prompt 的 text_str 直接驱动全序列分割）
    model.add_prompt(inference_state, frame_idx=0, text_str=prompt)

    # 3. propagate → 逐帧 mask
    H = inference_state["orig_height"]
    W = inference_state["orig_width"]
    masks = np.zeros((inference_state["num_frames"], H, W), dtype=bool)
    for frame_idx, out in model.propagate_in_video(inference_state):
        if out is None:
            continue
               # propagate_in_video 产出的是后处理字典（out_obj_ids / out_binary_masks 等），
        # 不是原始 obj_id_to_mask。out_binary_masks: (N,H,W) bool，取或合成单帧 EEF mask
        masks[frame_idx] |= out["out_binary_masks"].any(axis=0)
    return masks


def track_gt( video_gt,
    prompt: str = DEFAULT_EEF_PROMPT,
    confidence: float = 0.0,
    max_frames: int | None = None,
) -> np.ndarray:
    """GT 视频上的文本 prompt 逐帧分割。"""
    return track_masks(video_gt, prompt, confidence, max_frames)


def track_pred(
    video_pred,
    prompt: str = DEFAULT_EEF_PROMPT,
    confidence: float = 0.0,
    max_frames: int | None = None,
) -> np.ndarray:
    """Pred 视频上的文本 prompt 逐帧分割（同一 prompt、同一 checkpoint）。"""
    return track_masks(video_pred, prompt, confidence, max_frames)


def verify_independent_tracking(video_gt, video_pred, prompt: str = DEFAULT_EEF_PROMPT):
    """确保 GT/Pred 独立推理——禁止拼接视频后共同分割。"""
    M_gt = track_masks(video_gt, prompt)
    M_pred = track_masks(video_pred, prompt)
    assert not np.array_equal(M_gt[1:], M_gt[0]), "GT mask 全序列不变 — 检查分割是否生效"
    assert not np.array_equal(M_gt, M_pred), "GT/Pred masks 完全一致 — 检查独立分割设置"
    return M_gt, M_pred


def write_segment_video(video_frames, masks, out_path: str, opacity: float = 0.4,
                        fps: float = 0.0) -> str:
    """把逐帧 mask 覆盖叠加到视频上，写出 segment video（绿色覆盖 + 黄色轮廓）。

    video_frames: (T,H,W,3) uint8 BGR 或 list；masks: (T,H,W) bool；fps<=0 用 30。
    """
    import cv2

    frames = [np.asarray(f) for f in video_frames]
    T = len(frames)
    H, W = frames[0].shape[:2]
    out_fps = fps if fps and fps > 0 else 30.0
    color = np.array([0, 255, 0], dtype=np.float32)               # 绿色覆盖
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (W, H))
    for t in range(T):
        ov = frames[t].copy()
        m = masks[t]
        if m.sum() > 0:
            base = frames[t][m].astype(np.float32)
            ov[m] = ((1.0 - opacity) * base + opacity * color).astype(np.uint8)
            cnts, _ = cv2.findContours(
                m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(ov, cnts, -1, (0, 255, 255), 1)      # 黄色轮廓
        writer.write(ov)
    writer.release()
    return out_path


# ---------------------------------------------------------------------------
# CLI：输入 video → 输出对应的 segment video（SAM3 逐帧 mask 覆盖可视化）
#   python experiments/action_following/sam_tracking.py --video in.mp4 --out seg.mp4
#   python experiments/action_following/sam_tracking.py --video in.mp4 --out seg.mp4 --prompt "left gripper" --confidence 0
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    """输入 video，输出对应的 segment video。"""
    import argparse
    import cv2

    parser = argparse.ArgumentParser(
        description="SAM3 video 文本 prompt 逐帧 EEF 分割 → 可视化 segment video"
    )
    parser.add_argument(
        "--video",
        default=os.path.join(_PROJECT_ROOT, "gesim_video_gen_examples", "sample_0_res", "head.mp4"),
        help="输入视频路径",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(
            _PROJECT_ROOT, "gesim_video_gen_examples", "sample_0_res", "segment_video_sam3.mp4"
        ),
        help="输出 segment 视频路径",
    )
    parser.add_argument("--prompt", default=DEFAULT_EEF_PROMPT, help="文本 prompt（如 'left gripper'）")
    parser.add_argument("--confidence", type=float, default=0.1, help="文本检测置信度阈值，0 = 不过滤")
    parser.add_argument("--opacity", type=float, default=0.4, help="mask 覆盖透明度 (0~1)")
    parser.add_argument("--fps", type=float, default=0.0, help="输出 fps；0 = 跟随输入视频")
    parser.add_argument("--max-frames", type=int, default=150, help="只取前 N 帧（缓解 CUDA OOM）；0 = 全序列")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"[ERROR] 视频不存在: {args.video}")
        sys.exit(1)
    if not 0.0 <= args.opacity <= 1.0:
        print(f"[ERROR] --opacity 需在 0~1: {args.opacity}")
        sys.exit(1)

    # 1. 读输入视频
    cap = cv2.VideoCapture(args.video)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    if not frames:
        print(f"[ERROR] 读不到帧: {args.video}")
        sys.exit(1)
    video = np.stack(frames)
    if args.max_frames and args.max_frames > 0:
        video = video[: args.max_frames]
    T, H, W = video.shape[:3]
    print(f"[INFO] 输入: {T} 帧, {W}x{H}, fps={src_fps:.2f}")

    # 2. 文本 prompt 参数
    print(f"[INFO] 文本 prompt: {args.prompt!r}, confidence={args.confidence}")

    # 3. SAM3 video 文本 prompt 逐帧分割
    print(f"[INFO] 开始 SAM3 video 文本 prompt 分割 … checkpoint 存在={os.path.exists(SAM3_CKPT)}")
    M = track_masks(video, args.prompt, args.confidence)
    print(f"[INFO] 完成: masks shape={M.shape}, 每帧非零像素="
          f"{[int(M[t].sum()) for t in range(T)]}")

    # 4. 可视化 → segment video
    fps = args.fps if args.fps > 0 else src_fps
    write_segment_video(video, M, args.out, opacity=args.opacity, fps=fps)
    print(f"[DONE] segment video 已写出: {args.out} ({T} 帧)")
