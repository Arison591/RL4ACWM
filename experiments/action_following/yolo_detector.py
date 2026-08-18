"""
YOLO-World EEF detector: 逐帧检测 EEF → bbox 中心轨迹（GT / Pred 独立推理）。

依赖: ultralytics + checkpoints/yoloworld.pt（agibot-world/EWMBench-model，AgiBot-World 微调）。
类别: cls=0 左夹爪, cls=1 右夹爪；每帧每类取置信度最高 bbox 的中心点。
实现与 EWMBench/processing/detection_tracking.py 同款选框逻辑。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
from ultralytics import YOLO

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ASSET_ROOT = os.environ.get("AWM_ASSET_ROOT", _PROJECT_ROOT)
_MODEL_ROOT = os.environ.get(
    "AWM_MODEL_ROOT",
    os.environ.get("TEMPFLOW_MODEL_ROOT", _ASSET_ROOT),
)
YOLO_CKPT = os.path.join(_MODEL_ROOT, "yoloworld-EWMBench-v0.1.pt")

_yolo = None


def reset_eef_detector_state() -> None:
    """Reset video-local tracker state without unloading the detector weights."""
    if _yolo is None:
        return
    predictor = getattr(_yolo, "predictor", None)
    for tracker in getattr(predictor, "trackers", ()):
        reset = getattr(tracker, "reset", None)
        if callable(reset):
            reset()


def get_eef_detector(ckpt: str | None = None, device: str | None = None) -> YOLO:
    """YOLO EEF detector 懒加载单例。"""
    global _yolo
    if _yolo is None:
        ckpt = ckpt or YOLO_CKPT
        if device is None:
            device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        _yolo = YOLO(ckpt).to(device)
    return _yolo


def detect_eef_bbox(frame: np.ndarray, arm: str = "left", conf: float = 0.8):
    """
    单帧检测 EEF，返回最佳 bbox 像素坐标 [cx, cy, w, h]（YOLO 中心格式）；未检测到返回 None。
    """
    model = get_eef_detector()
    want_cls = 0 if arm == "left" else 1

    # We select the highest-confidence class box independently per frame;
    # keeping ByteTrack state cannot improve this score and leaks across videos.
    results = model.track(frame, persist=False, conf=conf)
    boxes = results[0].boxes

    best = None  # (xywh, conf)
    if boxes is not None and len(boxes) > 0:
        clses = boxes.cls.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        for i in range(len(boxes)):
            if int(clses[i]) == want_cls and (best is None or confs[i] > best[1]):
                best = (boxes.xywh[i].cpu().tolist(), float(confs[i]))
    return best[0] if best is not None else None


def detect_eef_box_tl(frame: np.ndarray, arm: str = "left", conf: float = 0.8):
    """
    单帧检测 EEF，返回 SAM3 video box prompt 需要的左上角 xywh [x1, y1, w, h]；
    未检测到返回 None。供 sam_tracking（video 分割种子）与 eval 使用。
    """
    box = detect_eef_bbox(frame, arm=arm, conf=conf)
    if box is None:
        return None
    cx, cy, w, h = box
    return [cx - w / 2, cy - h / 2, w, h]


def extract_trajectory(video_frames: np.ndarray, arm: str = "left", conf: float = 0.8):
    """
    在单段视频上逐帧检测 EEF，取 bbox 中心作为轨迹点。

    Args:
        video_frames: (T,H,W,3) uint8 BGR
        arm: "left" | "right"  对应 cls=0 / cls=1
        conf: 检测置信度阈值

    Returns:
        traj: (T,2) float32  [xc, yc]，检测失败帧为 NaN
        valid: (T,) bool
    """
    reset_eef_detector_state()
    centers, valid = [], []
    try:
        for frame in video_frames:
            box = detect_eef_bbox(frame, arm=arm, conf=conf)
            if box is None:
                centers.append((np.nan, np.nan))
                valid.append(False)
            else:
                centers.append((box[0], box[1]))
                valid.append(True)
    finally:
        reset_eef_detector_state()

    return np.array(centers, dtype=np.float32), np.array(valid, dtype=bool)


def track_gt(video_gt: np.ndarray, arm: str = "left"):
    """GT 视频上的轨迹提取。"""
    return extract_trajectory(video_gt, arm=arm)


def track_pred(video_pred: np.ndarray, arm: str = "left"):
    """Pred 视频上的轨迹提取（同一 checkpoint、同一 conf 阈值）。"""
    return extract_trajectory(video_pred, arm=arm)


def write_detection_video(
    video_frames,
    out_path: str,
    arm: str = "all",
    conf: float = 0.8,
    fps: float = 0.0,
) -> int:
    """逐帧 YOLO 检测 → bbox/类别/中心点可视化 → 写出 video，返回命中 bbox 总数。

    video_frames: (T,H,W,3) uint8 BGR 或 list；arm: all/left/right；fps<=0 用 30。
    """
    import cv2

    frames = [np.asarray(f) for f in video_frames]
    T = len(frames)
    H, W = frames[0].shape[:2]
    out_fps = fps if fps and fps > 0 else 30.0

    cls_names = {0: "left_gripper", 1: "right_gripper"}
    cls_colors = {0: (0, 255, 0), 1: (0, 200, 255)}          # BGR: 绿=左, 黄=右
    want_cls = None if arm == "all" else (0 if arm == "left" else 1)

    model = get_eef_detector()
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (W, H))
    n_det = 0
    for frame in frames:
        results = model.track(frame, persist=True, conf=conf)
        boxes = results[0].boxes
        ov = frame.copy()
        if boxes is not None and len(boxes) > 0:
            clses = boxes.cls.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()
            for i in range(len(boxes)):
                c = int(clses[i])
                if want_cls is not None and c != want_cls:
                    continue
                n_det += 1
                color = cls_colors.get(c, (255, 255, 255))
                x1, y1, x2, y2 = map(int, xyxy[i])
                cv2.rectangle(ov, (x1, y1), (x2, y2), color, 2)
                label = f"{cls_names.get(c, c)} {confs[i]:.2f}"
                cv2.putText(ov, label, (x1, max(y1 - 6, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                cv2.circle(ov, (cx, cy), 3, (0, 0, 255), -1)   # 中心点(红)
        writer.write(ov)
    writer.release()
    return n_det


# ---------------------------------------------------------------------------
# CLI：输入 video → 逐帧 YOLO 检测 → 可视化 bbox/类别/中心点 → 输出 video
#   python experiments/action_following/yolo_detector.py --video in.mp4 --out det.mp4
#   python experiments/action_following/yolo_detector.py --video in.mp4 --arm left --conf 0.6
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    """输入 video，输出 YOLO 检测可视化后的 video。"""
    import argparse
    import cv2

    parser = argparse.ArgumentParser(
        description="YOLO EEF detector: 输入 video → 逐帧检测 → bbox/类别/中心点可视化 → 输出 video"
    )
    parser.add_argument("--video", required=True, help="输入视频路径")
    parser.add_argument("--out", default="yolo_detected.mp4", help="输出视频路径")
    parser.add_argument("--conf", type=float, default=0.8, help="检测置信度阈值")
    parser.add_argument("--arm", default="all", help="关注臂: left/right/all（all 显示全部检测框）")
    parser.add_argument("--fps", type=float, default=0.0, help="输出 fps；0 = 跟随输入视频")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"[ERROR] 视频不存在: {args.video}")
        sys.exit(1)

    print(f"YOLO checkpoint: {YOLO_CKPT}  存在={os.path.exists(YOLO_CKPT)}")
    model = get_eef_detector()
    print(f"[DIAG] detector 加载 OK: {type(model).__name__}  设备: {model.device}")

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
    T, H, W = video.shape[:3]
    print(f"[INFO] 输入: {T} 帧, {W}x{H}, fps={src_fps:.2f}")

    # 2. 逐帧检测 → 可视化
    fps = args.fps if args.fps > 0 else src_fps
    n_det = write_detection_video(video, args.out, arm=args.arm, conf=args.conf, fps=fps)
    print(f"[DONE] 检测可视化 video 已写出: {args.out} ({T} 帧, 命中 bbox 总数={n_det})")
