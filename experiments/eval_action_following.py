"""
Action Following 双指标评估 CLI（SAM Mean IoU + YOLO ATE）。

用法（需要 SAM3 + YOLO-World 权重；SAM 通路为统一文本 prompt 分割）:
    python experiments/eval_action_following.py \
        --gt_videos   head=gesim_video_gen_examples/sample_0_res/head.mp4 \
        --pred_videos head=gesim_video_gen_examples/sample_0_res/head.mp4 \
        --prompt "robot gripper" --confidence 0.1 --max-frames 150

产物（默认写到 experiments/output/）:
    segment_<cam>_gt.mp4 / segment_<cam>_pred.mp4   SAM3 分割可视化
    detect_<cam>_gt.mp4  / detect_<cam>_pred.mp4    YOLO 检测可视化
    metrics.json                                     Mean IoU + ATE 指标
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def _parse_video_map(s: str) -> dict:
    """支持三种格式：
    - 'head=a.mp4,side=b.mp4' → {'head': 'a.mp4', 'side': 'b.mp4'}
    - 'a.mp4,b.mp4'（无 cam= 前缀） → 用文件名主干作为 cam 名
    - '=/path/x.mp4' 或 'cam==/path/x.mp4'（前导 =） → 剥离 '='，cam 缺失时用文件名主干
    """
    out = {}
    for i, item in enumerate(s.split(",")):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            cam, _, path = item.partition("=")
            cam, path = cam.strip(), path.strip()
        else:
            cam, path = "", item
        path = path.lstrip("=").strip()  # 容错前导 '='
        if not cam:
            cam = os.path.splitext(os.path.basename(path))[0] or f"cam{i}"
        out[cam] = path
    return out


def _load_video(path: str):
    """读取视频 → ((T,H,W,3) uint8 BGR, fps)。"""
    if not os.path.exists(path):
        raise ValueError(f"Video file not found: {path}")
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()
    if len(frames) == 0:
        raise ValueError(f"Failed to read video: {path}")
    return np.stack(frames), fps


# 统一文本 prompt：不分 left/right，与 sam_tracking.DEFAULT_EEF_PROMPT 保持一致
DEFAULT_EEF_PROMPT = "robot gripper"

# 评估臂：固定 left + right（YOLO cls=0 / cls=1）
DEFAULT_ARMS = ("left", "right")

# 输出目录：segment / detect videos 与 metrics 均写到 {out_dir}/
_DEFAULT_OUT_DIR = os.path.join(_PROJECT_ROOT, "experiments", "output")


# ---------------------------------------------------------------------------
# 正常模式：全流程（SAM3 文本 prompt 分割通路 + YOLO ATE 通路）
# ---------------------------------------------------------------------------
def _run_full(gt_videos: dict, pred_videos: dict, arms, prompt: str,
              confidence: float, max_frames: int | None, out_dir: str) -> dict:
    from experiments.action_following.metrics_action_following import compute_all
    from experiments.action_following import sam_tracking, yolo_detector

    results = {}
    for cam, gt_path in gt_videos.items():
        pred_path = pred_videos.get(cam)
        if pred_path is None:
            print(f"[SKIP] cam={cam}: 缺少 pred_videos 对应路径")
            continue
        gt, fps_gt = _load_video(gt_path)
        pred, fps_pred = _load_video(pred_path)
        if max_frames is not None:
            gt = gt[:max_frames]
            pred = pred[:max_frames]
        if gt.shape != pred.shape:
            print(f"[WARN] cam={cam}: 分辨率/帧数不一致 {gt.shape} vs {pred.shape}")
        H, W = gt.shape[1], gt.shape[2]

        # 统一 prompt → SAM 分割只跟视频有关、跟 arm 无关：每条视频只分割一次，left/right 复用
        masks = (
            sam_tracking.track_gt(gt, prompt, confidence),
            sam_tracking.track_pred(pred, prompt, confidence),
        )
        # 产物①: SAM3 分割可视化 video（GT / Pred，绿色覆盖 + 黄色轮廓）
        for tag, video, m, fps_v in (("gt", gt, masks[0], fps_gt),
                                     ("pred", pred, masks[1], fps_pred)):
            out_v = os.path.join(out_dir, f"segment_{cam}_{tag}.mp4")
            sam_tracking.write_segment_video(video, m, out_v, fps=fps_v)
            print(f"[SAVE] {out_v}")
        # 产物②: YOLO 检测可视化 video（GT / Pred，显示全部 bbox）
        for tag, video, fps_v in (("gt", gt, fps_gt), ("pred", pred, fps_pred)):
            out_v = os.path.join(out_dir, f"detect_{cam}_{tag}.mp4")
            yolo_detector.write_detection_video(video, out_v, arm="all", conf=0.8, fps=fps_v)
            print(f"[SAVE] {out_v}")

        for arm in arms:
            metrics = compute_all(gt, pred, prompt, arm, H, W,
                                  confidence=confidence, masks=masks)
            results[(cam, arm)] = metrics
    return results


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Action Following 双指标评估 (SAM Mean IoU + YOLO ATE)")
    parser.add_argument("--gt_videos", default="", help="cam=path[,cam=path]")
    parser.add_argument("--pred_videos", default="", help="cam=path[,cam=path]")
    parser.add_argument(
        "--prompt", default=DEFAULT_EEF_PROMPT,
        help=f"统一文本 prompt（不分 left/right），如 'robot gripper'；默认 {DEFAULT_EEF_PROMPT!r}",
    )
    parser.add_argument("--confidence", type=float, default=0.1, help="SAM3 文本检测置信度阈值，0 = 不过滤")
    parser.add_argument("--max-frames", type=int, default=150, help="只取前 N 帧（缓解 CUDA OOM）；0 = 全序列")
    parser.add_argument("--out-dir", default=_DEFAULT_OUT_DIR,
                        help=f"输出目录（segment/detect videos 与 metrics）；默认 {_DEFAULT_OUT_DIR}")
    args = parser.parse_args()

    gt_videos = _parse_video_map(args.gt_videos)
    pred_videos = _parse_video_map(args.pred_videos)
    max_frames = args.max_frames if args.max_frames > 0 else None

    if not gt_videos or not pred_videos:
        print("需要 --gt_videos 与 --pred_videos（cam=path 逗号分隔）")
        sys.exit(1)
    results = _run_full(gt_videos, pred_videos, DEFAULT_ARMS, args.prompt,
                        args.confidence, max_frames, args.out_dir)

    from experiments.action_following.aggregate import reduce

    payload = reduce(results)
    payload["meta"] = {
        "normalized_by": "image_diagonal",
        "prompt": args.prompt,
        "confidence": args.confidence,
        "max_frames": max_frames,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, allow_nan=True)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
"""
python experiments/eval_action_following.py
--gt_videos head=gesim_video_gen_examples/sample_0_res/head.mp4 \
--pred_videos head=gesim_video_gen_examples/sample_0_res/head.mp4 \
--prompt "robot arm" --confidence 0.1 --max-frames 150
"""
