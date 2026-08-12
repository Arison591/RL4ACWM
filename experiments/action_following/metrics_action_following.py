"""
Action Following 双指标纯函数：SAM Mean IoU + YOLO ATE。

两条通路共用范式：共同首帧初始化 → 各自逐帧提取 → 同时刻直接比较。
本模块不依赖任何模型，输入即 (mask / 轨迹) 时序数据。
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Mask 通路 —— 核心: Mean IoU（v2 §2.4.1）
# ---------------------------------------------------------------------------
def mask_metrics(M_gt: np.ndarray, M_pred: np.ndarray, skip_t0: bool = True) -> dict:
    """
    M_gt, M_pred: (T,H,W) bool  逐帧 EEF mask

    Returns:
        mean_iou:           [核心] 逐帧 IoU 的时序均值（排除 t=0）
        iou_per_frame:      (T-1,) 逐帧 IoU（诊断用）
        mask_coverage:      GT 存在时 Pred 也非空的帧占比（v2 §2.4.3）
        mask_disappearance: GT 存在但 Pred 消失的帧占比（v2 §2.4.4）
    """
    T = M_gt.shape[0]
    t_start = 1 if skip_t0 else 0

    ious = []
    for t in range(t_start, T):
        inter = (M_gt[t] & M_pred[t]).sum()
        union = (M_gt[t] | M_pred[t]).sum()
        ious.append(float(inter / union) if union > 0 else 0.0)
    ious = np.array(ious)

    gt_nonempty = np.array([M_gt[t].sum() > 0 for t in range(t_start, T)])
    pred_nonempty = np.array([M_pred[t].sum() > 0 for t in range(t_start, T)])
    denom = gt_nonempty.sum()

    return {
        "mean_iou": float(ious.mean()) if ious.size else np.nan,
        "iou_per_frame": ious.tolist(),
        "mask_coverage": float((gt_nonempty & pred_nonempty).sum() / denom) if denom > 0 else np.nan,
        "mask_disappearance": float((gt_nonempty & ~pred_nonempty).sum() / denom) if denom > 0 else np.nan,
    }


# ---------------------------------------------------------------------------
# 轨迹通路 —— 核心: ATE（v2 §3.1/§3.2/§3.5）
# ---------------------------------------------------------------------------
def trajectory_metrics(
    traj_gt: np.ndarray,
    traj_pred: np.ndarray,
    diag: float | None = None,
    skip_t0: bool = True,
) -> dict:
    """
    traj_gt, traj_pred: (T,2) float  像素坐标，检测失败为 NaN
    diag: sqrt(H^2+W^2) 或 d_eef0；None = 不归一化（raw 像素）

    Returns:
        ate:          [核心] 双方均检测到帧上的平均欧氏距离 (px)
        ate_p95:      误差 95 分位（诊断用）
        ate_norm:     对角线归一化（若 diag 给定）
        det_coverage: 双方共同有效帧占比（必须并列报告，v2 §3.5）
        joint_frames: 共同有效帧数
        total_frames: 评估总帧数（排除首帧后）
    """
    T = traj_gt.shape[0]
    t_start = 1 if skip_t0 else 0

    valid = ~np.isnan(traj_gt[:, 0]) & ~np.isnan(traj_pred[:, 0])
    valid[:t_start] = False

    det_coverage = float(valid.sum() / max(T - t_start, 1))
    if valid.sum() == 0:
        return {
            "ate": np.nan, "ate_p95": np.nan, "ate_norm": np.nan,
            "det_coverage": det_coverage, "joint_frames": 0, "total_frames": T - t_start,
        }

    errors = np.linalg.norm(traj_pred[valid] - traj_gt[valid], axis=1)
    ate = float(errors.mean())

    result = {
        "ate": ate,
        "ate_p95": float(np.percentile(errors, 95)),
        "det_coverage": det_coverage,
        "joint_frames": int(valid.sum()),
        "total_frames": T - t_start,
    }
    if diag is not None:
        result["ate_norm"] = ate / diag
    return result


# ---------------------------------------------------------------------------
# 统一入口: 单 cam × 单 arm 两条通路全流程
# ---------------------------------------------------------------------------
def compute_all(
    gt_video: np.ndarray,
    pred_video: np.ndarray,
    prompt: str,
    arm: str,
    H: int,
    W: int,
    d_eef0: float | None = None,
    confidence: float = 0.0,
    masks: tuple | None = None,
) -> dict:
    """
    gt_video, pred_video: (T,H,W,3) uint8 BGR
    prompt: 文本 prompt（如 "robot gripper"），SAM3 video 直接以其分割
    arm: "left" | "right"
    H, W: int
    d_eef0: float | None  覆盖 EEF 对角线归一化（缺省用图像对角线）
    confidence: 文本检测置信度阈值；0 = 不过滤候选
    masks: 可选 (M_gt, M_pred) 预计算逐帧 mask，跳过 SAM 分割。
           统一 prompt 下分割结果与 arm 无关，多臂评估应每视频只算一次再复用。

    Returns: dict，含 mean_iou / mask_coverage / mask_disappearance /
             ate / ate_p95 / ate_norm / det_coverage
    """
    from . import sam_tracking, yolo_detector

    out = {}

    # ── SAM 通路（纯文本 prompt 分割，无首帧 mask）──
    if masks is None:
        M_gt = sam_tracking.track_gt(gt_video, prompt, confidence)
        M_pred = sam_tracking.track_pred(pred_video, prompt, confidence)
    else:
        M_gt, M_pred = masks
    out.update(mask_metrics(M_gt, M_pred))

    # ── YOLO 通路 ──
    traj_gt, _ = yolo_detector.track_gt(gt_video, arm)
    traj_pred, _ = yolo_detector.track_pred(pred_video, arm)
    diag = d_eef0 or float(np.sqrt(H * H + W * W))
    out.update(trajectory_metrics(traj_gt, traj_pred, diag=diag))

    return out


# ---------------------------------------------------------------------------
# 自测入口：合成数据驱动纯函数，打印 期望 vs 实际
#   python experiments/action_following/metrics_action_following.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    def _check(name, got, expect, tol=1e-6):
        ok = abs(got - expect) <= tol if isinstance(expect, (int, float)) else got == expect
        print(f"[{'PASS' if ok else 'FAIL'}] {name:34s} expect={expect!r:>10} got={got!r:>10}")
        return ok

    def _mask_rect(T, H, W, x0, y0, w, h):
        M = np.zeros((T, H, W), dtype=bool)
        for t in range(T):
            M[t, y0:y0 + h, x0 + t:x0 + t + w] = True
        return M

    n_ok = n_tot = 0

    # ── Mask 通路 ──
    M = _mask_rect(10, 64, 64, x0=10, y0=20, w=30, h=20)
    r = mask_metrics(M, M)
    n_tot += 1; n_ok += _check("mean_iou identical", r["mean_iou"], 1.0)
    n_tot += 1; n_ok += _check("mask_coverage identical", r["mask_coverage"], 1.0)
    n_tot += 1; n_ok += _check("mask_disappearance identical", r["mask_disappearance"], 0.0)

    r = mask_metrics(M, np.zeros_like(M))
    n_tot += 1; n_ok += _check("mean_iou disjoint", r["mean_iou"], 0.0)
    n_tot += 1; n_ok += _check("mask_disappearance disjoint", r["mask_disappearance"], 1.0)

    # 平移 15px：GT [10+t,40+t) vs Pred [25+t,55+t) → IoU = 15/45 = 1/3
    half = _mask_rect(10, 64, 64, x0=25, y0=20, w=30, h=20)
    n_tot += 1; n_ok += _check("mean_iou shift15px", mask_metrics(M, half)["mean_iou"], 1.0 / 3.0)

    # ── 轨迹通路 ──
    gt = np.zeros((10, 2), dtype=np.float32)
    pred = np.full((10, 2), 4.0, dtype=np.float32)
    r = trajectory_metrics(gt, pred, diag=100.0)
    n_tot += 1; n_ok += _check("ate shift4px = 4√2", r["ate"], 4.0 * np.sqrt(2))
    n_tot += 1; n_ok += _check("ate_norm = ate/100", r["ate_norm"], 4.0 * np.sqrt(2) / 100.0)
    n_tot += 1; n_ok += _check("det_coverage full", r["det_coverage"], 1.0)

    pred = np.full((10, 2), np.nan, dtype=np.float32)
    pred[2:7] = 0.0   # t=2..6 有效 → 5 帧
    r = trajectory_metrics(gt, pred, diag=100.0)
    n_tot += 1; n_ok += _check("joint_frames=5", r["joint_frames"], 5)
    n_tot += 1; n_ok += _check("det_coverage=5/9", r["det_coverage"], 5.0 / 9.0)
    n_tot += 1; n_ok += _check("ate nan=0", r["ate"], 0.0)

    print(f"\n自测 → {n_ok}/{n_tot} 通过")
