"""
多视角 × 双臂 × 多样本聚合。

reduce()              → per (cam, arm) + 跨视角均值
reduce_micro_mask()   → 直接累加所有帧 inter/union 的全局 MeanIoU（v2 §5.1）
"""

from __future__ import annotations

import numpy as np


def reduce(results: dict) -> dict:
    """
    results: dict[(cam, arm)] → dict of metric values
    return: {per_view, mean}
    """
    per_view = {}
    all_vals = {}

    for (cam, arm), metrics in results.items():
        per_view.setdefault(cam, {})[arm] = metrics
        for k, v in metrics.items():
            # 只聚合数值标量；bool 状态（如 init_failure）与列表型（如 iou_per_frame）不参与均值
            if isinstance(v, (int, float)) and not isinstance(v, bool) and not np.isnan(v):
                all_vals.setdefault(k, []).append(v)

    mean = {k: float(np.mean(v)) if v else np.nan for k, v in all_vals.items()}
    return {"per_view": per_view, "mean": mean}


def reduce_micro_mask(mask_pairs) -> float:
    """
    跨 (cam, arm) 累加 inter/union 后的全局 MeanIoU（v2 §5.1）。

    mask_pairs: iterable of (M_gt (T,H,W) bool, M_pred (T,H,W) bool)

    优点：不会让仅有少量 EEF 像素的帧与大量 EEF 像素的帧获得相同权重。
    """
    total_inter = 0
    total_union = 0
    for M_gt, M_pred in mask_pairs:
        for t in range(1, M_gt.shape[0]):
            total_inter += int((M_gt[t] & M_pred[t]).sum())
            total_union += int((M_gt[t] | M_pred[t]).sum())
    return float(total_inter / total_union) if total_union > 0 else np.nan


# ---------------------------------------------------------------------------
# 自测入口：合成 results 驱动 reduce / reduce_micro_mask
#   python experiments/action_following/aggregate.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    def _check(name, got, expect, tol=1e-6):
        ok = abs(got - expect) <= tol if isinstance(expect, (int, float)) else got == expect
        print(f"[{'PASS' if ok else 'FAIL'}] {name:40s} expect={expect!r:>10} got={got!r:>10}")
        return ok

    n_ok = n_tot = 0

    results = {
        ("head", "left"):      {"mean_iou": 0.50, "ate": 2.0, "det_coverage": 0.8, "iou_per_frame": [0.5]},
        ("head", "right"):     {"mean_iou": 0.70, "ate": 4.0, "det_coverage": 1.0, "iou_per_frame": [0.7]},
        ("cam2", "left"): {"init_failure": True},    # bool 状态 → 不参与 mean
        ("cam2", "right"): {"mean_iou": np.nan, "ate": 6.0, "det_coverage": np.nan, "iou_per_frame": []},
    }
    out = reduce(results)
    mean = out["mean"]
    n_tot += 1; n_ok += _check("mean_iou 排除 NaN/init_failure (0.5+0.7)/2", mean["mean_iou"], 0.6)
    n_tot += 1; n_ok += _check("ate (2+4+6)/3", mean["ate"], 4.0)
    n_tot += 1; n_ok += _check("det_coverage (0.8+1.0)/2", mean["det_coverage"], 0.9)
    n_tot += 1; n_ok += _check("init_failure 不进 mean", "init_failure" in mean, False)
    n_tot += 1; n_ok += _check("per_view 保留 init_failure", out["per_view"]["cam2"]["left"]["init_failure"], True)

    # reduce_micro_mask：两段各 4 帧，20×20 rect 平移 5px → 每帧 IoU = (15·20)/(25·20) = 0.6
    def _mask_rect(T, H, W, x0):
        M = np.zeros((T, H, W), dtype=bool)
        for t in range(T):
            M[t, 10:30, x0 + 5 * t:x0 + 5 * t + 20] = True
        return M

    pairs = []
    for x0 in (0, 20):   # x0 需保证 t=3 帧也全部落在 W=64 内，避免右边界裁剪
        pairs.append((_mask_rect(4, 64, 64, x0), _mask_rect(4, 64, 64, x0 + 5)))
    n_tot += 1; n_ok += _check("reduce_micro_mask 全局 MeanIoU", reduce_micro_mask(pairs), 0.6)

    print(f"\n聚合自测 → {n_ok}/{n_tot} 通过")
