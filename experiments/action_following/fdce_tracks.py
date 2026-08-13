"""
mask → 锚点采样 → 点轨迹 → 打包协议兼容 NPZ 轨迹包（纯 NumPy，可本地开发机测试）。

流程（报告 §1.5 / §3.4）：
    1. 前景 mask 首帧腐蚀（3x3 十字形，纯 numpy）；
    2. 在腐蚀后区域内按种子采样 ≤k 个锚点（GT/Pred 各自独立，同一种子策略）；
    3. 从 dense 点轨迹中抽取锚点像素的 (T,N,2) 轨迹 + (T,N) 可见性；
    4. 逐帧可见性再与"该点仍在前景 mask 内"取与（仅统计前景有效区域内的运动）；
    5. 对齐到协议帧数 max_frames（截断 / 重复最后一帧补齐），打包 NPZ。

约定：generated = Pred（模型 rollout），reference = GT（与 CD-LAM score-fdce 一致）。

dense 参数为"带 .tracks/.visibility 属性的对象"（如 cowtracker_tracking.DenseTracks），
或 (tracks, visibility) 二元组 —— 本模块不 import torch，纯 numpy。
"""

from __future__ import annotations

import os

import numpy as np


def _erode_mask(mask0, iterations: int = 1) -> np.ndarray:
    """3x3 十字形腐蚀（仅保留"自身 + 上下左右邻居"全为前景的像素）。

    纯 numpy 实现，避免引入 scipy/cv2。iterations 次迭代。
    """
    m = np.asarray(mask0, dtype=bool)
    for _ in range(iterations):
        up = np.zeros_like(m); up[:-1] = m[1:]
        down = np.zeros_like(m); down[1:] = m[:-1]
        left = np.zeros_like(m); left[:, :-1] = m[:, 1:]
        right = np.zeros_like(m); right[:, 1:] = m[:, :-1]
        m = m & up & down & left & right
    return m


def sample_anchors(mask0, k: int = 16, seed: int = 0) -> np.ndarray:
    """在首帧前景 mask 内采样 ≤k 个锚点。

    Args:
        mask0: (H,W) bool 前景 mask
        k: 期望锚点数（协议上限 16）
        seed: 采样随机种子（同视频 GT/Pred 使用相同种子策略）

    Returns:
        (N,2) int64 [y,x] 坐标；前景为空时返回 (0,2)。
    """
    m = np.asarray(mask0, dtype=bool)
    if m.ndim != 2:
        raise ValueError(f"mask0 需为 (H,W) bool，收到 {m.shape}")
    coords = np.argwhere(m)          # (P,2) [y,x]
    P = coords.shape[0]
    if P == 0:
        return np.zeros((0, 2), dtype=np.int64)
    n = max(1, min(k, P))
    rng = np.random.default_rng(seed)
    idx = rng.choice(P, size=n, replace=False)
    return coords[idx]


def _align_index(length: int, target: int) -> np.ndarray:
    """构造长度为 target 的帧索引：超出部分重复最后一帧（stationary 补齐）。

    length >= target 时截断取前 target 帧；length < target 时重复最后一帧。
    """
    if target < 1:
        raise ValueError(f"target 需 >= 1，收到 {target}")
    if length < 1:
        raise ValueError(f"length 需 >= 1，收到 {length}")
    if length >= target:
        return np.arange(target)
    return np.minimum(np.arange(target), length - 1)


def sample_tracks(dense, mask, anchors=None, *, k: int = 16, seed: int = 0):
    """从 dense 点轨迹抽取锚点轨迹。

    Args:
        dense: DenseTracks（带 .tracks/.visibility）或 (tracks, visibility) 二元组；
               tracks (T,H,W,2)，visibility (T,H,W) bool
        mask: (T,H,W) bool 前景 mask；逐帧可见性与该点是否仍在 mask 内取与
        anchors: (N,2) [y,x] 锚点；None 时由首帧腐蚀 mask 采样

    Returns:
        (tracks (T,N,2) float32, visibility (T,N) bool)
    """
    if isinstance(dense, tuple):
        dense_tracks, dense_vis = dense
    else:
        dense_tracks = dense.tracks
        dense_vis = dense.visibility
    tracks = np.asarray(dense_tracks)
    vis = np.asarray(dense_vis)
    if tracks.ndim != 4 or tracks.shape[-1] != 2:
        raise ValueError(f"dense tracks 需为 (T,H,W,2)，收到 {tracks.shape}")
    if vis.shape != tracks.shape[:3]:
        raise ValueError(f"dense visibility 需为 {tracks.shape[:3]}，收到 {vis.shape}")
    if anchors is None:
        anchors = sample_anchors(_erode_mask(np.asarray(mask)[0]), k=k, seed=seed)
    anchors = np.asarray(anchors)
    if anchors.ndim != 2 or anchors.shape[1] != 2:
        raise ValueError(f"anchors 需为 (N,2) [y,x]，收到 {anchors.shape}")
    if anchors.shape[0] == 0:
        return np.zeros((tracks.shape[0], 0, 2), dtype=np.float32), np.zeros(
            (tracks.shape[0], 0), dtype=bool
        )

    H, W = tracks.shape[1:3]
    if anchors[:, 0].max() >= H or anchors[:, 1].max() >= W:
        raise ValueError("锚点越界")

    T = tracks.shape[0]
    ys = anchors[:, 0]
    xs = anchors[:, 1]
    out_tracks = tracks[:, ys, xs].astype(np.float32)      # (T,N,2)
    out_vis = vis[:, ys, xs].copy()                        # (T,N) bool

    if mask is not None:
        M = np.asarray(mask, dtype=bool)
        if M.ndim != 3 or M.shape[1:3] != (H, W):
            raise ValueError(f"mask 需为 ({T},{H},{W}) bool，收到 {M.shape}")
        if M.shape[0] != T:
            M = M[_align_index(M.shape[0], T)]             # 对齐到 dense 帧数
        # CoWTracker 轨迹坐标为 (x, y)；按每帧轨迹位置查询前景 mask。
        finite = np.isfinite(out_tracks).all(axis=-1)
        px = np.rint(np.where(finite, out_tracks[..., 0], 0)).astype(np.int64)
        py = np.rint(np.where(finite, out_tracks[..., 1], 0)).astype(np.int64)
        inside = finite & (px >= 0) & (px < W) & (py >= 0) & (py < H)
        foreground = np.zeros_like(out_vis, dtype=bool)
        ti, ni = np.nonzero(inside)
        foreground[ti, ni] = M[ti, py[ti, ni], px[ti, ni]]
        out_vis &= foreground
    return out_tracks, out_vis


def build_track_bundle(
    M_gt,
    M_pred,
    dense_gt,
    dense_pred,
    *,
    k: int = 16,
    seed: int = 0,
    max_frames: int = 49,
) -> tuple[dict, dict]:
    """把 GT/Pred 的 mask 与 dense 点跟踪打包成协议兼容轨迹包。

    Args:
        M_gt / M_pred: (T,H,W) bool 前景 mask（sam_tracking 产物，GT/Pred 各自独立）
        dense_gt / dense_pred: 对应侧 dense 点跟踪（DenseTracks 或二元组）
        k: 每侧锚点数上限（协议 16）
        seed: 锚点采样种子
        max_frames: 协议帧数（首帧 + 48 rollout = 49）

    Returns:
        (bundle, meta)：
            bundle["generated_tracks"]    (max_frames, N_pred, 2)   <- Pred
            bundle["reference_tracks"]    (max_frames, N_gt, 2)     <- GT
            bundle["generated_visibility"]/["reference_visibility"] (max_frames, N) bool
            meta：锚点数、mask 覆盖率、帧数、seed、init_failure 等诊断字段
    """
    M_gt = np.asarray(M_gt, dtype=bool)
    M_pred = np.asarray(M_pred, dtype=bool)
    if M_gt.ndim != 3 or M_pred.ndim != 3:
        raise ValueError(f"mask 需为 (T,H,W)，收到 gt={M_gt.shape} pred={M_pred.shape}")

    gt_anchors = sample_anchors(_erode_mask(M_gt[0]), k=k, seed=seed)
    pred_anchors = sample_anchors(_erode_mask(M_pred[0]), k=k, seed=seed)

    gt_tracks, gt_vis = sample_tracks(dense_gt, M_gt, anchors=gt_anchors)
    pred_tracks, pred_vis = sample_tracks(dense_pred, M_pred, anchors=pred_anchors)

    g_idx = _align_index(pred_tracks.shape[0], max_frames)
    r_idx = _align_index(gt_tracks.shape[0], max_frames)
    bundle = {
        "generated_tracks": pred_tracks[g_idx],          # (max_frames, N_pred, 2)
        "reference_tracks": gt_tracks[r_idx],            # (max_frames, N_gt, 2)
        "generated_visibility": pred_vis[g_idx],
        "reference_visibility": gt_vis[r_idx],
    }
    meta = {
        "generated_tracks": int(pred_tracks.shape[1]),
        "reference_tracks": int(gt_tracks.shape[1]),
        "anchor_count_generated": int(pred_anchors.shape[0]),
        "anchor_count_reference": int(gt_anchors.shape[0]),
        "mask_coverage_generated": float(M_pred.mean(axis=(1, 2)).mean()),
        "mask_coverage_reference": float(M_gt.mean(axis=(1, 2)).mean()),
        "frames_generated": int(pred_tracks.shape[0]),
        "frames_reference": int(gt_tracks.shape[0]),
        "frames_used": max_frames,
        "k": k,
        "seed": seed,
        "init_failure": bool(pred_anchors.shape[0] == 0 or gt_anchors.shape[0] == 0),
    }
    return bundle, meta


def save_bundle(bundle: dict, path: str) -> str:
    """把轨迹包写到 NPZ（协议：allow_pickle=False，键为协议 key 集合）。"""
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **bundle)
    return path


# ---------------------------------------------------------------------------
# 自测入口：纯 numpy 合成数据（无 GPU）
#   python experiments/action_following/fdce_tracks.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from types import SimpleNamespace

    def _check(name, got, expect, tol=1e-9):
        ok = abs(got - expect) <= tol if isinstance(expect, (int, float)) else got == expect
        print(f"[{'PASS' if ok else 'FAIL'}] {name:50s} expect={expect!r:>10} got={got!r:>10}")
        return ok

    n_ok = n_tot = 0
    H = W = 40

    # 合成：HxW 方块前景，匀速向右移动 8 像素
    T = 8
    dense_tracks = np.zeros((T, H, W, 2), dtype=np.float32)
    dense_vis = np.ones((T, H, W), dtype=bool)
    for t in range(T):
        dense_tracks[t] += np.array([0.0, 0.0])  # 占位，下面按 y,x 填
    # 构造一个"像素 x 坐标=其初始 x+dx(t)"的密集轨迹场
    for y in range(H):
        for x in range(W):
            dx = 8.0 if (10 <= y <= 20 and 10 <= x <= 20) else 0.0
            dense_tracks[:, y, x, 0] = x + dx * np.arange(T) / (T - 1)   # x 方向位移
            dense_tracks[:, y, x, 1] = y                                  # y 不动

    M = np.zeros((T, H, W), dtype=bool)
    M[:, 10:21, 10:21] = True   # 方块前景全程

    # 1) 锚点采样：腐蚀后方块 (11..19)^2 = 81 像素，k=16 → 16 个锚点
    anchors = sample_anchors(_erode_mask(M[0]), k=16, seed=0)
    n_tot += 1; n_ok += _check("腐蚀后采样锚点数=16", int(anchors.shape[0]), 16)
    n_tot += 1; n_ok += _check("锚点全在腐蚀后区域内", bool((_erode_mask(M[0])[anchors[:, 0], anchors[:, 1]]).all()), True)

    # 2) sample_tracks：dense 对象（SimpleNamespace）→ (T,N,2)
    dense = SimpleNamespace(tracks=dense_tracks, visibility=dense_vis)
    tr, vis = sample_tracks(dense, M, anchors=anchors)
    n_tot += 1; n_ok += _check("tracks shape=(8,16,2)", tuple(tr.shape), (8, 16, 2))
    n_tot += 1; n_ok += _check("visibility 全 True（前景全程）", bool(vis.all()), True)
    # 锚点内第一个像素的轨迹：首帧 x0，末帧 x0+8
    y0, x0 = int(anchors[0, 0]), int(anchors[0, 1])
    n_tot += 1; n_ok += _check("锚点位移 x 方向 8px", float(tr[-1, 0, 0] - tr[0, 0, 0]), 8.0)
    n_tot += 1; n_ok += _check("锚点位移 y 方向 0px", float(tr[-1, 0, 1] - tr[0, 0, 1]), 0.0)

    # 3) 逐帧前景可见性门控：让锚点在第 4 帧离开 mask → 该帧 invisible
    vis_gated, vis_g = sample_tracks(dense, M, anchors=anchors[:1])
    M2 = M.copy(); M2[4:, 10:16, 10:16] = False   # 第 4 帧起锚点离开前景
    _, vis2 = sample_tracks(dense, M2, anchors=anchors[:1])
    n_tot += 1; n_ok += _check("前景门控：前 4 帧可见", list(map(bool, vis2[:4, 0])), [True] * 4)
    n_tot += 1; n_ok += _check("前景门控：第 4 帧起不可见", list(map(bool, vis2[4:, 0])), [False] * 4)

    # 4) 帧对齐：8 帧 → 49 帧（重复最后一帧补齐）
    idx = _align_index(8, 49)
    n_tot += 1; n_ok += _check("补齐后 49 帧", int(idx.shape[0]), 49)
    n_tot += 1; n_ok += _check("补齐尾部重复最后一帧", bool((idx[-3:] == 7).all()), True)

    # 5) build_track_bundle：generated=Pred / reference=GT，帧数对齐 49
    bundle, meta = build_track_bundle(M, M, dense, dense, k=16, seed=0, max_frames=49)
    n_tot += 1; n_ok += _check("bundle generated_tracks (49,16,2)",
        tuple(bundle["generated_tracks"].shape), (49, 16, 2))
    n_tot += 1; n_ok += _check("bundle reference_tracks (49,16,2)",
        tuple(bundle["reference_tracks"].shape), (49, 16, 2))
    n_tot += 1; n_ok += _check("bundle visibility bool", bool(bundle["generated_visibility"].dtype == bool), True)
    n_tot += 1; n_ok += _check("meta anchor_count=16", meta["anchor_count_generated"], 16)
    n_tot += 1; n_ok += _check("meta init_failure=False", meta["init_failure"], False)

    # 6) 无前景 → init_failure=True，轨迹为空
    empty = np.zeros((T, H, W), dtype=bool)
    b2, m2 = build_track_bundle(empty, M, dense, dense, k=16, seed=0, max_frames=49)
    n_tot += 1; n_ok += _check("无前景 init_failure=True", m2["init_failure"], True)
    n_tot += 1; n_ok += _check("无前景 reference_tracks (49,0,2)",
        tuple(b2["reference_tracks"].shape), (49, 0, 2))

    print(f"\nfdce_tracks 自测 → {n_ok}/{n_tot} 通过")
