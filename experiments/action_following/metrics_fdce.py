"""Foreground Displacement Chamfer Error (FDCE)，纯 NumPy 移植。

打分器只消费"已经算好的点轨迹"，不加载任何分割/跟踪模型；语义与
CD-LAM `cd_lam.metrics.fdce` 一致：

    1. 每条轨迹先平移掉自己的首帧起点（比较"位移量"，全局平移不惩罚）；
    2. 先算固定轨迹对的位移成本 c_ij（论文式 A.3：rollout 帧上取平均，
       s 从 1 开始，首帧本身不参与）；
    3. 再对 c_ij 矩阵做对称 Chamfer 归约（论文式 A.4）。

关键约束：必须先"固定配对再按时间平均"，再 Chamfer。禁止改成逐帧 Chamfer
再按时间平均——轨迹交叉的夹具（见自测）会给出接近 0 的错误分数，而
A.3 → A.4 的正确顺序给出 5.0。

命令入口（CD-LAM）：
    bash run.sh score-fdce --tracks evaluation/tracks/*.npz --output evaluation/fdce.json
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class FDCEResult:
    """FDCE 分数与可见性过滤后的有效计数。"""

    score: float
    generated_to_reference: float
    reference_to_generated: float
    generated_tracks: int
    reference_tracks: int
    valid_pairs: int


def _as_tracks(name: str, value: ArrayLike) -> FloatArray:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[-1] != 2:
        raise ValueError(f"{name} must have shape (T, N, 2), got {array.shape}")
    if array.shape[0] < 2:
        raise ValueError(f"{name} must contain frame 0 and at least one rollout frame")
    if array.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one point track")
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise TypeError(f"{name} must be a real numeric array, got dtype {array.dtype}")
    return np.asarray(array, dtype=np.float64)


def _as_visibility(
    name: str,
    value: Optional[ArrayLike],
    tracks: FloatArray,
    *,
    threshold: float,
) -> BoolArray:
    finite_coordinates = np.isfinite(tracks).all(axis=-1)
    if value is None:
        return finite_coordinates
    visibility = np.asarray(value)
    if visibility.shape != tracks.shape[:2]:
        raise ValueError(
            f"{name} must have shape {tracks.shape[:2]}, got {visibility.shape}"
        )
    if visibility.dtype == np.bool_:
        visible = visibility.copy()
    else:
        if not np.issubdtype(visibility.dtype, np.number) or np.iscomplexobj(visibility):
            raise TypeError(
                f"{name} must be boolean or real-valued confidence scores"
            )
        visible = np.isfinite(visibility) & (visibility >= threshold)
    return np.asarray(visible & finite_coordinates, dtype=bool)


def _filter_tracks(
    tracks: FloatArray,
    visibility: BoolArray,
    *,
    min_visible_fraction: float,
) -> tuple[FloatArray, BoolArray]:
    # Frame 0 是必需的：每条轨迹都以它作为位移基准。
    keep = visibility[0] & (visibility.mean(axis=0) >= min_visible_fraction)
    return tracks[:, keep], visibility[:, keep]


def pairwise_displacement_costs(
    generated_tracks: ArrayLike,
    reference_tracks: ArrayLike,
    generated_visibility: Optional[ArrayLike] = None,
    reference_visibility: Optional[ArrayLike] = None,
    *,
    visibility_threshold: float = 0.5,
    min_visible_fraction: float = 0.8,
    min_common_frames: int = 1,
) -> FloatArray:
    """计算论文式 A.3 的全部 (generated, reference) 轨迹对成本。

    Tracks shape ``(T, N, 2)``。每条轨迹先平移掉自己的 frame-0 点；配对成本为
    双方同时可见的 rollout 帧上位移误差的均值。Frame 0 本身不参与平均
    （式 A.3 对 rollout 步 1..H 求和）。

    低可见率轨迹在配对前先被丢弃（评估协议）；共同可见 rollout 帧不足
    ``min_common_frames`` 的配对记为 ``NaN``。
    """

    generated = _as_tracks("generated_tracks", generated_tracks)
    reference = _as_tracks("reference_tracks", reference_tracks)
    if generated.shape[0] != reference.shape[0]:
        raise ValueError(
            "generated_tracks and reference_tracks must have the same number of frames, "
            f"got {generated.shape[0]} and {reference.shape[0]}"
        )
    if not np.isfinite(visibility_threshold):
        raise ValueError("visibility_threshold must be finite")
    if not np.isfinite(min_visible_fraction) or not 0 <= min_visible_fraction <= 1:
        raise ValueError("min_visible_fraction must lie in [0, 1]")
    if isinstance(min_common_frames, bool) or int(min_common_frames) != min_common_frames:
        raise ValueError("min_common_frames must be a positive integer")
    min_common_frames = int(min_common_frames)
    if min_common_frames < 1:
        raise ValueError("min_common_frames must be a positive integer")
    if min_common_frames > generated.shape[0] - 1:
        raise ValueError(
            "min_common_frames cannot exceed the number of rollout frames "
            f"({generated.shape[0] - 1})"
        )

    generated_vis = _as_visibility(
        "generated_visibility", generated_visibility, generated,
        threshold=visibility_threshold,
    )
    reference_vis = _as_visibility(
        "reference_visibility", reference_visibility, reference,
        threshold=visibility_threshold,
    )
    generated, generated_vis = _filter_tracks(
        generated, generated_vis, min_visible_fraction=min_visible_fraction,
    )
    reference, reference_vis = _filter_tracks(
        reference, reference_vis, min_visible_fraction=min_visible_fraction,
    )
    if generated.shape[1] == 0:
        raise ValueError("no generated tracks survive the visibility filter")
    if reference.shape[1] == 0:
        raise ValueError("no reference tracks survive the visibility filter")

    generated_delta = generated - generated[0:1]
    reference_delta = reference - reference[0:1]
    distances = np.linalg.norm(
        generated_delta[1:, :, None, :] - reference_delta[1:, None, :, :],
        axis=-1,
    )
    jointly_visible = (
        generated_vis[1:, :, None] & reference_vis[1:, None, :] & np.isfinite(distances)
    )
    counts = jointly_visible.sum(axis=0)
    sums = np.where(jointly_visible, distances, 0.0).sum(axis=0)
    costs = np.full(counts.shape, np.nan, dtype=np.float64)
    valid = counts >= min_common_frames
    costs[valid] = sums[valid] / counts[valid]
    return costs


def symmetric_chamfer_from_costs(pair_costs: ArrayLike) -> tuple[float, float, float]:
    """对配对成本矩阵做论文式 A.4 的对称 Chamfer 归约。"""

    costs = np.asarray(pair_costs, dtype=np.float64)
    if costs.ndim != 2 or 0 in costs.shape:
        raise ValueError(f"pair_costs must be a non-empty 2D matrix, got {costs.shape}")
    if np.isinf(costs).any():
        raise ValueError("pair_costs must not contain infinity")

    finite = np.isfinite(costs)
    if not finite.any(axis=1).all():
        bad = np.flatnonzero(~finite.any(axis=1)).tolist()
        raise ValueError(f"generated track(s) have no valid reference comparison: {bad}")
    if not finite.any(axis=0).all():
        bad = np.flatnonzero(~finite.any(axis=0)).tolist()
        raise ValueError(f"reference track(s) have no valid generated comparison: {bad}")

    safe = np.where(finite, costs, np.inf)
    generated_to_reference = float(safe.min(axis=1).mean())
    reference_to_generated = float(safe.min(axis=0).mean())
    score = 0.5 * (generated_to_reference + reference_to_generated)
    return score, generated_to_reference, reference_to_generated


def foreground_displacement_chamfer_error(
    generated_tracks: ArrayLike,
    reference_tracks: ArrayLike,
    generated_visibility: Optional[ArrayLike] = None,
    reference_visibility: Optional[ArrayLike] = None,
    *,
    visibility_threshold: float = 0.5,
    min_visible_fraction: float = 0.8,
    min_common_frames: int = 1,
    return_details: bool = False,
) -> Union[float, FDCEResult]:
    """计算 FDCE（generated / reference 前景点轨迹）。"""

    costs = pairwise_displacement_costs(
        generated_tracks,
        reference_tracks,
        generated_visibility,
        reference_visibility,
        visibility_threshold=visibility_threshold,
        min_visible_fraction=min_visible_fraction,
        min_common_frames=min_common_frames,
    )
    score, generated_to_reference, reference_to_generated = symmetric_chamfer_from_costs(
        costs
    )
    if not return_details:
        return score
    return FDCEResult(
        score=score,
        generated_to_reference=generated_to_reference,
        reference_to_generated=reference_to_generated,
        generated_tracks=int(costs.shape[0]),
        reference_tracks=int(costs.shape[1]),
        valid_pairs=int(np.isfinite(costs).sum()),
    )


def symmetric_chamfer_distance(first: ArrayLike, second: ArrayLike) -> float:
    """两个点云之间的对称欧氏 Chamfer 距离。"""

    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    if first_array.ndim != 2 or second_array.ndim != 2:
        raise ValueError("point clouds must both have shape (N, D)")
    if first_array.shape[0] == 0 or second_array.shape[0] == 0:
        raise ValueError("point clouds must be non-empty")
    if first_array.shape[1] != second_array.shape[1]:
        raise ValueError("point clouds must share the coordinate dimension")
    if not np.isfinite(first_array).all() or not np.isfinite(second_array).all():
        raise ValueError("point clouds must contain only finite coordinates")
    distances = np.linalg.norm(first_array[:, None] - second_array[None, :], axis=-1)
    score, _, _ = symmetric_chamfer_from_costs(distances)
    return score


# 简短的常规别名。
fdce = foreground_displacement_chamfer_error


__all__ = [
    "FDCEResult",
    "fdce",
    "foreground_displacement_chamfer_error",
    "pairwise_displacement_costs",
    "symmetric_chamfer_distance",
    "symmetric_chamfer_from_costs",
]


# ---------------------------------------------------------------------------
# 自测入口：纯 numpy 合成数据（无 GPU）
#   python experiments/action_following/metrics_fdce.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    def _track_2d(*disp):
        """位移轨迹 (T,1,2)：frame0=(0,0)，后续帧给定位移 (dx,dy)。"""
        T = len(disp) + 1
        t = np.zeros((T, 1, 2), dtype=np.float64)
        for s, d in enumerate(disp, start=1):
            t[s, 0] = d
        return t

    def _track_1d(*disp):
        """1D 位移轨迹 (T,1,2)，位移沿 x 轴。"""
        return _track_2d(*[(d, 0.0) for d in disp])

    n_ok = n_tot = 0
    def _check(name, got, expect, tol=1e-9):
        ok = abs(got - expect) <= tol if isinstance(expect, (int, float)) else got == expect
        print(f"[{'PASS' if ok else 'FAIL'}] {name:50s} expect={expect!r:>10} got={got!r:>10}")
        return ok

    # 1) 相同轨迹 / 全局平移 → FDCE = 0
    same = _track_1d(1.0, 3.0, 2.0)
    n_tot += 1; n_ok += _check("相同轨迹 FDCE=0",
        foreground_displacement_chamfer_error(same, same), 0.0)
    shifted = _track_1d(1.0, 3.0, 2.0) + np.array([[[10.0, 20.0]]])
    n_tot += 1; n_ok += _check("全局平移 FDCE=0（位移比较）",
        foreground_displacement_chamfer_error(shifted, same), 0.0)

    # 2) 单轨迹解析：误差 {1, 3} → c00 = (1+3)/2 = 2
    g = _track_1d(1.0, 3.0)      # 位移 (1,0),(3,0)
    r = _track_1d(0.0, 0.0)      # 位移 (0,0),(0,0)
    details = foreground_displacement_chamfer_error(
        g, r, return_details=True
    )
    n_tot += 1; n_ok += _check("单轨迹 c00=(1+3)/2=2", details.score, 2.0)

    # 3) 固定配对先于 Chamfer：交叉轨迹 → 5.0（逐帧 Chamfer 会错误给出 ~0）
    #    ref0 静止；ref1 在 s=1 跳到 (10,0) 并保持；gen0 在 s=1 匹配 ref0、s=2 匹配 ref1
    ref0 = _track_1d(0.0, 0.0)
    ref1 = _track_1d(10.0, 10.0)
    gen0 = _track_1d(0.0, 10.0)
    ref_cross = np.concatenate([ref0, ref1], axis=1)  # (3,2,2)
    fixed = foreground_displacement_chamfer_error(gen0, ref_cross, return_details=True)
    n_tot += 1; n_ok += _check("交叉轨迹固定配对=5.0", fixed.score, 5.0)
    # 逐帧 Chamfer 的错误结果 ≈ 0，证明顺序不能颠倒：
    #   逐帧做法 = 每帧对参考集合取 min，再按时间平均 → 帧 1 匹配 ref0(0,0)、帧 2 匹配 ref1(10,0) → ≈0
    costs = pairwise_displacement_costs(gen0, ref_cross)          # (1,2)
    per_frame = float(
        np.linalg.norm(gen0[1:, 0, None, :] - ref_cross[1:, :, :], axis=-1).min(axis=1).mean()
    )
    n_tot += 1; n_ok += _check("逐帧 Chamfer 会错误≈0（对照）", per_frame, 0.0)
    n_tot += 1; n_ok += _check("固定配对 ≠ 逐帧 Chamfer", fixed.score, 5.0)

    # 4) 对称 Chamfer：[[0],[2]] vs [[0]] → 0.5
    n_tot += 1; n_ok += _check("对称 Chamfer [[0],[2]] vs [[0]]=0.5",
        symmetric_chamfer_distance(np.array([[0.0]]), np.array([[0.0], [2.0]])), 0.5)

    # 5) 可见性过滤：低可见率轨迹被丢弃
    #    gen: 2 条轨迹；轨迹 0 全程可见，轨迹 1 后半段消失 → 平均可见率 0.5 < 0.8 → 丢弃
    gv = np.concatenate([_track_1d(1.0, 2.0), _track_1d(0.0, 1.0)], axis=1)  # (3,2,2)
    vis = np.ones((3, 2), dtype=bool)
    vis[2, 1] = False   # 轨迹 1 在最后一帧不可见 → mean visibility = 2/3 ≈ 0.667 < 0.8
    rv = np.ones((3, 2), dtype=bool)
    details2 = foreground_displacement_chamfer_error(
        gv, gv, generated_visibility=vis, reference_visibility=rv, return_details=True
    )
    n_tot += 1; n_ok += _check("可见性过滤后 generated 剩 1 条",
        details2.generated_tracks, 1)

    print(f"\nFDCE 自测 → {n_ok}/{n_tot} 通过")
