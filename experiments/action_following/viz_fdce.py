"""FDCE / Action-Following 可视化：前景 mask + 锚点轨迹 + 命令轨迹 A(t) 叠加视频。

纯 numpy + cv2（cv2 仅在绘制/写入时懒加载，开发机无 cv2 也可 import / py_compile）。
不依赖 torch / SAM3 / CoWTracker。

用法（eval_action_following_fdce.py 内，--save-videos 开启后逐 arm 调用）：
    colors = viz_fdce.make_palette(max(N_gt, N_pred))
    panel_gt  = viz_fdce.render_frame(gt[i],  M_gt[i],  gt_tr,  gt_vis,  A, i, colors=colors, label="GT")
    panel_pred = viz_fdce.render_frame(pred[i], M_pred[i], pred_tr, pred_vis, A, i, colors=colors, label="Pred")
    viz_fdce.write_video(panels, out_path, fps=...)

绘制内容：
    前景 mask   绿色半透明覆盖（MASK_BLEND 混合）
    命令轨迹    红色折线 + 当前命令点（A(t) 含 NaN 帧自动断开）
    锚点轨迹    每锚点同色拖尾（最近 max_trail 帧）+ 当前帧圆点，按可见性显隐
    前景质心    可见锚点逐帧均值（黄色圆点），与命令点直观对照
"""

from __future__ import annotations

import os

import numpy as np

MASK_BLEND = 0.45
MASK_COLOR = (0, 200, 0)          # BGR 绿
COMMAND_COLOR = (0, 0, 255)       # BGR 红
CENTROID_COLOR = (0, 255, 255)    # BGR 黄
TRAIL_DEFAULT = 12                # 锚点拖尾默认最大帧数


def _pt(p) -> tuple[int, int]:
    return (int(round(float(p[0]))), int(round(float(p[1]))))


def make_palette(n: int) -> list[tuple[int, int, int]]:
    """按黄金比例生成 n 个稳定的 BGR 颜色（同视频两侧共用同一 palette，避免色差）。"""
    import colorsys  # noqa: PLC0415  (标准库，仅此处用)

    return [
        tuple(int(c * 255) for c in colorsys.hsv_to_rgb((i * 0.618033988749895) % 1.0, 0.85, 0.95)[::-1])
        for i in range(n)
    ]


def render_frame(
    bgr,
    mask_t,
    tracks,
    vis,
    command,
    t: int,
    *,
    colors=None,
    label: str | None = None,
    max_trail: int = TRAIL_DEFAULT,
) -> np.ndarray:
    """叠加一帧并返回 (H,W,3) uint8 BGR。

    Args:
        bgr:      (H,W,3) uint8 BGR 当前帧
        mask_t:   (H,W) bool 当前帧前景 mask
        tracks:   (T,N,2) float 锚点绝对像素坐标 (x,y)，全序列（画拖尾需要回看）
        vis:      (T,N) bool 锚点可见性
        command:  (T,2) 命令轨迹（可含 NaN），None = 不画
        t:        当前帧索引（0-based）
        colors:   make_palette(N) 结果；None = 全白
        label:    ASCII 文字标注（cv2.putText 不支持中文）
        max_trail: 拖尾最大回看帧数
    """
    import cv2  # noqa: PLC0415  (GPU 环境懒加载)

    # 前景 mask：绿色半透明混合
    out = bgr.astype(np.float32).copy()
    out[mask_t] = (1 - MASK_BLEND) * out[mask_t] + MASK_BLEND * np.asarray(MASK_COLOR, np.float32)
    out = out.astype(np.uint8)

    # 命令轨迹：红色折线（NaN 帧断开），当前命令点
    if command is not None and command.shape[0] > 1:
        cmd = np.asarray(command, dtype=np.float32)
        valid = ~np.isnan(cmd).any(axis=1)
        prev = None
        for j in range(min(len(cmd), t + 1)):
            if not valid[j]:
                prev = None
                continue
            p = _pt(cmd[j])
            if prev is not None:
                cv2.line(out, prev, p, COMMAND_COLOR, 2)
            prev = p
        if t < len(cmd) and valid[t]:
            cv2.circle(out, _pt(cmd[t]), 5, COMMAND_COLOR, -1)

    # 前景质心：可见锚点逐帧均值（黄色）
    n_anch = tracks.shape[1]
    if n_anch and vis[t].any():
        cv2.circle(out, _pt(tracks[t][vis[t]].mean(axis=0)), 4, CENTROID_COLOR, -1)

    # 锚点：同色拖尾 + 当前帧圆点
    for a in range(n_anch):
        if not vis[t, a]:
            continue
        color = colors[a] if colors is not None else (255, 255, 255)
        prev = None
        for j in range(max(0, t - max_trail), t + 1):
            if not vis[j, a]:
                prev = None
                continue
            p = _pt(tracks[j, a])
            if prev is not None:
                cv2.line(out, prev, p, color, 1)
            prev = p
        cv2.circle(out, _pt(tracks[t, a]), 3, color, -1)

    if label:
        cv2.putText(out, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    return out


def write_video(frames, out_path: str, fps: float = 10.0) -> str:
    """把叠加帧序列写成 mp4（cv2.VideoWriter）。返回绝对路径。"""
    import cv2  # noqa: PLC0415  (GPU 环境懒加载)

    if not frames:
        raise ValueError("frames 为空")
    frames = [np.ascontiguousarray(f, dtype=np.uint8) for f in frames]
    H, W = frames[0].shape[:2]
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (W, H))
    try:
        for f in frames:
            writer.write(f)
    finally:
        writer.release()
    return out_path


# ---------------------------------------------------------------------------
# 自测入口：纯 numpy（不加载 cv2），仅验证 palette
#   python experiments/action_following/viz_fdce.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    n_ok = n_tot = 0
    for n in (0, 1, 16):
        pal = make_palette(n)
        n_tot += 1
        n_ok += (len(pal) == n)
        if n:
            n_tot += 1
            n_ok += all(isinstance(c, tuple) and len(c) == 3 for c in pal)
            n_tot += 1
            n_ok += all(all(0 <= v <= 255 for v in c) for c in pal)
            n_tot += 1
            n_ok += (len(set(pal)) == n)   # 互不重复
    print(f"viz_fdce palette 自测 → {n_ok}/{n_tot} 通过")
