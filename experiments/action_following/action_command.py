"""
ACWM Action-Following 指标：把 actions.npy 还原成"命令轨迹" A(t)，与视频里
YOLO 检测到的 EEF 轨迹对比，量化"模型是否按命令运动"。

背景：GE-Sim（Cosmos）把机器人 action 通过轨迹图（utils/get_traj_maps.py）注入：
    actions[t]（左臂 pos+quat+grip / 右臂 pos+quat+grip）
      → SE(3) 位姿矩阵
      → w2c @ pose @ correct_matrix（+0.23m z 偏移）
      → intrinsics 透视投影
      → 头相机图像平面上的 EEF 原点像素坐标
这条投影出的 2D 路径就是模型被"命令"画的轨迹。本模块用纯 numpy 复刻这条投影
（不依赖 torch / cv2 / 模型），得到逐帧命令轨迹 A(t)；再与 YOLO 检测轨迹
对比得到 action-following 误差（af_ate 一族）。

与现有 Pred-vs-GT ATE 的区别：
    ate       = mean || T_pred − T_gt ||         重建保真度（Pred 是否还原 GT）
    af_ate    = mean || T_pred − A      ||       action-following（Pred 是否跟命令）
    af_ate_gt = mean || T_gt   − A      ||       参考基线（录制本身与命令的吻合度，
                                                  含 bbox-center↔EEF-origin 系统偏差）
"""

from __future__ import annotations

import numpy as np

DEFAULT_SAMPLE_SIZE = (384, 512)   # (H, W)，与 acwm_cosmos.yaml sample_size 一致
DEFAULT_ORI_SIZE = (480, 640)      # (H, W)，原始 GT 分辨率
EEF_Z_OFFSET = 0.23                # get_traj_maps 中 correct_matrix 的相机 z 偏移（米）


# ---------------------------------------------------------------------------
# 命令轨迹 A(t)：复刻 utils/get_traj_maps.py 的 EEF 原点投影（纯 numpy）
# ---------------------------------------------------------------------------
def quaternion_to_matrix(quats: np.ndarray) -> np.ndarray:
    """quats: (...,4) real-first -> (...,3,3)。与 get_traj_maps 相同。"""
    r, i, j, k = quats[..., 0], quats[..., 1], quats[..., 2], quats[..., 3]
    two_s = 2.0 / (quats * quats).sum(-1)
    o = np.stack((
        1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
        two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
        two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j),
    ), -1)
    return o.reshape(quats.shape[:-1] + (3, 3))


def pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    """pose: (T,7) [xyz + quat(xyzw)] -> (T,4,4) SE(3)。"""
    q = pose[:, 3:][:, [3, 0, 1, 2]]   # xyzw -> wxyz（get_traj_maps 的 real-first 约定）
    R = quaternion_to_matrix(q)
    M = np.broadcast_to(np.eye(4), (len(pose), 4, 4)).copy()
    M[:, :3, :3] = R
    M[:, :3, 3] = pose[:, :3]
    return M


def commanded_trajectory(
    actions: np.ndarray,
    c2w: np.ndarray,
    intrinsic: np.ndarray,
    sample_size: tuple[int, int] = DEFAULT_SAMPLE_SIZE,
    ori_size: tuple[int, int] = DEFAULT_ORI_SIZE,
    arm: str = "left",
) -> np.ndarray:
    """
    actions[t] 对应的 EEF 原点投影到相机图像平面的像素坐标（= GE-Sim 命令轨迹）。

    Args:
        actions:   (T,16) [left_xyz+quat+grip, right_xyz+quat+grip]（prep 的 actions.npy）
        c2w:       (T,4,4) 相机外参 world-from-camera（prep 的 extrinsic_<cam>.npy）
        intrinsic: (3,3) 原始分辨率内参（prep 的 intrinsic_<cam>.npy）
        sample_size: 评估分辨率 (H,W)，默认 384x512
        ori_size:  内参对应的原始分辨率 (H,W)，默认 480x640
        arm:       "left" | "right"

    Returns:
        A: (T,2) float32 命令轨迹（像素坐标）；投影无效（z<=0 在相机后方）帧为 NaN
    """
    h, w = sample_size
    oh, ow = ori_size
    sx, sy = w / ow, h / oh
    K = intrinsic.astype(np.float64).copy()
    K[0, 0] *= sx
    K[0, 2] *= sx
    K[1, 1] *= sy
    K[1, 2] *= sy

    w2c = np.linalg.inv(c2w.astype(np.float64))                     # (T,4,4)
    pose = actions[:, 0:7] if arm == "left" else actions[:, 8:15]   # (T,7)
    M = w2c @ pose_to_matrix(pose.astype(np.float64))               # (T,4,4) ee2cam

    correct = np.eye(4)
    correct[2, 3] = EEF_Z_OFFSET                                    # get_traj_maps 的 z 偏移
    M = M @ correct

    pt = M[:, :3, 3]                                                # (T,3) 相机系 EEF 原点
    A = np.full((len(actions), 2), np.nan, dtype=np.float32)
    ok = pt[:, 2] > 0
    A[ok] = (K @ pt[ok].T).T[:, :2] / pt[ok, 2:3]                   # (T,2) 像素
    return A


# ---------------------------------------------------------------------------
# Action-Following 指标：T_pred / T_gt 与命令轨迹 A 对比
# ---------------------------------------------------------------------------
def action_following_metrics(
    traj: np.ndarray,
    A: np.ndarray,
    diag: float | None = None,
    skip_t0: bool = True,
) -> dict:
    """
    traj: (T,2) YOLO 检测的 EEF 轨迹（检测失败为 NaN）
    A:    (T,2) 命令轨迹（投影无效为 NaN）
    diag: 归一化对角线（√(H²+W²)）；None = 不归一化

    Returns:
        af_ate:           [核心] traj 与命令 A 的平均欧氏距离 (px)
        af_ate_p95:       误差 95 分位（诊断）
        af_ate_norm:      对角线归一化（若 diag 给定）
        af_det_coverage:  双方有效帧占比（必须并列报告）
        af_joint_frames:  有效帧数
        af_total_frames:  评估总帧数（排除首帧后）
    """
    T = traj.shape[0]
    t_start = 1 if skip_t0 else 0

    valid = ~np.isnan(traj[:, 0]) & ~np.isnan(A[:, 0])
    valid[:t_start] = False

    det_coverage = float(valid.sum() / max(T - t_start, 1))
    if valid.sum() == 0:
        return {
            "af_ate": np.nan, "af_ate_p95": np.nan, "af_ate_norm": np.nan,
            "af_det_coverage": det_coverage,
            "af_joint_frames": 0, "af_total_frames": T - t_start,
        }

    errors = np.linalg.norm(traj[valid] - A[valid], axis=1)
    result = {
        "af_ate": float(errors.mean()),
        "af_ate_p95": float(np.percentile(errors, 95)),
        "af_det_coverage": det_coverage,
        "af_joint_frames": int(valid.sum()),
        "af_total_frames": T - t_start,
    }
    if diag is not None:
        result["af_ate_norm"] = result["af_ate"] / diag
    return result


# ---------------------------------------------------------------------------
# 自测入口：纯 numpy 合成数据，验证投影与指标
#   python experiments/action_following/action_command.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 1) 命令轨迹：相机正前方 1m、无旋转的 EEF，投影应在图像中心附近
    #    actions: [x,y,z, qx,qy,qz,qw, grip, ...]（世界系）
    T = 5
    actions = np.zeros((T, 16), dtype=np.float64)
    for t in range(T):
        actions[t, 0:7] = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]   # 静止在 z=1m
    c2w = np.broadcast_to(np.eye(4), (T, 4, 4)).copy()          # 相机在世界原点
    K = np.array([[600.0, 0.0, 320.0],
                  [0.0, 600.0, 240.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)           # 640x480 分辨率
    A = commanded_trajectory(actions, c2w, K, sample_size=(384, 512), ori_size=(480, 640))
    # 640->512 缩放 0.8 → cx = 320*0.8 = 256；480->384 → cy = 240*0.8 = 192
    cx, cy = 512 / 640 * 320, 384 / 480 * 240
    print(f"[CHECK] 静止 EEF 命令轨迹 A0 = ({A[0,0]:.2f}, {A[0,1]:.2f})，期望 ≈ ({cx:.2f}, {cy:.2f})")
    assert np.allclose(A[:, 0], cx, atol=0.5) and np.allclose(A[:, 1], cy, atol=0.5), "投影中心偏"
    assert np.allclose(np.nan_to_num(A), np.nan_to_num(A[0]), atol=1e-3), "静止应不动"
    print("[PASS] 命令轨迹投影与缩放正确")

    # 2) af 指标：traj = A + 3px 噪声 → af_ate ≈ 3
    traj = A + np.random.default_rng(0).normal(0, 3, A.shape).astype(np.float32)
    r = action_following_metrics(traj, A, diag=640.0)
    print(f"[CHECK] af_ate = {r['af_ate']:.2f}，期望 ≈ 3.0")
    assert abs(r["af_ate"] - 3.0) < 0.5, "af_ate 应约等于噪声量级"
    assert r["af_det_coverage"] == 1.0 and r["af_joint_frames"] == T - 1
    print("[PASS] action_following_metrics 数值正确")

    # 3) NaN 处理：一半帧检测失败 → af_det_coverage ≈ 0.5
    traj2 = traj.copy()
    traj2[2:4] = np.nan
    r2 = action_following_metrics(traj2, A, diag=640.0)
    print(f"[CHECK] 一半帧 NaN → af_det_coverage = {r2['af_det_coverage']:.2f}，期望 0.5")
    assert abs(r2["af_det_coverage"] - 0.5) < 1e-9, "det coverage 应约等于 0.5"
    print("[PASS] NaN 帧被正确排除")
    print("\n自测通过")
