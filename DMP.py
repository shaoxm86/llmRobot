#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Point-to-point spring-damper trajectories with RK4 integration.

Positional progress is derived from the actual trajectory and normalized to the
endpoint; quaternion SLERP uses that same progress. Rotation-only motion uses
a quintic time law. This module does not learn demonstration trajectories.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


class PositionDMP:
    """
    3D 空间位置 DMP (Dynamic Movement Primitives)

    使用临界阻尼二阶弹簧-阻尼系统生成点到点平滑轨迹:
        tau^2 * x'' = alpha * (beta * (goal - x) - tau * x')

    临界阻尼条件: beta = alpha / 4 (无过冲, 无振荡)

    V2 新增:
      - 自适应 tau: tau = clip(dist_scale * distance, tau_min, tau_max)
      - RK4 积分替代 Euler
      - 输出 progress 数组供 QuaternionDMP 同步
      - 速度限幅
    """

    def __init__(self, alpha=25.0, beta=6.25):
        """
        :param alpha: 弹簧常数, 越大收敛越快
        :param beta:  阻尼系数, alpha/4 = 临界阻尼
        """
        self.alpha = alpha
        self.beta = beta


    def generate_traj(self, start_pos, goal_pos, tau=None, dt=0.01,
                            tau_min=0.8, tau_max=5.0, dist_scale=3.0,
                            max_vel=None):
        """
        生成从 start_pos 到 goal_pos 的 3D 位置与速度平滑轨迹。

        :param start_pos:   起始三维坐标 [x, y, z]
        :param goal_pos:    目标三维坐标 [x, y, z]
        :param tau:         运动持续总时间(秒)。None 则按距离自适应。
        :param dt:          采样时间间隔(秒), 必须与执行端 dt 一致!
        :param tau_min:     自适应最小时长(秒), 防止 tau 过小导致积分不稳定
        :param tau_max:     自适应最大时长(秒)
        :param dist_scale:  距离→时长比例系数(秒/米), 1m→3s, 0.1m→0.3s
        :param max_vel:     最大速度限幅[m/s], None 则不限
        :return: (pos_traj, vel_traj, progress)
            pos_traj:   (N, 3) 位置轨迹
            vel_traj:   (N, 3) 速度轨迹
            progress:   (N,)   归一化进度 [0, 1], 供 QuaternionDMP 同步使用
        """
        start_pos = np.array(start_pos, dtype=np.float64)
        goal_pos = np.array(goal_pos, dtype=np.float64)

        if start_pos.shape != (3,) or goal_pos.shape != (3,):
            raise ValueError('Positions must be 3-vectors')
        if not np.isfinite(start_pos).all() or not np.isfinite(goal_pos).all():
            raise ValueError('Positions must be finite')
        if dt <= 0 or (tau is not None and tau <= 0):
            raise ValueError('dt and tau must be positive')
        if max_vel is not None and max_vel <= 0:
            raise ValueError('max_vel must be positive')

        # ===== 自适应 tau =====
        if tau is None:
            distance = np.linalg.norm(goal_pos - start_pos)
            tau = float(np.clip(dist_scale * distance, tau_min, tau_max))

        timesteps = max(int(tau / dt), 10)

        pos = start_pos.copy()
        vel = np.zeros(3, dtype=np.float64)

        pos_traj = []
        vel_traj = []


        def dynamics(p, v):
            """二阶弹簧-阻尼系统: tau^2 * x'' = alpha * (beta * (g - x) - tau * x')"""
            acc = (self.alpha * (self.beta * (goal_pos - p) - tau * v)) / (tau ** 2)
            return v, acc

        for _ in range(timesteps):
            pos_traj.append(pos.copy())
            vel_traj.append(vel.copy())


            # ===== RK4 积分 (比 Euler 精度高, 小 tau 下更稳定) =====
            # 状态向量: [pos, vel], 动力学: [vel, acc]
            k1_p, k1_v = dynamics(pos, vel)
            k2_p, k2_v = dynamics(pos + 0.5 * dt * k1_p,
                                  vel + 0.5 * dt * k1_v)
            k3_p, k3_v = dynamics(pos + 0.5 * dt * k2_p,
                                  vel + 0.5 * dt * k2_v)
            k4_p, k4_v = dynamics(pos + dt * k3_p,
                                  vel + dt * k3_v)

            pos += (dt / 6.0) * (k1_p + 2 * k2_p + 2 * k3_p + k4_p)
            vel += (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

            # 速度限幅
            if max_vel is not None:
                speed = np.linalg.norm(vel)
                if speed > max_vel:
                    vel = vel * (max_vel / speed)

        pos_traj = np.array(pos_traj)
        vel_traj = np.array(vel_traj)
        displacement = goal_pos - start_pos
        distance_sq = np.dot(displacement, displacement)
        if distance_sq > 1e-16:
            # Use actual positional progress, not an unrelated canonical decay.
            progress = (pos_traj - start_pos) @ displacement / distance_sq
            if progress[-1] <= 0 or np.any(np.diff(progress) < -1e-8):
                raise ValueError('Unstable or non-monotonic trajectory')
            scale = progress[-1]
            progress = progress / scale
            pos_traj = start_pos + progress[:, None] * displacement
            vel_traj = vel_traj / scale
            if max_vel is not None and np.max(np.linalg.norm(vel_traj, axis=1)) > max_vel * (1 + 1e-6):
                raise ValueError('Requested duration cannot satisfy max_vel; increase tau')
        else:
            t = np.linspace(0., 1., timesteps)
            progress = 10*t**3 - 15*t**4 + 6*t**5
        progress[0], progress[-1] = 0., 1.

        return pos_traj, vel_traj, progress


class QuaternionDMP:
    """
    3D 空间姿态四元数 DMP (基于 SLERP 球面线性插值)

    V2 核心改进:
      接受 PositionDMP 输出的 progress 数组, 用其驱动 SLERP 时间参数。
      这样旋转的速度曲线与位置完全同步(都呈钟形),
      彻底消除"位置快旋转慢"的问题。

    原理:
      V1: SLERP 的时间参数 t = linspace(0, 1, N) → 匀速旋转
      V2: SLERP 的时间参数 t = progress → 与位置同频加速减速
    """

    def __init__(self):
        pass

    def generate_traj(self, start_rot, goal_rot, timesteps=None, progress=None):
        """
        生成从 start_rot 到 goal_rot 的四元数与旋转矩阵插值轨迹。

        :param start_rot:  起始旋转矩阵 (3x3) 或 四元数 [x, y, z, w]
        :param goal_rot:   目标旋转矩阵 (3x3) 或 四元数 [x, y, z, w]
        :param timesteps:  插值总步数 (progress=None 时用线性时间)
        :param progress:   [推荐] PositionDMP 输出的归一化进度数组 (N,)
                           若提供, 用其驱动 SLERP, 实现旋转与位移同步
        :return: (quat_traj, rot_mat_traj)
            quat_traj:    (N, 4) 四元数轨迹 [x, y, z, w]
            rot_mat_traj: (N, 3, 3) 旋转矩阵轨迹
        """
        # 1. 统一转换成 SciPy Rotation 对象
        if isinstance(start_rot, np.ndarray) and start_rot.shape == (3, 3):
            r_start = R.from_matrix(start_rot)
        else:
            r_start = R.from_quat(start_rot)

        if isinstance(goal_rot, np.ndarray) and goal_rot.shape == (3, 3):
            r_goal = R.from_matrix(goal_rot)
        else:
            r_goal = R.from_quat(goal_rot)

        # 2. 构建 SLERP 对象
        key_times = [0.0, 1.0]
        key_rots = R.from_quat([r_start.as_quat(), r_goal.as_quat()])
        slerp = Slerp(key_times, key_rots)

        # 3. 确定插值时间序列
        if progress is not None:
            # 【V2 核心】用位置 DMP 的进度驱动 SLERP
            # progress 已归一化到 [0, 1], 裁剪防止数值越界
            times = np.clip(progress, 0.0, 1.0)
        else:
            # 回退: 线性时间 (与 V1 行为一致)
            if timesteps is None:
                timesteps = 100
            times = np.linspace(0.0, 1.0, timesteps)

        interpolated_rots = slerp(times)

        # 4. 提取四元数和旋转矩阵
        quat_traj = interpolated_rots.as_quat()
        rot_mat_traj = interpolated_rots.as_matrix()

        return quat_traj, rot_mat_traj


# =========================================================================
# 辅助函数: 根据距离和角度自适应计算 tau
# =========================================================================
def compute_adaptive_tau(start_pos, goal_pos, start_rot=None, goal_rot=None,
                         dist_scale=3.0, angle_scale=0.5,
                         tau_min=0.8, tau_max=5.0):
    """
    根据位移距离和旋转角度综合计算自适应 tau。

    取位移 tau 和旋转 tau 的较大值, 确保两者都能在规定时间内完成。

    :param dist_scale:  距离→时长系数 (秒/米)
    :param angle_scale: 角度→时长系数 (秒/弧度)
    :param tau_min:     最小时长
    :param tau_max:     最大时长
    :return: tau (秒)
    """
    # 位移距离
    distance = np.linalg.norm(
        np.array(goal_pos) - np.array(start_pos))
    tau_dist = dist_scale * distance

    # 旋转角度
    tau_angle = 0.0
    if start_rot is not None and goal_rot is not None:
        r_start = R.from_matrix(np.array(start_rot))
        r_goal = R.from_matrix(np.array(goal_rot))
        # 相对旋转的角度 (弧度)
        rel_angle = (r_start.inv() * r_goal).magnitude()
        tau_angle = angle_scale * rel_angle

    # 取较大值, 确保位移和旋转都能从容完成
    tau = max(tau_dist, tau_angle)
    tau = float(np.clip(tau, tau_min, tau_max))

    return tau