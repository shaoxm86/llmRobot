#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DMP_v2.py — 改进版 Dynamic Movement Primitives

V1 存在的问题:
  1. [致命] dt 不匹配: 主程序 generate_trajectory 用 dt=0.01 生成 200 点(2s),
     但 execute_cartesian_trajectory 用 dt=0.04 发送, 实际执行 200×0.04=8s,
     轨迹被拉长 4 倍。
  2. [核心] 位置-旋转速度曲线不同步: 位置 DMP 是临界阻尼弹簧(指数收敛,
     前 30% 时间走完 95% 路程), 旋转用线性 SLERP(匀速), 导致"位置快旋转慢"。
  3. [精度] Euler 积分: 小 tau 下数值不稳定, alpha=25/tau=0.5 时发散。
  4. [冗余] n_bfs=100 声明但从未使用: 无基函数、无强制项, 实际是 PD 控制器。
  5. [安全] 无速度限幅: 大距离+小 tau 可能生成超限速度。

V2 改进:
  1. 自适应 tau: 按起止距离动态调整运动时长, 远距慢、近距快。
  2. progress 同步: PositionDMP 输出归一化进度数组, QuaternionDMP 用其驱动
     SLERP, 位置和旋转速度曲线完全同步(都呈钟形)。
  3. RK4 积分: 4 阶 Runge-Kutta 替代 Euler, 小 tau 下数值稳定。
  4. 速度限幅: 可选最大速度约束。
  5. 清理 n_bfs: 点到点运动不需要基函数, 直接移除避免误导。
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
        self.cs_alpha = 1.0  # 规范化系统衰减系数

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

        # ===== 自适应 tau =====
        if tau is None:
            distance = np.linalg.norm(goal_pos - start_pos)
            tau = float(np.clip(dist_scale * distance, tau_min, tau_max))

        timesteps = max(int(tau / dt), 10)

        pos = start_pos.copy()
        vel = np.zeros(3, dtype=np.float64)

        pos_traj = []
        vel_traj = []
        x_traj = []  # 规范化变量轨迹, 用于计算 progress

        x = 1.0  # 规范化变量, 从 1 衰减到接近 0

        def dynamics(p, v):
            """二阶弹簧-阻尼系统: tau^2 * x'' = alpha * (beta * (g - x) - tau * x')"""
            acc = (self.alpha * (self.beta * (goal_pos - p) - tau * v)) / (tau ** 2)
            return v, acc

        for _ in range(timesteps):
            pos_traj.append(pos.copy())
            vel_traj.append(vel.copy())
            x_traj.append(x)

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

            # 规范化系统指数衰减
            x += (-self.cs_alpha * x / tau) * dt

        pos_traj = np.array(pos_traj)
        vel_traj = np.array(vel_traj)
        x_traj = np.array(x_traj)

        # ===== 计算归一化进度 =====
        # x 从 1 衰减到 x_final (接近 0 但不等于 0)
        # progress = (1 - x) / (1 - x_final), 保证 [0, 1] 严格归一化
        x_final = x_traj[-1]
        if x_final < 1.0 - 1e-6:
            progress = (1.0 - x_traj) / (1.0 - x_final)
        else:
            progress = np.ones(len(x_traj))

        # 兜底: 确保首尾精确为 0 和 1
        progress[0] = 0.0
        progress[-1] = 1.0

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