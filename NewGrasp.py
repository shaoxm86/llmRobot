#!/usr/bin/env python3
# Linux 可执行脚本头部声明，告诉系统用环境变量里的 python3 解释器运行本文件

# 基于CamDynamicsSamGrasp4.py修改

import os
import sys
import rospy
import cv2
import torch
import numpy as np
import threading
from scipy.spatial.transform import Rotation as R

from sensor_msgs.msg import Image, CameraInfo
from piper_msgs.msg import PiperEulerPose, PosCmd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # 当前脚本所在文件夹的上级目录的绝对路径
sys.path.append(os.path.join(ROOT_DIR))

from Model_initialization import model_init
from DMP import PositionDMP, QuaternionDMP
from Plotter_tools import RealtimeTrajectoryPlotter, VisualizeMask, update_grasp_view
from Position_detect import ObjectMotionDetector, quick_object_pos_in_base, FunObj2base
from PointCloud_Grasp import get_point_cloud, generate_grasps
from Piper_message import Color, Depth, CameraInfoMsg, End2base
from piper_pinocchio_mod2 import Arm_IK


# 轨迹执行，独立线程，不阻塞主循环
# is_executing 标志防止同一时间启动多个轨迹线程
# 轨迹执行结束后不解冻 motion_detector —— 解冻由主循环的距离判断驱动
is_executing = False  # 轨迹执行互斥锁

# 抓取状态锁
is_holding_object = False

# 中断当前 DMP 轨迹的信号事件
# 主循环在安全区检测到物体运动 + 轨迹执行中时 set，
# 轨迹线程在每次迭代中检查，若已 set 则 break 退出。
g_stop_traj = threading.Event()

def execute_cartesian_traj(pos_traj, rot_mat_traj, dt=0.05):
    """
    在独立线程中逐点发送笛卡尔位姿指令到 Piper 机械臂。

    参数:
        pos_traj:    位置轨迹 (N,3) [m]
        rot_mat_traj: 姿态轨迹 (N,3,3) 旋转矩阵
        dt:          控制周期 [s]，默认 0.01s = 100Hz

    注意:
        执行完毕后仅释放 is_executing 标志，并清空 g_dmp_target 全局变量。
        不解冻 motion_detector —— 解冻由主循环的距离判断驱动。
    """
    global is_executing, g_dmp_target_pos, g_dmp_target_rot, is_holding_object
    
    rate = rospy.Rate(1.0 / dt)  # 100 Hz 控制频率
    # pos_traj是完整的轨迹，按照 100 Hz 执行
    for i in range(len(pos_traj)):
        if rospy.is_shutdown(): break

        # 检测到中断信号 → 立即退出当前轨迹（g_stop_trajectory.is_set()：查询事件是否被设置（是否触发停止信号））
        if g_stop_traj.is_set():
            rospy.loginfo("[Trajectory] 检测到目标位姿变化，中断当前轨迹")
            break

        pos = pos_traj[i]
        rot = rot_mat_traj[i]
        # 旋转矩阵 → XYZ 欧拉角
        r = R.from_matrix(rot)
        euler = r.as_euler('xyz', degrees=False)
        # 构造 Piper 欧拉角位姿消息
        msg = PosCmd()
        msg.x, msg.y, msg.z = pos[0], pos[1], pos[2]
        msg.roll, msg.pitch, msg.yaw = euler[0], euler[1], euler[2]
        msg.gripper = 0.06
        # 发布到笛卡尔空间控制话题 /pos_cmd
        arm_pub.publish(msg)

        # 更新 DMP 期望位姿 → 主循环读取后在 Matplotlib 中绘制橙色菱形
        g_dmp_target_pos = pos.copy()
        g_dmp_target_rot = rot.copy()

        rate.sleep()

    rospy.sleep(1)

    # ---------- 夹紧夹爪 ---------- 
    msg = PosCmd()
    msg.x, msg.y, msg.z = pos[0], pos[1], pos[2]
    msg.roll, msg.pitch, msg.yaw = euler[0], euler[1], euler[2]
    msg.gripper = 0.03
    arm_pub.publish(msg)
    is_holding_object = True  # 标记：已成功抓取，进入搬运阶段
    print("夹紧夹爪...")
    rospy.sleep(2)
    
    # ---------- 到放置位置后张开夹爪 ---------- 
    # joint = [0.920, 1.231, -1.303, 0.0, 1.039, 0.0, 0.035]
    msg = PosCmd()
    msg.x, msg.y, msg.z = 0.1478, 0.194, 0.3935
    msg.roll, msg.pitch, msg.yaw = 3.14159, 0.6916, -2.221
    msg.gripper = 0.03
    arm_pub.publish(msg)
    print("放到指定位置...")
    rospy.sleep(1.5)
    # 张开夹爪
    msg = PosCmd()
    msg.x, msg.y, msg.z = 0.1478, 0.194, 0.3935
    msg.roll, msg.pitch, msg.yaw = 3.14159, 0.6916, -2.221
    msg.gripper = 0.08
    arm_pub.publish(msg)
    print("张开夹爪...")
    rospy.sleep(1)
    
    # ---------- 回到起始位置---------- 
    # joint = [0.920, 1.231, -1.303, 0.0, 1.039, 0.0, 0.035]
    msg = PosCmd()
    msg.x, msg.y, msg.z = 0.12, 0.0, 0.36
    msg.roll, msg.pitch, msg.yaw = 3.14159, 0.777, 3.15159
    msg.gripper = 0.06
    arm_pub.publish(msg)
    print("回到起始位置...")
    rospy.sleep(3)
    print("[Trajectory] 轨迹执行完成")
    # is_holding_object = False # 标记：已释放，恢复视觉

    rospy.signal_shutdown("Task Finish !!!")

    is_executing = False  # 执行完毕，释放互斥锁
    # 轨迹结束，清空期望位姿 → 橙色菱形消失
    g_dmp_target_pos = None
    g_dmp_target_rot = None
    print("[Trajectory] 轨迹执行完成")

import time
# DMP 轨迹工具函数 —— 封装"规划 + 启动执行线程"的重复代码
def plan_execute_traj(pos_dmp, quat_dmp, start_pos, start_rot, target_pos, target_rot, plotter):
    """
    统一的 DMP 轨迹规划 + 启动执行

    步骤:
      1. 位置 DMP:  起点→终点 3D 直线平滑轨迹 (tau=2s, dt=10ms)
      2. 姿态 DMP:  SLERP 四元数球面插值, timesteps 与位置轨迹对齐
      3. 更新 Matplotlib 可视化参考轨迹
      4. 启动独立线程执行笛卡尔轨迹
      
    返回: True [表示已启动]
    """
    # 位置 DMP: 生成从 start_pos 到 target_pos 的平滑轨迹
    # tau = 2.0 表示大约 2 秒到达，dt=0.01 表示 100 Hz
    Time1 = time.time()
    pos_traj, _, progress = pos_dmp.generate_traj(start_pos, target_pos, tau=None, dt=0.01, dist_scale=100.0)    
    Time2 = time.time()
    print("DMP规划耗时1： ", Time2-Time1)
    # 姿态 DMP: SLERP 四元数球面插值
    _, rot_mat_traj = quat_dmp.generate_traj(start_rot, target_rot, progress=progress) 
    Time3 = time.time()
    print("DMP规划耗时2： ", Time2-Time1)
    rospy.loginfo("[DMP] 轨迹: %s → %s, 共 %d 点", np.round(start_pos, 3), np.round(target_pos, 3), len(pos_traj))
    # 更新可视化中的参考轨迹
    plotter.reset_planned_traj(pos_traj, target_pos, target_rot)

    # 启动独立线程执行轨迹，不阻塞主循环
    # daemon=True，守护线程，生命周期完全依赖于主线程，只要主程序退出（比如按了 Ctrl + C），该守护线程无论是否执行完，都会被系统强制立即终止
    traj_thread = threading.Thread(target=execute_cartesian_traj, args=(pos_traj, rot_mat_traj, 0.01), daemon=True) 
    traj_thread.start()

    return True


if __name__ == '__main__':

    rospy.init_node('yoloworld_sam2_graspnet', anonymous=True) # ROS 节点初始化

    # 先运行 (Grasp) roslaunch realsense2_camera rs_aligned_depth.launch
    colorImage = Color()   # 订阅彩色图
    rospy.Subscriber('/camera/color/image_raw', Image, colorImage.Callback, queue_size=10)
    depthImage = Depth()   # 订阅对齐后的深度图
    rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, depthImage.Callback, queue_size=10)
    cam_info = CameraInfoMsg()  # 订阅相机内参
    rospy.Subscriber('/camera/color/camera_info', CameraInfo, cam_info.Callback, queue_size=10)
    robEnd2base = End2base()    # 订阅末端位姿
    rospy.Subscriber('/end_pose_euler', PiperEulerPose, robEnd2base.Callback, queue_size=10)

    # 发布 /pos_cmd 话题，笛卡尔空间控制方式，xyz[m], rxryrz[rad], gripper[0-0.04m]
    arm_pub = rospy.Publisher('/end_pose_cmdtry', PosCmd, queue_size=10) # /end_pose_cmdtry, /pos_cmd

    yolo, sam, net = model_init() # 初始化yolo, sam, GraspNet(net)

    # set parameters
    yolo.set_classes(["remote"])  # 指定检测目标类别
    ALPHA = 15 # DMP 参数
    GRASP_FREEZE_DISTANCE= 0.2 # 冻结距离
    GRASP_UNFREEZE_DISTANCE = 0.2 # 解冻距离
    
    # 初始化 DMP
    pos_dmp = PositionDMP(alpha=ALPHA, beta=ALPHA/4)
    quat_dmp = QuaternionDMP()
    # 初始化-可视化
    plotter = RealtimeTrajectoryPlotter()

    # ===== 运动检测器 =====
    # ema_alpha=0.25:  新值权重 25%，约 4 帧收敛; pos_threshold=0.03: 位移超 3cm 判定为运动
    # cooldown_sec=1.0:  两次重算至少间隔 1 秒; aspect_threshold=0.3,  # 宽高比变化 > 0.3 判定为旋转
    motion_detector = ObjectMotionDetector(GRASP_FREEZE_DISTANCE, GRASP_UNFREEZE_DISTANCE,
                                           ema_alpha=0.25, pos_threshold=0.05, aspect_threshold=0.5, cooldown_sec=0.5)
    
    # DMP 当前发布的期望位姿（全局变量，轨迹执行线程写入，主循环读取）
    g_dmp_target_pos = None  # DMP 当前发布的期望位置 [x,y,z]
    g_dmp_target_rot = None  # DMP 当前发布的期望姿态 (3x3)

    # ===== 状态变量（跨帧缓存） =====
    saved_boxes = None       # SAM2 追踪框（跨帧继承）
    cached_best_grasp = None  # 上一轮的最优抓取（物体不动时复用）
    cached_cloud_o3d = None   # 上一轮的点云（可视化复用）
    cached_pts_mean = None    # 上一轮的物体质心
    cached_rot_mat = None     # 上一轮的物体主轴旋转
    last_mask = None          # 上一帧的 SAM2 分割掩码
    prev_x_axis = None        # 物体 x 的方向轴
    vis = None                # 保留一个可视画面

    # 创建Arm_IK实例
    arm_ik = Arm_IK()

    # ---------- 回到起始位置---------- 
    # joint = [0.920, 1.231, -1.303, 0.0, 1.039, 0.0, 0.035]
    msg = PosCmd()
    msg.x, msg.y, msg.z = 0.12, 0.0, 0.36
    msg.roll, msg.pitch, msg.yaw = 3.14159, 0.777, 3.15159
    msg.gripper = 0.05
    arm_pub.publish(msg)
    print("回到起始位置...")
    rospy.sleep(2)

    rate = rospy.Rate(30)  # 主循环频率 30Hz

    while not rospy.is_shutdown():
        # 检查图像数据是否就绪
        if colorImage.BGRdata is None or depthImage.data is None: 
            rate.sleep()
            continue

        frame = colorImage.BGRdata.copy()
        T_curr = robEnd2base.T_end2base  # 当前link6→基座 齐次变换
        current_tcp_pos = T_curr[:3, 3].copy()  # 当前末端位置
        current_tcp_rot = T_curr[:3, :3].copy() # 当前末端姿态

        # ====================== 第 1 层：YOLO 目标检测 ================================
        # 用途：在画面中定位目标物体，输出检测框给 SAM2 作为初始跟踪提示
        # 注意：这里每帧都跑 YOLO 是为了在目标丢失后能重新检测到
        yolo_results = yolo(frame, verbose=False)
        boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
        if len(boxes) > 0: saved_boxes = boxes  # 更新追踪框
        else:
            # YOLO 未检出 → 显示画面并跳过本帧
            cv2.imshow('Result', frame)
            if cv2.waitKey(1) == ord('q'): break
            rate.sleep()
            continue

        # ===================== 第 2 层：SAM2 视频追踪（每帧必跑）=========================
        # 用途：对目标物体进行像素级分割，输出掩码 mask
        # autocast bf16: 混合精度推理，降低显存占用、提升速度
        # SAM2 的 tracked boxes 会随物体运动自动更新边界框
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            sam_results = sam(frame, bboxes=saved_boxes, verbose=False)

        if sam_results[0].masks is None:
            rate.sleep()
            continue

        # 提取掩码并更新追踪框
        masks = sam_results[0].masks.data.cpu().numpy()
        if sam_results[0].boxes is not None and len(sam_results[0].boxes) > 0:
            # SAM2 追踪更新后的新框 → 下一帧继续用这个框追踪
            saved_boxes = sam_results[0].boxes.xyxy.cpu().float().numpy()

        # 将掩码渲染到画面上（半透明蓝色覆盖）
        for m in masks:
            if m.shape[:2] != frame.shape[:2]:
                m = cv2.resize(m, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            frame = VisualizeMask(frame, m, [255, 0, 0])
        last_mask = masks[0]  # 缓存本帧掩码，供后续使用

        # ======================= 第 3 层：带距离保护的智能运动检测 ========================
        # 步骤 A: 快速估计物体在基座系下的位置
        quick_pos, pos_valid, _, aspect_ratio = quick_object_pos_in_base(saved_boxes, depthImage.data, cam_info, T_curr, mask=last_mask)

        # 步骤 B: 计算末端到物体的直线距离
        # 优先用 EMA 滤波位置（稳定），其次用原始测量值，都不可用则设为无穷大
        # motion_detector.filtered_pos: 经降噪处理后，物体当前真正所在的位置
        if pos_valid and motion_detector.filtered_pos is not None:
            dist_to_object = np.linalg.norm(current_tcp_pos - motion_detector.filtered_pos)
        elif pos_valid and motion_detector.filtered_pos is None:
            # 首次测量，filtered_pos 尚未初始化，直接用原始值
            dist_to_object = np.linalg.norm(current_tcp_pos - quick_pos)
        else:
            dist_to_object = float('inf')
        # print("当前位置到目标位置的距离为：", dist_to_object)

        # 步骤 C: 三区判断（带回滞，防止临界振荡）
        #  距离分布:  ── 冻结区 ── 回滞区 ─── 安全区 ─────
        #           0           10cm       15cm
        #           冻结检测      保持原状    EMA 滤波运动检测
        #           锁定目标      不换状态    物体动→重算，不动→复用
        if dist_to_object < GRASP_FREEZE_DISTANCE:
            # rospy.loginfo("冻结区...")
            # 进入冻结模式（幂等操作，只有未冻结时实际切换）
            if not motion_detector.frozen: motion_detector.freeze()
            # 冻结时不触发重算 —— 这是解决 "末端靠近物体时深度噪声误判" 的关键
            need_recompute = False
            pos_valid = False
        else:
            # rospy.loginfo("安全区...")
            # 如果之前被冻结了，现在解冻恢复检测
            if motion_detector.frozen: motion_detector.unfreeze()

        # 正常运动检测：EMA 滤波 + 位移判断
        # 如果正在搬运物体，直接跳过运动检测与重算
        if is_holding_object:
            need_recompute = False
        elif pos_valid:
            need_recompute, filtered_pos = motion_detector.update(quick_pos, raw_aspect=aspect_ratio)

        # ============================== 第 4 层：按需执行昂贵计算 ===================================
        # 仅当 need_recompute=True 时才走完整流水线：
        # get_point_cloud → SVD → GraspNet 推理 → 碰撞检测 → 打分排序 → DMP 规划
        if need_recompute:
            rospy.loginfo("[MainLoop] 触发完整流水线: get_point_cloud → GraspNet → DMP")

            # 步骤 1: 构建点云 + SVD 求物体主轴
            prev_x_axis, cloud_color_points, cloud_o3d, pts_mean, rot_mat = get_point_cloud(prev_x_axis, cam_info, colorImage.BGRdata, depthImage.data, last_mask)

            if cloud_o3d is not None and cloud_color_points is not None:
                # 缓存点云和位姿信息（物体不动时复用）
                cached_cloud_o3d = cloud_o3d
                cached_pts_mean = pts_mean
                cached_rot_mat = rot_mat

                # 步骤 2: GraspNet 生成最优抓取
                best_grasp = generate_grasps(net, cloud_color_points, cloud_o3d, vis, arm_ik, T_curr, visual=True, obj_center=pts_mean, obj_rot=rot_mat)

                if best_grasp is not None:
                    cached_best_grasp = best_grasp  # 缓存抓取结果

                    # 步骤 3: 坐标变换 + DMP 轨迹规划
                    target_grasp_cam = best_grasp[0]
                    start_pos = T_curr[:3, 3]
                    start_rot = T_curr[:3, :3]
                    _, target_pos, target_rot = FunObj2base(target_grasp_cam, T_curr)

                    # 步骤 4: 启动 / 中断并重启轨迹执行
                    #  无轨迹执行中 (not is_executing):  正常启动新轨迹
                    #  轨迹执行中 (is_executing):        设置中断信号，
                    #  轨迹线程检测后 break 退出 → 下帧恢复块自动重规划
                    if is_executing:
                        rospy.loginfo("[MainLoop] 物体运动，中断当前 DMP 轨迹以便重规划")
                        g_stop_traj.set() # 把 flag 设为True，给轨迹执行线程execute_cartesian_traj发送中断信号：要求终止正在运行的 DMP 轨迹
                    else:
                        is_executing = True
                        plan_execute_traj(pos_dmp, quat_dmp, start_pos, start_rot, target_pos, target_rot, plotter)
                        
        else:
            # 未触发重算：仅更新 Open3D 可视化（复用缓存数据）
            if cached_best_grasp is not None and cached_cloud_o3d is not None:
                grippers = cached_best_grasp.to_open3d_geometry_list()
                vis = update_grasp_view(vis, cached_cloud_o3d, grippers, obj_center=cached_pts_mean, obj_rot=cached_rot_mat)
                
        # ===== 重规划恢复块：上一帧因 DMP 执行中而中断，现在轨迹线程已退出 =======
        # 触发条件:
        #   1. g_stop_traj 被 set（说明上一帧需要重规划但轨迹在执行中）
        #   2. is_executing == False（轨迹线程已 break 退出并释放标志）
        # 此时用最新的 cached_best_grasp（已经在上一帧的 need_recompute 块中更新）
        # 重新规划并启动 DMP 轨迹
        # 注意：此块独立于 need_recompute，即使在冷却期也会执行
        if g_stop_traj.is_set() and not is_executing: # 已经发出中断，且旧轨迹已结束，现在可以执行重规划
            if cached_best_grasp is not None:
                # 存在有效的抓取缓存
                g_stop_traj.clear() # 把 flag 设为 False
                rospy.loginfo("[MainLoop] 重新规划 DMP 轨迹（目标已更新）")
                target_grasp_cam = cached_best_grasp[0]
                start_pos = T_curr[:3, 3]
                start_rot = T_curr[:3, :3]
                _, target_pos, target_rot = FunObj2base(target_grasp_cam, T_curr)
                is_executing = True
                plan_execute_traj(pos_dmp, quat_dmp, start_pos, start_rot, target_pos, target_rot, plotter)
            else: 
                # 没有可用抓取
                # GraspNet 失败 → 无有效目标，清除过期中断信号
                g_stop_traj.clear()

        # ======= 第 5 层：末端轨迹可视化（始终更新，无论是否重算） =========
        # 实时显示：机械臂末端实际走过的路径 vs DMP 规划的参考轨迹
        # 同时显示 DMP 当前发布的期望位姿（橙色菱形），沿红线逐点移动
        plotter.add_actual_point(current_tcp_pos)  # 追加到实际轨迹列表
        plotter.update_plot(current_pos=current_tcp_pos, current_rot=current_tcp_rot, dmp_target_pos=g_dmp_target_pos, dmp_target_rot=g_dmp_target_rot)

        # 显示带掩码渲染的图像
        cv2.imshow('Result', frame)
        if cv2.waitKey(1) == ord('q'): break
        
        rate.sleep()

    # ===== 清理 =====
    cv2.destroyAllWindows()
        