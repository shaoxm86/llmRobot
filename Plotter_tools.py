import cv2
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 可视化工具：将 SAM2 掩码以半透明彩色叠加到原图上
def VisualizeMask(image, mask, color, alpha=0.4):
    """
    将掩码区域以指定颜色半透明叠加到原图上。

    参数:
        image: BGR 图像
        mask:  二值掩码 (H,W)，>0.5 认为是前景
        color: BGR 颜色三通道 [B,G,R]
        alpha: 掩码透明度 0~1，0.4 = 40% 掩码 + 60% 原图
    """
    bool_mask = mask > 0.5  # 二值化
    colored_mask = image.copy()
    colored_mask[bool_mask] = color  # 掩码区域填充指定颜色
    # 两张图加权融合: result = mask_img × α + original × (1-α)
    return cv2.addWeighted(colored_mask, alpha, image, 1 - alpha, 0)


class RealtimeTrajectoryPlotter:
    def __init__(self, axis_scale=0.1):
        plt.ion()  # 开启交互模式，非阻塞绘图
        self.fig = plt.figure(figsize=(8, 7))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.axis_scale = axis_scale  # 位姿坐标轴箭头的长度（单位：米）
        # 内部变量
        self.planned_pos_traj = None  # DMP 规划路径 (N, 3)
        self.goal_pos = None          # 目标位置 [x, y, z]
        self.goal_rot = None          # 目标姿态 (3, 3)
        self.actual_history = []      # 机械臂实际运动轨迹点列表

    def reset_planned_traj(self, pos_traj, goal_pos, goal_rot=None):
        """当更新 DMP 轨迹时重置规划路径与实际历史路径"""
        self.planned_pos_traj = np.array(pos_traj)
        self.goal_pos = np.array(goal_pos)
        self.goal_rot = np.array(goal_rot) if goal_rot is not None else None
        self.actual_history = []  # 重置实际运动历史

    def add_actual_point(self, current_pos):
        """记录当前实时末端点"""
        self.actual_history.append(np.array(current_pos).copy())

    def _draw_frame(self, pos, rot_mat, linestyle='-', linewidth=2, alpha=1.0, prefix=''):
        """
        辅助函数：在指定位置绘制 RGB 三色 3D 坐标轴
        红色 = X 轴，绿色 = Y 轴，蓝色 = Z 轴
        """
        if pos is None or rot_mat is None:
            return
        # 提取旋转矩阵的列向量作为 X, Y, Z 轴方向
        x_dir = rot_mat[:, 0] * self.axis_scale
        y_dir = rot_mat[:, 1] * self.axis_scale
        z_dir = rot_mat[:, 2] * self.axis_scale
        # 绘制 X 轴 (红色)
        self.ax.quiver(pos[0], pos[1], pos[2], x_dir[0], x_dir[1], x_dir[2],
                       color='r', linestyle=linestyle, linewidth=linewidth*2, alpha=alpha)
        # 绘制 Y 轴 (绿色)
        self.ax.quiver(pos[0], pos[1], pos[2], y_dir[0], y_dir[1], y_dir[2],
                       color='g', linestyle=linestyle, linewidth=linewidth, alpha=alpha)
        # 绘制 Z 轴 (蓝色)
        self.ax.quiver(pos[0], pos[1], pos[2], z_dir[0], z_dir[1], z_dir[2],
                       color='b', linestyle=linestyle, linewidth=linewidth, alpha=alpha)

    def update_plot(self, current_pos=None, current_rot=None, dmp_target_pos=None, dmp_target_rot=None):
        """
        非阻塞刷新画布，绘制规划轨迹 vs 实际轨迹对比。

        V5 新增参数:
            dmp_target_pos: DMP 当前发布的期望位置 [x,y,z]，橙色菱形标记
            dmp_target_rot: DMP 当前发布的期望姿态 (3x3)，橙色坐标轴
            这两个值由 execute_cartesian_trajectory 线程每步写入全局变量，
            主循环读取后传入此处。轨迹执行完毕后置为 None，不再显示。
        """
        if self.planned_pos_traj is None:
            return
        self.ax.cla()  # 清空画布准备重绘
        self.ax.set_title("DMP Planned vs Actual Motion (Base Frame)", fontsize=11)
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_zlabel("Z (m)")

        # 1. DMP 规划的理想路径 (红色虚线)
        self.ax.plot(self.planned_pos_traj[:, 0], self.planned_pos_traj[:, 1],
                     self.planned_pos_traj[:, 2], 'r--', label='DMP Traj',
                     linewidth=1.5, alpha=0.7)

        # 2. DMP 当前发布的期望位姿 (橙色菱形 + 橙色坐标轴)
        #    这是正在发给机械臂的目标点位，沿红色虚线从头走到尾
        if dmp_target_pos is not None:
            self.ax.scatter(dmp_target_pos[0], dmp_target_pos[1], dmp_target_pos[2],
                            color='orange', marker='D', s=80, label='DMP Target')
            if dmp_target_rot is not None:
                self._draw_frame(dmp_target_pos, dmp_target_rot,
                                 linestyle='-', linewidth=2.5, alpha=0.9)

        # 3. 机械臂末端实际运动历史轨迹 (蓝色实线)
        if len(self.actual_history) > 1:
            actual_arr = np.array(self.actual_history)
            self.ax.plot(actual_arr[:, 0], actual_arr[:, 1], actual_arr[:, 2],
                         'b-', label='Actual Path', linewidth=2)

        # 4. 期望到达的目标位姿 (红色星号 + 虚线 RGB 坐标轴)
        if self.goal_pos is not None:
            self.ax.scatter(self.goal_pos[0], self.goal_pos[1], self.goal_pos[2],
                            color='red', marker='*', s=120, label='Goal Pose')
            if self.goal_rot is not None:
                self._draw_frame(self.goal_pos, self.goal_rot,
                                 linestyle='--', linewidth=2, alpha=0.8)

        # 5. 当前机械臂末端位姿 (绿色圆点 + 实线 RGB 坐标轴)
        if current_pos is not None:
            self.ax.scatter(current_pos[0], current_pos[1], current_pos[2],
                            color='green', marker='o', s=60, label='Current TCP')
            if current_rot is not None:
                self._draw_frame(current_pos, current_rot, linestyle='-', linewidth=2.5, alpha=1.0)

        # 自动调整坐标轴显示范围，保证各轴比例一致
        all_pts = []
        if self.planned_pos_traj is not None:
            all_pts.append(self.planned_pos_traj)
        if len(self.actual_history) > 0:
            all_pts.append(np.array(self.actual_history))
        if all_pts:
            all_pts = np.vstack(all_pts)
            max_range = np.array([all_pts[:, 0].max() - all_pts[:, 0].min(),
                                  all_pts[:, 1].max() - all_pts[:, 1].min(),
                                  all_pts[:, 2].max() - all_pts[:, 2].min()]).max() / 2.0
            mid_x = (all_pts[:, 0].max() + all_pts[:, 0].min()) * 0.5
            mid_y = (all_pts[:, 1].max() + all_pts[:, 1].min()) * 0.5
            mid_z = (all_pts[:, 2].max() + all_pts[:, 2].min()) * 0.5
            r = max(max_range, 0.15)  # 最小范围 15cm
            self.ax.set_xlim(mid_x - r, mid_x + r)
            self.ax.set_ylim(mid_y - r, mid_y + r)
            self.ax.set_zlim(max(0, mid_z - r), mid_z + r)
        self.ax.legend(loc='upper right', fontsize=8)
        self.ax.grid(True)
        # 刷新 3D 图形窗口
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        """一轮执行完毕后，关闭 Matplotlib 绘图窗口"""
        plt.close(self.fig)
        plt.ioff()  # 关闭交互模式

# 全局可视化器（Open3D 单例，全程只开一个窗口）
def init_vis_window(vis, window_name="Grasp Visualizer", width=900, height=700):
    """初始化 Open3D 可视化窗口（仅首次调用时创建，后续复用）"""
    if vis is None:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name, width, height)
        # 设置相机视角为俯视图
        view_ctl = vis.get_view_control()
        view_ctl.set_front([0, 0, -1])  # 相机面朝 Z 轴负方向（从上往下看）
        view_ctl.set_up([0, -1, 0])     # 设置相机的正上方向
    return vis

def update_grasp_view(vis, cloud, grippers, obj_center=None, obj_rot=None):
    """
    持续更新 Open3D 窗口内容，不关闭窗口。
    显示：物体点云 + 预测抓取夹爪模型 + 物体自身坐标系[RGB三轴]
    支持鼠标自由拖拽且不重置视角。
    """
    is_first_frame = (vis is None)
    if is_first_frame:
        vis = init_vis_window(vis)
        cam_params = None
    else:
        # 非第一帧：保存上一帧用户调整好的视角
        try:
            view_ctl = vis.get_view_control()
            cam_params = view_ctl.convert_to_pinhole_camera_parameters()
        except Exception:
            cam_params = None

    # 清空上一帧所有几何体
    vis.clear_geometries()
    
    # 添加点云（第一帧允许更新包围盒，之后禁止重置）
    vis.add_geometry(cloud, reset_bounding_box=is_first_frame)
    
    # 添加所有夹爪模型
    for g in grippers:
        vis.add_geometry(g, reset_bounding_box=False)
        
    # 绘制物体专属坐标系（RGB 三轴）
    if obj_center is not None and obj_rot is not None:
        obj_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.10, origin=[0, 0, 0])
        obj_frame.rotate(obj_rot, center=(0, 0, 0))
        obj_frame.translate(obj_center)
        vis.add_geometry(obj_frame, reset_bounding_box=False)

    view_ctl = vis.get_view_control()

    if is_first_frame:
        # 仅在第一帧加载好几何体后，设置初始俯视视角
        view_ctl.set_front([0, 0, -1])  # 相机面朝 Z 轴负方向
        view_ctl.set_up([0, -1, 0])     # 设置相机的正上方向
    elif cam_params is not None:
        # 后续帧：恢复用户通过鼠标改变后的视角
        view_ctl.convert_from_pinhole_camera_parameters(cam_params)

    # 刷新渲染与窗口事件
    vis.poll_events()
    vis.update_renderer()
    return vis

def close_vis(vis):
    """程序结束时销毁 Open3D 窗口"""
    if vis is not None:
        vis.destroy_window()
        vis = None