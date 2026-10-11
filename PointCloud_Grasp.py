from data_utils import CameraInfo as Camera_Info, create_point_cloud_from_depth_image
import numpy as np
import open3d as o3d
import torch
from piper_pinocchio_mod2 import PiperIK

from Grasp_geometry import T_EYEINHAND, grasp_to_link6

# 点云处理：深度图 → 3D 点云 → SVD 求物体主轴 → 采样标准化
# 这是流水线中最昂贵的计算（每帧都跑的话 CPU 吃不消），
# 记录上一帧 SVD 的 X 轴方向，用于相邻帧间避免 SVD 随机 180° 翻转

def get_point_cloud(prev_x_axis, cam_info, BGRcolorData, depthData, maskData, device=None):
    """
    根据 RGB 图、深度图、分割掩码，生成 GraspNet 输入与 Open3D 可视化点云。

    处理流程:
      1. 深度图 → 3D 点云（针孔投影）
      2. 掩码过滤（只保留目标区域 + 深度 < 1m）
      3. 离群点剔除（90 百分位，去掉最远 10% 的噪点）
      4. SVD 求物体三个主轴方向（长/宽/厚）
      5. 时间序列惯性锁（防止相邻帧 SVD 180° 翻转）
      6. 均匀采样到 NUM_POINT=3000 个点
      7. 转 GPU 张量

    返回: (prev_x_axis, cloud_color_points, cloud_o3d, pts_mean, rot_mat)
          - prev_x_axis: 更新后的长轴方向，用于下一帧惯性锁
          - cloud_color_points: dict, GraspNet 模型输入 (GPU Tensor)
          - cloud_o3d: Open3D 点云对象（可视化用）
          - pts_mean: 物体质心在相机系下 [x,y,z]
          - rot_mat: 物体主轴旋转矩阵 (3x3)，列向量为三个主轴方向
    """
    # -- 步骤 1: 标准化 + 构建相机模型 --
    color = BGRcolorData.astype(np.float32) / 255.0
    depth = depthData
    mask = maskData

    width, height, fx, fy, cx, cy = (
        cam_info.width, cam_info.height,
        cam_info.fx, cam_info.fy,
        cam_info.cx, cam_info.cy
    )
    intrinsic = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    Scale = 1000.0  # 深度因子（mm → m）
    camera = Camera_Info(width, height, intrinsic[0][0], intrinsic[1][1],
                         intrinsic[0][2], intrinsic[1][2], Scale)
    # 将深度图转换为 3D 点云（organized=True 保留图像结构）
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    # -- 步骤 2: 掩码过滤 --
    # 只保留掩码有效 + 深度 < 1 米的点
    if mask.shape != depth.shape or color.shape[:2] != depth.shape:
        raise ValueError('Color, depth and mask dimensions must match')
    mask = (mask > .5) & (depth > 0) & (depth < 1000) & np.isfinite(depth)
    cloud_masked = cloud[mask]
    color_masked = color[mask]

    if len(cloud_masked) <= 20:
        print("[Warning] cloud_masked points too little or empty!")
        return None, None, None, None, None

    # -- 步骤 3: 离群点剔除 --
    # 用中值 + 90 百分位距离过滤，丢掉最远 10% 的异常飘点
    obj_points = cloud_masked.copy()
    cent = np.median(obj_points, axis=0)
    dist = np.linalg.norm(obj_points - cent, axis=1)
    valid_idx = dist < np.percentile(dist, 90)
    
    obj_points = obj_points[valid_idx]
    color_points = color_masked[valid_idx]  # 【修改】同步按索引剔除颜色中的离群点

    if len(obj_points) <= 3:
        print("[Warning] Points after outlier removal too few for SVD!")
        return None, None, None, None, None

    # -- 步骤 4: SVD 分解求物体三个主轴 --
    pts_mean = np.mean(obj_points, axis=0)
    pts_cen = obj_points - pts_mean
    U, S, Vt = np.linalg.svd(pts_cen, full_matrices=False)
    x_axis = Vt[0, :]  # 长轴方向（对应最大奇异值）
    y_axis = Vt[1, :]  # 中轴方向
    z_axis = Vt[2, :]  # 短轴方向（对应最小奇异值）

    # -- 步骤 5: 时间序列惯性锁 --
    if prev_x_axis is None:
        # 第一帧：固定初始朝向（倾向朝图像上方/前景方向）
        if x_axis[1] > 0:
            x_axis = -x_axis
    else:
        if np.dot(x_axis, prev_x_axis) < 0:
            x_axis = -x_axis
    prev_x_axis = x_axis.copy()

    # 固定 Z 轴指向相机
    if z_axis[2] > 0:
        z_axis = -z_axis
        
    # 【修改】确保 X 轴严格归一化
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    # 右手定则重算 Y 轴：Y = Z × X
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    # 组装旋转矩阵 (3x3)，列向量 = [X轴, Y轴, Z轴]
    rot_mat = np.vstack([x_axis, y_axis, z_axis]).T

    # -- 步骤 6: 均匀采样 3000 个点 --
    # 【修改】基于离群点剔除后的 obj_points 和 color_points 进行采样
    NUM_POINT = 3000
    if len(obj_points) >= NUM_POINT:
        idxs = np.random.choice(len(obj_points), NUM_POINT, replace=False)
    else:
        idxs1 = np.arange(len(obj_points))
        idxs2 = np.random.choice(len(obj_points), NUM_POINT - len(obj_points), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
        
    cloud_sampled, color_sampled = obj_points[idxs], color_points[idxs]

    # -- 步骤 7: 构建输出 --
    # Open3D 点云（【修改】使用离群点剔除后的 obj_points / color_points）
    cloud_o3d = o3d.geometry.PointCloud()
    cloud_o3d.points = o3d.utility.Vector3dVector(obj_points.astype(np.float32))
    cloud_o3d.colors = o3d.utility.Vector3dVector(color_points.astype(np.float32))

    # GraspNet 输入（GPU 张量）
    device = torch.device(device if device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    
    # 【修改】保证内存连续性
    cloud_sampled_cont = np.ascontiguousarray(cloud_sampled[np.newaxis].astype(np.float32))
    cloud_sampled_tensor = torch.from_numpy(cloud_sampled_cont).to(device)
    
    cloud_color_points = dict()
    cloud_color_points['point_clouds'] = cloud_sampled_tensor
    cloud_color_points['cloud_colors'] = color_sampled

    return prev_x_axis, cloud_color_points, cloud_o3d, pts_mean, rot_mat

# -------------------------------------------------------------------------------
from graspnetAPI import GraspGroup
from graspnet import pred_decode
from collision_detector import ModelFreeCollisionDetector

def score_grasp(all_grasps, obj_center, obj_rot):

    O_x = obj_rot[:, 0] / np.linalg.norm(obj_rot[:, 0])
    O_y = obj_rot[:, 1] / np.linalg.norm(obj_rot[:, 1])
    O_z = obj_rot[:, 2] / np.linalg.norm(obj_rot[:, 2])

    scored_grasps = []

    for grasp in all_grasps:
        # 提取当前抓取姿态的 3 个轴方向 (列向量)
        G_x = grasp.rotation_matrix[:, 0] / np.linalg.norm(grasp.rotation_matrix[:, 0])
        G_y = grasp.rotation_matrix[:, 1] / np.linalg.norm(grasp.rotation_matrix[:, 1])
        G_z = grasp.rotation_matrix[:, 2] / np.linalg.norm(grasp.rotation_matrix[:, 2]) # grasp的 z 轴

        if np.dot(G_z, obj_rot[:,0]) > 0: # grasp_z 与物体的 x 轴同向
            # 计算 grasp 的 x,y,z 与物体的 -z,y,x 轴的重合度
            target_x, target_y, target_z = -O_z, O_y, O_x

        else: # grasp_z 与物体的 x 轴反向
            # 计算 grasp 的 x,y,z 与物体的 -z,-y,-x 轴的重合度
            target_x, target_y, target_z = -O_z, -O_y, -O_x

    # 计算 3 个轴的点积（余弦相似度，值域 [-1, 1]，越接近 1 说明越接近重合）
        align_x = np.dot(G_x, target_x)
        align_y = np.dot(G_y, target_y)
        align_z = np.dot(G_z, target_z)

        # 组合三轴方向重合度得分（平均重合度，值域 [-1, 1]）
        rot_alignment_score = (align_x + align_y + align_z) / 3.0

        # 辅助评分: 抓取中心点与物体质心距离衰减项 (距离越小得分越高)
        dist = np.linalg.norm(grasp.translation - obj_center)
        dist_score = np.exp(-10.0 * dist)

        # 综合评分: 重合度得分 (权重 70%) + 质心距离 (权重 20%) + GraspNet 原生分数 (权重 10%)
        # 你可以根据实际抓取成功率调整这里的权重比例
        total_score = (0.7 * rot_alignment_score) + (0.2 * dist_score) + (0.1 * grasp.score)

        scored_grasps.append({
            'grasp': grasp,
            'score': total_score,
            'rot_score': rot_alignment_score,
            'dist_score': dist_score,
            'graspnet_score': grasp.score
        })

    # 按照 total_score 降序排列 (得分高的排在最前面)
    scored_grasps.sort(key=lambda x: x['score'], reverse=True)

    return scored_grasps


import time
def generate_grasps(net, cloud_color_points, cloud, vis, arm_ik, T_end2base, visual=False, obj_center=None, obj_rot=None, scene_points=None, table_z=0., clearance=.004):
    """
    基于 GraspNet 神经网络，预测机械爪该怎么抓物体。

    完整流水线:
      1. GraspNet 前向推理 → 生成大量候选抓取
      2. 碰撞检测 → 剔除会撞到物体/桌面的抓取 (< 1cm)
      3. NMS 去重 → 去掉重叠的冗余预测
      4. 几何打分与排序 → score_grasp
      5. 选取得分最高的抓取作为最终结果

    返回: best_grasp (GraspGroup, 仅含 1 个最优抓取) 或 None
    """
    Time1 = time.time()
    # 步骤 1: GraspNet 前向推理（不计算梯度）
    with torch.no_grad():
        cloud_color_points = net(cloud_color_points)  # 前向推理
        grasp_preds = pred_decode(cloud_color_points)  # 解码网络输出
        # 转成抓取组（包含多个候选抓取，每个含 score/位置/旋转/宽度/高度/深度）
        gg = GraspGroup(grasp_preds[0].detach().cpu().numpy())
    print(f"原始解码抓取数: {len(gg)}")
    # 步骤 2: 碰撞检测
    # 创建无模型碰撞检测器，检查每个抓取会不会撞到点云中的物体
    COLLISION_THRESH = 0.01  # ModelFreeCollisionDetector 的占用比例阈值（非米）
    if COLLISION_THRESH > 0:
        voxel_size = 0.01  # 体素大小
        collision_thresh = COLLISION_THRESH
        mfcdetector = ModelFreeCollisionDetector(
            np.asarray(cloud.points) if scene_points is None else scene_points, voxel_size=voxel_size)
        # detect() 逐一检查每个抓取姿态是否碰撞
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=collision_thresh)
        # 只保留不碰撞的（collision_mask 为 False）
        gg = gg[~collision_mask]

    # 步骤 3: NMS 非极大值抑制 + 按置信度排序
    print(f"碰撞检测后抓取数: {len(gg)}")
    # 把空间中位置极度接近、旋转角度非常相似的“冗余抓取”合并为一个高分抓取
    # translation_thresh: 位置距离阈值（单位：米），值越小保留越多
    # rotation_thresh: 旋转角度阈值（单位：弧度），值越小保留越多
    gg = gg.nms(translation_thresh=0.005, rotation_thresh=np.deg2rad(3))
    print(f"NMS 去重后抓取数: {len(gg)}")
    gg.sort_by_score()
    all_grasps = list(gg)
    
    if len(all_grasps) == 0:
        print("\n[Warning] GraspNet predicted 0 grasps.")
        return None
    
    ##################
    # Convert base-frame downward table normal to the capture camera frame.
    vertical = (T_end2base @ T_EYEINHAND)[:3, :3].T @ np.array([0., 0., -1.])
    angle_threshold = np.deg2rad(25)  # 25 度的弧度值
    filtered = []
    for grasp in all_grasps:
        # 抓取的接近方向取
        approach_dir = grasp.rotation_matrix[:, 0]
        # 计算[抓取的接近方向]与[垂直方向]的夹角
        approach_dir = approach_dir / np.linalg.norm(approach_dir)
        cos_angle = np.clip(np.dot(approach_dir, vertical), -1.0, 1.0)
        angle = np.arccos(cos_angle) # 反余弦，返回弧度值
        if angle < angle_threshold: filtered.append(grasp)
    if len(filtered) == 0:
        print("\n[Warning] No grasp predictions within vertical angle threshold. Rejecting unsafe approach directions.")
        return None
    else:
        print(f"\nFiltered {len(filtered)} grasps within 25° of vertical out of {len(all_grasps)} total predictions.")

    # 对过滤后的抓取根据 score 排序（降序）
    filtered.sort(key=lambda g: g.score, reverse=True)

    # Score without allocating unused Open3D meshes for every candidate.
    ranked_filterd = score_grasp(filtered, obj_center, obj_rot)
    if len(ranked_filterd) == 0: return None
    print("len(ranked_filterd):", len(ranked_filterd))

    best_grasp = None   # 循环前初始化
    for item in ranked_filterd: # filtered ; ranked_filterd
    
        T_obj2base_forlink6 = grasp_to_link6(item['grasp'], T_end2base, table_z, clearance)

        if PiperIK(arm_ik, T_obj2base_forlink6): # reach: 目标位置可达，返回 True（可执行），有碰撞返回 False
            b_grasp = item['grasp']  # item['grasp']
            best_grasp = GraspGroup()
            best_grasp.add(b_grasp)

            x_dir = b_grasp.rotation_matrix[:, 0]
            cos_val = np.clip(np.dot(x_dir / np.linalg.norm(x_dir), [0, 0, 1]), -1.0, 1.0)
            actual_angle = np.rad2deg(np.arccos(cos_val))
            print(f"选中抓取的 X 轴与 [0,0,1] 实际夹角: {actual_angle:.2f}°")

            break

    Time2 = time.time()
    print("GraspNet的推理时间为：", Time2-Time1)

    if best_grasp is None:
        print("[Warning] No IK-feasible grasp found.")
        return None

    # Window ownership belongs to the caller: return the grasp, never create
    # an untracked visualizer here. The caller uses update_grasp_view's return.
    return best_grasp