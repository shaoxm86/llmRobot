import time
import rospy
import numpy as np

# 相机光心到关节 6 末端的齐次变换矩阵（眼在手上）
# 来源：SolidWorks 标定结果
T_EYEINHAND = np.array([
    [0,  0.9397, 0.342, -0.07037],
    [-1, 0,      0,      0.04228],
    [0, -0.342,  0.9397, 0.04115],
    [0,  0,      0,      1]
])

# 带冻结机制的运动检测器
class ObjectMotionDetector:
    """
    物体运动检测器 —— 回答"物体在基座系下是否真的移动了"。

    ===== 两层保护 =====
    1. EMA 指数移动平均滤波
       作用：对抗相机视角变化引入的高频噪声。
       公式：filtered = α × raw + (1-α) × filtered_prev
             其中 α = 0.25，新值权重 25%，旧值权重 75%。
       物理含义：相机视角变化产生的噪声是高频的（单帧跳 1-3cm），EMA 天然压制高频、保留低频（物体真动是持续的）。
       等效时间常数约 4 帧（~0.13s @30Hz）。

    2. 距离冻结机制
       作用：当机械臂末端距离物体 < 10cm 时，锁定目标不重算。
       原因：近距离下 D435i 深度噪声极大（可能跳 5-10cm），EMA 仍会逐渐漂移导致误判。此时机械臂马上就要抓到了，重算毫无意义。
       解冻：距离 > 15cm 时自动恢复检测（5cm 回滞防振荡）。

    ===== 冻结时的行为 =====
    - EMA 照常更新（保持 filtered_pos 准确，为后续解冻做准备）
    - update() 永远返回 need_recompute=False
    - 解冻仅由主循环的距离判断驱动，不由轨迹执行线程驱动
    """

    def __init__(self, GRASP_FREEZE_DISTANCE=0.10, GRASP_UNFREEZE_DISTANCE=0.15,
                 ema_alpha=0.25, pos_threshold=0.03,aspect_threshold=0.3, cooldown_sec=1.0):
        """
        参数:
            ema_alpha:        EMA 平滑系数（越小越平滑，越大越灵敏）
            pos_threshold:    位移阈值 [m]，滤波位置超此值判定为移动（默认 3cm）
            aspect_threshold: 宽高比变化阈值，超此值判定为旋转（默认 0.3）
                             例: 3.0→2.5 变化 0.5 > 0.3 → 触发
            cooldown_sec:     冷却时间 [s]，两次触发重算的最小间隔（默认 1s）
        """
        self.ema_alpha = ema_alpha
        self.pos_threshold = pos_threshold
        self.aspect_threshold = aspect_threshold  # 【V7 新增】
        self.cooldown_sec = cooldown_sec

        self.filtered_pos = None      # EMA 滤波后的物体位置 [x,y,z]
        self.filtered_aspect = None   # 【V7 新增】EMA 滤波后的 bbox 宽高比
        self.last_grasp_pos = None    # 上一次触发重算时的物体位置（参考点）
        self.last_grasp_aspect = None # 【V7 新增】上一次触发重算时的宽高比（参考点）
        self.last_grasp_time = 0.0    # 上一次触发重算的时间戳
        self.frozen = False           # 冻结标志：True 时不触发重算

        # 
        self.grasp_freeze_distance = GRASP_FREEZE_DISTANCE
        self.grasp_unfreeze_distance = GRASP_UNFREEZE_DISTANCE

    def freeze(self):
        """进入冻结模式：后续 update() 始终返回 need_recompute=False"""
        if not self.frozen:
            self.frozen = True
            rospy.loginfo("[MotionDetector] 冻结检测 (末端距物体 < %.0f cm)", self.grasp_freeze_distance * 100)

    def unfreeze(self):
        """
        退出冻结模式，恢复运动检测。
        【关键】解冻时用当前滤波位置 + 滤波宽高比重置参考点，
        避免解冻瞬间因为 EMA 历史积累的微小漂移而误判运动。
        """
        if self.frozen:
            self.frozen = False
            # 解冻时用当前滤波值作为新参考点，避免解冻瞬间误判
            if self.filtered_pos is not None:
                self.last_grasp_pos = self.filtered_pos.copy()
            if self.filtered_aspect is not None:
                self.last_grasp_aspect = self.filtered_aspect  # 【V7】
            self.last_grasp_time = time.time()
            rospy.loginfo("[MotionDetector] 恢复检测 (末端距物体 > %.0f cm)",
                          self.grasp_unfreeze_distance * 100)

    def update(self, raw_pos_in_base, raw_aspect=None, current_time=None):
        """
        每帧调用，输入原始测量位置 + bbox 宽高比，输出是否需要重算。

        参数:
            raw_pos_in_base: 物体在基座系下的原始测量位置 [x,y,z]
            raw_aspect:      bbox 宽高比 w/h（可选，V7 新增，用于姿态检测）
            current_time:    当前时间戳（None 则自动取 time.time()）

        返回: (need_recompute: bool, filtered_pos: np.ndarray)

        内部逻辑顺序:
          1. EMA 滤波位置（无论是否冻结都更新）
          2. EMA 滤波宽高比（V7 新增，同样始终更新）
          3. 首次初始化 → 返回 True（需要算第一次抓取）
          4. 冻结检查 → 返回 False（锁定状态不重算）
          5. 冷却期检查 → 返回 False（1 秒内不重复触发）
          6. 位移超过阈值 → 返回 True（物体平移了）
          7. 宽高比变化超过阈值 → 返回 True（物体旋转了，V7 新增）
          8. 正常 → 返回 False
        """
        if current_time is None:
            current_time = time.time()

        # ===== 步骤 1: EMA 滤波位置 =====
        # 冻结时也照常更新 —— 保持 filtered_pos 准确，
        # 这样解冻时才能用正确的当前位置重置参考点
        if self.filtered_pos is None:
            self.filtered_pos = raw_pos_in_base.copy()
        else:
            # EMA 公式: new_filtered = α × raw + (1-α) × old_filtered
            self.filtered_pos = (self.ema_alpha * raw_pos_in_base +
                                 (1 - self.ema_alpha) * self.filtered_pos)

        # ===== 步骤 2: EMA 滤波宽高比（V7 新增） =====
        # 宽高比检测旋转: 物体旋转 → bbox 变宽或变高 → aspect 变化
        # 同样始终更新（冻结时也更新，为解冻备用）
        if raw_aspect is not None and raw_aspect > 0:
            if self.filtered_aspect is None:
                self.filtered_aspect = raw_aspect
            else:
                self.filtered_aspect = (self.ema_alpha * raw_aspect +
                                        (1 - self.ema_alpha) * self.filtered_aspect)

        # ===== 步骤 3: 首次初始化 =====
        # 第一次运行，还没有参考点，强制返回 True 触发首轮完整的抓取计算
        if self.last_grasp_pos is None:
            self.last_grasp_pos = self.filtered_pos.copy()
            if self.filtered_aspect is not None:
                self.last_grasp_aspect = self.filtered_aspect  # 【V7】
            self.last_grasp_time = current_time
            return True, self.filtered_pos

        # ===== 步骤 4: 冻结检查 =====
        # 这是 V4 的核心：冻结状态下永远不触发重算
        if self.frozen:
            return False, self.filtered_pos

        # ===== 步骤 5: 冷却期检查 =====
        # 防止连续多帧超过阈值导致重复触发重算
        time_since_last = current_time - self.last_grasp_time
        if time_since_last < self.cooldown_sec:
            return False, self.filtered_pos

        # ===== 步骤 6: 位移检测 =====
        # 比较当前滤波位置与上次重算时的参考位置，
        # 欧氏距离超过阈值 → 物体真的移动了 → 触发重算
        displacement = np.linalg.norm(self.filtered_pos - self.last_grasp_pos)
        if displacement > self.pos_threshold:
            # 更新参考点为当前位置（下次从新位置开始比较）
            self.last_grasp_pos = self.filtered_pos.copy()
            if self.filtered_aspect is not None:
                self.last_grasp_aspect = self.filtered_aspect  # 【V7】
            self.last_grasp_time = current_time
            rospy.loginfo("[MotionDetector] 检测到物体运动！位移: %.1f cm",
                          displacement * 100)
            return True, self.filtered_pos

        # ===== 步骤 7: 姿态检测 =====
        # 比较当前滤波宽高比与上次重算时的参考宽高比，
        # 变化超过阈值 → 物体旋转了（即使位置没变） → 触发重算
        if (self.filtered_aspect is not None and
                self.last_grasp_aspect is not None):
            aspect_change = abs(self.filtered_aspect - self.last_grasp_aspect)
            if aspect_change > self.aspect_threshold:
                self.last_grasp_pos = self.filtered_pos.copy()
                self.last_grasp_aspect = self.filtered_aspect
                self.last_grasp_time = current_time
                rospy.loginfo("[MotionDetector] 检测到姿态变化: "
                              "宽高比 %.2f → %.2f (变化 %.2f)",
                              self.last_grasp_aspect, self.filtered_aspect,
                              aspect_change)
                return True, self.filtered_pos

        # ===== 步骤 8: 正常，无需重算 =====
        return False, self.filtered_pos

    def reset(self):
        """完全重置检测器状态（轨迹取消 / 新一轮检测等场景）"""
        self.filtered_pos = None
        self.filtered_aspect = None
        self.last_grasp_pos = None
        self.last_grasp_aspect = None
        self.last_grasp_time = 0.0


# 轻量级物体位置快速估计
# 设计意图：每次循环都要知道"物体在基座系下的大致位置"来做距离判断，
#          但不需要每次都建完整点云 + SVD（那太贵了）。
#          这里只用 SAM2 追踪框的 bbox 中心 + 掩码区域的中值深度，
#          通过针孔相机模型投影到 3D，再经手眼变换转到基座系。
#          计算量：O(N) 取中值 + 一次矩阵乘法，远轻于 get_point_cloud 的 SVD。
def quick_object_pos_in_base(saved_boxes, depth_data, cam_info, T_end2base, mask=None):
    """
    仅使用 SAM2 追踪框 + 中值深度，快速估计物体在基座系下的位置。
    不建点云、不做 SVD —— 计算量极小，适合每帧调用。

    参数:
        saved_boxes: SAM2 追踪框 [x1,y1,x2,y2]
        depth_data:  深度图 (H,W) uint16, 单位 mm
        cam_info:    相机内参 (fx,fy,cx,cy,width,height)
        T_end2base:  末端在基座系下的齐次变换 (4x4)
        mask:        SAM2 分割掩码（可选，用于更精确的深度采样）

    返回: (position_in_base [3,], success_bool, median_depth_m, aspect_ratio)
          success=False 时 position 为 None, aspect_ratio 为 0.0
          aspect_ratio = w/h, 宽高比 > 1 表示横向长条, < 1 表示纵向长条
    """
    if saved_boxes is None or len(saved_boxes) == 0:
        return None, False, 0.0, 0.0

    x1, y1, x2, y2 = saved_boxes[0].astype(int)
    # 边界裁剪，防止越界
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(depth_data.shape[1] - 1, x2), min(depth_data.shape[0] - 1, y2)

    # 优先用 mask 区域取深度（更精确），否则用 bbox 区域
    if mask is not None and mask.shape[:2] == depth_data.shape[:2]:
        # 取掩码内有效深度值（排除深度为 0 的无效点）
        region_depth = depth_data[(mask > 0.5) & (depth_data > 0)]
    else:
        bbox_depth = depth_data[y1:y2, x1:x2]
        region_depth = bbox_depth[bbox_depth > 0]

    # 有效深度点太少则放弃
    if len(region_depth) < 20:
        return None, False, 0.0, 0.0

    # 中值深度比均值更鲁棒，不受远处噪点影响
    median_depth_mm = np.median(region_depth)
    median_depth_m = median_depth_mm / 1000.0  # mm → m

    # 【重要】深度 < 8cm 时 D435i 测量噪声过大，拒绝使用
    if median_depth_m < 0.08:
        return None, False, median_depth_m, 0.0

    # bbox 中心像素坐标
    cx_bbox = (x1 + x2) / 2.0
    cy_bbox = (y1 + y2) / 2.0

    # 针孔相机模型：像素坐标 → 相机坐标系 3D
    # X_cam = (u - cx) * Z / fx, Y_cam = (v - cy) * Z / fy, Z_cam = median_depth
    X_cam = (cx_bbox - cam_info.cx) * median_depth_m / cam_info.fx
    Y_cam = (cy_bbox - cam_info.cy) * median_depth_m / cam_info.fy
    Z_cam = median_depth_m

    pt_cam = np.array([X_cam, Y_cam, Z_cam])

    # 相机坐标系 → 基座坐标系
    # 变换链: T_obj2base = T_end2base @ T_eyeinhand @ T_obj2cam
    T_obj2cam = np.eye(4)
    T_obj2cam[:3, 3] = pt_cam
    T_obj2base = T_end2base @ T_EYEINHAND @ T_obj2cam

    # 【V7 新增】bbox 宽高比作为姿态代理
    # 物体旋转 → bbox 变宽或变高 → aspect_ratio 变化
    # 例如: 水平香蕉 w/h≈3, 旋转 90° → w/h≈0.33
    bbox_w = max(x2 - x1, 1)
    bbox_h = max(y2 - y1, 1)
    aspect_ratio = float(bbox_w) / float(bbox_h)

    return T_obj2base[:3, 3], True, median_depth_m, aspect_ratio

def FunObj2base(grasp_in_cam, T_end2base):
    """
    将 GraspNet 预测的相机系抓取位姿转换为基座系下 link6 的目标位姿。

    处理流程:
      1. 物体系抓取 → 基座系（T_end2base × T_eyeinhand × T_obj2cam）
      2. 夹爪坐标系调整（旋转矩阵修正：GraspNet x→夹爪 z, GraspNet z→夹爪 x）
      3. 桌面保护（夹爪低于桌面时抬升 0.5cm）
      4. 夹爪→link6 偏移补偿（Z 方向 13cm）

    返回: (T_obj2base_forlink6, target_pos, target_rot)
          target_pos [x,y,z] 单位 m, target_rot (3x3)
    """
    # 步骤 1: 物体→相机 齐次矩阵
    T_obj2cam = np.eye(4)
    T_obj2cam[0:3, 0:3] = grasp_in_cam.rotation_matrix  # GraspNet 预测的抓取姿态
    T_obj2cam[0:3, 3] = grasp_in_cam.translation         # GraspNet 预测的抓取位置
    # 物体→基座
    T_obj2base = T_end2base @ T_EYEINHAND @ T_obj2cam

    # 步骤 2: 夹爪坐标系调整
    # GraspNet 输出的抓取坐标系与 Piper 夹爪坐标系定义不同：
    #   GraspNet: x=接近方向, y=夹爪宽度方向, z=夹爪长度方向
    #   Piper:    x=夹爪长度方向, y=夹爪宽度方向, z=接近方向
    # 旋转映射: GraspNet_x → Piper_z, GraspNet_y_取反 → Piper_y, GraspNet_z → Piper_x
    R_Trans = np.eye(4)
    grasp_z = T_obj2base[0:3,2]
    grasp_z = grasp_z / np.linalg.norm(grasp_z)
    if np.dot(grasp_z, [1,0,0]) > 0:
        R_Trans[0:3, 0:3] = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    else:
        R_Trans[0:3, 0:3] = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]]) # 我写的
        
    T_obj2base_forGripper = T_obj2base @ R_Trans

    # 步骤 3: 桌面保护
    if T_obj2base_forGripper[2, 3] <= 0.0:
        T_obj2base_forGripper[2, 3] = 0.003  # 低于桌面时抬升 0.3cm

    # 步骤 4: 夹爪→link6 Z 轴偏移补偿（13cm = 夹爪末端到 link6 法兰盘的距离）
    T_gripper2link6 = np.eye(4)
    T_gripper2link6[2, 3] = 0.130  # 【重要】夹爪到 link6 的 Z 轴方向距离
    # 夹爪目标位姿 → link6 目标位姿
    T_obj2base_forlink6 = T_obj2base_forGripper @ np.linalg.inv(T_gripper2link6)

    target_pos = T_obj2base_forlink6[:3, 3]  # link6 目标位置 [x,y,z]
    target_rot = T_obj2base_forlink6[:3, :3]  # link6 目标姿态 (3x3)
    return T_obj2base_forlink6, target_pos, target_rot