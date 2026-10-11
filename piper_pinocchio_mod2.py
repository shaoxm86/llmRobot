import os
import casadi                    # 非线性优化与符号计算库
import numpy as np
import pinocchio as pin          # 机器人动力学/运动学库
from pinocchio import casadi as cpin                  # Pinocchio 的 CasADi 符号接口（用于将运动学函数转换为优化变量）
from pinocchio.robot_wrapper import RobotWrapper      # 机器人封装类
from scipy.spatial.transform import Rotation


exit_flag = False

class Arm_IK:
    def __init__(self, urdf_path=None):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)
        cur_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        urdf_path = urdf_path or os.environ.get('PIPER_URDF')
        if not urdf_path or not os.path.isfile(urdf_path):
            raise FileNotFoundError('Set ~urdf_path or PIPER_URDF to the Piper URDF')
        # print(urdf_path) # /home/sxm/catkin_ws/src/piper_ros-noetic/src/piper_description/urdf/piper_description.urdf
    
        self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path)
        # 锁死夹爪/末端相关的关节 ("joint7", "joint8")，将 8 自由度机器人简化为 6 自由度臂
        self.mixed_jointsToLockIDs = ["joint7", "joint8"]
        # buildReducedRobot 将 joint7、joint8（夹爪两个手指关节）固定在零位并从运动学树中消除
        self.reduced_robot = self.robot.buildReducedRobot(list_of_joints_to_lock=self.mixed_jointsToLockIDs, reference_configuration=np.array([0] * self.robot.model.nq))

        # 为末端执行器定义一个新的参考系 'ee' (end-effector) 并附加到 joint6 上
        q = Rotation.from_euler('xyz', [0, 0, 0]).as_quat()
        self.reduced_robot.model.addFrame(
            pin.Frame('ee',
                      self.reduced_robot.model.getJointId('joint6'),
                      pin.SE3(pin.Quaternion(q[3], q[0], q[1], q[2]), np.array([0.0, 0.0, 0.0])),
                      pin.FrameType.OP_FRAME))

        # 构建碰撞检测模型，并手动添加碰撞对（判定第 4~9 个连杆和第 0~2 个连杆间是否会碰撞）
        self.geom_model = pin.buildGeomFromUrdf(self.robot.model, urdf_path, pin.GeometryType.COLLISION)
        for i in range(4, min(10, len(self.geom_model.geometryObjects))):
            for j in range(min(3, len(self.geom_model.geometryObjects))):
                if i != j:
                    self.geom_model.addCollisionPair(pin.CollisionPair(i, j))
        self.geometry_data = pin.GeometryData(self.geom_model)

        # 记录关节数据的缓存数组
        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.history_data = np.zeros(self.reduced_robot.model.nq)

        # --- 使用 CasADi 建立正运动学与误差数学模型 ---
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        # 定义符号变量：cq (关节角), cTf (4x4 目标齐次变换矩阵)
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)
        self.cTf = casadi.SX.sym("tf", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # Get the hand joint ID and define the error function
        # 定义末端执行器的误差表达式 (包含位置与姿态的 6-DOF 空间偏差)
        self.gripper_id = self.reduced_robot.model.getFrameId("ee")
        self.error = casadi.Function(
            "error",
            [self.cq, self.cTf],
            [
                casadi.vertcat(
                    cpin.log6(
                        self.cdata.oMf[self.gripper_id].inverse() * cpin.SE3(self.cTf)
                    ).vector,
                )
            ],
        )

        # --- 建立求解 IK 的非线性优化问题 (NLP) ---
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq) # 优化变量：目标关节角
        # self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)   # 参数：上一帧关节角（用于平滑）
        self.param_tf = self.opti.parameter(4, 4)
        # self.totalcost = casadi.sumsqr(self.error(self.var_q, self.param_tf))
        error_vec = self.error(self.var_q, self.param_tf)
        pos_error = error_vec[:3]  # 取前3个值为位置误差
        ori_error = error_vec[3:]  # 取后3个值为姿态误差
        # 设置位置和姿态的权重
        weight_position = 1.0  # 位置权重
        weight_orientation = 0.1  # 姿态权重
        # 总成本函数
        self.totalcost = casadi.sumsqr(weight_position * pos_error) + casadi.sumsqr(weight_orientation * ori_error)
        # 正则化项
        self.regularization = casadi.sumsqr(self.var_q) # 正则化：惩罚关节角偏离零位，使解偏向中心位形
        # self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last) # 平滑项：减少相邻两帧跳变

        # 设置约束条件：关节角度必须在机器人的物理极限范围内
        self.opti.subject_to(self.opti.bounded(
            self.reduced_robot.model.lowerPositionLimit,
            self.var_q,
            self.reduced_robot.model.upperPositionLimit)
        )
        # print("self.reduced_robot.model.lowerPositionLimit:", self.reduced_robot.model.lowerPositionLimit)
        # print("self.reduced_robot.model.upperPositionLimit:", self.reduced_robot.model.upperPositionLimit)
        # 设置优化目标函数：最小化 (20*加权误差 + 0.01*正则化)
        self.opti.minimize(20 * self.totalcost + 0.01 * self.regularization)
        # self.opti.minimize(20 * self.totalcost + 0.01 * self.regularization + 0.1 * self.smooth_cost) # for smooth
        
        # 配置 IPOPT 求解器（最大迭代50次，精度 1e-4，静音输出）
        opts = {
            'ipopt': {
                'print_level': 0,
                'max_iter': 50,
                'tol': 1e-4
            },
            'print_time': False
        }
        self.opti.solver("ipopt", opts)

    # 核心算法
    def ik_fun(self, target_pose, gripper=0, motorstate=None, motorV=None):
        gripper = np.array([gripper/2.0, -gripper/2.0]) # gripper=0.04（开 4cm）→ [0.02, -0.02]
        # 如果传入当前电机关节角（6 维），用它覆盖 init_data。init_data 是优化的初值猜测，给出一个好的起点能显著加速 IPOPT 收敛、减少解的跳变。
        if motorstate is not None: self.init_data = motorstate
        self.opti.set_initial(self.var_q, self.init_data) # 以当前状态作为优化初值

        self.opti.set_value(self.param_tf, target_pose)          # 传入目标齐次变换矩阵
        # self.opti.set_value(self.var_q_last, self.init_data) # for smooth

        try:
            # sol = self.opti.solve()
            sol = self.opti.solve_limited() # solve_limited()：带最大迭代次数上限的求解，避免优化无限卡死
            sol_q = self.opti.value(self.var_q) # 提取算出的关节角度

            # ========== 新增：可达性校验 ==========
            # 求解"成功"只代表收敛到了最优解，不代表误差为零
            # 必须显式检查末端误差是否在容差内
            err_vec = np.array(self.error(sol_q, target_pose)).flatten()
            pos_err = np.linalg.norm(err_vec[:3])   # 位置误差范数 (m)
            ori_err = np.linalg.norm(err_vec[3:])   # 姿态误差范数 (rad)

            pos_tol = 0.01    # 位置容差 1cm，按需调整
            ori_tol = 0.10    # 姿态容差约 5.7°，按需调整

            if pos_err > pos_tol or ori_err > ori_tol:
                print(f"Target unreachable: pos_err={pos_err:.4f} m, " f"ori_err={ori_err:.4f} rad")
                return None, '', False
            # ========== 新增结束 ==========

            # # 安全校验：若单帧关节角突变超过 30 度，认为解异常，触发保护机制重置初值
            if self.init_data is not None:
                max_diff = max(abs(self.history_data - sol_q)) # 计算当前解 sol_q 与上一帧解 self.history_data 之间的最大绝对差值
                # print("max_diff:", max_diff)
                self.init_data = sol_q
                if max_diff > 30.0/180.0*3.1415:
                    # print("Excessive changes in joint angle:", max_diff)
                    self.init_data = np.zeros(self.reduced_robot.model.nq)
            else:
                self.init_data = sol_q
            self.history_data = sol_q # 将当前解存入 history_data，供下一次调用做突变检测

            if motorV is not None:
                v = motorV * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0
            
            # 调用 Pinocchio 的 RNEA（Recursive Newton-Euler Algorithm）计算逆动力学
            tau_ff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v, np.zeros(self.reduced_robot.model.nv))

            # 调用 check_self_collision 方法，在完整 8-DOF 模型上执行碰撞检测
            is_collision = self.check_self_collision(sol_q, gripper) # 返回 True（有碰撞）或 False（无碰撞）

            # 返回三元组：6 维关节角解，6 维重力补偿力矩，无碰撞返回 True（可执行），有碰撞返回 False（不可执行）
            return sol_q, tau_ff, not is_collision

        except Exception as e:
            print(f"ERROR in convergence, plotting debug info.{e}")
            # sol_q = self.opti.debug.value(self.var_q)   # return original value
            return None, '', False

    def validate_path(self, positions, rotations, gripper=0., max_joint_step=.15, cancelled=lambda: False):
        """Check every Cartesian sample, not just the endpoint.

        This checks IK, limits, selected self-collision pairs and solution jumps;
        environmental collisions require an external full-arm planning checker.
        """
        previous = None
        for pos, rot in zip(positions, rotations):
            if cancelled():
                return False
            target = np.eye(4)
            target[:3, 3], target[:3, :3] = pos, rot
            joints, reach = self.get_ik_solution(target, gripper)
            if not reach:
                return False
            joints = np.asarray(joints)
            if previous is not None and np.max(np.abs(joints - previous)) > max_joint_step:
                return False
            previous = joints
        return True

    def check_self_collision(self, q, gripper=np.array([0, 0])):
        pin.forwardKinematics(self.robot.model, self.robot.data, np.concatenate([q, gripper], axis=0))
        pin.updateGeometryPlacements(self.robot.model, self.robot.data, self.geom_model, self.geometry_data)
        collision = pin.computeCollisions(self.geom_model, self.geometry_data, False)
        # print("collision:", collision)
        return collision

    def get_ik_solution(self, targetT, gripper=0.):
        
        # 返回三元组：6 维关节角解，6 维重力补偿力矩，无碰撞返回 True（可执行），有碰撞返回 False（不可执行）
        sol_q, tau_ff, get_result = self.ik_fun(targetT, gripper) # .homogeneous 是齐次变换矩阵（4×4）
        
        if get_result:
            print("IK Solution:", sol_q)
            return sol_q, True
        else:
            return [], False
    
def PiperIK(arm_ik, targetT):
    
    # 调用Arm_IK类的逆解函数
    sol_q, reach = arm_ik.get_ik_solution(targetT)

    if reach: print("找到可达的逆解...")

    return reach
