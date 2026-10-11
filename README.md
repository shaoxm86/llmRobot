# llmRobot

ROS 1 / Piper 眼在手上视觉抓取程序，入口为 `NewGrasp.py`。

## 功能与修正

- 接近阶段默认使用在线 DMP 积分，目标变化只更新目标位置和目标姿态，保留位置、速度、四元数、角速度及时间尺度。搬运和返回继续使用固定目标的离线路径。
- 将抓取执行拆为 `APPROACH → CLOSING → TRANSPORT → RELEASING → RETURN → COMPLETE`，目标移动在线更新吸引点，目标丢失可取消接近轨迹；取消和异常均清理状态，取消不进入抓放阶段。
- 取消时尝试发送当前位置保持指令，保留当前夹爪开度。是否真正停止取决于驱动对笛卡尔话题的处理，不能替代驱动急停。
- 到位由新鲜末端反馈判断；闭合与释放分别需要新的实际状态反馈。闭合后失败或取消必须人工恢复，不会自动重试未知夹持状态。
- 彩色图、深度图和采集时末端位姿按时间戳匹配；内参就绪、维度匹配和数据新鲜度不足时不规划。缓存的是基座坐标目标，重规划不会重新套用移动后的相机变换。
- 冻结时继续更新 EMA；可选冻结 10 cm、解冻 15 cm，距离计算使用夹爪点；在线模式默认关闭近距离目标冻结。点云/GraspNet 失败后静止目标也会重试，目标丢失超时会取消接近。
- 位移与旋转使用同一实际轨迹进度；仅旋转时使用平滑五次时间律。规划与执行周期一致，并验证有限值和速度上限。
- 候选抓取使用整个可见场景做夹爪碰撞检查，接近方向在基座系中约束为向下；IK 与实际目标统一使用同一个手眼/夹爪偏移转换。
- 接近、搬运和返回路径逐点进行 IK、关节解跳变和已有自碰撞对检查，并检查夹爪扫掠点云碰撞。完整机械臂环境碰撞需要下述外部检查器。
- 统一窗口所有权，支持关闭可视化；正确处理深度单位、图像行填充、字节序和零/NaN 深度；SAM 明确采用逐帧分割，不再声称视频跟踪。

## 运行前提

需要 ROS 1（`rospy`、`sensor_msgs`、`std_msgs`、`piper_msgs`）、PyTorch、Ultralytics、GraspNet baseline 及其自定义算子、graspnetAPI、OpenCV、Open3D、SciPy、NumPy、CasADi，以及带 CasADi 支持的 Pinocchio。版本和硬件部署需在原机器人环境核对；本云环境没有 ROS、模型资源或机器人硬件，尚未完成实机验证。

设置 `GRASPNET_BASELINE_ROOT` 为 baseline 根目录，默认使用仓库同级的 `graspnet-baseline`。

模型、URDF 可以用 ROS 私有参数指定，也支持环境变量：

| ROS 参数 | 环境变量 |
| --- | --- |
| `~yolo_weights` | `YOLO_WORLD_WEIGHTS` |
| `~sam_weights` | `SAM_WEIGHTS` |
| `~graspnet_checkpoint` | `GRASPNET_CHECKPOINT` |
| `~urdf_path` | `PIPER_URDF` |

仅加载可信来源的权重；GraspNet 使用 `torch.load(..., weights_only=True)`。`~device` 可选择 `cuda:0` 或 `cpu`，默认按 PyTorch CUDA 可用性选择；baseline 自定义算子的 CPU 支持取决于其实现。

## 驱动与反馈接口

- 输入：`/camera/color/image_raw`、`/camera/aligned_depth_to_color/image_raw`、`/camera/color/camera_info`、`/end_pose_euler`。
- 命令：`~command_topic` 默认 `/end_pose_cmdtry`，类型 `piper_msgs/PosCmd`，位置为米，欧拉角为弧度。驱动必须接收 link6 目标位姿；若它接收 TCP 位姿，应先统一驱动坐标定义。
- `~grasp_feedback_topic` 默认 `/grasp_confirmed`，类型 `std_msgs/Bool`。需要硬件适配节点根据实际夹爪状态/抓取传感器发布：`True` 表示确认持物，`False` 表示确认释放。必须持续更新，且确认接收时间晚于本次夹爪命令。发布命令本身不能作为确认。没有这项反馈，程序在闭合后超时停止，不搬运。
- 默认要求末端消息有非零 `header.stamp`。驱动未提供时间戳时，建议修正驱动；显式设置 `~allow_unstamped_pose:=true` 可使用接收时间近似匹配，但不代表准确采集时刻同步。
- `~sync_slop` 默认 0.05 秒；`~max_sensor_age` 默认 0.5 秒；`~feedback_timeout` 默认 5 秒。
- 夹爪默认 `~open_width:=0.06`、`~close_width:=0.03`，必须按实际驱动允许范围设置。程序不假定原注释所述范围正确。
- `~place_pose`、`~home_pose` 为 `[x,y,z,roll,pitch,yaw]`；原固定位置保留为默认值，但不再启动时盲目发送初始位姿。
- `~visualize` 在有 DISPLAY 时默认启用，否则关闭。`MPLBACKEND` 可显式指定绘图后端。

## 完整路径环境碰撞接口

`~path_checker` 设置为 `python_module:function`。函数签名为：

```python
def check_path(link6_poses, opening, scene_points_base):
    # link6_poses: (N,4,4)，基座系全路径，位置米。
    # opening: 当前夹爪开度，米。
    # scene_points_base: (M,3)，基座系可见环境点云，米。
    # 必须检查全机械臂、桌面、障碍物以及夹持物体，成功才返回 True。
    return planner.validate(...)
```

检查器应读取真实关节状态作为种子，并检查与驱动实际插补一致的路径，包括采样之间的运动；本仓库的 IK 结果只用作校验，执行仍通过笛卡尔驱动完成。

没有完整检查器时，默认拒绝运动。`~allow_partial_collision_check:=true` 仅显式允许已有的 IK、自碰撞对和夹爪点云检查，不能据此声称全机械臂路径无碰撞。单视角点云不覆盖遮挡区域；搬运时夹持物的几何信息应由完整检查器补齐。

## 离线回归测试

```bash
cd /workspace/llmRobot
PYTHONDONTWRITEBYTECODE=1 /workspace/llmRobot-venv/bin/python -m unittest discover -s tests -v
```

测试使用模拟机械臂、模拟时钟及 ROS/图像库隔离接口，覆盖中断、异常、到位/抓取/释放反馈、坐标缓存、轨迹同步、冻结回滞、时间戳匹配、图像解码和无效深度。测试不会发布真实 ROS 控制指令。


## 在线 DMP 目标更新

旧 `PositionDMP.generate_traj()` / `QuaternionDMP.generate_traj()` 保留给搬运、返回及离线验证。接近阶段默认使用 `DMP.OnlinePoseDMP`：

```text
位置: tau² * x'' = alpha * (beta * (g - x) - tau * x')
姿态: tau² * omega' = alpha * (beta * Log(R_goal * R_current^-1) - tau * omega)
```

位置为三维向量，姿态内部为单位四元数，线速度和角速度分别为三维向量（角速度在基座系表达）。每个周期在线积分一个下一目标点；视觉线程通过 `executor.update_goal(base_frame_link6_pose)` 更新吸引点。更新目标时不重建执行线程、不清零速度、不更改 tau，也不重置运动相位。该点到点系统没有示教学习的基函数或强制项。

目标发生跳变时位置、姿态及速度状态不会直接跳变，但加速度会改变；默认线/角加速度和速度都有上限，不保证 jerk 连续。原 DMP 输出的是参考状态，实际机械臂运动取决于驱动插补及反馈跟踪能力；跟踪偏差过大时停止，而非重置状态追赶。

视觉刷新期间继续朝已知目标执行，但暂不允许闭合夹爪；新抓取有效时更新 g / R_goal，失败或持续丢失时取消接近。默认每 0.5 秒重新估计六自由度抓取姿态，不依赖检测框宽高比发现所有旋转；这仍是离散的模型估计，不能保证对所有物体旋转都准确跟踪。候选筛选与执行校验使用独立 IK 实例，场景点云及碰撞检测器在独立线程更新，避免模型刷新直接持有执行校验锁。

| ROS 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `~online_approach` | `true` | 接近阶段在线积分；false 恢复离线路径与中断重规划 |
| `~online_tau` | `2.0` 秒 | 在线运动固定时间尺度；目标更新时保持不变 |
| `~control_dt` | `0.01` 秒 | 积分/发布目标周期 |
| `~max_linear_speed` | `0.10` m/s | 参考线速度上限 |
| `~max_linear_acceleration` | `0.25` m/s² | 参考线加速度上限 |
| `~max_angular_speed` | `0.5` rad/s | 参考角速度上限 |
| `~max_angular_acceleration` | `1.0` rad/s² | 参考角加速度上限 |
| `~grasp_refresh_interval` | `0.5` 秒 | 抓取目标周期刷新间隔；0 关闭周期刷新，保留运动触发 |
| `~goal_position_threshold` | `0.005` 米 | 接受新目标位置的变化阈值 |
| `~goal_rotation_threshold` | `0.05` 弧度 | 接受新目标姿态的变化阈值 |
| `~freeze_target_near_grasp` | `false` | 是否在近距离冻结视觉目标更新 |
| `~goal_quiet_time` | `0.3` 秒 | 目标未变化至少此时间，且参考速度收敛、实际末端到位，才闭合 |
| `~approach_timeout` | `60.0` 秒 | 单次在线接近最大持续时间 |
| `~max_online_step_delay` | `0.2` 秒 | 单次积分/校验的延迟上限，超限停止 |

只需更新位置而不改变姿态时，传入当前目标旋转和新位置构成的齐次矩阵即可。进入 CLOSING 或后续阶段后拒绝目标更新。

每次发布前校验当前实际位姿、上一参考位姿与下一参考位姿，默认完整路径检查器要求保留。在线校验为同步调用，计算超过 10 ms 会记录周期超时，超过配置的延迟上限会失败；不会批量补发过期点。因此“目标周期 100 Hz”不是已验证的实际实时频率，需在真实硬件上测量检查器耗时。外部 `~path_checker` 会高频接收短路径，输入场景点云只读，必须支持该调用方式。

在线回归测试覆盖：中途更换位置/姿态并收敛到新目标、状态和 tau 不重置、速度/加速度限制、跨 ±180° 的最短旋转、校验期间目标更新、无效下一步拒绝、反馈偏差、取消以及视觉刷新期间禁止闭合。

### 五次多项式目标过渡

在线目标更新现在默认先经过 `QuinticPoseGoal`，再作为 DMP 的吸引点。`~goal_transition_time` 默认 **0.5 秒**。

当旧目标静止时，位置过渡等价于：

```text
g(t) = g_B + (10u³ - 15u⁴ + 6u⁵)(g_C - g_B)
u = clip((t - t_c) / T_s, 0, 1)
```

边界位置、速度和加速度连续。姿态在旋转向量坐标中做五次过渡，并转换为单位四元数；不会直接线性插值四元数分量。

如果上次过渡尚未结束就收到新目标，新多项式继承当前吸引点的位置/速度/加速度及姿态/角速度/角加速度，末端导数设为零，避免反复从零导数开始造成跳变。同一目标重复提交不重启过渡。过渡未结束时，不允许因暂时接近旧吸引点而判定接近完成。

注意：五次过渡是 C2，不保证吸引点自身的 jerk 在端点连续。在未触发限幅的理想位置 DMP 中，加速度取决于 g、x、v，jerk 取决于 g'、v、x''；因此该过渡消除了直接切换 g 引起的加速度/jerk 跳变。速度/加速度限幅切换、姿态对数在 π 附近的分支变化、离散控制和实际驱动仍可能破坏更高阶连续性，不能据此保证真实机械臂的全程 jerk 连续。继承非零导数的过渡可能短暂超出新旧目标连线，实际发布路径仍须通过原有校验。

较大的 T_s 更平滑，但目标跟随滞后更大。这里 T_s 按成功提交的 DMP 积分步推进；执行发生周期超时时实际过渡时间会延长，不会跳过中间状态追赶墙钟。


## 真机前的代码复核

增加了执行线程独立的目标观测过期检查，`~max_target_estimate_age` 默认 1.0 秒，按输入图像采集时间计算。即使视觉主循环阻塞，在线执行也不会无限追逐旧目标；超过此阈值会停止接近。模型较慢时需根据实测时延设置该参数，不能将持续超时误认为目标跟踪正常。

点云张量使用 GraspNet 实际设备，避免指定 CPU 或第二张 GPU 时输入设备不一致。搬运/返回不再通过快速补发轨迹点追赶计算延迟。移除了候选筛选中未使用的批量夹爪渲染，以及执行 IK 的每点成功打印，以减少不必要开销。无效内参、空环境点云和非法时序参数会被拒绝。

本仓库的离线检查不代表所有 ROS/Piper/GraspNet/Pinocchio 版本组合兼容。实验机仍须提供权重、URDF、依赖及前述反馈/检查器。当前代码不会自动接入任意机械臂驱动的夹爪反馈；`/grasp_confirmed` 需要实际测量适配器。请核对 `/end_pose_cmdtry` 的位姿定义、夹爪开度范围和手眼标定；默认位置/标定来自原项目，尚未在你的实机上核验。实际停止、插补行为以及 100 Hz 达标情况由真机实验确认。
