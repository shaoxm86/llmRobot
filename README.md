# llmRobot

ROS 1 / Piper 眼在手上视觉抓取程序，入口为 `NewGrasp.py`。

## 本次修正

- 将抓取执行拆为 `APPROACH → CLOSING → TRANSPORT → RELEASING → RETURN → COMPLETE`，目标移动/丢失可取消接近轨迹；取消和异常均清理状态，取消不进入抓放阶段。
- 取消时尝试发送当前位置保持指令，保留当前夹爪开度。是否真正停止取决于驱动对笛卡尔话题的处理，不能替代驱动急停。
- 到位由新鲜末端反馈判断；闭合与释放分别需要新的实际状态反馈。闭合后失败或取消必须人工恢复，不会自动重试未知夹持状态。
- 彩色图、深度图和采集时末端位姿按时间戳匹配；内参就绪、维度匹配和数据新鲜度不足时不规划。缓存的是基座坐标目标，重规划不会重新套用移动后的相机变换。
- 冻结时继续更新 EMA；冻结 10 cm、解冻 15 cm，距离计算使用夹爪点。点云/GraspNet 失败后静止目标也会重试，目标丢失超时会取消接近。
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
