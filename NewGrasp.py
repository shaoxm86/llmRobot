#!/usr/bin/env python3
"""ROS visual grasping: synchronized perception, base-frame targets, cancellable execution."""
import importlib
import os
import threading
import time
from contextlib import nullcontext
import cv2
import numpy as np
import rospy
import torch
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool
from piper_msgs.msg import PiperEulerPose, PosCmd
from Model_initialization import model_init
from DMP import PositionDMP, QuaternionDMP
from Plotter_tools import RealtimeTrajectoryPlotter, VisualizeMask, update_grasp_view, close_vis
from Position_detect import ObjectMotionDetector, quick_object_pos_in_base, FunObj2base
from PointCloud_Grasp import get_point_cloud, generate_grasps
from Piper_message import Color, Depth, CameraInfoMsg, End2base
from Grasp_geometry import T_EYEINHAND, gripper_position
from Grasp_runtime import GraspExecutor, SampleBuffer
from piper_pinocchio_mod2 import Arm_IK


def configured_pose(name, default):
    values = np.asarray(rospy.get_param(name, default), dtype=float)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError(name + ' must contain finite xyz[m], roll/pitch/yaw[rad]')
    pose = np.eye(4)
    pose[:3, 3], pose[:3, :3] = values[:3], R.from_euler('xyz', values[3:]).as_matrix()
    return pose


def scene_from_depth(depth, info):
    """Full visible scene, including table and objects outside the target mask."""
    y, x = np.indices(depth.shape)
    valid = (depth > 0) & np.isfinite(depth) & (depth < 2000)
    z = depth[valid] / 1000.
    return np.column_stack(((x[valid] - info.cx) * z / info.fx,
                            (y[valid] - info.cy) * z / info.fy, z))


def main():
    rospy.init_node('yoloworld_sam2_graspnet', anonymous=True)
    dt = float(rospy.get_param('~control_dt', .01))
    sync_slop = float(rospy.get_param('~sync_slop', .05))
    max_age = float(rospy.get_param('~max_sensor_age', .5))
    retry_sec = float(rospy.get_param('~retry_sec', 1.))
    lost_sec = float(rospy.get_param('~target_lost_timeout', .5))
    visualize = bool(rospy.get_param('~visualize', bool(os.environ.get('DISPLAY'))))
    freeze = float(rospy.get_param('~freeze_distance', .10))
    unfreeze = float(rospy.get_param('~unfreeze_distance', .15))
    if not 0 <= freeze < unfreeze or min(dt, sync_slop, max_age, retry_sec, lost_sec) <= 0:
        raise ValueError('Invalid timing or freeze/unfreeze parameters')
    table_z = float(rospy.get_param('~table_z', 0.))
    clearance = float(rospy.get_param('~table_clearance', .004))
    color, depth, info = Color(), Depth(), CameraInfoMsg()
    receipt_pose = bool(rospy.get_param('~allow_unstamped_pose', False))
    if receipt_pose:
        rospy.logwarn('Unstamped pose uses receipt time: capture-time synchronization is approximate')
    end = End2base(receipt_pose, rospy.get_time, max_age)
    confirmations = SampleBuffer()
    rospy.Subscriber('/camera/color/image_raw', Image, color.Callback, queue_size=5)
    rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, depth.Callback, queue_size=5)
    rospy.Subscriber('/camera/color/camera_info', CameraInfo, info.Callback, queue_size=1)
    rospy.Subscriber('/end_pose_euler', PiperEulerPose, end.Callback, queue_size=100)
    # External adapter must publish actual measured holding/release status.
    rospy.Subscriber(rospy.get_param('~grasp_feedback_topic', '/grasp_confirmed'), Bool,
                     lambda msg: confirmations.add(time.monotonic(), bool(msg.data)), queue_size=10)
    publisher = rospy.Publisher(rospy.get_param('~command_topic', '/end_pose_cmdtry'), PosCmd, queue_size=1)
    yolo, sam, net = model_init(rospy.get_param('~yolo_weights', None),
                              rospy.get_param('~sam_weights', None),
                              rospy.get_param('~graspnet_checkpoint', None),
                              rospy.get_param('~device', None))
    yolo.set_classes(rospy.get_param('~target_classes', ['remote']))
    arm_ik = Arm_IK(rospy.get_param('~urdf_path', None))
    ik_lock = threading.Lock()
    position_dmp, quaternion_dmp = PositionDMP(alpha=15, beta=15/4), QuaternionDMP()
    detector = ObjectMotionDetector(freeze, unfreeze, ema_alpha=.25,
        pos_threshold=.05, aspect_threshold=.5, cooldown_sec=.5)
    plotter = RealtimeTrajectoryPlotter() if visualize else None
    vis = None
    scene_lock = threading.Lock()
    scene_base = None
    scene_received = None
    path_checker_name = rospy.get_param('~path_checker', '')
    path_checker = None
    if path_checker_name:
        module, function = path_checker_name.rsplit(':', 1)
        path_checker = getattr(importlib.import_module(module), function)
    partial_checks = bool(rospy.get_param('~allow_partial_collision_check', False))
    if path_checker is None and not partial_checks:
        rospy.logwarn('Configure ~path_checker for full-arm/environment collision checks; motion is disabled')

    def publish(pose, opening):
        if publisher.get_num_connections() == 0:
            raise RuntimeError('机械臂控制话题没有订阅者')
        euler = R.from_matrix(pose[:3, :3]).as_euler('xyz')
        msg = PosCmd()
        msg.x, msg.y, msg.z = pose[:3, 3]
        msg.roll, msg.pitch, msg.yaw = euler
        msg.gripper = opening
        publisher.publish(msg)

    def confirmed(since, expected):
        sample = confirmations.latest()
        return sample is not None and sample[0] > since and sample[1] == expected

    def planner(start, target):
        distance = np.linalg.norm(target[:3, 3] - start[:3, 3])
        angle = R.from_matrix(start[:3, :3].T @ target[:3, :3]).magnitude()
        max_speed = float(rospy.get_param('~max_linear_speed', .10))
        max_angular = float(rospy.get_param('~max_angular_speed', .5))
        if min(max_speed, max_angular) <= 0:
            raise ValueError('Speed limits must be positive')
        tau = max(.8, 4*distance/max_speed, 4*angle/max_angular)
        for _ in range(6):
            positions, _, progress = position_dmp.generate_traj(start[:3, 3], target[:3, 3], tau=tau, dt=dt)
            _, rotations = quaternion_dmp.generate_traj(start[:3, :3], target[:3, :3], progress=progress)
            linear = np.max(np.linalg.norm(np.diff(positions, axis=0), axis=1)) / dt
            angular = np.max(R.from_matrix(np.transpose(rotations[:-1], (0, 2, 1)) @ rotations[1:]).magnitude()) / dt
            if linear <= max_speed and angular <= max_angular:
                return positions, rotations
            tau *= 1.5
        raise RuntimeError('Unable to satisfy trajectory speed limits')

    def validate(positions, rotations, opening):
        from graspnetAPI import GraspGroup
        from collision_detector import ModelFreeCollisionDetector
        with scene_lock:
            points = None if scene_base is None else scene_base.copy()
            received = scene_received
        if points is None or time.monotonic() - received > max_age:
            return False
        if path_checker is None and not partial_checks:
            return False
        with ik_lock:
            if not arm_ik.validate_path(positions, rotations, opening, cancelled=executor.cancel.is_set):
                return False
        # Validate the entire gripper sweep against the full visible scene.
        poses = np.tile(np.eye(4), (len(positions), 1, 1))
        poses[:, :3, :3], poses[:, :3, 3] = rotations, positions
        gripper_points = positions + .130 * rotations[:, :, 2]
        if np.any(gripper_points[:, 2] < table_z + clearance - 1e-6):
            return False
        # GraspNet X approaches; Piper Z approaches, Y reverses.
        correction = np.array([[0., 0., 1.], [0., -1., 0.], [1., 0., 0.]])
        grasp_rotations = rotations @ correction.T
        data = np.zeros((len(positions), 17))
        data[:, 0:4] = [1., opening, .02, .04]
        data[:, 4:13] = grasp_rotations.reshape(-1, 9)
        data[:, 13:16], data[:, 16] = gripper_points, -1
        collision = ModelFreeCollisionDetector(points, voxel_size=.01).detect(
            GraspGroup(data), approach_dist=0., collision_thresh=.01)
        if np.any(collision):
            return False
        # Adapter additionally checks whole arm, table, obstacles and carried object.
        if executor.cancel.is_set():
            return False
        checked = path_checker is None or bool(path_checker(poses, opening, points))
        with scene_lock:
            fresh = scene_received is not None and time.monotonic() - scene_received <= max_age
        return checked and fresh

    open_width = float(rospy.get_param('~open_width', .06))
    close_width = float(rospy.get_param('~close_width', .03))
    if not 0 <= close_width < open_width:
        raise ValueError('Require 0 <= close_width < open_width in metres')
    executor = GraspExecutor(publish, end.feedback_buffer.latest, validate, confirmed,
        shutdown=rospy.is_shutdown, dt=dt, feedback_age=max_age,
        timeout=float(rospy.get_param('~feedback_timeout', 5.)),
        open_width=open_width, close_width=close_width,
        log=rospy.logerr)
    place = configured_pose('~place_pose', [.1478, .194, .3935, 3.14159, .6916, -2.221])
    home = configured_pose('~home_pose', [.12, 0., .36, 3.14159, .777, 3.15159])
    cached_target = cached_grasp = cached_cloud = cached_center = cached_rot = None
    previous_axis = None
    pending_target = False
    last_attempt = 0.
    last_valid = time.monotonic()
    last_frame_stamp = None
    rate = rospy.Rate(30)

    def lost_target():
        nonlocal cached_target, pending_target
        state, _, _, _ = executor.snapshot()
        if time.monotonic() - last_valid >= lost_sec:
            cached_target, pending_target = None, False
            detector.reset()
            if state == 'APPROACH':
                executor.interrupt()

    try:
        while not rospy.is_shutdown():
            need_recompute = False  # Never reuse the preceding frame's decision.
            state, holding, desired, error = executor.snapshot()
            if state == 'COMPLETE':
                if rospy.get_param('~shutdown_on_complete', True):
                    rospy.signal_shutdown('Task complete after measured release and return')
                    break
                rate.sleep()
                continue
            rgb = color.buffer.latest()
            calibration = info.snapshot()
            if rgb is None or calibration is None:
                lost_target(); rate.sleep(); continue
            stamp, image = rgb
            dep = depth.buffer.nearest(stamp, sync_slop)
            capture_end = end.buffer.nearest(stamp, sync_slop)
            if (dep is None or capture_end is None or stamp == last_frame_stamp or
                    not 0 <= rospy.get_time() - stamp <= max_age):
                lost_target(); rate.sleep(); continue
            last_frame_stamp = stamp
            raw_frame, depth_frame, capture_pose = image.copy(), dep[1].copy(), capture_end[1].copy()
            if (raw_frame.shape[:2] != depth_frame.shape or
                    depth_frame.shape != (calibration.height, calibration.width)):
                rospy.logwarn_throttle(2., 'RGB/depth/intrinsics dimensions do not match')
                lost_target(); rate.sleep(); continue
            camera_scene = scene_from_depth(depth_frame, calibration)
            camera_to_base = capture_pose @ T_EYEINHAND
            with scene_lock:
                scene_base = camera_scene @ camera_to_base[:3, :3].T + camera_to_base[:3, 3]
                scene_received = time.monotonic()
            feedback = end.feedback_buffer.latest()
            if feedback is None or time.monotonic() - feedback[0] > max_age:
                lost_target(); rate.sleep(); continue
            current = feedback[1]
            frame = raw_frame.copy()
            # During closing/transport/release/return, camera target disappearance
            # is expected; continue refreshing scene and feedback, not detection.
            if state not in ('CLOSING', 'TRANSPORT', 'RELEASING', 'RETURN') and not holding:
                results = yolo(raw_frame, verbose=False)
                boxes = results[0].boxes.xyxy.cpu().numpy()
                if len(boxes) == 0:
                    lost_target(); rate.sleep(); continue
                # Single-object workflow: highest-confidence detection and its mask.
                index = int(results[0].boxes.conf.argmax().item())
                boxes = boxes[index:index+1]
                use_cuda = str(sam.device).startswith('cuda')
                context = torch.autocast('cuda', dtype=torch.bfloat16) if use_cuda and torch.cuda.is_bf16_supported() else nullcontext()
                with context:
                    segmentation = sam(raw_frame, bboxes=boxes, verbose=False)
                masks = segmentation[0].masks
                if masks is None or len(masks.data) == 0:
                    lost_target(); rate.sleep(); continue
                mask = masks.data[0].cpu().numpy()
                if mask.shape != depth_frame.shape:
                    mask = cv2.resize(mask, (depth_frame.shape[1], depth_frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                quick_pos, valid, _, aspect = quick_object_pos_in_base(boxes, depth_frame, calibration, capture_pose, mask)
                if not valid:
                    lost_target(); rate.sleep(); continue
                last_valid = time.monotonic()
                frame = VisualizeMask(frame, mask, [255, 0, 0])
                # Update EMA even while frozen. Apply transitions using this frame's
                # filtered position and the actual gripper point, not link6.
                need_recompute, filtered = detector.update_with_distance(
                    quick_pos, gripper_position(current), aspect,
                    can_freeze=cached_target is not None)
                if cached_target is None and time.monotonic() - last_attempt >= retry_sec:
                    need_recompute = True
                if need_recompute:
                    # Cancel before expensive inference; never keep following a
                    # known obsolete target while waiting for a new prediction.
                    if state == 'APPROACH':
                        executor.interrupt()
                    cached_target, pending_target = None, False
                    last_attempt = time.monotonic()
                    previous_axis, inputs, cloud, center, rotation = get_point_cloud(
                        previous_axis, calibration, raw_frame, depth_frame, mask)
                    if inputs is not None:
                        with ik_lock:
                            grasp = generate_grasps(net, inputs, cloud, None, arm_ik, capture_pose,
                                obj_center=center, obj_rot=rotation, scene_points=camera_scene,
                                table_z=table_z, clearance=clearance)
                        if grasp is not None:
                            cached_target, _, _ = FunObj2base(grasp[0], capture_pose, table_z, clearance)
                            cached_grasp, cached_cloud, cached_center, cached_rot = grasp, cloud, center, rotation
                            pending_target = True
                    if cached_target is None:
                        detector.reset()  # Static objects retry after failed planning.
                if pending_target and not executor.active() and cached_target is not None:
                    if executor.start(cached_target, place, home, planner):
                        pending_target = False
                if state == 'FAILED' and not executor.active() and executor.recoverable:
                    cached_target = None
                    detector.reset()
            if visualize:
                if cached_grasp is not None:
                    vis = update_grasp_view(vis, cached_cloud, cached_grasp.to_open3d_geometry_list(),
                                            obj_center=cached_center, obj_rot=cached_rot)
                _, _, desired, _ = executor.snapshot()
                plotter.add_actual_point(current[:3, 3])
                plotter.update_plot(current[:3, 3], current[:3, :3],
                    None if desired is None else desired[:3, 3],
                    None if desired is None else desired[:3, :3])
                cv2.imshow('Result', frame)
                if cv2.waitKey(1) == ord('q'):
                    break
            rate.sleep()
    finally:
        executor.close()
        if vis is not None:
            close_vis(vis)
        if plotter is not None:
            plotter.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
