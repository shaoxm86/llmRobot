"""Shared camera/gripper/link6 transforms (positions in metres)."""
import numpy as np

T_EYEINHAND = np.array([[0, .9397, .342, -.07037],
                       [-1, 0, 0, .04228],
                       [0, -.342, .9397, .04115], [0, 0, 0, 1.]])
# Round-off in the measured calibration must not create a non-rigid transform.
u, _, vt = np.linalg.svd(T_EYEINHAND[:3, :3])
T_EYEINHAND[:3, :3] = u @ vt
GRIPPER_OFFSET = .130


def grasp_to_link6(grasp, end_to_base, table_z=0., clearance=.004):
    pose = np.eye(4)
    pose[:3, :3] = grasp.rotation_matrix
    pose[:3, 3] = grasp.translation
    pose = end_to_base @ T_EYEINHAND @ pose
    if pose[0, 2] > 0:
        correction = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    else:
        correction = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
    pose[:3, :3] = pose[:3, :3] @ correction
    pose[2, 3] = max(pose[2, 3], table_z + clearance)
    pose[:3, 3] -= GRIPPER_OFFSET * pose[:3, 2]
    return pose


def gripper_position(link6):
    return link6[:3, 3] + GRIPPER_OFFSET * link6[:3, 2]
