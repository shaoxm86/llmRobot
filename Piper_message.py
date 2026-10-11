import numpy as np
import cv2
from Transformation import PosEuler2PosRotation


class Color():
    """RGB 图像订阅回调，存 BGR 格式的 numpy 数组"""
    def __init__(self):
        self.height = 0
        self.width = 0
        self.encoding = ''
        self.RGBdata = None  # RGB 格式 numpy 数组 (H,W,3)
        self.BGRdata = None  # BGR 格式 numpy 数组（OpenCV 显示用）

    def Callback(self, msg):
        self.height = msg.height  # 720
        self.width = msg.width    # 1280
        self.encoding = msg.encoding  # rgb8
        # ROS Image msg → numpy
        self.RGBdata = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            self.height, self.width, 3)
        # RGB → BGR（OpenCV 显示用）
        self.BGRdata = cv2.cvtColor(self.RGBdata, cv2.COLOR_RGB2BGR)

class Depth():
    """对齐到彩色图坐标系的深度图订阅回调，单位 mm"""
    def __init__(self):
        self.height = 0
        self.width = 0
        self.encoding = ''
        self.data = None  # numpy 深度图 (H,W) uint16，单位 mm

    def Callback(self, msg):
        self.height = msg.height
        self.width = msg.width
        self.encoding = msg.encoding  # 16UC1: 16 位无符号单通道
        self.data = np.frombuffer(msg.data, dtype=np.uint16).reshape(
            self.height, self.width)

class CameraInfoMsg():
    """相机内参订阅回调"""
    def __init__(self):
        self.fx = 0.0  # X 方向焦距 [像素]
        self.fy = 0.0  # Y 方向焦距 [像素]
        self.cx = 0.0  # 主点 X 坐标 [像素]
        self.cy = 0.0  # 主点 Y 坐标 [像素]
        self.width = 0
        self.height = 0

    def Callback(self, msg):
        self.width = msg.width
        self.height = msg.height
        # K 矩阵: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        self.fx = msg.K[0]
        self.fy = msg.K[4]
        self.cx = msg.K[2]
        self.cy = msg.K[5]


class End2base():
    """机械臂末端 link6 在基座系下的位姿订阅回调"""
    def __init__(self):
        self.T_end2base = np.eye(4)  # 末端→基座 齐次变换矩阵

    def Callback(self, msg):
        # Piper 欧拉角位姿格式: [x, y, z, roll, pitch, yaw]
        robotstates = [msg.x, msg.y, msg.z, msg.roll, msg.pitch, msg.yaw]
        self.T_end2base = PosEuler2PosRotation(robotstates)