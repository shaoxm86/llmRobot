"""Decode ROS messages without assuming tightly packed image rows."""
import threading
import time
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from Grasp_runtime import SampleBuffer


def message_stamp(msg, allow_receipt=False, receipt=None):
    header = getattr(msg, 'header', None)
    if header is not None and header.stamp.to_sec() > 0:
        return header.stamp.to_sec()
    if allow_receipt:
        return receipt
    return None


def decode_image(msg, dtype, channels=1):
    dtype = np.dtype(dtype).newbyteorder('>' if msg.is_bigendian else '<')
    row_bytes = msg.width * channels * dtype.itemsize
    if msg.step < row_bytes or len(msg.data) < msg.height * msg.step:
        raise ValueError('Image step/data length is invalid')
    rows = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    packed = rows[:, :row_bytes].copy().reshape(-1).view(dtype)
    shape = (msg.height, msg.width, channels) if channels > 1 else (msg.height, msg.width)
    return packed.reshape(shape).astype(dtype.newbyteorder('='), copy=False)


class Color:
    def __init__(self):
        self.BGRdata = self.RGBdata = None
        self.buffer = SampleBuffer()

    def Callback(self, msg):
        if msg.encoding not in ('rgb8', 'bgr8'):
            raise ValueError('Expected rgb8 or bgr8 color image')
        image = decode_image(msg, np.uint8, 3)
        self.BGRdata = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if msg.encoding == 'rgb8' else image
        stamp = message_stamp(msg)
        if stamp is not None:
            self.buffer.add(stamp, self.BGRdata)


class Depth:
    def __init__(self):
        self.data = None
        self.buffer = SampleBuffer()

    def Callback(self, msg):
        if msg.encoding == '16UC1':
            self.data = decode_image(msg, np.uint16)
        elif msg.encoding == '32FC1':
            metres = decode_image(msg, np.float32)
            self.data = np.where(np.isfinite(metres) & (metres > 0), metres * 1000., 0.)
        else:
            raise ValueError('Expected 16UC1 (mm) or 32FC1 (m) depth image')
        stamp = message_stamp(msg)
        if stamp is not None:
            self.buffer.add(stamp, self.data)


class CameraInfoMsg:
    def __init__(self):
        self.fx = self.fy = self.cx = self.cy = 0.
        self.width = self.height = 0
        self.lock = threading.Lock()
        self.ready = False

    def Callback(self, msg):
        with self.lock:
            self.width, self.height = msg.width, msg.height
            self.fx, self.fy, self.cx, self.cy = msg.K[0], msg.K[4], msg.K[2], msg.K[5]
            self.ready = self.fx > 0 and self.fy > 0 and self.width > 0 and self.height > 0

    def snapshot(self):
        from types import SimpleNamespace
        with self.lock:
            if not self.ready:
                return None
            return SimpleNamespace(**{key: getattr(self, key) for key in
                ('width', 'height', 'fx', 'fy', 'cx', 'cy')})


class End2base:
    def __init__(self, allow_receipt=False, ros_clock=None, max_age=.5):
        self.T_end2base = None
        self.buffer = SampleBuffer()
        self.feedback_buffer = SampleBuffer()
        self.allow_receipt, self.ros_clock = allow_receipt, ros_clock
        self.max_age = max_age

    def Callback(self, msg):
        values = np.array([msg.x, msg.y, msg.z, msg.roll, msg.pitch, msg.yaw])
        if not np.isfinite(values).all():
            return
        pose = np.eye(4)
        pose[:3, 3] = values[:3]
        pose[:3, :3] = Rotation.from_euler('xyz', values[3:]).as_matrix()
        receipt = self.ros_clock() if self.ros_clock is not None else None
        stamp = message_stamp(msg, self.allow_receipt, receipt)
        source_stamp = message_stamp(msg)
        if (source_stamp is not None and receipt is not None and
                not -.05 <= receipt - source_stamp <= self.max_age):
            return  # A newly received message can still contain stale robot state.
        self.T_end2base = pose
        self.feedback_buffer.add(time.monotonic(), pose)
        if stamp is not None:
            self.buffer.add(stamp, pose)
