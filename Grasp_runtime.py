"""Hardware-independent execution state machine and stamped sample buffer."""
import threading
import time
from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation as R


class SampleBuffer:
    def __init__(self, size=100):
        self.lock = threading.Lock()
        self.samples = deque(maxlen=size)

    def add(self, stamp, value):
        with self.lock:
            self.samples.append((float(stamp), value))

    def latest(self):
        with self.lock:
            return self.samples[-1] if self.samples else None

    def nearest(self, stamp, tolerance):
        with self.lock:
            if not self.samples:
                return None
            sample = min(self.samples, key=lambda s: abs(s[0] - stamp))
            return sample if abs(sample[0] - stamp) <= tolerance else None


class Cancelled(Exception):
    pass


class GraspExecutor:
    """Publish only validated paths; confirmation is independent of commands.

    publish(pose, opening), feedback()->(monotonic_receipt, pose),
    validate(positions, rotations, opening), confirmed(since, expected).
    """
    def __init__(self, publish, feedback, validate, confirmed, shutdown=lambda: False,
                 sleep=time.sleep, clock=time.monotonic, dt=.01, timeout=5.,
                 feedback_age=.5, pos_tol=.01, rot_tol=.10, open_width=.06,
                 close_width=.03, finish=lambda: None, log=lambda msg: None):
        self.publish, self.feedback, self.validate = publish, feedback, validate
        self.confirmed, self.shutdown = confirmed, shutdown
        self.sleep, self.clock, self.dt, self.timeout = sleep, clock, dt, timeout
        self.feedback_age, self.pos_tol, self.rot_tol = feedback_age, pos_tol, rot_tol
        self.open_width, self.close_width = open_width, close_width
        self.finish, self.log = finish, log
        self.lock = threading.Lock()
        self.cancel = threading.Event()
        self.thread = None
        self.state = 'IDLE'
        self.error = None
        self.holding = False
        self.target = None
        self.gripper_commanded = False
        self.recoverable = True
        self.last_opening = open_width
        self.has_published = False

    def snapshot(self):
        with self.lock:
            return self.state, self.holding, self.target, self.error

    def active(self):
        with self.lock:
            return self.thread is not None and self.thread.is_alive()

    def interrupt(self):
        self.cancel.set()

    def check(self):
        if self.cancel.is_set() or self.shutdown():
            raise Cancelled()

    def current_pose(self):
        sample = self.feedback()
        if sample is None or self.clock() - sample[0] > self.feedback_age:
            raise RuntimeError('末端反馈缺失或过期')
        return sample[1].copy()

    def set_state(self, state):
        with self.lock:
            self.state = state

    def wait_pose(self, target):
        deadline = self.clock() + self.timeout
        while self.clock() < deadline:
            self.check()
            pose = self.current_pose()
            pos_error = np.linalg.norm(pose[:3, 3] - target[:3, 3])
            angle = R.from_matrix(pose[:3, :3].T @ target[:3, :3]).magnitude()
            if pos_error <= self.pos_tol and angle <= self.rot_tol:
                return
            self.sleep(self.dt)
        raise RuntimeError('等待末端到位超时')

    def wait_gripper(self, since, expected):
        deadline = self.clock() + self.timeout
        while self.clock() < deadline:
            self.check()
            self.current_pose()  # stale robot feedback must stop the task
            if self.confirmed(since, expected):
                return
            self.sleep(self.dt)
        raise RuntimeError('夹爪/抓取反馈未确认，停止后续运动')

    def move(self, target, opening, planner):
        self.check()
        start = self.current_pose()
        positions, rotations = planner(start, target)
        if len(positions) < 2 or len(positions) != len(rotations):
            raise ValueError('轨迹长度无效')
        if not np.isfinite(positions).all() or not np.isfinite(rotations).all():
            raise ValueError('轨迹包含非有限值')
        if not self.validate(positions, rotations, opening):
            raise RuntimeError('路径可达性或碰撞检查失败')
        # Validation may be expensive: refuse a start pose that has moved meanwhile.
        actual = self.current_pose()
        if (np.linalg.norm(actual[:3, 3] - start[:3, 3]) > self.pos_tol or
                R.from_matrix(actual[:3, :3].T @ start[:3, :3]).magnitude() > self.rot_tol):
            raise RuntimeError('路径验证期间起点发生变化，请重新规划')
        deadline = self.clock()
        for pos, rot in zip(positions, rotations):
            self.check()
            self.current_pose()
            pose = np.eye(4)
            pose[:3, 3], pose[:3, :3] = pos, rot
            self.publish(pose, opening)
            self.last_opening = opening
            self.has_published = True
            with self.lock:
                self.target = pose.copy()
            deadline += self.dt
            self.sleep(max(0., deadline - self.clock()))
        self.wait_pose(target)

    def start(self, target, place, home, planner):
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                return False
            if self.holding or not self.recoverable or self.state == 'COMPLETE':
                return False
            self.cancel.clear()
            self.error = None
            self.state = 'APPROACH'
            self.gripper_commanded = False
            self.has_published = False
            self.thread = threading.Thread(target=self.run,
                args=(target.copy(), place.copy(), home.copy(), planner), daemon=False)
            self.thread.start()
            return True

    def run(self, target, place, home, planner):
        try:
            self.move(target, self.open_width, planner)
            self.check()
            self.set_state('CLOSING')
            since = self.clock()
            self.gripper_commanded = True
            self.last_opening = self.close_width
            self.publish(target, self.close_width)
            self.wait_gripper(since, True)
            with self.lock:
                self.holding = True
            self.set_state('TRANSPORT')
            self.move(place, self.close_width, planner)
            self.check()
            self.set_state('RELEASING')
            since = self.clock()
            self.last_opening = self.open_width
            self.publish(place, self.open_width)
            self.wait_gripper(since, False)
            with self.lock:
                self.holding = False
            self.set_state('RETURN')
            self.move(home, self.open_width, planner)
            self.set_state('COMPLETE')
            self.finish()
        except Cancelled:
            self.hold_position()
            self.set_state('CANCELLED')
        except Exception as exc:
            self.hold_position()
            with self.lock:
                self.error = str(exc)
                self.state = 'FAILED'
            self.log(str(exc))
        finally:
            with self.lock:
                self.target = None
                # After closing, cancellation/failure requires operator recovery.
                self.recoverable = not self.gripper_commanded

    def hold_position(self):
        """Best-effort hold; never close/release as a side effect of cancellation.

        Actual stopping semantics belong to the downstream robot driver.
        """
        if not self.has_published or self.shutdown():
            return
        try:
            self.publish(self.current_pose(), self.last_opening)
        except Exception as exc:
            self.log('Cannot publish hold command: ' + str(exc))

    def close(self):
        self.interrupt()
        thread = self.thread
        if thread is not None:
            thread.join()
