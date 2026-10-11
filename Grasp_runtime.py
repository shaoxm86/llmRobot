"""Hardware-independent execution state machine and stamped sample buffer."""
import copy
import threading
import time
from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation as R
from DMP import OnlinePoseDMP


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
                 close_width=.03, finish=lambda: None, log=lambda msg: None,
                 online_approach=False, online_dmp_options=None, goal_quiet_time=.3,
                 approach_timeout=60., max_step_delay=.2, max_tracking_error=.05,
                 max_rotation_tracking_error=.3, approach_health=lambda: True):
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
        self.online_approach = online_approach
        self.online_dmp_options = dict(online_dmp_options or {})
        self.goal_quiet_time = goal_quiet_time
        self.approach_timeout = approach_timeout
        self.max_step_delay = max_step_delay
        self.max_tracking_error = max_tracking_error
        self.max_rotation_tracking_error = max_rotation_tracking_error
        self.approach_goal = None
        self.goal_revision = 0
        self.goal_updated_at = self.clock()
        self.goal_refresh_pending = False
        self.online_controller = None
        self.control_overruns = 0
        self.approach_health = approach_health
        if (not np.isfinite(dt) or dt <= 0 or
                not all(np.isfinite(value) and value > 0 for value in
                    (timeout, feedback_age, approach_timeout, max_step_delay,
                     max_tracking_error, max_rotation_tracking_error)) or
                not np.isfinite(goal_quiet_time) or goal_quiet_time < 0):
            raise ValueError('Invalid execution timing or tracking limits')

    def snapshot(self):
        with self.lock:
            return self.state, self.holding, self.target, self.error

    def active(self):
        with self.lock:
            return self.thread is not None and self.thread.is_alive()

    def update_goal(self, target):
        """Thread-safe update of g and q_g only; do not restart the executor."""
        target = OnlinePoseDMP.checked_pose(target)
        with self.lock:
            if (not self.online_approach or self.state != 'APPROACH' or
                    self.cancel.is_set()):
                return False
            changed = (self.approach_goal is None or
                np.linalg.norm(target[:3, 3] - self.approach_goal[:3, 3]) > 1e-6 or
                R.from_matrix(target[:3, :3] @ self.approach_goal[:3, :3].T).magnitude() > 1e-6)
            if changed:
                self.approach_goal = target.copy()
                self.goal_revision += 1
                self.goal_updated_at = self.clock()
            return True

    def set_goal_refresh_pending(self, pending):
        with self.lock:
            if self.state == 'APPROACH':
                self.goal_refresh_pending = bool(pending)

    def move_online(self, target):
        """Integrate live DMP state; validate each next command before publishing.

        The validation API is synchronous. A slow checker causes an overrun or
        task failure, never unvalidated commands or a burst of catch-up samples.
        """
        self.check()
        start = self.current_pose()
        controller = OnlinePoseDMP(start, target, **self.online_dmp_options)
        with self.lock:
            if self.approach_goal is None:
                self.approach_goal = target.copy()
            self.online_controller = controller
        started = self.clock()
        while self.clock() - started < self.approach_timeout:
            tick = self.clock()
            self.check()
            actual = self.current_pose()
            if not self.approach_health():
                raise RuntimeError('目标观测过期或丢失，停止在线接近')
            with self.lock:
                goal, revision = self.approach_goal.copy(), self.goal_revision
            # Copy for speculative integration: rejected steps never mutate state.
            candidate = copy.deepcopy(controller)
            candidate.set_goal(goal)
            previous = controller.pose()
            proposed = candidate.step(self.dt)
            if (np.linalg.norm(actual[:3, 3] - previous[:3, 3]) > self.max_tracking_error or
                    R.from_matrix(actual[:3, :3] @ previous[:3, :3].T).magnitude() > self.max_rotation_tracking_error):
                raise RuntimeError('实际末端偏离在线 DMP 状态，停止执行')
            positions = np.array([actual[:3, 3], previous[:3, 3], proposed[:3, 3]])
            rotations = np.array([actual[:3, :3], previous[:3, :3], proposed[:3, :3]])
            valid = self.validate(positions, rotations, self.open_width)
            self.check()
            elapsed = self.clock() - tick
            if elapsed > self.max_step_delay:
                raise RuntimeError('在线路径检查超过控制延迟上限')
            actual = self.current_pose()
            if not self.approach_health():
                raise RuntimeError('路径检查期间目标观测过期')
            if (np.linalg.norm(actual[:3, 3] - previous[:3, 3]) > self.max_tracking_error or
                    R.from_matrix(actual[:3, :3] @ previous[:3, :3].T).magnitude() > self.max_rotation_tracking_error):
                raise RuntimeError('路径检查期间实际末端偏离在线 DMP 状态')
            with self.lock:
                if revision != self.goal_revision:
                    # A new goal arrived during validation: discard this proposal,
                    # keep x/v/q/omega, and validate a step toward the newest goal.
                    continue
                if not valid:
                    raise RuntimeError('在线 DMP 下一步路径检查失败')
                self.publish(proposed, self.open_width)
                self.last_opening = self.open_width
                self.has_published = True
                controller = candidate
                self.online_controller = controller
                self.target = proposed.copy()
            actual = self.current_pose()
            measured_arrival = (np.linalg.norm(actual[:3, 3] - goal[:3, 3]) <= self.pos_tol and
                R.from_matrix(actual[:3, :3] @ goal[:3, :3].T).magnitude() <= self.rot_tol)
            with self.lock:
                if (revision == self.goal_revision and not self.goal_refresh_pending and
                        self.clock() - self.goal_updated_at >= self.goal_quiet_time and
                        controller.settled() and measured_arrival):
                    # Atomically stop accepting goal updates before closing.
                    self.state = 'CLOSING'
                    return goal.copy()
            elapsed = self.clock() - tick
            if elapsed > self.dt:
                self.control_overruns += 1
                if self.control_overruns == 1 or self.control_overruns % 100 == 0:
                    self.log('在线控制周期超时: %.4fs (目标 %.4fs)' % (elapsed, self.dt))
            self.sleep(max(0., self.dt - elapsed))
        raise RuntimeError('在线接近超过最大持续时间')

    def online_preview(self, horizon=.5):
        """Visualization-only rollout of a copy; execution state is untouched."""
        with self.lock:
            if self.online_controller is None or self.state != 'APPROACH':
                return None
            controller = copy.deepcopy(self.online_controller)
            goal = self.approach_goal.copy()
        controller.set_goal(goal)
        poses = [controller.pose()]
        for _ in range(max(1, int(horizon / self.dt))):
            poses.append(controller.step(self.dt))
        return np.array(poses), goal

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
            # Do not burst old samples to catch up after a blocking publish.
            deadline = max(deadline + self.dt, self.clock() + self.dt)
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
            self.approach_goal = target.copy()
            self.goal_revision += 1
            self.goal_updated_at = self.clock()
            self.goal_refresh_pending = False
            self.online_controller = None
            self.control_overruns = 0
            self.thread = threading.Thread(target=self.run,
                args=(target.copy(), place.copy(), home.copy(), planner), daemon=False)
            self.thread.start()
            return True

    def run(self, target, place, home, planner):
        try:
            if self.online_approach:
                target = self.move_online(target)
            else:
                self.move(target, self.open_width, planner)
                self.set_state('CLOSING')
            self.check()
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
