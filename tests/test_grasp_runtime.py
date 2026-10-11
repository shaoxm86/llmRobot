import unittest
import numpy as np
from scipy.spatial.transform import Rotation as R
from DMP import PositionDMP, QuaternionDMP
from Grasp_runtime import GraspExecutor, SampleBuffer
from Grasp_geometry import grasp_to_link6, gripper_position
from types import SimpleNamespace


class FakeRobot:
    def __init__(self):
        self.now = 1.
        self.pose = np.eye(4)
        self.commands = []
        self.paths = []
        self.confirm = True
        self.confirm_calls = []
        self.on_sleep = lambda: None
        self.stale = False
        self.reach = True
        self.follow = True

    def publish(self, pose, opening):
        self.commands.append((pose.copy(), opening))
        if self.follow:
            self.pose = pose.copy()

    def feedback(self):
        return (self.now - 10 if self.stale else self.now), self.pose.copy()

    def sleep(self, delay):
        self.now += max(delay, .001)
        self.on_sleep()

    def validate(self, positions, rotations, opening):
        self.paths.append(opening)
        return self.reach

    def confirmed(self, since, expected):
        self.confirm_calls.append(expected)
        return self.confirm and self.now > since

    def planner(self, start, target):
        return np.array([start[:3, 3], target[:3, 3]]), np.array([start[:3, :3], target[:3, :3]])

    def executor(self):
        return GraspExecutor(self.publish, self.feedback, self.validate, self.confirmed,
            sleep=self.sleep, clock=lambda: self.now, timeout=.05)

    def run(self, executor):
        target, place, home = [np.eye(4) for _ in range(3)]
        target[0, 3], place[0, 3] = .2, .4
        executor.run(target, place, home, self.planner)


class ExecutionTests(unittest.TestCase):
    def test_mid_approach_cancel_never_closes_or_transports(self):
        robot = FakeRobot(); executor = robot.executor()
        executor.set_state('APPROACH')
        robot.on_sleep = executor.interrupt
        robot.run(executor)
        self.assertEqual(executor.state, 'CANCELLED')
        self.assertFalse(executor.holding)
        self.assertEqual(robot.paths, [.06])
        self.assertTrue(all(width == .06 for _, width in robot.commands))
        self.assertIsNone(executor.target)
        self.assertTrue(executor.recoverable)

    def test_cancel_before_first_point_does_not_use_unbound_pose(self):
        robot = FakeRobot(); executor = robot.executor(); executor.interrupt()
        robot.run(executor)
        self.assertEqual(executor.state, 'CANCELLED')
        self.assertEqual(robot.commands, [])

    def test_no_confirmation_no_transport_and_no_automatic_retry(self):
        robot = FakeRobot(); robot.confirm = False; executor = robot.executor()
        robot.run(executor)
        self.assertEqual(executor.state, 'FAILED')
        self.assertEqual(robot.paths, [.06])
        self.assertFalse(executor.holding)
        self.assertFalse(executor.recoverable)
        self.assertFalse(executor.start(np.eye(4), np.eye(4), np.eye(4), robot.planner))

    def test_success_requires_close_and_release_feedback_and_three_paths(self):
        robot = FakeRobot(); executor = robot.executor(); robot.run(executor)
        self.assertEqual(executor.state, 'COMPLETE')
        self.assertFalse(executor.holding)
        self.assertEqual(robot.paths, [.06, .03, .06])
        self.assertIn(True, robot.confirm_calls)
        self.assertIn(False, robot.confirm_calls)

    def test_cancel_while_closing_never_transports_or_claims_holding(self):
        robot = FakeRobot(); robot.confirm = False; executor = robot.executor()
        robot.on_sleep = lambda: executor.interrupt() if executor.state == 'CLOSING' else None
        robot.run(executor)
        self.assertEqual(executor.state, 'CANCELLED')
        self.assertEqual(robot.paths, [.06])
        self.assertFalse(executor.holding)
        self.assertFalse(executor.recoverable)
        self.assertEqual(robot.commands[-1][1], .03)

    def test_release_without_confirmation_never_returns_or_claims_release(self):
        robot = FakeRobot(); executor = robot.executor()
        executor.confirmed = lambda since, expected: expected and robot.now > since
        robot.run(executor)
        self.assertEqual(executor.state, 'FAILED')
        self.assertTrue(executor.holding)
        self.assertEqual(robot.paths, [.06, .03])
        self.assertFalse(executor.recoverable)

    def test_unreachable_path_publishes_nothing(self):
        robot = FakeRobot(); robot.reach = False; executor = robot.executor()
        robot.run(executor)
        self.assertEqual(executor.state, 'FAILED')
        self.assertEqual(robot.commands, [])

    def test_stale_feedback_publishes_nothing(self):
        robot = FakeRobot(); robot.stale = True; executor = robot.executor()
        robot.run(executor)
        self.assertEqual(executor.state, 'FAILED')
        self.assertEqual(robot.commands, [])

    def test_position_not_reached_never_closes(self):
        robot = FakeRobot(); robot.follow = False; executor = robot.executor()
        robot.run(executor)
        self.assertEqual(executor.state, 'FAILED')
        self.assertTrue(all(width == .06 for _, width in robot.commands))

    def test_offline_delayed_publish_does_not_burst_catch_up_points(self):
        robot = FakeRobot(); executor = robot.executor()
        times = []
        original = robot.publish
        def delayed_publish(command, opening):
            times.append(robot.now)
            original(command, opening)
            if len(times) == 1:
                robot.now += .05
        executor.publish = delayed_publish
        target = np.eye(4)
        def planner(start, goal):
            return np.zeros((4,3)), np.tile(np.eye(3),(4,1,1))
        executor.move(target, .06, planner)
        self.assertGreaterEqual(times[1] - times[0], .06 - 1e-12)
        self.assertGreaterEqual(min(np.diff(times[1:])), .01 - 1e-12)

    def test_publish_exception_cleans_state(self):
        robot = FakeRobot(); executor = robot.executor()
        def failure(*args):
            raise RuntimeError('driver error')
        executor.publish = failure
        robot.run(executor)
        self.assertEqual(executor.state, 'FAILED')
        self.assertIsNone(executor.target)


class GeometryTests(unittest.TestCase):
    def test_cached_base_target_does_not_follow_moving_camera(self):
        grasp = SimpleNamespace(rotation_matrix=np.eye(3), translation=np.array([0., 0., .5]))
        capture = np.eye(4)
        target = grasp_to_link6(grasp, capture)
        cached = target.copy()
        capture[0, 3] += .1
        self.assertTrue(np.allclose(target, cached))
        self.assertFalse(np.allclose(cached, grasp_to_link6(grasp, capture)))

    def test_link6_compensation_preserves_gripper_clearance(self):
        grasp = SimpleNamespace(rotation_matrix=np.eye(3), translation=np.array([0., 0., -.5]))
        target = grasp_to_link6(grasp, np.eye(4))
        self.assertAlmostEqual(gripper_position(target)[2], .004)
        self.assertAlmostEqual(np.linalg.det(target[:3, :3]), 1.)

    def test_position_and_rotation_use_same_progress(self):
        positions, _, progress = PositionDMP(15, 15/4).generate_traj([0,0,0], [1,0,0], tau=2.)
        _, rotations = QuaternionDMP().generate_traj(np.eye(3), R.from_euler('z', 90, degrees=True).as_matrix(), progress=progress)
        angle_progress = R.from_matrix(rotations).magnitude() / (np.pi / 2)
        np.testing.assert_allclose(positions[:,0], progress)
        np.testing.assert_allclose(angle_progress, progress, atol=1e-10)
        np.testing.assert_allclose(positions[-1], [1,0,0])

    def test_rotation_only_has_smooth_progress(self):
        positions, _, progress = PositionDMP().generate_traj([0,0,0], [0,0,0], tau=1.)
        self.assertTrue(np.isfinite(progress).all())
        self.assertTrue((np.diff(progress) >= 0).all())
        np.testing.assert_array_equal(positions, np.zeros_like(positions))

    def test_timestamp_matching_rejects_unsynchronized_pose(self):
        buffer = SampleBuffer(); buffer.add(1., 'old'); buffer.add(2., 'new')
        self.assertIsNone(buffer.nearest(1.5, .05))
        self.assertEqual(buffer.nearest(2.01, .05)[1], 'new')


if __name__ == '__main__':
    unittest.main()
