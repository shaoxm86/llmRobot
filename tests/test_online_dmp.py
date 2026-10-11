import unittest
import numpy as np
from scipy.spatial.transform import Rotation as R
from DMP import OnlinePoseDMP, QuinticPoseGoal
from test_grasp_runtime import FakeRobot


def pose(position=(0., 0., 0.), yaw=0.):
    result = np.eye(4)
    result[:3, 3] = position
    result[:3, :3] = R.from_euler('z', yaw).as_matrix()
    return result


class OnlineDMPTests(unittest.TestCase):
    def test_goal_update_preserves_position_velocity_quaternion_and_angular_velocity(self):
        dmp = OnlinePoseDMP(pose(), pose((.2, 0, 0), .5))
        for _ in range(40):
            dmp.step(.01)
        state = (dmp.position.copy(), dmp.velocity.copy(), dmp.orientation.as_quat().copy(), dmp.angular_velocity.copy())
        tau = dmp.tau
        dmp.set_goal(pose((.1, .2, .1), -.5))
        for before, after in zip(state, (dmp.position, dmp.velocity, dmp.orientation.as_quat(), dmp.angular_velocity)):
            np.testing.assert_array_equal(before, after)
        self.assertEqual(dmp.tau, tau)
        for _ in range(1500):
            dmp.step(.01)
        np.testing.assert_allclose(dmp.position, [.1, .2, .1], atol=.001)
        self.assertLess((dmp.orientation * R.from_euler('z', -.5).inv()).magnitude(), .001)
        self.assertTrue(dmp.settled())

    def test_goal_jump_respects_speed_and_acceleration_limits(self):
        dmp = OnlinePoseDMP(pose(), pose((1, 0, 0), 1.))
        last_position = dmp.position.copy()
        last_velocity, last_angular = dmp.velocity.copy(), dmp.angular_velocity.copy()
        for step in range(500):
            if step == 100:
                dmp.set_goal(pose((-1, 1, 0), -2.))
            current = dmp.step(.01)
            self.assertLessEqual(np.linalg.norm(dmp.velocity), .100001)
            self.assertLessEqual(np.linalg.norm(dmp.angular_velocity), .500001)
            self.assertLessEqual(np.linalg.norm(dmp.velocity - last_velocity) / .01, .250001)
            self.assertLessEqual(np.linalg.norm(dmp.angular_velocity - last_angular) / .01, 1.000001)
            self.assertLessEqual(np.linalg.norm(dmp.position-last_position), .001001)
            self.assertAlmostEqual(np.linalg.norm(dmp.orientation.as_quat()), 1.)
            last_position = current[:3, 3].copy()
            last_velocity, last_angular = dmp.velocity.copy(), dmp.angular_velocity.copy()

    def test_rotation_only_shortest_path_crosses_pi_without_long_spin(self):
        dmp = OnlinePoseDMP(pose(yaw=np.deg2rad(179)), pose(yaw=np.deg2rad(-179)))
        total_rotation = 0.
        previous = dmp.orientation
        for _ in range(800):
            dmp.step(.01)
            total_rotation += (dmp.orientation * previous.inv()).magnitude()
            previous = dmp.orientation
        self.assertLess(total_rotation, np.deg2rad(3))
        self.assertTrue(dmp.settled())
        np.testing.assert_array_equal(dmp.position, [0,0,0])

    def test_nonrigid_goal_rejected_without_mutating_state(self):
        dmp = OnlinePoseDMP(pose(), pose((.1,0,0)))
        invalid = np.eye(4); invalid[0,0] = 2.
        with self.assertRaises(ValueError):
            dmp.set_goal(invalid)
        np.testing.assert_array_equal(dmp.goal_position, [.1,0,0])


class QuinticGoalTests(unittest.TestCase):
    def test_rest_to_rest_matches_screenshot_polynomial(self):
        transition = QuinticPoseGoal(pose(), duration=.5)
        transition.retarget(pose((1,2,3), .8))
        transition.step(.125)
        u = .25
        blend = 10*u**3 - 15*u**4 + 6*u**5
        current = transition.sample()[0]
        np.testing.assert_allclose(current[:3,3], blend*np.array([1,2,3]))
        self.assertAlmostEqual(R.from_matrix(current[:3,:3]).as_rotvec()[2], .8*blend)

    def test_endpoints_have_zero_velocity_and_acceleration(self):
        transition = QuinticPoseGoal(pose(), duration=.5)
        transition.retarget(pose((1,0,0), .8))
        start = transition.sample()
        for derivative in start[1:]:
            np.testing.assert_allclose(derivative, np.zeros(3), atol=1e-10)
        transition.step(.5)
        finish = transition.sample()
        np.testing.assert_allclose(finish[0], pose((1,0,0), .8))
        for derivative in finish[1:]:
            np.testing.assert_allclose(derivative, np.zeros(3), atol=1e-10)

    def test_retarget_mid_transition_preserves_linear_and_angular_derivatives(self):
        transition = QuinticPoseGoal(pose(), duration=.5)
        destination = pose((.2,.1,0))
        destination[:3,:3] = R.from_euler('xyz',[.4,.3,.2]).as_matrix()
        transition.retarget(destination)
        transition.step(.17)
        before = transition.sample()
        new = pose((-.1,.2,.1))
        new[:3,:3] = R.from_euler('xyz',[-.2,.5,-.4]).as_matrix()
        transition.retarget(new)
        after = transition.sample()
        for old, new in zip(before, after):
            np.testing.assert_allclose(old, new, atol=1e-8)
        transition.step(.5)
        np.testing.assert_allclose(transition.sample()[0], transition.destination)

    def test_rotation_derivatives_agree_with_numerical_pose_derivative(self):
        transition = QuinticPoseGoal(pose(), duration=.8)
        target = pose((.1,0,0))
        target[:3,:3] = R.from_euler('xyz',[.6,-.4,.3]).as_matrix()
        transition.retarget(target)
        epsilon = 1e-5
        transition.elapsed = .27
        _, _, _, velocity, acceleration = transition.sample()
        transition.elapsed = .27-epsilon
        before, _, _, w_before, _ = transition.sample()
        transition.elapsed = .27+epsilon
        after, _, _, w_after, _ = transition.sample()
        numerical_velocity = (R.from_matrix(after[:3,:3]) * R.from_matrix(before[:3,:3]).inv()).as_rotvec()/(2*epsilon)
        numerical_acceleration = (w_after-w_before)/(2*epsilon)
        np.testing.assert_allclose(velocity, numerical_velocity, atol=1e-7)
        np.testing.assert_allclose(acceleration, numerical_acceleration, atol=1e-6)

    def test_many_retargets_keep_pose_and_derivatives_finite(self):
        rng = np.random.default_rng(1234)
        dmp = OnlinePoseDMP(pose(), pose((.1,0,0)))
        for index in range(500):
            if index % 20 == 0:
                target = pose(rng.uniform(-.2,.2,3))
                target[:3,:3] = R.from_rotvec(rng.uniform(-1.,1.,3)).as_matrix()
                dmp.set_goal(target)
            output = dmp.step(.01)
            self.assertTrue(np.isfinite(output).all())
            self.assertTrue(np.isfinite(dmp.goal_filter.sample()[1:]).all())
            self.assertAlmostEqual(np.linalg.det(output[:3,:3]), 1.)
            self.assertLessEqual(np.linalg.norm(dmp.velocity), .100001)

    def test_identical_goal_does_not_restart_transition(self):
        transition = QuinticPoseGoal(pose(), duration=.5)
        target = pose((1,0,0))
        transition.retarget(target); transition.step(.2)
        before = transition.sample()
        transition.retarget(target)
        self.assertAlmostEqual(transition.elapsed, .2)
        np.testing.assert_allclose(transition.sample()[0], before[0])

    def test_goal_switch_preserves_ideal_dmp_acceleration_and_jerk(self):
        dmp = OnlinePoseDMP(pose(), pose((.1,0,0)), max_acceleration=100.)
        for _ in range(20):
            dmp.step(.01)
        def derivatives():
            goal, goal_velocity, _, _, _ = dmp.goal_filter.sample()
            a = dmp.alpha * (dmp.beta*(goal[:3,3]-dmp.position)-dmp.tau*dmp.velocity)/dmp.tau**2
            jerk = dmp.alpha*(dmp.beta*(goal_velocity-dmp.velocity)-dmp.tau*a)/dmp.tau**2
            return a, jerk
        before = derivatives()
        dmp.set_goal(pose((-.1,.2,0)))
        after = derivatives()
        np.testing.assert_allclose(before, after, atol=1e-10)
        self.assertFalse(dmp.settled())


class OnlineExecutionTests(unittest.TestCase):
    def executor(self, robot):
        executor = robot.executor()
        executor.online_approach = True
        executor.set_state('APPROACH')
        return executor

    def test_changed_goal_reached_without_restarting_dmp_and_closes_at_new_goal(self):
        robot = FakeRobot(); executor = self.executor(robot)
        new_goal = pose((.1, .15, 0), .4)
        updates = []
        def change_goal():
            if robot.now > 1.3 and not updates and executor.state == 'APPROACH':
                controller = executor.online_controller
                state = (controller.position.copy(), controller.velocity.copy(),
                         controller.orientation.as_quat().copy(), controller.angular_velocity.copy())
                self.assertTrue(executor.update_goal(new_goal))
                for before, after in zip(state, (controller.position, controller.velocity,
                                                controller.orientation.as_quat(), controller.angular_velocity)):
                    np.testing.assert_array_equal(before, after)
                updates.append(robot.now)
        robot.on_sleep = change_goal
        robot.run(executor)
        self.assertEqual(executor.state, 'COMPLETE')
        self.assertEqual(len(updates), 1)
        close_pose = next(command for command, opening in robot.commands if opening == .03)
        np.testing.assert_allclose(close_pose, new_goal)
        self.assertIsNone(executor.error)
        self.assertGreater(len(robot.commands), 100)

    def test_pending_perception_refresh_blocks_closing_but_keeps_motion_alive(self):
        robot = FakeRobot(); executor = self.executor(robot)
        executor.set_goal_refresh_pending(True)
        def release_refresh():
            if robot.now >= 10.:
                executor.set_goal_refresh_pending(False)
        robot.on_sleep = release_refresh
        close_times = []
        original_publish = executor.publish
        def publish(command, opening):
            if opening == .03:
                close_times.append(robot.now)
            original_publish(command, opening)
        executor.publish = publish
        robot.run(executor)
        self.assertEqual(executor.state, 'COMPLETE')
        self.assertGreaterEqual(close_times[0], 10.)

    def test_visualization_preview_never_mutates_execution_state(self):
        robot = FakeRobot(); executor = self.executor(robot)
        executor.online_controller = OnlinePoseDMP(pose(), pose((.2,0,0)))
        executor.approach_goal = pose((.2,0,0))
        before = executor.online_controller.pose()
        preview, goal = executor.online_preview()
        self.assertGreater(preview[-1,0,3], 0.)
        np.testing.assert_array_equal(executor.online_controller.pose(), before)

    def test_cancelled_online_approach_never_closes(self):
        robot = FakeRobot(); executor = self.executor(robot)
        robot.on_sleep = executor.interrupt
        robot.run(executor)
        self.assertEqual(executor.state, 'CANCELLED')
        self.assertTrue(all(opening == .06 for _, opening in robot.commands))

    def test_invalid_next_step_does_not_advance_controller(self):
        robot = FakeRobot(); executor = self.executor(robot)
        robot.reach = False
        robot.run(executor)
        self.assertEqual(executor.state, 'FAILED')
        self.assertEqual(robot.commands, [])
        np.testing.assert_array_equal(executor.online_controller.position, [0,0,0])

    def test_goal_changed_during_validation_discards_old_proposal(self):
        robot = FakeRobot(); executor = self.executor(robot)
        new_goal = pose((0., .1, 0.))
        calls = []
        def validate(*args):
            if not calls:
                self.assertTrue(executor.update_goal(new_goal))
            calls.append(1)
            return True
        executor.validate = validate
        robot.on_sleep = executor.interrupt
        robot.run(executor)
        self.assertEqual(executor.state, 'CANCELLED')
        self.assertEqual(len(calls), 2)
        # The new goal is now blended from the previous attractor, so x need
        # not immediately become zero. A nonzero y proves the new target was
        # used, and elapsed=.01 proves the rejected proposal was not committed.
        self.assertGreater(robot.commands[0][0][1,3], 0.)
        self.assertAlmostEqual(executor.online_controller.goal_filter.elapsed, .01)

    def test_target_update_rejected_after_closing_begins(self):
        robot = FakeRobot(); executor = self.executor(robot)
        executor.set_state('CLOSING')
        self.assertFalse(executor.update_goal(pose((.2,0,0))))

    def test_processing_delay_failure_never_publishes_invalid_late_point(self):
        robot = FakeRobot(); executor = self.executor(robot)
        def slow_validator(*args):
            robot.now += .3
            return True
        executor.validate = slow_validator
        robot.run(executor)
        self.assertEqual(executor.state, 'FAILED')
        self.assertEqual(robot.commands, [])

    def test_expired_observation_stops_even_when_perception_loop_is_blocked(self):
        robot = FakeRobot(); executor = self.executor(robot)
        executor.approach_health = lambda: robot.now < 1.05
        robot.run(executor)
        self.assertEqual(executor.state, 'FAILED')
        self.assertIn('目标观测', executor.error)
        self.assertTrue(all(opening == .06 for _, opening in robot.commands))

    def test_target_expires_during_validation_never_publishes(self):
        robot = FakeRobot(); executor = self.executor(robot)
        executor.approach_health = lambda: robot.now < 1.05
        def validate(*args):
            robot.now += .1
            return True
        executor.validate = validate
        robot.run(executor)
        self.assertEqual(executor.state, 'FAILED')
        self.assertEqual(robot.commands, [])

    def test_tracking_error_stops_instead_of_resetting_state(self):
        robot = FakeRobot(); robot.follow = False; executor = self.executor(robot)
        robot.run(executor)
        self.assertEqual(executor.state, 'FAILED')
        self.assertIn('偏离', executor.error)
        self.assertTrue(all(opening == .06 for _, opening in robot.commands))


if __name__ == '__main__':
    unittest.main()
