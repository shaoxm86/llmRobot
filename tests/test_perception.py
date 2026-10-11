"""Exercise actual callback/geometry code with ROS logging and cv2 isolated."""
import ast
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename, stubs):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


ros = SimpleNamespace(loginfo=lambda *a: None)
position = load('position_test', 'Position_detect.py', {'rospy': ros})
messages = load('messages_test', 'Piper_message.py', {'cv2': SimpleNamespace()})


def image(data, width, height, step, encoding='16UC1', bigendian=False):
    return SimpleNamespace(data=data, width=width, height=height, step=step,
                           encoding=encoding, is_bigendian=bigendian,
                           header=SimpleNamespace(stamp=SimpleNamespace(to_sec=lambda: 1.)))


class PerceptionTests(unittest.TestCase):
    def test_frozen_detector_still_updates_ema(self):
        detector = position.ObjectMotionDetector(ema_alpha=.25)
        detector.update(np.array([0., 0., 0.]))
        detector.freeze()
        need, filtered = detector.update(np.array([1., 0., 0.]))
        self.assertFalse(need)
        self.assertAlmostEqual(filtered[0], .25)

    def test_hysteresis_band_does_not_unfreeze(self):
        detector = position.ObjectMotionDetector(.1, .15)
        detector.update(np.zeros(3)); detector.freeze()
        need, _ = detector.update_with_distance(np.zeros(3), [.12, 0, 0])
        self.assertFalse(need)
        self.assertTrue(detector.frozen)
        need, _ = detector.update_with_distance(np.zeros(3), [.16, 0, 0])
        self.assertTrue(need)
        self.assertFalse(detector.frozen)

    def test_initial_frozen_detector_never_triggers_grasp(self):
        detector = position.ObjectMotionDetector(); detector.freeze()
        need, _ = detector.update(np.zeros(3))
        self.assertFalse(need)

    def test_reset_clears_frozen_state_for_retry(self):
        detector = position.ObjectMotionDetector(); detector.freeze(); detector.reset()
        self.assertFalse(detector.frozen)
        self.assertTrue(detector.update(np.zeros(3))[0])

    def test_missing_intrinsics_rejects_position(self):
        info = SimpleNamespace(fx=0., fy=0.)
        result = position.quick_object_pos_in_base(np.array([[0,0,5,5]]), np.ones((6,6)), info, np.eye(4))
        self.assertFalse(result[1])

    def test_zero_and_nan_depth_rejected(self):
        info = SimpleNamespace(fx=100., fy=100., cx=3., cy=3.)
        for depth in [np.zeros((6,6)), np.full((6,6), np.nan)]:
            self.assertFalse(position.quick_object_pos_in_base(np.array([[0,0,5,5]]), depth, info, np.eye(4))[1])

    def test_image_row_padding_and_big_endian(self):
        rows = np.array([[100, 200, 999], [300, 400, 999]], dtype='>u2')
        msg = image(rows.tobytes(), 2, 2, 6, bigendian=True)
        np.testing.assert_array_equal(messages.decode_image(msg, np.uint16), [[100,200],[300,400]])

    def test_float_depth_converts_metres_to_mm(self):
        depth = messages.Depth()
        data = np.array([[.5, np.nan, -1]], dtype=np.float32)
        depth.Callback(image(data.tobytes(), 3, 1, 12, '32FC1'))
        np.testing.assert_array_equal(depth.data, [[500,0,0]])

    def test_camera_info_not_ready_until_valid(self):
        info = messages.CameraInfoMsg()
        self.assertIsNone(info.snapshot())
        info.Callback(SimpleNamespace(width=2, height=2, K=[100,0,1,0,100,1,0,0,1]))
        self.assertEqual(info.snapshot().fx, 100)

    def test_unstamped_pose_requires_explicit_fallback(self):
        msg = SimpleNamespace(x=0., y=0., z=.2, roll=0., pitch=0., yaw=0.)
        end = messages.End2base(); end.Callback(msg)
        self.assertIsNone(end.buffer.latest())
        self.assertIsNotNone(end.feedback_buffer.latest())
        fallback = messages.End2base(True, lambda: 2.)
        fallback.Callback(msg)
        self.assertEqual(fallback.buffer.latest()[0], 2.)

    def test_recent_receipt_does_not_make_old_pose_fresh(self):
        msg = SimpleNamespace(x=0., y=0., z=.2, roll=0., pitch=0., yaw=0.,
            header=SimpleNamespace(stamp=SimpleNamespace(to_sec=lambda: 1.)))
        end = messages.End2base(False, lambda: 10.)
        end.Callback(msg)
        self.assertIsNone(end.feedback_buffer.latest())
        self.assertIsNone(end.T_end2base)

    def test_scene_contains_obstacles_outside_target_mask(self):
        tree = ast.parse((ROOT/'NewGrasp.py').read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'scene_from_depth')
        ns = {'np': np}
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'NewGrasp.py', 'exec'), ns)
        info = SimpleNamespace(fx=100., fy=100., cx=0., cy=0.)
        scene = ns['scene_from_depth'](np.array([[500., 1000., 0.]]), info)
        self.assertEqual(len(scene), 2)
        self.assertTrue(np.allclose(scene[:,2], [.5,1.]))

    def test_point_cloud_excludes_zero_depth(self):
        tree = ast.parse((ROOT/'PointCloud_Grasp.py').read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'get_point_cloud')
        ns = {'np': np, 'Camera_Info': lambda *a: None,
              'create_point_cloud_from_depth_image': lambda depth, camera, organized: np.zeros((*depth.shape,3))}
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'PointCloud_Grasp.py', 'exec'), ns)
        info = SimpleNamespace(width=10, height=10, fx=100., fy=100., cx=5., cy=5.)
        result = ns['get_point_cloud'](None, info, np.zeros((10,10,3)), np.zeros((10,10)), np.ones((10,10)))
        self.assertEqual(result, (None,)*5)

    def test_path_rejects_bad_intermediate_ik_and_joint_jump(self):
        tree = ast.parse((ROOT/'piper_pinocchio_mod2.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'validate_path')
        ns = {'np': np}
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'ik.py', 'exec'), ns)
        for solutions in [iter([(np.zeros(6), True), ([], False)]),
                          iter([(np.zeros(6), True), (np.ones(6), True)])]:
            fake = SimpleNamespace(get_ik_solution=lambda *args: next(solutions))
            self.assertFalse(ns['validate_path'](fake, np.zeros((2,3)), np.tile(np.eye(3),(2,1,1))))


if __name__ == '__main__':
    unittest.main()
