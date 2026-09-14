import json
import os
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from core.line_follower import LineDetector, LineFollower
from core.traffic_control import TrafficControlRunner
from debug_web import CONFIG_SCHEMA, DebugWebServer


class IntegrationTests(unittest.TestCase):
    def test_batched_row_segments_preserve_empty_full_and_split_rows(self):
        rng = np.random.default_rng(42)
        mask = (rng.random((80, 100)) > .4).astype(np.uint8)*255
        mask[0] = 0
        mask[1] = 255
        actual = LineDetector._row_segments(mask)
        for y, row in enumerate(mask):
            xs = np.flatnonzero(row)
            groups = np.split(xs, np.flatnonzero(np.diff(xs) > 1)+1)
            expected = [(int(g[0]), int(g[-1]), len(g)) for g in groups if len(g)]
            self.assertEqual(actual[y], expected)

    def test_corner_stem_batch_fit_keeps_outliers_out_with_one_least_squares(self):
        points = [(100.0 + 0.25*y, y, 20) for y in range(50)]
        points = [(x + (40 if y % 5 == 0 else 0), y, w)
                  for x, y, w in points]
        expected, expected_inliers = LineDetector._robust_linear_fit(
            points, residual_limit=3.0)
        with patch('core.line_follower.np.polyfit', wraps=np.polyfit) as fit:
            actual, inliers = LineDetector._batch_corner_stem_fit(
                points, residual_limit=3.0)
        np.testing.assert_allclose(actual, expected, atol=1e-9)
        self.assertEqual(inliers, expected_inliers)
        self.assertEqual(len(inliers), 40)
        self.assertEqual(fit.call_count, 1)

    def runner(self, speed=300):
        chassis = Mock()
        chassis.send_speed.return_value = True
        dashboard = Mock()
        dashboard.heartbeat_age.return_value = 0
        dashboard.get_traffic_status.return_value = {
            'state': 'ready', 'age_sec': 0, 'detections': []}
        f = SimpleNamespace(base_speed=speed, max_z=800, chassis=chassis, web_debug=dashboard)
        r = TrafficControlRunner(f)
        r.armed = True
        r.active = True
        r.last_line_valid = True
        r.last_feedback = r.last_tick = time.monotonic()
        return r

    def test_motion_is_disarmed_by_default(self):
        r = self.runner()
        r.armed = False
        self.assertEqual(r.send((300, 100, -240), threading.Event()), (0, 0, 0))
        r.f.chassis.send_speed.assert_called_with(0, 0, 0)

    def test_arm_requires_fresh_dependencies_and_disarm_stops(self):
        r = self.runner()
        r.armed = False
        r.last_feedback = time.monotonic()
        result = r.set_armed(True, 120)
        self.assertTrue(result['chassis_armed'])
        self.assertEqual(r.f.base_speed, 120)
        self.assertEqual(r.send((120, 0, 0), threading.Event()), (120, 0, 0))
        result = r.set_armed(False)
        self.assertFalse(result['chassis_armed'])
        self.assertEqual(r.send((120, 0, 0), threading.Event()), (0, 0, 0))

        r.last_feedback = None
        with self.assertRaisesRegex(RuntimeError, '底盘反馈'):
            r.set_armed(True, 120)

    def test_forward_speed_range_accepts_400_and_rejects_401(self):
        r = self.runner(0)
        r.armed = False
        r.last_feedback = time.monotonic()

        result = r.set_armed(True, 400)

        self.assertTrue(result['chassis_armed'])
        self.assertEqual(r.f.base_speed, 400)
        r.set_armed(False)
        with self.assertRaisesRegex(ValueError, '1~400'):
            r.set_armed(True, 401)
        self.assertEqual(CONFIG_SCHEMA['speed'][2], 400)

    def test_command_gate_estop_covers_all_axes(self):
        r = self.runner()
        event = threading.Event()
        self.assertEqual(r.send((0,100,0), event), (0,100,0))
        event.set()
        self.assertEqual(r.send((300,100,-240), event), (0,0,0))
        r.f.chassis.send_speed.assert_called_with(0,0,0)

    def test_zero_speed_gate_covers_turns_and_lateral(self):
        r = self.runner(0)
        self.assertEqual(r.send((0,100,240), threading.Event()), (0,0,0))

    def test_watchdog_heartbeat_fault_is_latched(self):
        r = self.runner()
        r.f.web_debug.heartbeat_age.return_value = 4
        event = threading.Event()
        r.check_safety(time.monotonic(), event)
        self.assertIn('心跳', r.fault)
        r.f.web_debug.heartbeat_age.return_value = 0
        self.assertEqual(r.send((300,0,0), event), (0,0,0))

    def test_watchdog_feedback_and_camera_block(self):
        for feedback in (True, False):
            r = self.runner()
            if feedback:
                r.last_feedback -= 1
            else:
                r.last_tick -= 1
            r.check_safety(time.monotonic(), threading.Event())
            self.assertTrue(r.fault)
            r.f.chassis.send_speed.assert_called_with(0,0,0)

    def test_send_failure_latches_stop(self):
        r = self.runner()
        r.f.chassis.send_speed.return_value = False
        self.assertEqual(r.send((100,0,0), threading.Event()), (0,0,0))
        self.assertIn('串口', r.fault)

    def test_full_loop_static_validation_and_finally_stop(self):
        camera, chassis, web = Mock(), Mock(), Mock()
        camera.read.return_value = np.zeros((480,640,3),np.uint8)
        chassis.send_speed.return_value = True
        chassis.read_status.return_value = {'real_x':0,'real_y':0,'real_z':0,'ang_vel_z':0}
        web.heartbeat_age.return_value = 0
        web.get_traffic_status.return_value = {'state':'ready','age_sec':0,'sequence':1,'detections':[]}
        web.restart_requested = False
        f = LineFollower(camera,chassis,base_speed=0,web_debug=web,startup_frames=1)
        f.frame_interval = 0
        f.detector = Mock(work_width=320,crop_top_frac=.6,crop_bottom_frac=.5,binary_mode='otsu')
        f.detector.process.return_value = {'is_valid':True,'error_px':0,'angle_deg':0}
        r = TrafficControlRunner(f)
        r.run(max_frames=4)
        self.assertEqual(web.update.call_count,4)
        self.assertTrue(all(call.args==(0,0,0) for call in chassis.send_speed.call_args_list))
        self.assertEqual(chassis.stop.call_count,2)
        self.assertEqual(web.update.call_args.args[2]['behavior']['speed_ceiling_mm_s'],0)

    def test_camera_drop_latches_and_cannot_resume(self):
        camera, chassis, web = Mock(), Mock(), Mock()
        frame = np.zeros((480,640,3),np.uint8)
        camera.read.side_effect = [frame,None,frame]
        chassis.send_speed.return_value = True
        chassis.read_status.return_value = {'real_x':0,'real_y':0,'real_z':0,'ang_vel_z':0}
        web.heartbeat_age.return_value = 0
        web.get_traffic_status.return_value = {'state':'ready','age_sec':0,'sequence':1,'detections':[]}
        web.restart_requested = False
        f = LineFollower(camera,chassis,base_speed=100,web_debug=web,startup_frames=1)
        f.frame_interval = 0
        f.detector = Mock(work_width=320,crop_top_frac=.6,crop_bottom_frac=.5,binary_mode='otsu')
        f.detector.process.return_value = {'is_valid':True,'error_px':0,'angle_deg':0}
        r = TrafficControlRunner(f)
        r.armed = True
        r.run(max_frames=3)
        self.assertIn('相机', r.fault)
        self.assertTrue(all(call.args==(0,0,0) for call in chassis.send_speed.call_args_list[1:]))

    def test_traffic_loop_executes_remembered_right_corner_in_blind_zone(self):
        camera, chassis, web = Mock(), Mock(), Mock()
        frame = np.zeros((480, 640, 3), np.uint8)
        camera.read.return_value = frame
        chassis.send_speed.return_value = True
        chassis.read_status.return_value = {
            'real_x': 0, 'real_y': 0, 'real_z': 0, 'ang_vel_z': 0}
        web.heartbeat_age.return_value = 0
        web.get_traffic_status.return_value = {
            'state': 'ready', 'age_sec': 0, 'sequence': 1,
            'detections': []}
        web.restart_requested = False
        follower = LineFollower(
            camera, chassis, base_speed=250, web_debug=web,
            startup_frames=1, ramp_frames=0, corner_delay_frames=1,
            corner_delay_distance_m=0.0)
        follower.frame_interval = 0
        follower.detector = Mock(
            work_width=320, crop_top_frac=.6, crop_bottom_frac=.5,
            binary_mode='otsu')
        follower.detector.process.side_effect = [
            {'is_valid': True, 'corner_dir': 0,
             'error_px': 0, 'angle_deg': 0},
            {'is_valid': True, 'corner_dir': 1, 'corner_y_ratio': .7,
             'corner_span': 180, 'error_px': 0, 'angle_deg': 0},
            {'is_valid': True, 'corner_dir': 1, 'corner_y_ratio': .7,
             'corner_span': 180, 'error_px': 0, 'angle_deg': 0},
            {'is_valid': True, 'corner_dir': 1, 'corner_y_ratio': .88,
             'corner_span': 180, 'error_px': 0, 'angle_deg': 0},
            {'is_valid': False, 'corner_dir': 0,
             'error_px': 0, 'angle_deg': 0},
        ]
        runner = TrafficControlRunner(follower)
        runner.armed = True

        runner.run(max_frames=5)

        commands = [call.args for call in chassis.send_speed.call_args_list]
        confirm_updates = [call.args[2] for call in web.update.call_args_list
                           if call.args[2]['state'] == 'corner-confirm-right']
        self.assertEqual(len(confirm_updates), 1)
        self.assertTrue(all(update['speed'] <= 125
                            for update in confirm_updates))
        self.assertTrue(any(x == 0 and y == 0 and z < 0
                            for x, y, z in commands))
        self.assertTrue(any(call.args[2]['state'] == 'corner-right'
                            for call in web.update.call_args_list))

    def test_main_path_fit_is_robust_but_strictly_linear(self):
        points = [(40, 20, 8), (50, 40, 9), (60, 60, 10),
                  (70, 80, 11), (250, 50, 8)]
        coeffs, inliers = LineDetector._robust_linear_fit(points)

        self.assertEqual(coeffs[0], 0.0)
        self.assertAlmostEqual(coeffs[1], 0.5, places=5)
        self.assertAlmostEqual(coeffs[2], 30.0, places=5)
        self.assertEqual(len(inliers), 4)

    def test_near_curve_is_validated_before_linear_direction_fit(self):
        # Regression for a real right-angle approach: a global straight-line
        # RANSAC fit keeps the distant diagonal but discards every bottom row.
        # Those bottom rows are still the continuous tape and must determine
        # near-field presence and the local steering tangent.
        points = [
            (51, 168, 27), (61, 174, 35), (71, 180, 43),
            (83, 187, 50), (94, 193, 47), (108, 200, 47),
            (122, 206, 45), (138, 213, 59), (140, 219, 62),
            (140, 226, 65), (140, 232, 58), (140, 239, 54),
        ]
        global_fit, global_inliers = LineDetector._robust_linear_fit(points)
        self.assertIsNotNone(global_fit)
        self.assertFalse(any(y >= 220 for _, y, _ in global_inliers))

        local_fit, local_points = LineDetector._fit_local_direction(points)

        self.assertIsNotNone(local_fit)
        self.assertTrue(any(y >= 220 for _, y, _ in local_points))
        self.assertAlmostEqual(float(np.polyval(local_fit, 239)), 140,
                               delta=6)

    def test_crossing_arms_and_near_geometry(self):
        d = LineDetector()
        for left, right in [(True,True), (True,False), (False,True)]:
            binary = np.zeros((132,320),np.uint8)
            binary[:,156:164] = 255
            binary[88:95,80 if left else 156:240 if right else 164] = 255
            result = d._detect_l_corner(binary, 108)
            self.assertTrue(result['junction_near'])
            self.assertEqual(result['junction_left'], left)
            self.assertEqual(result['junction_right'], right)
            self.assertEqual(result['corner_dir'], 0)

    def test_preview_piecewise_l_detects_arm_above_control_roi(self):
        """A forward-facing camera sees the L arm before it enters tracking ROI."""
        frame = np.full((480, 640, 3), 225, np.uint8)
        cv2.line(frame, (320, 479), (320, 170), (20, 20, 20), 32)
        cv2.line(frame, (320, 170), (560, 170), (20, 20, 20), 32)
        detector = LineDetector(
            roi_top_ratio=.45, crop_bottom_frac=.70,
            crop_top_frac=.90, track_half=60, binary_mode='otsu')

        result = detector.process(frame)

        self.assertTrue(result['is_valid'])
        self.assertEqual(result['corner_dir'], 1)
        # It is only an early preview; the maneuver must not start yet.
        self.assertLess(result['corner_y_ratio'], .52)

    def test_preview_piecewise_l_detects_thick_perspective_corner(self):
        """A thick real-camera L must not lose its arm during skeletonization."""
        detector = LineDetector()
        binary = np.zeros((197, 320), np.uint8)
        cv2.fillPoly(binary, [np.asarray([
            (122, 196), (157, 196), (158, 121), (162, 95),
            (290, 101), (291, 97), (144, 90),
        ], dtype=np.int32)], 255)

        result = detector._detect_piecewise_corner(
            binary, preview_top=43, control_top=108,
            full_height=240, anchor_x=160)

        self.assertEqual(result['corner_dir'], 1)

    def test_preview_reconnects_only_nearby_left_l_arm(self):
        """A glare-sized break in the real L junction must not erase its arm."""
        detector = LineDetector()
        binary = np.zeros((197, 320), np.uint8)
        cv2.fillPoly(binary, [np.asarray([
            (54, 107), (164, 111), (164, 123), (54, 119),
        ], dtype=np.int32)], 255)
        cv2.fillPoly(binary, [np.asarray([
            (187, 117), (180, 129), (171, 196), (206, 196),
            (204, 160), (193, 151), (201, 141), (199, 119),
        ], dtype=np.int32)], 255)

        result = detector._detect_piecewise_corner(
            binary, preview_top=43, control_top=108,
            full_height=240, anchor_x=160)

        self.assertEqual(result['corner_dir'], -1)

    def test_preview_does_not_bridge_distant_or_two_sided_arm(self):
        detector = LineDetector()
        stem = np.zeros((197, 320), np.uint8)
        cv2.fillPoly(stem, [np.asarray([
            (187, 117), (180, 129), (171, 196), (206, 196),
            (204, 160), (193, 151), (201, 141), (199, 119),
        ], dtype=np.int32)], 255)
        for arm_left, arm_right in ((30, 135), (60, 270)):
            binary = stem.copy()
            cv2.rectangle(binary, (arm_left, 107),
                          (arm_right, 123), 255, -1)
            result = detector._detect_piecewise_corner(
                binary, preview_top=43, control_top=108,
                full_height=240, anchor_x=160)
            self.assertEqual(result['corner_dir'], 0)

        # Two separate one-sided arms near the same knee are an ambiguous
        # broken T, not permission to choose the closer branch as an L.
        binary = stem.copy()
        cv2.rectangle(binary, (55, 107), (164, 123), 255, -1)
        cv2.rectangle(binary, (217, 107), (305, 123), 255, -1)
        result = detector._detect_piecewise_corner(
            binary, preview_top=43, control_top=108,
            full_height=240, anchor_x=160)
        self.assertEqual(result['corner_dir'], 0)

    def test_near_split_cannot_move_preview_l_corner_to_vehicle(self):
        """Keep direction and distance from the same corner observation."""
        frame = np.full((480, 640, 3), 225, np.uint8)
        cv2.line(frame, (320, 479), (320, 170), (20, 20, 20), 32)
        cv2.line(frame, (320, 170), (560, 170), (20, 20, 20), 32)
        detector = LineDetector(
            roi_top_ratio=.45, crop_bottom_frac=.70,
            crop_top_frac=.90, track_half=60, binary_mode='otsu')
        detector._detect_split_branches = Mock(return_value=[{
            'direction': 'right', 'target_x': 245.0, 'split_y': 220.0,
            'points': [(160, 220, 20), (190, 190, 15), (245, 160, 12)],
        }])

        result = detector.process(frame)

        self.assertEqual(result['corner_dir'], 1)
        self.assertLess(result['corner_point'][1], result['roi_top'])
        self.assertLess(result['corner_y_ratio'], .52)
        self.assertFalse(result['junction_left'])
        self.assertFalse(result['junction_straight'])
        self.assertFalse(result['junction_right'])
        self.assertEqual(result['branch_candidates'], [])

    def test_preview_piecewise_corner_rejects_smooth_curve(self):
        frame = np.full((480, 640, 3), 225, np.uint8)
        points = np.asarray([
            (320, 479), (319, 430), (315, 380), (305, 330),
            (288, 285), (265, 245), (235, 210), (200, 180),
        ], dtype=np.int32)
        cv2.polylines(frame, [points], False, (20, 20, 20), 32)
        detector = LineDetector(
            roi_top_ratio=.45, crop_bottom_frac=.70,
            crop_top_frac=.90, track_half=60, binary_mode='otsu')

        result = detector.process(frame)

        self.assertTrue(result['is_valid'])
        self.assertEqual(result['corner_dir'], 0)

    def test_preview_piecewise_corner_rejects_t_junction(self):
        frame = np.full((480, 640, 3), 225, np.uint8)
        cv2.line(frame, (320, 479), (320, 170), (20, 20, 20), 32)
        cv2.line(frame, (80, 170), (560, 170), (20, 20, 20), 32)
        detector = LineDetector(
            roi_top_ratio=.45, crop_bottom_frac=.70,
            crop_top_frac=.90, track_half=60, binary_mode='otsu')

        result = detector.process(frame)

        self.assertTrue(result['is_valid'])
        self.assertEqual(result['corner_dir'], 0)

    def test_broad_porous_floor_shadow_is_not_a_track_or_l_turn(self):
        frame = np.full((480, 640, 3), 220, np.uint8)
        roi_top = 288
        cv2.rectangle(frame, (270, 410), (370, 479), (30, 30, 30), -1)
        cv2.line(frame, (320, 420), (240, 330), (30, 30, 30), 24)
        cv2.rectangle(frame, (160, roi_top), (500, 330), (30, 30, 30), -1)
        for x in range(220, 480, 45):
            cv2.rectangle(frame, (x, 302), (x+25, 330),
                          (220, 220, 220), -1)
        detector = LineDetector(
            roi_top_ratio=.6, crop_bottom_frac=.4,
            crop_top_frac=.6, track_half=60, binary_mode='otsu')

        result = detector.process(frame)

        self.assertFalse(result['is_valid'])
        self.assertEqual(result['corner_dir'], 0)

    def test_wide_near_curve_connected_to_distant_cross_line_is_kept(self):
        frame = np.full((480, 640, 3), 225, np.uint8)
        cv2.line(frame, (32, 288), (608, 288), (25, 25, 25), 14)
        centers = np.asarray([
            (520, 288, 70), (500, 330, 80), (450, 380, 100),
            (390, 430, 120), (340, 479, 130),
        ], dtype=np.float64)
        left = np.column_stack((centers[:, 0] - centers[:, 2] / 2,
                                centers[:, 1]))
        right = np.column_stack((centers[:, 0] + centers[:, 2] / 2,
                                 centers[:, 1]))[::-1]
        cv2.fillPoly(frame, [np.rint(np.vstack((left, right))).astype(np.int32)],
                     (25, 25, 25))
        model = {
            'horizontal_fov_deg': 100, 'camera_height_m': .23,
            'pitch_down_deg': 8, 'segmentation_scale': 1.8,
            'min_width_mm': 30, 'max_width_mm': 60,
        }
        detector = LineDetector(
            roi_top_ratio=.6, crop_bottom_frac=.7,
            crop_top_frac=.9, track_half=60, binary_mode='otsu',
            line_width_model=model, enforce_width=False)

        result = detector.process(frame)

        self.assertTrue(result['is_valid'])
        self.assertGreaterEqual(len(result['points']), 3)

    def test_real_width_calibration_accepts_only_30_to_60_mm_tape(self):
        model = {
            'horizontal_fov_deg': 100, 'camera_height_m': .23,
            'pitch_down_deg': 8, 'segmentation_scale': 1.8,
            'min_width_mm': 30, 'max_width_mm': 60,
        }

        def detect(width_factor):
            frame = np.full((480, 640, 3), 225, np.uint8)
            ys = np.arange(320, 480)
            progress = (ys-320)/159.0
            centers = 290+30*progress+8*np.sin(progress*np.pi)
            widths = (14+16*progress)*width_factor
            left = np.column_stack((centers-widths/2, ys))
            right = np.column_stack((centers+widths/2, ys))[::-1]
            polygon = np.rint(np.vstack((left, right))).astype(np.int32)
            cv2.fillPoly(frame, [polygon], (0, 0, 0))
            detector = LineDetector(
                roi_top_ratio=.6, crop_bottom_frac=.5,
                crop_top_frac=.7, track_half=80, binary_mode='otsu',
                line_width_model=model)
            return detector.process(frame)

        too_small = detect(.45)
        nominal = detect(1.0)
        too_large = detect(1.8)

        self.assertFalse(too_small['is_valid'])
        self.assertTrue(nominal['is_valid'])
        self.assertAlmostEqual(nominal['line_width_mm'], 50, delta=3)
        self.assertFalse(too_large['is_valid'])

    def test_relaxed_metric_guard_keeps_solid_tape_but_rejects_extremes(self):
        model = {
            'horizontal_fov_deg': 100, 'camera_height_m': .23,
            'pitch_down_deg': 8, 'segmentation_scale': .55,
            'min_width_mm': 10, 'max_width_mm': 100,
        }

        def detect(raw_pixel_width):
            frame = np.full((480, 640, 3), 225, np.uint8)
            left = 320 - raw_pixel_width // 2
            cv2.rectangle(frame, (left, 288),
                          (left + raw_pixel_width - 1, 479),
                          (0, 0, 0), -1)
            detector = LineDetector(
                roi_top_ratio=.6, crop_bottom_frac=.7,
                crop_top_frac=.9, track_half=80, binary_mode='otsu',
                line_width_model=model, enforce_width=True)
            return detector.process(frame)

        too_thin = detect(4)
        nominal = detect(50)
        too_wide = detect(200)

        self.assertFalse(too_thin['is_valid'])
        self.assertTrue(nominal['is_valid'])
        self.assertGreaterEqual(nominal['line_width_mm'], 10)
        self.assertLessEqual(nominal['line_width_mm'], 100)
        self.assertFalse(too_wide['is_valid'])

    def test_glare_gap_inside_straight_tape_does_not_create_curve(self):
        frame = np.full((480, 640, 3), 225, np.uint8)
        tape = np.asarray([
            (298, 288), (342, 288), (360, 479), (280, 479),
        ], dtype=np.int32)
        cv2.fillPoly(frame, [tape], (10, 10, 10))
        # A bright floor reflection removes the middle/right part of the
        # otherwise straight ribbon for several consecutive scan rows.
        cv2.rectangle(frame, (310, 350), (340, 420), (245, 245, 245), -1)
        model = {
            'horizontal_fov_deg': 100, 'camera_height_m': .23,
            'pitch_down_deg': 8, 'segmentation_scale': .55,
            'min_width_mm': 10, 'max_width_mm': 100,
        }
        detector = LineDetector(
            roi_top_ratio=.6, crop_bottom_frac=.7,
            crop_top_frac=.9, track_half=60, binary_mode='otsu',
            line_width_model=model, enforce_width=True)

        result = detector.process(frame)

        self.assertTrue(result['is_valid'])
        self.assertGreaterEqual(len(result['points']), 10)
        self.assertLess(abs(result['error_px']), 5)
        self.assertLess(abs(result['angle_deg']), 5)

    def test_bounded_hysteresis_restores_attached_reflective_tape_only(self):
        gray = np.full((60, 100), 220, np.uint8)
        gray[:, 40:60] = 20
        # Specular glare lifts one side of the tape above the strict threshold.
        gray[20:40, 50:60] = 125
        # A similarly grey but disconnected floor mark must not be introduced.
        gray[20:40, 82:92] = 125
        # Nor may a weak scratch touching the outside of the tape grow merely
        # because it is connected to a strict tape seed.
        gray[5:15, 35:40] = 125
        inside = np.ones_like(gray, dtype=bool)
        detector = LineDetector(polarity='black')
        strict = detector._apply_global_threshold(gray, inside, 100)

        repaired = detector._recover_reflective_tape(
            gray, strict, inside, threshold=100)

        self.assertTrue(np.all(repaired[22:38, 52:58] == 255))
        self.assertTrue(np.all(repaired[22:38, 84:90] == 0))
        self.assertTrue(np.all(repaired[7:13, 36:39] == 0))

    def test_reflective_tape_end_is_extended_only_toward_near_field(self):
        gray = np.full((60, 100), 220, np.uint8)
        gray[:42, 40:60] = 20
        # Reflection erases the final near-field portion of an otherwise
        # vertical stem, while its grey pixels remain connected below it.
        gray[42:, 40:60] = 205
        inside = np.ones_like(gray, dtype=bool)
        detector = LineDetector(polarity='black')
        strict = detector._apply_global_threshold(gray, inside, 100)

        repaired = detector._recover_reflective_tape(
            gray, strict, inside, threshold=100)

        self.assertTrue(np.all(repaired[48:58, 43:57] == 255))
        self.assertTrue(np.all(repaired[48:58, :35] == 0))
        self.assertTrue(np.all(repaired[48:58, 65:] == 0))

    def test_branch_candidates_keep_relative_direction_ids(self):
        corner = {'corner_point': (160, 150), 'corner_span': 180,
                  'junction_left': True, 'junction_straight': True,
                  'junction_right': True}
        candidates = LineDetector._branch_candidates(corner, 320)
        self.assertEqual([item['direction'] for item in candidates],
                         ['left', 'straight', 'right'])
        self.assertLess(candidates[0]['target_x'], candidates[1]['target_x'])
        self.assertLess(candidates[1]['target_x'], candidates[2]['target_x'])

    def test_y_fork_produces_simultaneous_straight_and_right_paths(self):
        frame = np.full((480, 640, 3), 225, np.uint8)
        cv2.line(frame, (320, 479), (320, 300), (0, 0, 0), 36)
        cv2.line(frame, (320, 300), (320, 215), (0, 0, 0), 24)
        cv2.line(frame, (320, 300), (540, 220), (0, 0, 0), 24)
        kwargs = dict(roi_top_ratio=.45, crop_bottom_frac=.7,
                      crop_top_frac=.9, track_half=90,
                      scan_start_ratio=.1, binary_mode='otsu',
                      n_scan_rows=20)
        straight = LineDetector(**kwargs)
        straight.path_preference = 'continuation'
        result = straight.process(frame)
        self.assertEqual([item['direction']
                          for item in result['branch_candidates']],
                         ['straight', 'right'])
        right = LineDetector(**kwargs)
        right.path_preference = 'right'
        selected = right.process(frame)
        self.assertTrue(selected['is_valid'])
        self.assertEqual([item['direction']
                          for item in selected['branch_candidates']],
                         ['straight', 'right'])
        self.assertNotEqual(selected['fit_coeffs'], result['fit_coeffs'])
        self.assertGreater(selected['error_px'], 50)

    def test_locked_right_path_hands_off_to_forward_after_vehicle_turns(self):
        kwargs = dict(roi_top_ratio=.45, crop_bottom_frac=.7,
                      crop_top_frac=.9, track_half=90,
                      scan_start_ratio=.1, binary_mode='otsu',
                      n_scan_rows=20)
        detector = LineDetector(**kwargs)
        initial = np.full((480, 640, 3), 225, np.uint8)
        cv2.line(initial, (320, 479), (320, 300), (0, 0, 0), 36)
        cv2.line(initial, (320, 300), (320, 215), (0, 0, 0), 24)
        cv2.line(initial, (320, 300), (540, 220), (0, 0, 0), 24)
        detector.path_preference = 'right'
        selected = detector.process(initial)
        self.assertEqual(selected['selected_branch_direction'], 'right')

        # After entering the right branch, that same physical road is forward
        # in the camera and the old incoming road appears on the left.
        aligned = np.full((480, 640, 3), 225, np.uint8)
        cv2.line(aligned, (320, 479), (320, 300), (0, 0, 0), 36)
        cv2.line(aligned, (320, 300), (320, 215), (0, 0, 0), 24)
        cv2.line(aligned, (320, 300), (100, 220), (0, 0, 0), 24)
        handed_off = detector.process(aligned)
        self.assertIsNone(handed_off['selected_branch_direction'])
        self.assertTrue(handed_off['is_valid'])

    def test_selected_side_branch_raises_steering_without_slow_ramp(self):
        runner = self.runner()
        runner.f.kp = 12.0
        runner.f.kd = 1.5
        runner.f.ka = 3.5
        runner.f.err_alpha = .6
        runner.f.z_rate_limit = 120
        runner.f.z_invert = True
        runner.policy.state = 'memorized'
        runner.policy.task = 'right'
        command = runner._tracking({
            'is_valid': True, 'error_px': 70, 'angle_deg': 60,
            'selected_branch_direction': 'right'}, .1)
        self.assertEqual(abs(command[2]), 800)

    def test_tracking_slows_on_early_curve_before_large_lateral_error(self):
        runner = self.runner()
        runner.f.kp = 12.0
        runner.f.kd = 0.0
        runner.f.ka = 3.5
        runner.f.err_alpha = 1.0
        runner.f.z_rate_limit = 800
        runner.f.z_invert = True
        runner.policy.state = 'driving'
        command = runner._tracking({
            'is_valid': True, 'error_px': 15, 'angle_deg': 10,
            'selected_branch_direction': None}, .1)
        self.assertLessEqual(command[0], 225)
        self.assertGreater(command[0], 0)

    def test_tracking_targets_are_delayed_by_odometry_not_frames(self):
        follower = SimpleNamespace(
            base_speed=300, max_z=800, kp=12.0, kd=0.0, ka=0.0,
            err_alpha=1.0, z_rate_limit=800, z_invert=True)
        runner = TrafficControlRunner(follower, control_delay_m=.10)
        det = {'is_valid': True, 'error_px': 20, 'angle_deg': 0,
               'selected_branch_direction': None}
        self.assertEqual(runner._tracking(det, .05, 0.00, 'main')[2], 0)
        self.assertEqual(runner._tracking(det, .05, 0.05, 'main')[2], 0)
        self.assertNotEqual(runner._tracking(det, .05, 0.10, 'main')[2], 0)

    def test_route_change_discards_queued_straight_targets(self):
        follower = SimpleNamespace(
            base_speed=300, max_z=800, kp=12.0, kd=0.0, ka=0.0,
            err_alpha=1.0, z_rate_limit=800, z_invert=True)
        runner = TrafficControlRunner(follower, control_delay_m=.10)
        straight = {'is_valid': True, 'error_px': -30, 'angle_deg': 0,
                    'selected_branch_direction': None}
        right = {'is_valid': True, 'error_px': 60, 'angle_deg': 40,
                 'selected_branch_direction': 'right'}
        runner._tracking(straight, .05, 0.00, 'main')
        self.assertEqual(runner._tracking(right, .05, 0.05,
                                          'branch:right')[2], 0)
        command = runner._tracking(right, .05, 0.15, 'branch:right')
        self.assertLess(command[2], 0)

    def test_lost_line_clears_spatial_control_queue(self):
        follower = SimpleNamespace(
            base_speed=300, max_z=800, kp=12.0, kd=0.0, ka=0.0,
            err_alpha=1.0, z_rate_limit=800, z_invert=True)
        runner = TrafficControlRunner(follower, control_delay_m=.10)
        valid = {'is_valid': True, 'error_px': 30, 'angle_deg': 0,
                 'selected_branch_direction': None}
        runner._tracking(valid, .05, 0.00, 'main')
        runner._tracking({'is_valid': False}, .05, 0.05, 'main')
        self.assertEqual(runner._tracking(valid, .05, 0.11, 'main')[2], 0)

    def test_line_following_states_trace_from_near_connected_line(self):
        runner = self.runner()
        for state in ('driving', 'roadblock-memorized',
                      'people-crossing', 'cross-follow', 'junction-wait',
                      ):
            runner.policy.state = state
            self.assertEqual(runner._detector_path_preference(), 'continuation')
        runner.policy.state = 'memorized'
        runner.policy.task = 'right'
        self.assertEqual(runner._detector_path_preference(), 'continuation')
        runner.policy.task = 'left'
        self.assertEqual(runner._detector_path_preference(), 'continuation')
        runner.policy.locked_branch = 'right'
        self.assertEqual(runner._detector_path_preference(), 'right')
        runner.policy.locked_branch = 'left'
        self.assertEqual(runner._detector_path_preference(), 'left')
        runner.policy.locked_branch = None
        runner.policy.state = 'red-hold'
        self.assertIsNone(runner._detector_path_preference())
    def test_ring_branch_preference_only_when_opted_in(self):
        d = LineDetector(scan_start_ratio=.25)
        binary = np.zeros((132,320),np.uint8)
        binary[:,120:130] = 255
        binary[:,175:190] = 255
        inside = np.ones_like(binary,dtype=bool)
        standard = d._scan_lines(binary,108,320,inside,None)
        self.assertTrue(all(p[0]>170 for p in standard))
        d.path_preference = 'continuation'
        ring = d._scan_lines(binary,108,320,inside,125)
        self.assertTrue(any(p[0]<140 for p in ring))
        # The left/inner diameter must not attract a right-hand main stem.
        ring = d._scan_lines(binary,108,320,inside,182)
        self.assertTrue(all(p[0]>170 for p in ring))

    def test_web_heartbeat_and_behavior_status(self):
        web = DebugWebServer('127.0.0.1',0, emergency_callback=Mock())
        web.start()
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            self.assertGreater(web.heartbeat_age(),3)
            with opener.open(urllib.request.Request(web.url+'/api/heartbeat',method='POST'), timeout=2) as response:
                self.assertTrue(json.load(response)['ok'])
            self.assertLess(web.heartbeat_age(),1)
            web.update(None, {}, {'traffic_control': True, 'behavior': {'exit_count': 2},
                                  'lateral_speed':100})
            with opener.open(web.url+'/api/status',timeout=2) as response:
                status = json.load(response)
            self.assertTrue(status['traffic']['control_enabled'])
            self.assertEqual(status['behavior']['exit_count'],2)
            with opener.open(urllib.request.Request(web.url+'/api/emergency-stop',method='POST'),timeout=2):
                pass
            self.assertEqual(web.get_status()['lateral_speed'],0)
        finally:
            web.stop()

    def test_stale_web_form_preserves_new_server_config_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'web_config.json')
            config = {name: bounds[1] for name, bounds in CONFIG_SCHEMA.items()}
            config.update(binary_mode='otsu', polarity='black',
                          cross_lateral_distance_m=.35,
                          cross_lateral_speed=90,
                          control_delay_m=.10)
            web = DebugWebServer(config=config, config_path=path)
            stale = dict(config)
            stale.pop('cross_lateral_distance_m')
            stale.pop('cross_lateral_speed')
            saved = web.save_config(stale)
            self.assertEqual(saved['cross_lateral_distance_m'], .35)
            self.assertEqual(saved['cross_lateral_speed'], 90)
            self.assertEqual(saved['control_delay_m'], .10)

    def test_web_chassis_arm_requires_explicit_confirmation(self):
        callback = Mock(return_value={
            'chassis_armed': True, 'chassis_target_speed': 100,
            'chassis_state': 'ready'})
        web = DebugWebServer('127.0.0.1', 0,
                             emergency_callback=Mock(),
                             chassis_arm_callback=callback)
        web.start()
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            def request(body):
                return urllib.request.Request(
                    web.url + '/api/chassis/arm', method='POST',
                    headers={'Content-Type': 'application/json'},
                    data=json.dumps(body).encode())
            with self.assertRaises(urllib.error.HTTPError) as error:
                opener.open(request({'armed': True, 'speed': 100}), timeout=2)
            self.assertEqual(error.exception.code, 400)
            with opener.open(request({
                    'armed': True, 'speed': 100,
                    'confirmation': 'ENABLE_CHASSIS'}), timeout=2) as response:
                self.assertTrue(json.load(response)['chassis_armed'])
            callback.assert_called_once_with(True, 100)
        finally:
            web.stop()


if __name__ == '__main__':
    unittest.main()
