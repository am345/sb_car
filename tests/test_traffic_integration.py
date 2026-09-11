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

    def test_generic_tracker_does_not_replace_curve_with_corner_state(self):
        camera, chassis = Mock(), Mock()
        camera.read.return_value = np.zeros((480, 640, 3), np.uint8)
        chassis.send_speed.return_value = True
        chassis.read_status.return_value = {
            'real_x': 0, 'real_y': 0, 'real_z': 0, 'ang_vel_z': 0}
        follower = LineFollower(
            camera, chassis, base_speed=300, startup_frames=1,
            ramp_frames=0, err_alpha=1.0, z_rate_limit=800)
        follower.frame_interval = 0
        follower.detector = Mock(
            work_width=320, crop_top_frac=.6, crop_bottom_frac=.5,
            binary_mode='otsu')
        follower.detector.process.return_value = {
            'is_valid': True, 'error_px': 25, 'angle_deg': 20,
            'line_type': 'black',
            # A tight continuous curve can resemble a one-sided L in image
            # space. Generic tracking must keep steering instead of entering
            # the old advance-then-rotate terrain state machine.
            'corner_dir': 1, 'corner_y_ratio': .7, 'corner_span': 100,
        }

        with patch('core.line_follower.cv2.destroyAllWindows'):
            follower.run(max_frames=3)

        moving = [call.args for call in chassis.send_speed.call_args_list
                  if call.args[0] > 0]
        self.assertTrue(moving)
        self.assertTrue(any(abs(command[2]) > 0 for command in moving))

    def test_generic_tracker_slows_for_preview_curvature(self):
        camera, chassis = Mock(), Mock()
        camera.read.return_value = np.zeros((480, 640, 3), np.uint8)
        chassis.send_speed.return_value = True
        chassis.read_status.return_value = {
            'real_x': 0, 'real_y': 0, 'real_z': 0, 'ang_vel_z': 0}
        follower = LineFollower(
            camera, chassis, base_speed=300, startup_frames=1,
            ramp_frames=0, err_alpha=1.0, z_rate_limit=800)
        follower.frame_interval = 0
        follower.detector = Mock(
            work_width=320, crop_top_frac=.6, crop_bottom_frac=.5,
            binary_mode='otsu')
        follower.detector.process.return_value = {
            'is_valid': True, 'error_px': 0, 'angle_deg': 0,
            'path_curvature': .02, 'line_type': 'black',
        }

        with patch('core.line_follower.cv2.destroyAllWindows'):
            follower.run(max_frames=3)

        moving = [call.args for call in chassis.send_speed.call_args_list
                  if call.args[0] > 0]
        self.assertTrue(moving)
        self.assertLess(max(command[0] for command in moving), 200)
        self.assertTrue(any(abs(command[2]) > 0 for command in moving))

    def test_centerline_trace_preserves_a_sharp_bend_as_one_path(self):
        binary = np.zeros((132, 320), np.uint8)
        cv2.line(binary, (160, 131), (160, 65), 255, 12)
        cv2.line(binary, (160, 65), (260, 65), 255, 12)

        path = LineDetector._trace_centerline(binary)

        self.assertGreater(len(path), 20)
        self.assertGreater(path[-1][0], 240)
        self.assertAlmostEqual(path[-1][1], 65, delta=5)
        self.assertGreater(
            LineDetector._lookahead_index(path, 100),
            LineDetector._lookahead_index(path, 40))

    def test_centerline_trace_rejects_line_disconnected_from_vehicle(self):
        binary = np.zeros((132, 320), np.uint8)
        cv2.line(binary, (30, 25), (290, 25), 255, 12)
        self.assertEqual(LineDetector._trace_centerline(binary), [])

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
