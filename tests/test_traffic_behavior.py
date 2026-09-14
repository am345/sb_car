import unittest

from core.traffic_behavior import SignEvents, TrafficBehavior


class Scene:
    def __init__(self, ceiling=300, cross_distance=.25, follow_line=True,
                 max_z=800):
        self.p = TrafficBehavior(ceiling, max_z=max_z, follow_line=follow_line,
                                 cross_lateral_distance_m=cross_distance)
        self.now = 10.0
        self.seq = 0
        self.pose = {'odom_yaw_total_deg': 0., 'odom_distance_m': 0.}
        self.det = {'is_valid': True, 'error_px': 0, 'angle_deg': 0,
                    'junction_near': False, 'branch_candidates': [],
                    'line_end_candidate': False, 'crosswalk_active': False}
        self.measured = {'real_x': 0, 'real_y': 0, 'real_z': 0}
        self.command = (0, 0, 0)

    def tick(self, labels=(), dt=.05, fresh=True, tracking=(300, 0, 40)):
        self.now += dt
        self.seq += 1
        result = {'state': 'ready', 'sequence': self.seq,
                  'age_sec': 0 if fresh else 2,
                  'detections': [{'label': label, 'confidence': .9}
                                 for label in labels]}
        self.command = self.p.step(self.now, result, self.det, self.pose,
                                   self.measured, tracking)
        return self.command

    def confirm(self, label):
        for _ in range(3):
            self.tick([label])

    def junction(self, *directions):
        self.det['junction_near'] = True
        self.det['branch_candidates'] = [
            {'direction': direction, 'target_x': 160 + index * 40}
            for index, direction in enumerate(directions)]


class BehaviorTests(unittest.TestCase):
    def test_confirmation_requires_new_frames(self):
        e = SignEvents()
        result = {'state': 'ready', 'age_sec': 0, 'sequence': 1,
                  'detections': [{'label': 'left', 'confidence': .9}]}
        self.assertEqual(e.update(result, 1), [])
        self.assertEqual(e.update(result, 1.1), [])
        result['sequence'] = 2
        self.assertEqual(e.update(result, 1.2), [])
        result['sequence'] = 3
        self.assertEqual(e.update(result, 1.3), ['left'])

    def test_speed_policy_supports_400_and_red40_reduces_to_200(self):
        s = Scene(400)
        for _ in range(30): s.tick(tracking=(400, 0, 40))
        self.assertEqual(s.p.cruise, 400)
        self.assertEqual(s.command[0], 400)
        s.confirm('red40')
        self.assertEqual(s.p.cruise, 200)
        self.assertEqual(s.command[0], 200)
        s.confirm('black40')
        for _ in range(15): s.tick(tracking=(400, 0, 40))
        self.assertEqual(s.p.cruise, 400)
        self.assertEqual(s.command[0], 400)

    def test_red_never_releases_on_disappearance_only_green(self):
        s = Scene()
        for _ in range(25): s.tick()
        s.confirm('red')
        for _ in range(30): s.tick()
        self.assertEqual(s.p.state, 'red-hold')
        for _ in range(20): s.tick()
        self.assertEqual(s.p.state, 'red-hold')
        s.tick(['green']); s.tick(['green'])
        self.assertEqual(s.p.state, 'red-hold')
        s.tick(['green'])
        self.assertEqual(s.p.state, 'driving')

    def test_unassigned_junction_keeps_normal_tracking_without_branch_lock(self):
        s = Scene()
        s.junction('left', 'straight', 'right')
        for _ in range(3): s.tick()
        self.assertEqual(s.p.state, 'driving')
        self.assertIsNone(s.p.locked_branch)
        self.assertGreater(s.command[0], 0)
        self.assertFalse(s.p.fault)

    def test_unassigned_t_junction_does_not_start_branch_matching(self):
        s = Scene()
        s.junction('left', 'right')
        for _ in range(3): s.tick()
        self.assertEqual(s.p.state, 'driving')
        self.assertIsNone(s.p.locked_branch)
        self.assertFalse(s.p.fault)

    def test_preconfirmed_left_selects_requested_branch(self):
        s = Scene()
        s.confirm('left')
        s.junction('left', 'right')
        for _ in range(3): s.tick()
        self.assertEqual(s.p.state, 'branch-follow')
        self.assertEqual(s.p.locked_branch, 'left')

    def test_right_selects_and_follows_detected_branch(self):
        s = Scene()
        s.confirm('right')
        self.assertEqual(s.p.state, 'memorized')
        s.junction('straight', 'right')
        for _ in range(3):
            s.tick(tracking=(120, 0, 90))
        self.assertEqual(s.p.state, 'branch-follow')
        self.assertEqual(s.p.locked_branch, 'right')
        self.assertEqual(s.command[0], 300)
        self.assertFalse(s.p.fault)
        s.det['junction_near'] = False
        s.det['branch_candidates'] = []
        s.pose['odom_distance_m'] = .31
        for _ in range(22): s.tick()
        self.assertEqual(s.p.state, 'driving')

    def test_confirmed_right_locks_on_first_near_frame(self):
        s = Scene()
        s.confirm('right')
        s.junction('straight', 'right')
        s.tick(tracking=(300, 0, 180))
        self.assertEqual(s.p.state, 'branch-follow')
        self.assertEqual(s.p.locked_branch, 'right')
        self.assertEqual(s.command, (300, 0, 180))

    def test_right_branch_ignores_short_clear_gap_before_releasing(self):
        s = Scene()
        s.confirm('right')
        s.junction('straight', 'right')
        for _ in range(3):
            s.tick()
        self.assertEqual(s.p.state, 'branch-follow')
        s.det['junction_near'] = False
        s.det['branch_candidates'] = []
        for _ in range(10):
            s.tick()
        self.assertEqual(s.p.state, 'branch-follow')
        s.pose['odom_distance_m'] = .16
        for _ in range(10):
            s.tick()
        self.assertEqual(s.p.state, 'branch-follow')
        s.junction('straight', 'right')
        for _ in range(3):
            s.tick()
        s.det['junction_near'] = False
        s.det['branch_candidates'] = []
        s.pose['odom_distance_m'] = .31
        for _ in range(22):
            s.tick()
        self.assertEqual(s.p.state, 'driving')

    def test_branch_progress_prevents_fixed_timeout_while_fork_remains(self):
        s = Scene()
        s.confirm('right')
        s.junction('straight', 'right')
        for _ in range(3):
            s.tick()
        for _ in range(200):
            s.pose['odom_distance_m'] += .005
            s.tick()
        self.assertEqual(s.p.state, 'branch-follow')
        self.assertFalse(s.p.fault)

    def test_branch_stops_when_commanded_motion_has_no_progress(self):
        s = Scene()
        s.confirm('right')
        s.junction('straight', 'right')
        for _ in range(3):
            s.tick()
        for _ in range(45):
            s.tick()
        self.assertIn('里程无进展', s.p.fault)

    def test_requested_branch_absent_stops(self):
        s = Scene()
        s.confirm('left')
        s.junction('straight', 'right')
        for _ in range(3): s.tick()
        self.assertIn('目标支路不存在', s.p.fault)

    def test_branch_blind_window_is_half_second(self):
        s = Scene()
        s.confirm('left')
        s.junction('left', 'right')
        for _ in range(3): s.tick()
        s.det['is_valid'] = False
        for _ in range(9):
            self.assertFalse(s.p.fault)
            s.tick(dt=.05)
        s.tick(dt=.1)
        self.assertTrue(s.p.fault)

    def test_people_restores_only_after_region_seen_then_cleared(self):
        s = Scene()
        s.confirm('red40')
        s.confirm('people')
        self.assertEqual(s.p.state, 'people-crossing')
        self.assertEqual(s.p.cruise, 200)
        s.det['crosswalk_active'] = True
        s.tick()
        s.det['crosswalk_active'] = False
        s.tick(); s.tick()
        self.assertEqual(s.p.state, 'people-crossing')
        s.tick()
        self.assertEqual(s.p.state, 'driving')
        self.assertEqual(s.p.cruise, 200)

    def test_cross_uses_line_end_odometry_lateral_distance_and_reacquire(self):
        s = Scene(cross_distance=.25)
        s.confirm('cross')
        self.assertEqual(s.p.state, 'cross-follow')
        s.det['line_end_candidate'] = True
        for distance in (.00, .01, .03):
            s.pose['odom_distance_m'] = distance
            s.tick()
        self.assertEqual(s.p.state, 'cross-lateral')
        self.assertEqual(s.tick(), (0, 100, 0))
        s.pose['odom_distance_m'] += .25
        self.assertEqual(s.tick(), (0, 0, 0))
        self.assertEqual(s.p.state, 'cross-reacquire')
        for _ in range(4): s.tick()
        self.assertEqual(s.p.state, 'driving')

    def test_cross_without_configured_distance_stops(self):
        s = Scene(cross_distance=0)
        s.confirm('cross')
        s.det['line_end_candidate'] = True
        for distance in (0, .01, .03):
            s.pose['odom_distance_m'] = distance
            s.tick()
        self.assertIn('未配置', s.p.fault)

    def test_round_and_turn_are_explicitly_unsupported(self):
        for label in ('round', 'turn'):
            with self.subTest(label=label):
                s = Scene()
                s.confirm(label)
                self.assertTrue(s.p.fault)
                self.assertEqual(s.command, (0, 0, 0))


if __name__ == '__main__':
    unittest.main()
