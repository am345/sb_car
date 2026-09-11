import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from core.traffic_control import TrafficControlRunner
from core.traffic_signs import TrafficSignWorker
from test_traffic_behavior import Scene


class RoadblockTests(unittest.TestCase):
    def test_decoder_accepts_eleven_classes_and_preserves_ten(self):
        for classes in (10, 11):
            names = {i: ('roadblock' if i == 10 else str(i))
                     for i in range(classes)}
            output = np.zeros((1, 4 + classes, 8400), np.float32)
            output[0, :4, 0] = [320, 320, 80, 80]
            output[0, 4 + classes - 1, 0] = .9
            result = TrafficSignWorker.decode(
                output, 1, 0, 0, (640, 640, 3), names)
            self.assertEqual(result[0]['label'], names[classes - 1])

    def test_decoder_rejects_metadata_mismatch(self):
        with self.assertRaises(ValueError):
            TrafficSignWorker.decode(
                np.zeros((1, 15, 8400), np.float32), 1, 0, 0,
                (640, 640, 3), {i: str(i) for i in range(10)})

    def test_box_is_associated_with_nearest_branch(self):
        follower = SimpleNamespace(
            base_speed=300, max_z=800,
            detector=SimpleNamespace(work_width=320))
        runner = TrafficControlRunner(follower)
        det = {'branch_candidates': [
            {'direction': 'left', 'target_x': 80},
            {'direction': 'right', 'target_x': 240},
        ]}
        result = {'frame_width': 640, 'detections': [
            {'label': 'roadblock', 'confidence': .9, 'box': [100, 10, 180, 100]}
        ]}
        runner._associate_roadblock(result, det)
        self.assertEqual(det['blocked_branch'], 'left')

    def test_ambiguous_box_does_not_guess_branch(self):
        follower = SimpleNamespace(
            base_speed=300, max_z=800,
            detector=SimpleNamespace(work_width=320))
        runner = TrafficControlRunner(follower)
        det = {'branch_candidates': [
            {'direction': 'left', 'target_x': 120},
            {'direction': 'right', 'target_x': 200},
        ]}
        result = {'frame_width': 640, 'detections': [
            {'label': 'roadblock', 'confidence': .9, 'box': [300, 10, 340, 100]}
        ]}
        runner._associate_roadblock(result, det)
        self.assertIsNone(det['blocked_branch'])

    def test_ground_point_is_matched_against_entire_branch_polyline(self):
        follower = SimpleNamespace(
            base_speed=300, max_z=800,
            detector=SimpleNamespace(work_width=320))
        runner = TrafficControlRunner(follower)
        det = {'branch_candidates': [
            {'direction': 'straight', 'target_x': 80, 'target_y': 20,
             'points': [(160, 160), (170, 120), (80, 20)]},
            {'direction': 'right', 'target_x': 280, 'target_y': 40,
             'points': [(210, 160), (240, 100), (280, 40)]},
        ]}
        result = {'frame_width': 640, 'detections': [
            {'label': 'roadblock', 'confidence': .91,
             'box': [251, 2, 424.65, 291.28]},
            # A low-confidence second box must not block the right branch.
            {'label': 'roadblock', 'confidence': .28,
             'box': [557.61, 40.51, 640, 259.07]},
        ]}
        runner._associate_roadblock(result, det)
        self.assertEqual(det['blocked_branches'], ['straight'])
        self.assertEqual(det['blocked_branch'], 'straight')
        self.assertEqual(len(det['roadblock_associations']), 1)

    def test_blocked_branch_is_excluded_and_other_branch_locked(self):
        s = Scene()
        s.junction('left', 'right')
        s.det['blocked_branch'] = 'right'
        s.confirm('roadblock')
        for _ in range(3): s.tick(['roadblock'])
        self.assertEqual(s.p.state, 'branch-follow')
        self.assertEqual(s.p.locked_branch, 'left')
        self.assertFalse(s.p.fault)

    def test_missing_association_fails_stopped_at_fork(self):
        s = Scene()
        s.confirm('roadblock')
        s.junction('left', 'right')
        for _ in range(3): s.tick()
        self.assertIn('无法确定路障', s.p.fault)
        self.assertEqual(s.command, (0, 0, 0))

    def test_multiple_blocked_branches_stop_instead_of_guessing(self):
        s = Scene()
        s.junction('straight', 'right')
        s.det['blocked_branches'] = ['straight', 'right']
        s.confirm('roadblock')
        for _ in range(3):
            s.tick(['roadblock'])
        self.assertIn('目标支路不唯一', s.p.fault)
        self.assertEqual(s.command, (0, 0, 0))


if __name__ == '__main__':
    unittest.main()
