"""Regression checks for floor shadows being accepted as black tape."""
import unittest

import cv2
import numpy as np

from core.line_follower import LineDetector


class OtsuContrastTests(unittest.TestCase):
    @staticmethod
    def detector(**kwargs):
        return LineDetector(enforce_width=False, line_width_model={
            'horizontal_fov_deg': 100., 'camera_height_m': .23,
            'pitch_down_deg': 8., 'segmentation_scale': .55,
            'min_width_mm': 10., 'max_width_mm': 100.}, **kwargs)

    def test_floor_gradient_is_invalid(self):
        gray = np.tile(np.linspace(90, 150, 640, dtype=np.uint8), (480, 1))
        result = self.detector().process(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        self.assertFalse(result['is_valid'])

    def test_tape_with_reflection_is_valid(self):
        frame = np.full((480, 640, 3), 170, np.uint8)
        cv2.line(frame, (320, 200), (320, 479), (35, 35, 35), 38)
        cv2.rectangle(frame, (301, 340), (315, 365), (175, 175, 175), -1)
        result = self.detector().process(frame)
        self.assertTrue(result['is_valid'])
        self.assertGreaterEqual(result['contrast_support_rows'], result['contrast_required_rows'])

    @staticmethod
    def one_sided_candidate():
        work = np.full((100, 100, 3), 140, np.uint8)
        work[:, :50] = 100
        binary = np.zeros((100, 100), np.uint8)
        binary[:, 20:50] = 255
        result = dict(is_valid=True, binary=binary, roi_top=0,
                      points=[(35, y, 30) for y in (20, 40, 60, 80)], corner_dir=1)
        return work, result

    def test_one_sided_candidate_clears_tracking_and_corner(self):
        detector = self.detector()
        detector._prev_cx = 35
        work, candidate = self.one_sided_candidate()
        result = detector._check_otsu_contrast(work, candidate)
        self.assertFalse(result['is_valid'])
        self.assertEqual(result['corner_dir'], 0)
        self.assertEqual(result['contrast_support_rows'], 0)
        self.assertIsNone(detector._prev_cx)
        self.assertIs(result['binary'], candidate['binary'])

    def test_fixed_and_white_modes_are_unchanged(self):
        for options in ({'binary_mode': 'fixed'}, {'polarity': 'white'}):
            work, result = self.one_sided_candidate()
            self.assertIs(self.detector(**options)._check_otsu_contrast(work, result), result)

    def test_unseen_side_does_not_count(self):
        work = np.full((100, 100, 3), 140, np.uint8)
        work[:, :30] = 30
        binary = np.zeros((100, 100), np.uint8)
        binary[:, :30] = 255
        result = dict(binary=binary, roi_top=0, points=[(15, 50, 30)])
        self.assertEqual(LineDetector._measure_side_contrast(work, result), [])


if __name__ == '__main__':
    unittest.main()
