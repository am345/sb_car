import math
import unittest

from core.corner_maneuver import CornerManeuver


class CornerManeuverTests(unittest.TestCase):
    def test_requested_default_maneuver_parameters(self):
        maneuver = CornerManeuver()
        self.assertEqual(maneuver.advance_speed, 150)
        self.assertAlmostEqual(maneuver.advance_distance_m, 0.20)
        self.assertAlmostEqual(math.degrees(maneuver.turn_target_radians), 80.0)
        self.assertEqual(maneuver.turn_speed, 800)
        self.assertEqual(maneuver.confirm_frames, 2)
        self.assertAlmostEqual(maneuver.confirm_speed_ratio, 0.5)
        self.assertAlmostEqual(maneuver.turn_gate_y_ratio, 0.88)
        self.assertEqual(maneuver.reacquire_frames, 3)

    def test_corner_slows_during_confirmation_then_starts_maneuver(self):
        maneuver = CornerManeuver()

        for count in (1,):
            result = maneuver.step(
                self.corner(1), 0.05, enabled=True, base_speed=200,
                max_z=800, z_invert=True)
            self.assertIsNone(result)
            self.assertTrue(maneuver.confirming)
            self.assertEqual(maneuver.confirm_count, count)
            self.assertEqual(maneuver.confirmation_speed(200), 100)

        result = maneuver.step(
            self.corner(1), 0.05, enabled=True, base_speed=200,
            max_z=800, z_invert=True)
        self.assertEqual(result.state, 'corner-approach-right')
        self.assertIsNone(result.command)
        self.assertFalse(maneuver.confirming)
        self.assertTrue(maneuver.active)

    def test_confirmed_distant_corner_is_latched_until_blind_zone(self):
        """Confirmation must not be lost just because the corner starts far away."""
        maneuver = CornerManeuver(confirm_frames=3,
                                  advance_distance_m=0.25)
        distant = self.corner(1)
        distant['corner_y_ratio'] = 0.25

        for _ in range(3):
            result = maneuver.step(
                distant, 0.05, enabled=True, base_speed=300,
                max_z=800, z_invert=True, odom_distance_m=1.0)

        self.assertTrue(maneuver.active)
        self.assertEqual(result.state, 'corner-approach-right')
        blind = maneuver.step(
            {'is_valid': False, 'corner_dir': 0}, 0.05, enabled=True,
            base_speed=300, max_z=800, z_invert=True,
            odom_distance_m=1.10)
        self.assertEqual(blind.state, 'corner-delay-right')
        self.assertEqual(blind.command, (150, 0, 0))

    def test_corner_confirmation_resets_when_candidate_disappears(self):
        maneuver = CornerManeuver(confirm_frames=3)
        maneuver.step(self.corner(1), 0.05, enabled=True,
                      base_speed=200, max_z=800, z_invert=True)
        maneuver.step({'is_valid': True, 'corner_dir': 0}, 0.05,
                      enabled=True, base_speed=200, max_z=800,
                      z_invert=True)
        self.assertFalse(maneuver.confirming)
        self.assertEqual(maneuver.confirm_count, 0)

    def test_one_missing_frame_decays_but_keeps_corner_evidence(self):
        maneuver = CornerManeuver(confirm_frames=3)
        for _ in range(2):
            maneuver.step(self.corner(1), 0.05, enabled=True,
                          base_speed=200, max_z=800, z_invert=True)
        maneuver.step({'is_valid': True, 'corner_dir': 0}, 0.05,
                      enabled=True, base_speed=200, max_z=800,
                      z_invert=True)
        self.assertTrue(maneuver.confirming)
        self.assertEqual(maneuver.confirm_direction, 1)
        self.assertEqual(maneuver.confirm_count, 1)

    def test_bottom_virtual_gate_starts_turn_before_distance_limit(self):
        maneuver = CornerManeuver(confirm_frames=1,
                                  advance_distance_m=0.25)
        corner = self.corner(1)
        corner['corner_y_ratio'] = 0.90
        result = maneuver.step(
            corner, 0.05, enabled=True, base_speed=200, max_z=800,
            z_invert=True, yaw_total_deg=0, odom_distance_m=1.0)
        self.assertEqual(result.state, 'corner-right')
        self.assertEqual(result.command[0], 0)

    @staticmethod
    def corner(direction=1):
        return {
            'is_valid': True,
            'corner_dir': direction,
            'corner_y_ratio': 0.70,
            'corner_span': 180,
            'error_px': 0,
            'angle_deg': 0,
        }

    def test_right_corner_survives_blind_zone_and_keeps_turning_right(self):
        maneuver = CornerManeuver(advance_frames=1, advance_speed=40,
                                  turn_degrees=78, turn_speed=300,
                                  confirm_frames=1)

        gate = dict(self.corner(1), corner_y_ratio=.88)
        approach = maneuver.step(gate, 0.05, enabled=True,
                                 base_speed=250, max_z=800,
                                 z_invert=True, yaw_total_deg=0)
        blind = maneuver.step({'is_valid': False, 'corner_dir': 0}, 0.05,
                              enabled=True, base_speed=250, max_z=800,
                              z_invert=True, yaw_total_deg=0)

        self.assertEqual(approach.state, 'corner-right')
        self.assertEqual(blind.state, 'corner-right')
        self.assertEqual(blind.command[0], 0)
        self.assertLess(blind.command[2], 0)
        self.assertTrue(maneuver.active)

    def test_left_corner_turn_sign_matches_existing_chassis_convention(self):
        maneuver = CornerManeuver(advance_frames=0, confirm_frames=1)
        result = maneuver.step(dict(self.corner(-1), corner_y_ratio=.88), 0.05, enabled=True,
                               base_speed=250, max_z=800,
                               z_invert=True, yaw_total_deg=0)
        self.assertGreater(result.command[2], 0)

    def test_measured_distance_controls_turn_and_ignores_frame_limit(self):
        maneuver = CornerManeuver(
            advance_frames=1, advance_speed=40, advance_distance_m=0.10,
            confirm_frames=1)

        result = maneuver.step(
            self.corner(1), 0.05, enabled=True, base_speed=250,
            max_z=800, z_invert=True, yaw_total_deg=0,
            odom_distance_m=2.0)
        self.assertEqual(result.state, 'corner-approach-right')
        self.assertIsNone(result.command)

        for _ in range(20):
            result = maneuver.step(
                {'is_valid': False, 'corner_dir': 0}, 0.05, enabled=True,
                base_speed=250, max_z=800, z_invert=True,
                yaw_total_deg=0, odom_distance_m=2.05)
            self.assertEqual(result.state, 'corner-delay-right')
            self.assertEqual(result.command, (40, 0, 0))

        result = maneuver.step(
            {'is_valid': False, 'corner_dir': 0}, 0.05, enabled=True,
            base_speed=250, max_z=800, z_invert=True,
            yaw_total_deg=0, odom_distance_m=2.101)
        self.assertEqual(result.state, 'corner-right')
        self.assertEqual(result.command[0], 0)

    def test_visible_distant_corner_does_not_consume_blind_advance_distance(self):
        maneuver = CornerManeuver(
            advance_distance_m=0.25, confirm_frames=1,
            turn_gate_y_ratio=0.88)
        distant = self.corner(1)
        distant['corner_y_ratio'] = 0.25
        result = maneuver.step(
            distant, 0.05, enabled=True, base_speed=300,
            max_z=800, z_invert=True, odom_distance_m=1.0)
        self.assertEqual(result.state, 'corner-approach-right')

        # The large preview can keep the same L visible for well over 0.25 m.
        # That approach distance must not be mistaken for blind-zone travel.
        nearer = self.corner(1)
        nearer['corner_y_ratio'] = 0.64
        result = maneuver.step(
            nearer, 0.05, enabled=True, base_speed=300,
            max_z=800, z_invert=True, odom_distance_m=1.30)
        self.assertEqual(result.state, 'corner-approach-right')

        result = maneuver.step(
            {'is_valid': False, 'corner_dir': 0}, 0.05, enabled=True,
            base_speed=300, max_z=800, z_invert=True,
            odom_distance_m=1.31)
        self.assertEqual(result.state, 'corner-delay-right')
        result = maneuver.step(
            {'is_valid': False, 'corner_dir': 0}, 0.05, enabled=True,
            base_speed=300, max_z=800, z_invert=True,
            odom_distance_m=1.56)
        self.assertEqual(result.state, 'corner-right')
        self.assertEqual(result.command[0], 0)

    def test_corner_reappearing_below_gate_resumes_approach_then_turns_at_gate(self):
        maneuver = CornerManeuver(advance_frames=1, advance_speed=40,
                                  confirm_frames=1)
        result = maneuver.step(
            self.corner(1), 0.05, enabled=True, base_speed=250,
            max_z=800, z_invert=True, odom_distance_m=1.00)
        self.assertEqual(result.state, 'corner-approach-right')
        self.assertIsNone(result.command)

        result = maneuver.step(
            {'is_valid': False, 'corner_dir': 0}, 0.05, enabled=True,
            base_speed=250, max_z=800, z_invert=True,
            odom_distance_m=1.19)
        self.assertEqual(result.state, 'corner-delay-right')

        result = maneuver.step(
            {'is_valid': False, 'corner_dir': 0}, 0.05, enabled=True,
            base_speed=250, max_z=800, z_invert=True,
            odom_distance_m=1.195)
        self.assertEqual(result.state, 'corner-delay-right')
        self.assertEqual(result.command, (40, 0, 0))
        for ratio, state in ((.879, 'corner-approach-right'), (.88, 'corner-right')):
            result = maneuver.step(
                dict(self.corner(1), corner_y_ratio=ratio), .05, enabled=True,
                base_speed=250, max_z=800, z_invert=True, odom_distance_m=4.0)
            self.assertEqual(result.state, state)
        self.assertLess(result.command[2], 0)

    def test_reacquired_forward_line_finishes_after_actual_yaw_progress(self):
        maneuver = CornerManeuver(advance_frames=0, confirm_frames=1)
        maneuver.step(dict(self.corner(1), corner_y_ratio=.88), 0.05, enabled=True,
                      base_speed=250, max_z=800,
                      z_invert=True, yaw_total_deg=10)
        result = None
        for _ in range(3):
            result = maneuver.step({
                'is_valid': True, 'corner_dir': 0,
                'error_px': 10, 'angle_deg': 5,
            }, 0.05, enabled=True, base_speed=250, max_z=800,
                z_invert=True, yaw_total_deg=-40)

        self.assertIsNone(result)
        self.assertFalse(maneuver.active)
        self.assertGreater(maneuver.exit_frames, 0)

    def test_disabled_controller_cancels_corner(self):
        maneuver = CornerManeuver(advance_frames=0, confirm_frames=1)
        maneuver.step(self.corner(1), 0.05, enabled=True,
                      base_speed=250, max_z=800, z_invert=True)

        self.assertIsNone(maneuver.step({}, 0.05, enabled=False,
                                        base_speed=250, max_z=800,
                                        z_invert=True))
        self.assertFalse(maneuver.active)

    def test_no_odometry_uses_frame_fallback_then_retains_confirmed_direction(self):
        maneuver = CornerManeuver(advance_frames=2, advance_distance_m=.2,
                                  confirm_frames=1)
        maneuver.step(self.corner(1), .05, enabled=True, base_speed=200,
                      max_z=800, z_invert=True)
        result = maneuver.step({'is_valid': False}, .05, enabled=True,
                               base_speed=200, max_z=800, z_invert=True)
        self.assertEqual(result.command, (150, 0, 0))
        result = maneuver.step(dict(self.corner(-1), corner_y_ratio=.99), .05,
                               enabled=True, base_speed=200, max_z=800, z_invert=True)
        self.assertEqual(result.state, 'corner-right')
        self.assertLess(result.command[2], 0)


if __name__ == '__main__':
    unittest.main()
