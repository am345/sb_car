import math
import unittest

from core.trajectory_memory import (
    GroundProjector, TrajectoryMemory, trajectory_geometry)


def pose(x=0.0, y=0.0, yaw=0.0, distance=0.0):
    return {
        'odom_x_m': x,
        'odom_y_m': y,
        'odom_yaw_deg': yaw,
        'odom_distance_m': distance,
    }


class TrajectoryMemoryTests(unittest.TestCase):
    def test_real_and_sim_camera_use_same_work_image_geometry_builder(self):
        geometry = trajectory_geometry(
            640, 480, 320, 90, .14, 8.02, .115)
        self.assertEqual(geometry['image_width'], 320)
        self.assertEqual(geometry['image_height'], 240)

    def test_bottom_pixel_matches_simulated_camera_blind_distance(self):
        projector = GroundProjector(
            320, 240, horizontal_fov_deg=90,
            camera_height_m=.14,
            pitch_down_deg=math.degrees(.14),
            camera_forward_m=.115)

        forward, right = projector.project(159.5, 239)

        self.assertAlmostEqual(forward, .256, delta=.006)
        self.assertAlmostEqual(right, 0, delta=1e-6)

    def test_recall_uses_outgoing_slope_after_reaching_vertex(self):
        memory = TrajectoryMemory(max_travel_without_vision_m=.6)
        memory.observe_vehicle_path(
            [(.25, 0), (.35, 0), (.45, 0), (.45, .10), (.45, .20)],
            pose())

        cue = memory.recall(pose(x=.46, distance=.46))

        self.assertIsNotNone(cue)
        self.assertAlmostEqual(cue['heading_error_deg'], -90, delta=2)
        self.assertGreater(cue['target_right_m'], .08)

    def test_rotation_changes_heading_error_not_stored_world_slope(self):
        memory = TrajectoryMemory(max_travel_without_vision_m=.6)
        memory.observe_vehicle_path(
            [(.25, 0), (.45, 0), (.65, 0)], pose())

        cue = memory.recall(pose(yaw=45, distance=0))

        self.assertAlmostEqual(cue['heading_error_deg'], -45, delta=1e-6)

    def test_memory_expires_after_unobserved_translation(self):
        memory = TrajectoryMemory(max_travel_without_vision_m=.6)
        memory.observe_vehicle_path(
            [(.25, 0), (.45, 0), (.65, 0)], pose())

        self.assertIsNone(memory.recall(pose(x=.7, distance=.61)))


if __name__ == '__main__':
    unittest.main()
