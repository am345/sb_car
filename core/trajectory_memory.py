#!/usr/bin/env python3
"""Short odometry-referenced memory for visually observed line slopes."""

import math


class GroundProjector:
    """Flat-ground pinhole projection; vehicle axes are forward/right."""

    def __init__(self, image_width, image_height, horizontal_fov_deg,
                 camera_height_m, pitch_down_deg, camera_forward_m=0.0):
        self.width = int(image_width)
        self.height = int(image_height)
        self.hfov_deg = float(horizontal_fov_deg)
        self.camera_height_m = float(camera_height_m)
        self.pitch = math.radians(float(pitch_down_deg))
        self.camera_forward_m = float(camera_forward_m)
        if self.width < 2 or self.height < 2 or self.camera_height_m <= 0:
            raise ValueError('invalid camera geometry')
        hfov = math.radians(self.hfov_deg)
        if not 0 < hfov < math.pi:
            raise ValueError('horizontal field of view must be between 0 and 180 degrees')
        self.cx = (self.width - 1) / 2.0
        self.cy = (self.height - 1) / 2.0
        self.focal = self.cx / math.tan(hfov / 2.0)

    def project(self, pixel_x, pixel_y):
        right_ray = (float(pixel_x) - self.cx) / self.focal
        down_ray = (float(pixel_y) - self.cy) / self.focal
        ray_down = math.sin(self.pitch) + down_ray * math.cos(self.pitch)
        if ray_down <= 1e-6:
            return None
        scale = self.camera_height_m / ray_down
        forward_ray = math.cos(self.pitch) - down_ray * math.sin(self.pitch)
        forward = self.camera_forward_m + scale * forward_ray
        if forward <= 0:
            return None
        return forward, scale * right_ray

    def project_path(self, pixel_path):
        result = []
        for point in pixel_path:
            projected = self.project(point[0], point[1])
            if projected is not None:
                result.append(projected)
        return result


class TrajectoryMemory:
    """Remember one short ordered trajectory and its tangent at every point."""

    def __init__(self, max_travel_without_vision_m=.6, join_distance_m=.20):
        self.max_travel_without_vision_m = float(max_travel_without_vision_m)
        self.join_distance_m = float(join_distance_m)
        self.reset()

    def reset(self):
        self._points = []
        self._last_observation_distance = None

    @staticmethod
    def _pose(pose):
        return (float(pose.get('odom_x_m', 0)),
                float(pose.get('odom_y_m', 0)),
                math.radians(float(pose.get('odom_yaw_deg', 0))),
                float(pose.get('odom_distance_m', 0)))

    @classmethod
    def _to_world(cls, point, pose):
        x, y, yaw, _ = cls._pose(pose)
        forward, right = point
        left = -right
        return (x + forward * math.cos(yaw) - left * math.sin(yaw),
                y + forward * math.sin(yaw) + left * math.cos(yaw))

    @classmethod
    def _to_vehicle(cls, point, pose):
        x, y, yaw, _ = cls._pose(pose)
        dx, dy = point[0] - x, point[1] - y
        forward = dx * math.cos(yaw) + dy * math.sin(yaw)
        left = -dx * math.sin(yaw) + dy * math.cos(yaw)
        return forward, -left

    @staticmethod
    def _with_headings(points):
        records = []
        for index, point in enumerate(points):
            if index + 1 < len(points):
                other = points[index + 1]
            else:
                other = points[index - 1]
                point, other = other, point
            heading = math.atan2(other[1] - point[1],
                                 other[0] - point[0])
            actual = points[index]
            records.append((actual[0], actual[1], heading))
        return records

    def _advance(self, pose):
        while len(self._points) > 2:
            first = self._to_vehicle(self._points[0], pose)
            second = self._to_vehicle(self._points[1], pose)
            dx, dy = second[0] - first[0], second[1] - first[1]
            length_sq = dx * dx + dy * dy
            if length_sq <= 1e-9:
                self._points.pop(0)
                continue
            passed = -(first[0] * dx + first[1] * dy) / length_sq
            if passed < 1.0:
                break
            self._points.pop(0)

    def observe_vehicle_path(self, vehicle_path, pose):
        if len(vehicle_path) < 2:
            return False
        world = [self._to_world(point, pose) for point in vehicle_path]
        fresh = self._with_headings(world)
        self._advance(pose)
        if self._points:
            first = fresh[0]
            distances = [math.hypot(point[0] - first[0],
                                    point[1] - first[1])
                         for point in self._points]
            join = min(range(len(distances)), key=distances.__getitem__)
            if distances[join] <= self.join_distance_m:
                fresh = self._points[:join] + fresh
        self._points = fresh
        self._last_observation_distance = float(
            pose.get('odom_distance_m', 0))
        return True

    def observe_pixels(self, pixel_path, pose, projector):
        return self.observe_vehicle_path(projector.project_path(pixel_path), pose)

    def recall(self, pose):
        if self._last_observation_distance is None:
            return None
        _, _, yaw, distance = self._pose(pose)
        if (distance - self._last_observation_distance >
                self.max_travel_without_vision_m):
            return None
        self._advance(pose)
        if len(self._points) < 2:
            return None
        current = self._points[0]
        target = self._points[1]
        forward, right = self._to_vehicle(target, pose)
        heading_error = math.atan2(
            math.sin(current[2] - yaw), math.cos(current[2] - yaw))
        return {
            'target_forward_m': forward,
            'target_right_m': right,
            'heading_error_deg': math.degrees(heading_error),
            'remaining_points': len(self._points),
        }
