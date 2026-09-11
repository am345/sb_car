#!/usr/bin/env python3
"""基于底盘实测速度与 IMU Z 轴角速度的轻量二维里程计。"""

import math
import threading
import time


class ImuOdometry:
    """积分车体速度得到世界坐标，并用静止期数据估计陀螺零偏。"""

    def __init__(self, velocity_deadband=5.0, gyro_deadband=8.0):
        self.velocity_deadband = float(velocity_deadband)
        self.gyro_deadband = float(gyro_deadband)
        self._lock = threading.Lock()
        self._gyro_bias = None
        self.reset()

    def reset(self):
        with self._lock:
            self._x_m = 0.0
            self._y_m = 0.0
            self._yaw_rad = 0.0
            self._yaw_total_rad = 0.0
            self._distance_m = 0.0
            self._last_t = None

    def update(self, status, timestamp=None):
        now = time.monotonic() if timestamp is None else float(timestamp)
        body_x = float(status.get('real_x', 0.0))
        body_y = float(status.get('real_y', 0.0))
        chassis_z = float(status.get('real_z', 0.0))
        gyro_raw = float(status.get('ang_vel_z', 0.0))

        stopped = (math.hypot(body_x, body_y) < self.velocity_deadband and
                   abs(chassis_z) < 0.02)
        with self._lock:
            if stopped:
                if self._gyro_bias is None:
                    self._gyro_bias = gyro_raw
                else:
                    self._gyro_bias = 0.98 * self._gyro_bias + 0.02 * gyro_raw
            elif self._gyro_bias is None:
                self._gyro_bias = 0.0

            if self._last_t is None:
                self._last_t = now
                return self._snapshot_unlocked()
            dt = now - self._last_t
            self._last_t = now
            if dt <= 0.0 or dt > 0.5:
                return self._snapshot_unlocked()

            vx = 0.0 if abs(body_x) < self.velocity_deadband else body_x
            vy = 0.0 if abs(body_y) < self.velocity_deadband else body_y
            corrected_gyro = gyro_raw - (self._gyro_bias or 0.0)
            if abs(corrected_gyro) < self.gyro_deadband:
                corrected_gyro = 0.0

            # real_z 是底盘协议已换算的 rad/s，避免把未标定的 IMU
            # 原始数据直接当成角速度。底盘右转为正，世界坐标左转为正。
            yaw_rate = 0.0 if abs(chassis_z) < 0.01 else -chassis_z
            yaw_mid = self._yaw_rad + yaw_rate * dt * 0.5
            world_vx = vx * math.cos(yaw_mid) - vy * math.sin(yaw_mid)
            world_vy = vx * math.sin(yaw_mid) + vy * math.cos(yaw_mid)
            self._x_m += world_vx * dt / 1000.0
            self._y_m += world_vy * dt / 1000.0
            self._distance_m += math.hypot(vx, vy) * dt / 1000.0
            self._yaw_total_rad += yaw_rate * dt
            self._yaw_rad = self._yaw_total_rad
            self._yaw_rad = math.atan2(
                math.sin(self._yaw_rad), math.cos(self._yaw_rad))
            return self._snapshot_unlocked()

    def snapshot(self):
        with self._lock:
            return self._snapshot_unlocked()

    def _snapshot_unlocked(self):
        return {
            'odom_x_m': self._x_m,
            'odom_y_m': self._y_m,
            'odom_yaw_deg': math.degrees(self._yaw_rad),
            'odom_yaw_total_deg': math.degrees(self._yaw_total_rad),
            'odom_distance_m': self._distance_m,
            'odom_gyro_bias': self._gyro_bias or 0.0,
        }
