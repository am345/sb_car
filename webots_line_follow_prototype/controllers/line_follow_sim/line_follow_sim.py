#!/usr/bin/env python3
"""PROTOTYPE adapter: run the deployed line follower inside Webots.

No detector or control law is duplicated here.  The adapters only translate:
Webots BGRA camera frames -> OpenCV BGR frames, and the real chassis protocol
(mm/s, mm/s, mrad/s) -> the prototype's four mecanum wheel velocities.
"""
import json
import hashlib
import logging
import math
import os
import sys
import time
from pathlib import Path

from controller import Supervisor


TIME_STEP = 50
CAMERA_PERIOD = 30
WHEEL_RADIUS = 0.04
HALF_WHEELBASE = 0.10
HALF_TRACK = 0.13
START_TRANSLATION = [-2.175, 0.0, 0.04]

CONTROLLER_DIR = Path(__file__).resolve().parent
PROTOTYPE_ROOT = CONTROLLER_DIR.parents[1]
WORKSPACE_ROOT = PROTOTYPE_ROOT.parent
PRODUCTION_ROOT = WORKSPACE_ROOT / "work" / "ipc_webui_patch"
VISION_DEPS = WORKSPACE_ROOT / "work" / "video_analysis_deps"
sys.path.insert(0, str(VISION_DEPS))
sys.path.insert(0, str(PRODUCTION_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from debug_web import DebugWebServer  # noqa: E402
from core.line_follower import LineFollower  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s %(name)s] %(message)s",
)

EXPECTED_CORE_HASHES = {
    "core/line_follower.py": "cf44916e114819664e2d1f82cdce002eb6b41f2c7a4b53fb0b70f879651a5916",
    "core/odometry.py": "69b5645836cedfe7684597742ba2fc4cee84fc11c3553e861bb8135e46aa186b",
}


def verify_production_source():
    actual = {}
    for relative_path, expected in EXPECTED_CORE_HASHES.items():
        path = PRODUCTION_ROOT / relative_path
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        actual[relative_path] = digest
        if digest != expected:
            raise RuntimeError(
                f"Production source mismatch: {relative_path} is {digest}, expected {expected}"
            )
    logging.getLogger(__name__).info("实机/仿真核心源码哈希一致: %s", actual)
    return actual


SOURCE_HASHES = verify_production_source()


class WebotsCamera:
    """Present a Webots Camera through the deployed USBCamera.read() seam."""

    def __init__(self, robot, device):
        self.robot = robot
        self.device = device
        self.width = device.getWidth()
        self.height = device.getHeight()
        self.frames = 0
        device.enable(CAMERA_PERIOD)

    def read(self):
        if self.robot.step(TIME_STEP) == -1:
            raise KeyboardInterrupt
        raw = self.device.getImage()
        if raw is None:
            return None
        bgra = np.frombuffer(raw, dtype=np.uint8).reshape(
            (self.height, self.width, 4)
        )
        self.frames += 1
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)

    def release(self):
        pass


class WebotsMecanumChassis:
    """Translate real chassis command units into four-wheel mecanum IK."""

    def __init__(self, robot):
        self.motors = {
            "front_left": robot.getDevice("front_left_wheel_joint"),
            "front_right": robot.getDevice("front_right_wheel_joint"),
            "back_left": robot.getDevice("back_left_wheel_joint"),
            "back_right": robot.getDevice("back_right_wheel_joint"),
        }
        for motor in self.motors.values():
            motor.setPosition(float("inf"))
            motor.setVelocity(0.0)
        self.last_command = {"x_mm_s": 0, "y_mm_s": 0, "z_mrad_s": 0}
        self.last_wheels = {name: 0.0 for name in self.motors}
        self.halted = False

    def send_speed(self, x, y, z):
        if self.halted:
            x = y = z = 0
        vx = float(x) / 1000.0
        vy = float(y) / 1000.0
        omega = float(z) / 1000.0
        lever = HALF_TRACK + HALF_WHEELBASE
        wheel_speeds = {
            "front_left": (vx - vy - lever * omega) / WHEEL_RADIUS,
            "front_right": (vx + vy + lever * omega) / WHEEL_RADIUS,
            "back_left": (vx + vy - lever * omega) / WHEEL_RADIUS,
            "back_right": (vx - vy + lever * omega) / WHEEL_RADIUS,
        }
        for name, speed in wheel_speeds.items():
            self.motors[name].setVelocity(speed)
        self.last_command = {
            "x_mm_s": int(round(float(x))),
            "y_mm_s": int(round(float(y))),
            "z_mrad_s": int(round(float(z))),
        }
        self.last_wheels = wheel_speeds
        return True

    def read_status(self):
        # The first prototype stage validates vision/control commands only.
        # Simulated encoder/IMU feedback belongs in the dynamics stage.
        return None

    def stop(self):
        self.send_speed(0, 0, 0)

    def emergency_stop(self):
        self.halted = True
        self.stop()


class TraceSink:
    """Expose the deployed controller state without changing its code."""

    restart_requested = False

    def __init__(self, robot, chassis):
        self.robot = robot
        self.chassis = chassis
        self.trace = []
        self.frames = 0
        self.valid_frames = 0
        self.last_detection = {}
        self.last_control = {}
        self.next_trace_time = 0.0
        self.next_capture_time = 0.0
        self.capture_dir = os.environ.get("WEBOTS_SIM_CAPTURE_DIR", "")
        self.self_node = robot.getSelf()
        self.last_position = self.self_node.getPosition()
        self.last_sim_time = robot.getTime()
        self.wall_start = time.monotonic()
        self.sim_start = robot.getTime()
        self.actual_speed_mm_s = 0.0
        self.realtime_factor = 0.0

    def update(self, frame, detection, control):
        now = self.robot.getTime()
        position = self.self_node.getPosition()
        dt = now - self.last_sim_time
        if dt > 0:
            dx = position[0] - self.last_position[0]
            dy = position[1] - self.last_position[1]
            instant_speed = math.hypot(dx, dy) * 1000.0 / dt
            self.actual_speed_mm_s = (
                0.85 * self.actual_speed_mm_s + 0.15 * instant_speed
            )
        self.last_position = position
        self.last_sim_time = now
        wall_elapsed = max(1e-6, time.monotonic() - self.wall_start)
        self.realtime_factor = (now - self.sim_start) / wall_elapsed

        self.frames += 1
        if detection.get("is_valid"):
            self.valid_frames += 1
        self.last_detection = {
            "is_valid": bool(detection.get("is_valid")),
            "error_px": round(float(detection.get("error_px", 0.0)), 3),
            "angle_deg": round(float(detection.get("angle_deg", 0.0)), 3),
            "roi_top": int(detection.get("roi_top", 0)),
            "points": len(detection.get("points") or []),
        }
        self.last_control = {
            "state": str(control.get("state", "")),
            "speed_mm_s": int(control.get("speed", 0)),
            "turn_mrad_s": int(round(float(control.get("turn", 0.0)))),
            "lost_count": int(control.get("lost_count", 0)),
            "started": bool(control.get("started", False)),
        }

        if now >= self.next_trace_time:
            entry = {
                "time_s": round(now, 3),
                **self.last_detection,
                **self.last_control,
                **self.chassis.last_command,
                "wheel_rad_s": {
                    name: round(speed, 3)
                    for name, speed in self.chassis.last_wheels.items()
                },
            }
            self.trace.append(entry)
            print("\r" + json.dumps(entry, ensure_ascii=False), end="", flush=True)
            self.next_trace_time += 1.0

        if self.capture_dir and now >= self.next_capture_time:
            directory = Path(self.capture_dir)
            directory.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(directory / f"camera_{round(now):03d}.jpg"), frame)
            self.next_capture_time += 1.0
        return {
            "sim_time_s": round(now, 3),
            "sim_actual_speed_mm_s": round(self.actual_speed_mm_s, 2),
            "sim_realtime_factor": round(self.realtime_factor, 3),
            "sim_position_m": [round(value, 4) for value in position],
        }


class WebAndTraceSink:
    """Send the same simulated frame/result to the existing WebUI and report."""

    restart_requested = False

    def __init__(self, trace, web, simulation_control):
        self.trace = trace
        self.web = web
        self.simulation_control = simulation_control

    def update(self, frame, detection, control):
        sim_status = self.trace.update(frame, detection, control)
        web_control = dict(control)
        web_control.update({
            "simulation_mode": True,
            "vision_only": False,
            "traffic_control": False,
            "chassis_connected": True,
            "chassis_armed": self.simulation_control["armed"],
            "chassis_target_speed": self.simulation_control["target_speed"],
            **sim_status,
        })
        self.web.update(frame, detection, web_control)


robot = Supervisor()
camera_device = robot.getDevice("camera")
camera = WebotsCamera(robot, camera_device)
chassis = WebotsMecanumChassis(robot)
trace = TraceSink(robot, chassis)
self_node = robot.getSelf()
simulation_control = {"armed": False, "target_speed": 100}
follower_holder = {}


def set_simulation_chassis(enabled, speed=None):
    follower = follower_holder.get("follower")
    if follower is None:
        raise RuntimeError("仿真循迹控制器尚未就绪")
    if enabled:
        target = int(speed if speed is not None else simulation_control["target_speed"])
        if not 1 <= target <= 300:
            raise ValueError("仿真速度必须在1~300 mm/s之间")
        simulation_control.update(armed=True, target_speed=target)
        chassis.halted = False
        follower.base_speed = target
        follower._reset_tracking_state()
        return {
            "chassis_armed": True,
            "chassis_target_speed": target,
            "chassis_state": f"Webots虚拟底盘已启动，目标速度 {target} mm/s",
        }
    simulation_control["armed"] = False
    follower.base_speed = 0
    follower._reset_tracking_state()
    chassis.stop()
    return {
        "chassis_armed": False,
        "chassis_target_speed": simulation_control["target_speed"],
        "chassis_state": "Webots虚拟底盘已停用",
    }


def emergency_stop_simulation():
    simulation_control["armed"] = False
    follower = follower_holder.get("follower")
    if follower is not None:
        follower.base_speed = 0
        follower._reset_tracking_state()
    chassis.emergency_stop()

web_config = {
    "speed": 100,
    "max_z": 800,
    "kp": 12.0,
    "kd": 1.2,
    "ka": 3.5,
    "err_alpha": 0.6,
    "z_rate": 120.0,
    "exposure": 150,
    "roi_top": 0.45,
    "scan_start": 0.25,
    "crop_bottom": 0.50,
    "crop_top": 0.60,
    "track_half": 60.0,
    "startup_frames": 5,
    "ramp_frames": 20,
    "corner_delay_frames": 10,
    "corner_delay_speed": 40,
    "corner_turn_degrees": 78.0,
    "corner_turn_speed": 300,
    "lost_hold": 10,
    "search_frames": 15,
    "threshold": 100,
    "adaptive_block": 31,
    "adaptive_c": 8.0,
    "cross_lateral_distance_m": 0.0,
    "cross_lateral_speed": 100,
    "control_delay_m": 0.10,
    "binary_mode": "otsu",
    "polarity": "black",
}
web = DebugWebServer(
    "127.0.0.1",
    int(os.environ.get("WEBOTS_WEB_PORT", "9092")),
    stream_fps=8.0,
    config=web_config,
    emergency_callback=emergency_stop_simulation,
    chassis_arm_callback=set_simulation_chassis,
)
web.start()
output = WebAndTraceSink(trace, web, simulation_control)

# Match the deployed service: --speed 100; all other values are its defaults.
follower = LineFollower(
    camera,
    chassis,
    base_speed=0,
    max_z=800,
    kp=12.0,
    kd=1.2,
    ka=3.5,
    err_alpha=0.6,
    z_rate_limit=120.0,
    startup_frames=5,
    ramp_frames=20,
    work_width=320,
    roi_top_ratio=0.45,
    n_scan_rows=12,
    scan_start_ratio=0.25,
    crop_bottom_frac=0.50,
    crop_top_frac=0.60,
    track_half=60.0,
    polarity="black",
    binary_mode="otsu",
    z_invert=True,
    # Webots fast mode advances 50 ms per iteration; LineFollower's existing
    # 20 Hz limiter then keeps simulated time and wall time approximately 1:1.
    target_fps=20,
    web_debug=output,
)
follower_holder["follower"] = follower

max_seconds = float(os.environ.get("WEBOTS_SIM_MAX_SECONDS", "0"))
max_frames = None
if max_seconds > 0:
    max_frames = max(1, math.ceil(max_seconds * 1000.0 / TIME_STEP))

try:
    follower.run(max_frames=max_frames)
finally:
    chassis.stop()
    position = self_node.getField("translation").getSFVec3f()
    result = {
        "time_s": round(robot.getTime(), 3),
        "camera": {
            "width": camera.width,
            "height": camera.height,
            "field_of_view_deg": 90.0,
            "pitch_down_deg": round(math.degrees(0.14), 2),
            "source_format": "Webots BGRA",
            "detector_format": "OpenCV BGR",
            "detector_work_width": 320,
        },
        "production_source_hashes": SOURCE_HASHES,
        "frames": trace.frames,
        "valid_frames": trace.valid_frames,
        "valid_ratio": (
            round(trace.valid_frames / trace.frames, 4) if trace.frames else 0.0
        ),
        "position_m": [round(value, 4) for value in position],
        "last_detection": trace.last_detection,
        "last_control": trace.last_control,
        "trace": trace.trace,
    }
    print("\nTEST_RESULT " + json.dumps(result, ensure_ascii=False), flush=True)
    report_path = os.environ.get("WEBOTS_SIM_REPORT", "")
    if report_path:
        Path(report_path).write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    view_report_path = os.environ.get("WEBOTS_VIEW_REPORT", "")
    if view_report_path:
        robot.exportImage(view_report_path, 90)
    web.stop()
    if max_frames is not None:
        robot.simulationQuit(0)
