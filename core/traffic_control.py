"""Opt-in traffic control loop; existing detector, PD parameters and odometry.

The legacy/manual loop is unchanged. This loop owns the serial command channel
exclusively and adds an independent watchdog for a blocked camera/control loop.
"""
import logging
import math
import threading
import time
from collections import deque

from core.line_follower import LineFollower
from core.traffic_behavior import TrafficBehavior

logger = logging.getLogger(__name__)


class TrafficControlRunner:
    def __init__(self, follower, advance_m=0.10, sign_only=False,
                 cross_lateral_distance_m=0.0, cross_lateral_speed=100,
                 control_delay_m=0.10):
        self.f = follower
        self.sign_only = bool(sign_only)
        self.advance_m = float(advance_m)
        self.cross_lateral_distance_m = float(cross_lateral_distance_m)
        self.cross_lateral_speed = int(cross_lateral_speed)
        self.control_delay_m = float(control_delay_m)
        if (not math.isfinite(self.control_delay_m) or
                not 0 <= self.control_delay_m <= 0.5):
            raise ValueError('路径延迟距离必须在0~0.5m')
        self.policy = self._new_policy()
        self.lock = threading.RLock()
        self.guard_done = threading.Event()
        self.last_tick = time.monotonic()
        self.last_feedback = None
        self.last_line_valid = False
        # Hardware may be connected, but motion is never authorized on startup.
        # Only an explicit WebUI action can arm this process.
        self.armed = False
        self.active = False
        self.fault = ''
        self.last_fault = ''
        self.control_queue = deque(maxlen=400)
        self.control_route = None
        self.control_target = None
        self.control_last_distance = None
        self.start_count = 0
        self.lost_since = None
        self.started = False
        self.last_loop_state = None
        self.last_command = (0, 0, 0)
        LineFollower.reset_tracking_controller(self.f)

    def _new_policy(self):
        return TrafficBehavior(self.f.base_speed, self.advance_m, self.f.max_z,
                               follow_line=not self.sign_only,
                               cross_lateral_distance_m=self.cross_lateral_distance_m,
                               cross_lateral_speed=self.cross_lateral_speed)

    def _associate_roadblock(self, result, det):
        """Associate every obstacle ground contact with candidate polylines."""
        candidates = det.get('branch_candidates') or []
        width = result.get('frame_width')
        boxes = [d for d in result.get('detections', [])
                 if d.get('label') == 'roadblock' and
                 d.get('confidence', 0) >= 0.5 and len(d.get('box', ())) == 4]
        if not boxes or len(candidates) < 2 or not width:
            det['blocked_branch'] = None
            det['blocked_branches'] = []
            det['roadblock_associations'] = []
            return
        scale = self.f.detector.work_width / float(width)
        blocked = set()
        associations = []
        for obstacle in boxes:
            x1, _, x2, y2 = obstacle['box']
            # Bottom-centre approximates the cone contact point on the ground.
            obstacle_x = (float(x1)+float(x2))*0.5*scale
            obstacle_y = float(y2)*scale
            ranked = sorted((self._distance_to_candidate(
                                  obstacle_x, obstacle_y, item),
                              item['direction'])
                             for item in candidates)
            corridor = max(20.0, min(60.0,
                                     abs(float(x2)-float(x1))*scale*0.5))
            unambiguous = (len(ranked) == 1 or
                           ranked[1][0]-ranked[0][0] >= 12.0)
            branch = (ranked[0][1]
                      if ranked[0][0] <= corridor and unambiguous else None)
            if branch is not None:
                blocked.add(branch)
            associations.append({
                'branch': branch,
                'distance_px': round(float(ranked[0][0]), 1),
                'corridor_px': round(float(corridor), 1),
                'confidence': float(obstacle.get('confidence', 0)),
                'ground_point': (round(obstacle_x, 1), round(obstacle_y, 1)),
            })
        det['blocked_branches'] = sorted(blocked)
        det['blocked_branch'] = (next(iter(blocked))
                                 if len(blocked) == 1 else None)
        det['roadblock_associations'] = associations

    @staticmethod
    def _distance_to_candidate(x, y, candidate):
        points = candidate.get('points') or []
        if len(points) < 2:
            return math.hypot(float(candidate['target_x'])-x,
                              float(candidate.get('target_y', y))-y)
        best = float('inf')
        for first, second in zip(points, points[1:]):
            ax, ay = float(first[0]), float(first[1])
            bx, by = float(second[0]), float(second[1])
            dx, dy = bx-ax, by-ay
            denom = dx*dx + dy*dy
            t = 0.0 if denom <= 1e-9 else max(
                0.0, min(1.0, ((x-ax)*dx + (y-ay)*dy)/denom))
            px, py = ax+t*dx, ay+t*dy
            best = min(best, math.hypot(x-px, y-py))
        return best

    def _detector_path_preference(self):
        # A locked branch is an explicit route decision and wins. Otherwise,
        # every line-following state traces from the line directly under the
        # vehicle toward the distance. This prevents a longer transverse arm
        # at a crossing from stealing the current connected path.
        if self.policy.locked_branch is not None:
            return self.policy.locked_branch
        if self.policy.state in (
                'driving', 'memorized', 'roadblock-memorized',
                'people-crossing', 'cross-follow', 'junction-wait'):
            return 'continuation'
        return None

    def _reset_control_state(self):
        self.policy = self._new_policy()
        self.active = False
        self.started = False
        self.start_count = 0
        self.lost_since = None
        LineFollower.reset_tracking_controller(self.f)
        self._clear_control_queue()
        self.last_command = (0, 0, 0)

    def _clear_control_queue(self):
        self.control_queue.clear()
        self.control_route = None
        self.control_target = None
        self.control_last_distance = None

    def set_armed(self, enabled, speed=None):
        """Arm or disarm physical motion; every arm starts a fresh policy."""
        with self.lock:
            if not enabled:
                self.armed = False
                self.fault = ''
                self.last_fault = ''
                self._reset_control_state()
                if not self.f.chassis.send_speed(0, 0, 0):
                    raise RuntimeError('底盘停车命令发送失败')
                logger.warning('网页已停用底盘，所有运动命令归零')
                return {'chassis_armed': False,
                        'chassis_target_speed': int(self.f.base_speed),
                        'chassis_state': '底盘已停用'}

            if isinstance(speed, bool):
                raise ValueError('启动速度类型错误')
            try:
                target_speed = int(speed)
            except (TypeError, ValueError, OverflowError):
                raise ValueError('启动速度必须是1~300 mm/s的整数') from None
            if not 1 <= target_speed <= 300:
                raise ValueError('启动速度必须在1~300 mm/s之间')

            now = time.monotonic()
            traffic = self.f.web_debug.get_traffic_status()
            blockers = []
            if self.f.web_debug.heartbeat_age() > 2:
                blockers.append('网页心跳过期')
            if (self.last_feedback is None or
                    now - self.last_feedback > 0.5):
                blockers.append('底盘反馈过期')
            if (traffic.get('state') != 'ready' or
                    traffic.get('age_sec', 1e9) > 1):
                blockers.append('模型结果未就绪')
            if not self.sign_only and not self.last_line_valid:
                blockers.append('未检测到有效线路')
            if blockers:
                raise RuntimeError('无法启动底盘：' + '、'.join(blockers))

            self.f.base_speed = target_speed
            self.fault = ''
            self.last_fault = ''
            self._reset_control_state()
            self.armed = True
            logger.warning('网页已启动底盘，速度上限=%d mm/s', target_speed)
            return {'chassis_armed': True,
                    'chassis_target_speed': target_speed,
                    'chassis_state': '底盘已授权，等待起步确认'}

    def trip(self, reason):
        with self.lock:
            if not self.fault:
                self.fault = reason
                self.last_fault = reason
                logger.error('交通控制保护停车: %s', reason)
            self.armed = False
            self.active = False
            self.started = False
            self._clear_control_queue()
            self.f.chassis.send_speed(0, 0, 0)

    def _watchdog(self, stop_event):
        while not self.guard_done.wait(0.1):
            self.check_safety(time.monotonic(), stop_event)

    def check_safety(self, now, stop_event):
        if stop_event.is_set():
            self.trip('急停')
        elif self.armed and self.active:
            if now-self.last_tick > 0.7:
                self.trip('控制循环/相机阻塞')
            elif self.last_feedback is None or now-self.last_feedback > 0.5:
                self.trip('底盘反馈超时')
            elif self.f.web_debug.heartbeat_age() > 3:
                self.trip('网页心跳超时')

    def send(self, command, stop_event):
        with self.lock:
            if (not self.armed or self.fault or stop_event.is_set() or
                    self.f.base_speed <= 0):
                command = (0, 0, 0)
            if not self.f.chassis.send_speed(*command):
                self.trip('串口发送失败')
                command = (0, 0, 0)
            self.last_command = command
            self.last_tick = time.monotonic()
            return command

    def _tracking(self, det, dt, odom_distance_m=None, route_id='main'):
        if self.sign_only:
            # Straight cruise; no visual steering, including when no line exists.
            return (300, 0, 0)
        if not det.get('is_valid'):
            LineFollower.reset_tracking_controller(self.f)
            self._clear_control_queue()
            return (0, 0, 0)
        f = self.f
        target = {
            'error_px': float(det['error_px']),
            'angle_deg': float(det['angle_deg']),
            'path_curvature': float(det.get('path_curvature', 0.0)),
            'memory_active': bool(det.get('memory_active')),
            'selected_branch_direction': det.get('selected_branch_direction'),
        }
        memory_active = bool(det.get('memory_active'))
        if memory_active:
            # A recalled cue is already expressed at the current vehicle pose;
            # delaying it again would apply the stored slope too late.
            self._clear_control_queue()
        elif odom_distance_m is not None and self.control_delay_m > 0:
            distance = float(odom_distance_m)
            if not math.isfinite(distance):
                self._clear_control_queue()
                return (0, 0, 0)
            route = str(route_id or 'main')
            backwards = (self.control_last_distance is not None and
                         distance < self.control_last_distance-1e-4)
            if route != self.control_route or backwards:
                self._clear_control_queue()
                self.control_route = route
                LineFollower.reset_tracking_controller(self.f)
            self.control_last_distance = distance
            sample = dict(target, distance=distance, route=route)
            self.control_queue.append(sample)
            matured = None
            while (self.control_queue and
                   distance-self.control_queue[0]['distance'] >=
                   self.control_delay_m-1e-9):
                matured = self.control_queue.popleft()
            if matured is not None:
                self.control_target = matured
            target = (self.control_target or {
                'error_px': 0.0, 'angle_deg': 0.0,
                'path_curvature': 0.0, 'memory_active': False,
                'selected_branch_direction': None})
        effective_det = dict(det)
        effective_det.update(target)
        limit = abs(f.max_z)
        rate_limit = f.z_rate_limit
        if (self.policy.task in ('left', 'right') and
                target.get('selected_branch_direction') == self.policy.task):
            rate_limit = limit
        steering = LineFollower.compute_tracking_steering(
            f, effective_det, dt, rate_limit=rate_limit)
        z = steering['turn']
        base = min(self.policy.cruise, self.policy.ceiling)
        if self.policy.state == 'branch-follow':
            return (round(base), 0, round(z))
        speed = LineFollower.compute_tracking_speed(
            f, effective_det, z, base_speed=base, ramp=1.0)
        return (speed, 0, round(z))

    def run(self, max_frames=None, stop_event=None):
        f = self.f
        stop_event = stop_event or threading.Event()
        if f.web_debug is None:
            raise ValueError('交通控制必须启用 WebUI 和心跳')
        thread = threading.Thread(target=self._watchdog, args=(stop_event,), daemon=True)
        f.chassis.stop()
        thread.start()
        previous = time.monotonic()
        frame_count = 0
        last_state = None
        try:
            while not stop_event.is_set():
                frame = f.camera.read()
                now = time.monotonic()
                dt, previous = max(0.001, now-previous), now
                if frame is None:
                    self.trip('相机掉线')
                status = f.chassis.read_status()
                if status is not None:
                    self.last_feedback = now
                    f._last_chassis_status = status
                    f.odometry.update(status, timestamp=now)
                f.detector.path_preference = self._detector_path_preference()
                det = f.detector.process(frame)
                pose = f.odometry.snapshot()
                if status is not None:
                    det, _ = f._apply_trajectory_memory(det, pose)
                self.last_line_valid = bool(det.get('is_valid'))
                result = f.web_debug.get_traffic_status()
                self._associate_roadblock(result, det)
                ready = (frame is not None and result.get('state') == 'ready'
                         and result.get('age_sec', 1e9) <= 1
                         and self.last_feedback is not None and now-self.last_feedback <= 0.5
                         and f.web_debug.heartbeat_age() <= 3)
                if self.armed:
                    self.start_count = (self.start_count + 1 if ready and
                                        (self.sign_only or det.get('is_valid')) else 0)
                    if (not self.started and
                            self.start_count >= max(1, f.startup_frames)):
                        self.started = self.active = True
                else:
                    self.start_count = 0
                    self.started = self.active = False
                    self.lost_since = None
                command = (0, 0, 0)
                if self.armed and self.started:
                    if not ready:
                        self.trip('相机/识别/反馈/心跳不可用')
                    if self.sign_only or det.get('is_valid') or self.policy.state in (
                            'branch-follow', 'cross-lateral', 'cross-reacquire',
                            'junction-wait', 'red-brake', 'red-hold'):
                        self.lost_since = None
                    else:
                        self.lost_since = now if self.lost_since is None else self.lost_since
                        if now-self.lost_since > 2:
                            self.trip('持续失线')
                    route_id = ('branch:' + self.policy.locked_branch
                                if self.policy.locked_branch else
                                ('inactive:' + self.policy.state
                                 if self.policy.state in (
                                     'red-brake', 'red-hold', 'fault',
                                     'junction-wait', 'cross-lateral',
                                     'cross-reacquire') else 'main'))
                    tracking = self._tracking(
                        det, dt, pose['odom_distance_m'], route_id)
                    if not self.fault:
                        with self.lock:
                            command = self.policy.step(now, result, det, pose,
                                                       f._last_chassis_status or {}, tracking)
                            if self.policy.fault:
                                self.trip(self.policy.fault)
                command = self.send(command, stop_event)
                waiting = '等待网页心跳/模型/反馈' + ('' if self.sign_only else '/线路')
                if not self.armed:
                    state = ('保护停车：' + self.last_fault
                             if self.last_fault else '底盘未启用')
                else:
                    state = self.fault or (self.policy.state if self.started else waiting)
                if state != last_state:
                    LineFollower.reset_tracking_controller(f)
                    logger.info('交通状态=%s task=%s exit=%s', state, self.policy.task, self.policy.exits)
                    last_state = state
                web_det = dict(det, work_width=f.detector.work_width,
                               crop_top_frac=f.detector.crop_top_frac,
                               crop_bottom_frac=f.detector.crop_bottom_frac)
                f.web_debug.update(frame, web_det, {
                    'state': state, 'traffic_control': True, 'vision_only': False,
                    'sign_only': self.sign_only,
                    'chassis_connected': True, 'chassis_armed': self.armed,
                    'chassis_target_speed': int(f.base_speed),
                    'chassis_fault': self.last_fault,
                    'behavior': {**self.policy.status(),
                                 'fault': self.last_fault or self.policy.fault},
                    'speed': command[0], 'lateral_speed': command[1], 'turn': command[2],
                    'frame_count': frame_count, 'fps': 1/dt,
                    'error_px': det.get('error_px', 0), 'angle_deg': det.get('angle_deg', 0),
                    'binary_mode': f.detector.binary_mode, 'max_z': abs(f.max_z),
                    'control_delay_m': self.control_delay_m,
                    'control_queue_depth': len(self.control_queue),
                    'started': self.started, 'start_seen': self.start_count,
                    'startup_frames': f.startup_frames, 'heartbeat_age_sec': f.web_debug.heartbeat_age(),
                    'feedback_age_sec': None if self.last_feedback is None else now-self.last_feedback,
                    'junction_left': det.get('junction_left', False),
                    'junction_right': det.get('junction_right', False),
                    'junction_near': det.get('junction_near', False),
                    'manual_state': '交通控制模式：手动目标与里程清零已锁定',
                    **f.odometry.snapshot(),
                })
                if f.web_debug.restart_requested:
                    break
                frame_count += 1
                if max_frames is not None and frame_count >= max_frames:
                    break
                stop_event.wait(max(0, f.frame_interval-(time.monotonic()-now)))
        except KeyboardInterrupt:
            pass
        finally:
            self.guard_done.set()
            thread.join(timeout=1)
            with self.lock:
                f.chassis.stop()
            f.detector.path_preference = None
