"""Traffic-sign and line-following policy.

No hardware access lives here. Required visual conditions fail stopped instead
of being replaced by timed turns. Chassis z is right-positive; positive y is
the configured left translation.
"""
import math


LABELS = ('black40', 'cross', 'green', 'left', 'people', 'red',
          'red40', 'right', 'round', 'turn', 'roadblock')
PRIORITY = ('roadblock', 'red', 'red40', 'people', 'black40', 'round', 'turn',
            'left', 'right', 'cross', 'green')


class SignEvents:
    """Confirm each class on three new frames and rearm after it clears."""

    def __init__(self):
        self.sequence = None
        self.counts = dict.fromkeys(LABELS, 0)
        self.last_seen = dict.fromkeys(LABELS, -1e9)
        self.locks = {}
        self.last_frame = -1e9

    def update(self, result, now):
        if result.get('state') != 'ready' or result.get('age_sec', 1e9) > 1.0:
            self.counts = dict.fromkeys(LABELS, 0)
            for lock in self.locks.values():
                lock['absent_since'] = None
            return []
        sequence = result.get('sequence')
        if sequence is None or sequence == self.sequence:
            return []
        self.sequence = sequence
        if now - self.last_frame > 1.0:
            self.counts = dict.fromkeys(LABELS, 0)
            for lock in self.locks.values():
                lock['absent_since'] = None
        self.last_frame = now
        visible = {d.get('label') for d in result.get('detections', [])
                   if d.get('confidence', 0) > 0.6}
        ready = []
        for label in LABELS:
            lock = self.locks.get(label)
            if lock is not None:
                self.counts[label] = 0
                if lock['until'] is not None and now >= lock['until']:
                    if label in visible:
                        lock['absent_since'] = None
                    elif lock['absent_since'] is None:
                        lock['absent_since'] = now
                    elif now - lock['absent_since'] >= 1.5:
                        del self.locks[label]
                if label in visible:
                    self.last_seen[label] = now
                continue
            if label in visible:
                self.last_seen[label] = now
                self.counts[label] += 1
                if self.counts[label] >= 3:
                    ready.append(label)
            else:
                self.counts[label] = 0
        return [label for label in PRIORITY if label in ready]

    def consume(self, labels, now):
        for label in labels:
            if label not in self.locks:
                self.begin(label, now)
                self.complete(label, now, 'ignored')

    def begin(self, label, now):
        if label not in self.locks:
            self.locks[label] = {'started_at': now, 'until': None,
                                 'absent_since': None, 'outcome': 'executing'}
            self.counts[label] = 0

    def complete(self, label, now, outcome='completed'):
        lock = self.locks.get(label)
        if lock is not None and lock['until'] is None:
            lock.update(until=now + 5.0, absent_since=None, outcome=outcome)

    def release(self, label):
        self.locks.pop(label, None)
        self.counts[label] = 0
        self.last_seen[label] = -1e9

    def status(self, now):
        result = {}
        for label, lock in self.locks.items():
            remaining = None if lock['until'] is None else max(0, lock['until'] - now)
            phase = ('executing' if remaining is None else
                     ('cooldown' if remaining > 0 else 'waiting-clear'))
            absent = lock['absent_since']
            result[label] = {
                'phase': phase,
                'remaining_sec': remaining,
                'clear_remaining_sec': (1.5 if absent is None else
                                        max(0, 1.5 - (now - absent))),
                'outcome': lock['outcome'],
            }
        return result


class TrafficBehavior:
    """State machine joining sign decisions to line-geometry observations."""

    def __init__(self, speed_ceiling=0, advance_m=0.10, max_z=240,
                 follow_line=True, cross_lateral_distance_m=0.0,
                 cross_lateral_speed=100):
        self.follow_line = bool(follow_line)
        # The configured forward-speed ceiling is bounded independently from
        # lateral and angular motion limits.
        self.ceiling = max(0, min(400, int(speed_ceiling)))
        self.advance_m = float(advance_m)  # retained for config compatibility
        if not math.isfinite(self.advance_m) or not 0 <= self.advance_m <= 0.5:
            raise ValueError('路口前进距离必须在 0~0.5 m')
        self.max_z = max(0, min(800, abs(int(max_z))))
        self.cross_lateral_distance_m = float(cross_lateral_distance_m)
        if (not math.isfinite(self.cross_lateral_distance_m) or
                not 0 <= self.cross_lateral_distance_m <= 2.0):
            raise ValueError('cross横移距离必须在 0~2.0 m')
        self.cross_lateral_speed = max(1, min(300, int(cross_lateral_speed)))

        self.events = SignEvents()
        self.state = 'driving'
        self.task = None
        self.action_label = None
        self.cruise = 400
        self.command_x = 0.0
        self.last_time = None
        self.deadline = 0.0
        self.fault = ''
        self.log = []
        self.exits = 0  # legacy WebUI field; roundabout execution is disabled

        self.near_count = 0
        self.forward_count = 0
        self.green_count = 0
        self.people_seen = False
        self.people_clear_count = 0
        self.locked_branch = None
        self.blocked_branch = None
        self.blocked_branches = set()
        self.branch_started_yaw = None
        self.branch_started_distance = None
        self.branch_progress_distance = None
        self.branch_progress_time = None
        self.branch_clear_since = None
        self.branch_lost_since = None
        self.last_branch_command = (0, 0, 0)
        self.line_end_count = 0
        self.line_end_start_distance = None
        self.lateral_start_distance = None
        self.reacquire_count = 0

    def _state(self, state, now, timeout=None):
        self.state = state
        self.deadline = now + timeout if timeout is not None else 0.0
        self.near_count = self.forward_count = self.reacquire_count = 0
        self.log.append({'time': round(now, 3), 'state': state,
                         'task': self.task, 'branch': self.locked_branch})
        self.log = self.log[-30:]

    def _complete_action(self, now, outcome='completed'):
        if self.action_label is not None:
            self.events.complete(self.action_label, now, outcome)
            self.action_label = None

    def _finish(self, now):
        self._complete_action(now)
        self.task = None
        self.locked_branch = None
        self.blocked_branch = None
        self.blocked_branches = set()
        self.branch_started_yaw = None
        self.branch_started_distance = None
        self.branch_progress_distance = None
        self.branch_progress_time = None
        self.branch_clear_since = None
        self.branch_lost_since = None
        self.line_end_count = 0
        self.line_end_start_distance = None
        self.lateral_start_distance = None
        self._state('driving', now)

    def stop(self, reason, now):
        if not self.fault:
            self._complete_action(now, 'interrupted')
            self.events.complete('people', now, 'interrupted')
            self.fault = reason
            self._state('fault', now)
        self.command_x = 0.0
        return (0, 0, 0)

    @staticmethod
    def _visible(result, label):
        return any(d.get('label') == label and d.get('confidence', 0) > 0.6
                   for d in result.get('detections', []))

    @staticmethod
    def _branches(det):
        candidates = det.get('branch_candidates')
        if candidates is not None:
            return {str(item['direction']) for item in candidates
                    if item.get('direction') in ('left', 'straight', 'right')}
        result = set()
        if det.get('junction_left'):
            result.add('left')
        if det.get('junction_straight'):
            result.add('straight')
        if det.get('junction_right'):
            result.add('right')
        return result

    @staticmethod
    def _forward_line(det):
        return (det.get('is_valid', False) and
                abs(float(det.get('error_px', 999))) < 45 and
                abs(float(det.get('angle_deg', 999))) < 30)

    def _begin_task(self, label, now, det, pose):
        self.events.begin(label, now)
        self.action_label = label
        self.task = label
        if label in ('round', 'turn'):
            return self.stop(('环岛' if label == 'round' else '掉头') +
                             '逻辑尚未实现，已安全停车', now)
        if not self.follow_line:
            return self.stop('无循迹时不能安全选择支路', now)
        if label == 'roadblock':
            observed = set(det.get('blocked_branches') or [])
            if det.get('blocked_branch'):
                observed.add(det['blocked_branch'])
            self.blocked_branches = observed
            self.blocked_branch = (next(iter(observed))
                                   if len(observed) == 1 else None)
            self._state('roadblock-memorized', now, 12)
        elif label == 'cross':
            self._state('cross-follow', now, 20)
        else:
            self._state('memorized', now, 12)
        return None

    def _select_branch(self, requested, det, pose, now):
        available = self._branches(det)
        if requested == 'roadblock':
            blocked = set(self.blocked_branches)
            blocked.update(det.get('blocked_branches') or [])
            if det.get('blocked_branch'):
                blocked.add(det['blocked_branch'])
            blocked &= available
            if not blocked:
                return self.stop('无法确定路障所在支路', now)
            self.blocked_branches = blocked
            self.blocked_branch = (next(iter(blocked))
                                   if len(blocked) == 1 else None)
            free = available - blocked
            if len(free) != 1:
                return self.stop('路障支路排除后目标支路不唯一', now)
            selected = next(iter(free))
        else:
            selected = requested
            if selected not in available:
                return self.stop('目标支路不存在: ' + selected, now)
        self.locked_branch = selected
        self.branch_started_yaw = pose['odom_yaw_total_deg']
        self.branch_started_distance = pose['odom_distance_m']
        self.branch_progress_distance = pose['odom_distance_m']
        self.branch_progress_time = now
        self.branch_clear_since = None
        self.branch_lost_since = None
        self.last_branch_command = (0, 0, 0)
        self._state('branch-follow', now)
        return None

    def step(self, now, result, det, pose, measured, tracking):
        dt = 0.05 if self.last_time is None else max(0, min(0.1, now - self.last_time))
        self.last_time = now
        if self.fault:
            return (0, 0, 0)
        if self.ceiling <= 0:
            return (0, 0, 0)
        if not all(math.isfinite(float(pose.get(k, float('nan')))) for k in
                   ('odom_yaw_total_deg', 'odom_distance_m')):
            return self.stop('里程反馈无效', now)
        if result.get('state') != 'ready' or result.get('age_sec', 1e9) > 1.0:
            return self.stop('识别数据过期或不可用', now)

        previous_sequence = self.events.sequence
        events = self.events.update(result, now)
        new_perception = self.events.sequence != previous_sequence
        red_visible = self._visible(result, 'red')
        green_visible = self._visible(result, 'green')

        # Red is a latch. Disappearance is not permission to move.
        if self.state in ('red-brake', 'red-hold'):
            if new_perception:
                self.green_count = (self.green_count + 1
                                    if green_visible and not red_visible else 0)
            self.events.consume([e for e in events if e not in ('red', 'green')], now)
        elif 'red' in events:
            self._complete_action(now, 'interrupted')
            self.events.begin('red', now)
            self.action_label = 'red'
            self.task = 'red'
            self.green_count = 0
            self._state('red-brake', now, 5)

        if self.state not in ('red-brake', 'red-hold'):
            for label in ('red40', 'black40'):
                if label in events:
                    self.events.begin(label, now)
                    self.cruise = 200 if label == 'red40' else 400
                    self.events.complete(label, now)
            if 'people' in events and self.state == 'driving':
                self.events.begin('people', now)
                self.action_label = 'people'
                self.task = 'people'
                self.people_seen = False
                self.people_clear_count = 0
                self._state('people-crossing', now, 20)

            task_events = [e for e in events
                           if e in ('roadblock', 'round', 'turn', 'left', 'right', 'cross')]
            if task_events and self.state in ('driving', 'junction-wait'):
                outcome = self._begin_task(task_events[0], now, det, pose)
                if outcome is not None:
                    return outcome

        if self.deadline and now >= self.deadline:
            return self.stop('动作超时: ' + self.state, now)

        cap = min(self.cruise, self.ceiling)
        branches = self._branches(det)
        near = bool(det.get('junction_near')) and len(branches) >= 2
        self.near_count = self.near_count + 1 if near else 0

        if self.state == 'red-brake':
            self.command_x = max(0.0, self.command_x - 300 * dt)
            stopped = (self.command_x <= 0 and
                       abs(measured.get('real_x', 999)) < 10 and
                       abs(measured.get('real_y', 999)) < 10 and
                       abs(measured.get('real_z', 999)) < 0.03)
            if stopped:
                self._state('red-hold', now)
            return (round(self.command_x), 0, 0)
        if self.state == 'red-hold':
            self.command_x = 0.0
            if self.green_count >= 3:
                self.events.release('red')
                self.events.begin('green', now)
                self.events.complete('green', now)
                self.action_label = None
                self.task = None
                self.green_count = 0
                self._state('driving', now)
            return (0, 0, 0)

        if self.state == 'people-crossing':
            # Prefer an explicit crossing-region detector when available. The
            # confirmed people sign is also usable as the entry observation;
            # its subsequent disappearance across fresh frames marks passage.
            active = (bool(det.get('crosswalk_active', False)) or
                      self._visible(result, 'people'))
            if new_perception:
                self.people_seen = self.people_seen or active
                self.people_clear_count = (self.people_clear_count + 1
                                           if self.people_seen and not active else 0)
            if self.people_clear_count >= 3:
                self._finish(now)
            cap = min(200, self.ceiling)

        if self.state == 'junction-wait':
            self.command_x = 0.0
            return (0, 0, 0)

        required_near = (1 if self.state == 'memorized' and
                         self.task in ('left', 'right') else 3)
        if (self.state in ('memorized', 'roadblock-memorized') and
                self.near_count >= required_near):
            outcome = self._select_branch(self.task, det, pose, now)
            if outcome is not None:
                return outcome

        if self.state == 'branch-follow':
            yaw_delta = abs(pose['odom_yaw_total_deg'] - self.branch_started_yaw)
            travelled = max(0.0, pose['odom_distance_m'] -
                            self.branch_started_distance)
            if yaw_delta > 120:
                return self.stop('支路锁定后航向变化过大', now)
            if (pose['odom_distance_m'] - self.branch_progress_distance >=
                    0.02):
                self.branch_progress_distance = pose['odom_distance_m']
                self.branch_progress_time = now
            elif now - self.branch_progress_time > 2.0:
                return self.stop('支路循迹里程无进展超过2秒', now)
            if det.get('is_valid'):
                self.branch_lost_since = None
                command = (cap, tracking[1], tracking[2])
                self.last_branch_command = command
            else:
                self.branch_lost_since = (now if self.branch_lost_since is None
                                          else self.branch_lost_since)
                if now - self.branch_lost_since > 0.5:
                    return self.stop('转弯中目标线丢失超过0.5秒', now)
                command = self.last_branch_command
            # The instruction applies to one fork only. Once that fork has
            # disappeared and the captured route is the sole continuous line,
            # release the semantic left/right lock before a later fork appears.
            fork_cleared = (travelled >= 0.30 and det.get('is_valid') and
                            not near and len(branches) < 2)
            if fork_cleared:
                if self.branch_clear_since is None:
                    self.branch_clear_since = now
            else:
                self.branch_clear_since = None
            if (self.branch_clear_since is not None and
                    now - self.branch_clear_since >= 1.0):
                self._finish(now)
            self.command_x = float(command[0])
            return (round(command[0]), int(command[1]), round(command[2]))

        if self.state == 'cross-follow':
            candidate = bool(det.get('line_end_candidate', False))
            if candidate:
                if self.line_end_start_distance is None:
                    self.line_end_start_distance = pose['odom_distance_m']
                self.line_end_count += 1
            else:
                self.line_end_count = 0
                self.line_end_start_distance = None
            progressed = (self.line_end_start_distance is not None and
                          pose['odom_distance_m'] - self.line_end_start_distance >= 0.02)
            if self.line_end_count >= 3 and progressed:
                if self.cross_lateral_distance_m <= 0:
                    return self.stop('cross横移距离未配置', now)
                self.lateral_start_distance = pose['odom_distance_m']
                self._state('cross-lateral', now, 8)
                self.command_x = 0.0
                return (0, 0, 0)

        if self.state == 'cross-lateral':
            travelled = pose['odom_distance_m'] - self.lateral_start_distance
            self.command_x = 0.0
            if travelled >= self.cross_lateral_distance_m:
                self._state('cross-reacquire', now, 5)
                return (0, 0, 0)
            return (0, min(self.cross_lateral_speed, self.ceiling), 0)

        if self.state == 'cross-reacquire':
            stable = self._forward_line(det)
            self.reacquire_count = self.reacquire_count + 1 if stable else 0
            self.command_x = 0.0
            if self.reacquire_count >= 4:
                self._finish(now)
            return (0, 0, 0)

        command = tracking if det.get('is_valid') else (0, 0, 0)
        target_x = min(cap, max(0, command[0]))
        self.command_x = min(target_x, self.command_x + 300 * dt)
        return (round(self.command_x), int(command[1]), round(command[2]))

    def status(self):
        return {
            'enabled': True, 'state': self.state, 'task': self.task,
            'mode': 'line-and-signs' if self.follow_line else 'sign-only',
            'round_display_only': True,
            'cruise_mm_s': self.cruise, 'speed_ceiling_mm_s': self.ceiling,
            'locked_branch': self.locked_branch,
            'blocked_branch': self.blocked_branch,
            'blocked_branches': sorted(self.blocked_branches),
            'branch_clear_sec': (0.0 if self.branch_clear_since is None else
                                 max(0.0, self.last_time-
                                     self.branch_clear_since)),
            'cross_lateral_distance_m': self.cross_lateral_distance_m,
            'fault': self.fault,
            'sign_locks': self.events.status(self.last_time or 0.0),
            'log': list(self.log),
            'exit_count': 0,
        }
