"""Shared, bounded right-angle maneuver for real and simulated line following."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CornerCommand:
    command: tuple
    state: str


class CornerManeuver:
    """Approach a confirmed L turn with a measured blind-zone advance."""

    def __init__(self, advance_frames=10, advance_speed=150,
                 advance_distance_m=0.20,
                 turn_degrees=80.0, turn_speed=800,
                 trigger_y_ratio=0.52, exit_frames=15,
                 reacquire_degrees=43.0, max_turn_frames=110,
                 confirm_frames=2, confirm_speed_ratio=0.50,
                 turn_gate_y_ratio=0.88, reacquire_frames=3):
        self.advance_frames = max(0, int(advance_frames))
        self.advance_speed = max(0, int(advance_speed))
        self.advance_distance_m = max(0.0, float(advance_distance_m))
        self.turn_target_radians = math.radians(
            max(10.0, min(180.0, float(turn_degrees))))
        self.turn_speed = max(50, min(1000, int(turn_speed)))
        self.trigger_y_ratio = float(trigger_y_ratio)
        self.exit_frame_count = max(0, int(exit_frames))
        self.reacquire_radians = math.radians(float(reacquire_degrees))
        self.max_turn_frames = max(1, int(max_turn_frames))
        self.confirm_frames = max(1, int(confirm_frames))
        self.confirm_speed_ratio = max(
            0.0, min(1.0, float(confirm_speed_ratio)))
        self.turn_gate_y_ratio = max(
            self.trigger_y_ratio, min(1.0, float(turn_gate_y_ratio)))
        self.reacquire_frames = max(1, int(reacquire_frames))
        self.reset(clear_exit=True)

    @property
    def active(self):
        return self.direction != 0

    @property
    def turn_radians(self):
        return self._turn_radians

    @property
    def confirming(self):
        return not self.active and self._confirm_count > 0

    @property
    def confirm_direction(self):
        return self._confirm_direction

    @property
    def confirm_count(self):
        return self._confirm_count

    def confirmation_speed(self, base_speed):
        return max(0, int(round(float(base_speed) *
                                self.confirm_speed_ratio)))

    @property
    def exit_frames(self):
        return self._exit_frames

    def reset(self, clear_exit=True):
        self.direction = 0
        self.phase = ''
        self.frames = 0
        self._turn_radians = 0.0
        self._turn_start_yaw_deg = None
        self._advance_start_distance_m = None
        self._confirm_direction = 0
        self._confirm_count = 0
        self._reacquire_count = 0
        if clear_exit or not hasattr(self, '_exit_frames'):
            self._exit_frames = 0

    @staticmethod
    def _finite_yaw(value):
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return value if math.isfinite(value) else None

    def _finish(self):
        self.reset(clear_exit=False)
        self._exit_frames = self.exit_frame_count

    def step(self, det, dt, *, enabled, base_speed, max_z, z_invert,
             yaw_total_deg=None, odom_distance_m=None):
        if not enabled:
            self.reset(clear_exit=True)
            return None

        observed = int(det.get('corner_dir', 0) or 0)
        if self._exit_frames > 0 and not self.active:
            self._exit_frames -= 1
            observed = 0

        if not self.active:
            candidate = (-1 if observed < 0 else 1) if (
                det.get('is_valid') and observed) else 0
            if candidate:
                if candidate == self._confirm_direction:
                    self._confirm_count = min(
                        self.confirm_frames, self._confirm_count + 1)
                elif self._confirm_count > 0:
                    # An opposite one-frame classification should weaken the
                    # existing hypothesis, not instantly flip the maneuver.
                    self._confirm_count -= 1
                    if self._confirm_count == 0:
                        self._confirm_direction = candidate
                        self._confirm_count = 1
                else:
                    self._confirm_direction = candidate
                    self._confirm_count = 1
            else:
                # Reflections commonly erase one frame of tape.  Decay the
                # evidence instead of imposing a brittle consecutive-frame
                # requirement.
                self._confirm_count = max(0, self._confirm_count - 1)
                if self._confirm_count == 0:
                    self._confirm_direction = 0

        # Remember the confirmed direction while approaching its visual gate.
        if (not self.active and
                self._confirm_count >= self.confirm_frames):
            self.direction = self._confirm_direction
            self.phase = 'advance'
            self.frames = 0
            self._turn_radians = 0.0
            self._turn_start_yaw_deg = None
            self._reacquire_count = 0
            self._confirm_direction = 0
            self._confirm_count = 0

            self._advance_start_distance_m = self._finite_yaw(odom_distance_m)

        if not self.active:
            return None

        visual_ready = (
            self.phase == 'advance' and det.get('is_valid') and
            observed == self.direction and
            float(det.get('corner_y_ratio', 0.0)) >=
            self.turn_gate_y_ratio)
        distance_ready = bool(visual_ready)
        if self.phase == 'advance':
            distance = self._finite_yaw(odom_distance_m)
            # While the confirmed corner remains visible, restart the blind
            # distance origin. Normal visible approach does not spend it.
            if (not visual_ready and det.get('is_valid') and
                    observed == self.direction and distance is not None):
                self._advance_start_distance_m = distance
            if distance is not None and self._advance_start_distance_m is not None:
                distance_ready = (distance_ready or
                                  distance-self._advance_start_distance_m >=
                                  self.advance_distance_m)
            elif self._advance_start_distance_m is None:
                distance_ready = distance_ready or self.frames >= self.advance_frames

        if self.phase == 'advance' and not distance_ready:
            self.frames += 1
            if det.get('is_valid') and observed == self.direction:
                # While the corner remains visible, let the normal line
                # controller keep steering toward it at confirmation speed.
                state = ('corner-approach-left' if self.direction < 0
                         else 'corner-approach-right')
                return CornerCommand(None, state)
            speed = min(self.advance_speed, max(0, int(base_speed)))
            state = ('corner-delay-left' if self.direction < 0
                     else 'corner-delay-right')
            return CornerCommand((speed, 0, 0), state)

        if self.phase == 'advance':
            self.phase = 'turn'
            self.frames = 0
            self._turn_start_yaw_deg = self._finite_yaw(yaw_total_deg)

        yaw = self._finite_yaw(yaw_total_deg)
        if yaw is not None and self._turn_start_yaw_deg is not None:
            self._turn_radians = math.radians(abs(yaw-self._turn_start_yaw_deg))

        reacquired_now = (
            self._turn_radians >= self.reacquire_radians and
            det.get('is_valid') and observed == 0 and
            abs(float(det.get('angle_deg', 0.0))) < 35 and
            abs(float(det.get('error_px', 0.0))) < 55)
        self._reacquire_count = (self._reacquire_count + 1
                                 if reacquired_now else 0)
        if self._reacquire_count >= self.reacquire_frames:
            self._finish()
            return None

        self.frames += 1
        if (self._turn_radians >= self.turn_target_radians or
                self.frames > self.max_turn_frames):
            self._finish()
            return CornerCommand((0, 0, 0), 'corner-exit')

        turn_limit = min(abs(float(max_z)), float(self.turn_speed))
        raw_z = self.direction * min(turn_limit, 100.0+self.frames*15.0)
        if yaw is None or self._turn_start_yaw_deg is None:
            self._turn_radians += abs(raw_z)*max(0.001, float(dt))/1000.0
        z = -raw_z if z_invert else raw_z
        if float(base_speed) <= 0:
            z = 0.0
        state = 'corner-left' if self.direction < 0 else 'corner-right'
        return CornerCommand((0, 0, int(round(z))), state)
