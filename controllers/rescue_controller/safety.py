"""D-4: LiDAR-based stop/slow. Safety overrides drive output.

LiDAR (LDS-01): 360 beams, index 180 = front, 0 = back, 90 = left, 270 = right
(same convention as controllers/tb3_lidar).
"""
import math

ROBOT_RADIUS = 0.105
STOP_DIST = 0.20         # m, front-sector obstacle distance that triggers a stop
SLOW_DIST = 0.40
MIN_SLOW_SCALE = 0.45    # never crawl slower than this fraction while merely 'slowing'
FRONT_HALF_ANGLE = 15    # beams on each side of index 180 (narrow: doorframes beside us must not count)
STOP_TICKS_FOR_REPLAN = 15
BACKUP_AFTER_TICKS = 45  # still blocked after this long -> reverse a little
BACKUP_TICKS = 20
BACKUP_WHEEL = -1.5      # rad/s
ESCAPE_SPIN = 1.0        # rad/s per wheel when turning away from a blocked front
# Measured in Webots: index runs CLOCKWISE, index 180 = front, 90 = left, 270 = right.
LEFT_SLICE = (60, 120)
RIGHT_SLICE = (240, 300)
REAR_CLEAR = 0.25        # m needed behind us to reverse


class SafetyMonitor:
    def __init__(self, stop_dist=STOP_DIST, slow_dist=SLOW_DIST,
                 half_angle=FRONT_HALF_ANGLE, replan_ticks=STOP_TICKS_FOR_REPLAN):
        self.stop_dist = stop_dist
        self.slow_dist = slow_dist
        self.half_angle = half_angle
        self.replan_ticks = replan_ticks
        self.stopped_ticks = 0
        self.stopped = False
        self.replan = False
        self.backup_left = 0

    @staticmethod
    def _side_open(ranges, sl):
        vals = [r for r in ranges[sl[0]:sl[1]] if r is not None and not math.isnan(r) and not math.isinf(r)]
        return sum(vals) / len(vals) if vals else 3.5

    def rear_min(self, ranges):
        n = len(ranges)
        if n == 0:
            return 0.0
        best = float("inf")
        for i in range(-self.half_angle, self.half_angle + 1):
            r = ranges[i % n]
            if r is None or math.isnan(r) or math.isinf(r) or r <= 0.0:
                continue
            best = min(best, r)
        return best if math.isfinite(best) else 0.0

    def front_min(self, ranges):
        n = len(ranges)
        if n == 0:
            return float("inf")
        c = n // 2
        best = float("inf")
        for i in range(c - self.half_angle, c + self.half_angle + 1):
            r = ranges[i % n]
            if r is None or math.isnan(r) or math.isinf(r) or r <= 0.0:
                continue
            best = min(best, r)
        return best

    def safe_wheels(self, ranges, left, right):
        """Return (left, right, stopped, replan). Call once per control step.

        Only forward motion is limited; rotation is kept so the robot can turn
        away once stopped. If we stay blocked for a long time we back up a little.
        """
        if self.backup_left > 0:
            if self.rear_min(ranges) <= REAR_CLEAR:
                self.backup_left = 0
                self.stopped, self.replan = True, True
                return 0.0, 0.0, True, True
            self.backup_left -= 1
            self.stopped, self.replan = True, True
            return BACKUP_WHEEL, BACKUP_WHEEL, True, True

        if not ranges:
            self.stopped, self.replan = True, True
            return 0.0, 0.0, True, True
        d = self.front_min(ranges)
        forward = (left + right) / 2.0
        if forward < 0 and self.rear_min(ranges) <= REAR_CLEAR:
            self.stopped, self.replan = True, True
            return 0.0, 0.0, True, True
        if forward <= 0 or d >= self.slow_dist:
            self.stopped, self.replan = False, False
            self.stopped_ticks = 0
            return left, right, False, False

        if d <= self.stop_dist:
            spin = (right - left) / 2.0
            if abs(spin) < 0.3:
                # blocked and not already turning: turn toward the more open side
                spin = ESCAPE_SPIN if self._side_open(ranges, LEFT_SLICE) >= self._side_open(ranges, RIGHT_SLICE) else -ESCAPE_SPIN
            left, right = -spin, spin
            self.stopped = True
            self.stopped_ticks += 1
            self.replan = self.stopped_ticks >= self.replan_ticks
            if self.stopped_ticks >= BACKUP_AFTER_TICKS and self.rear_min(ranges) > REAR_CLEAR:
                self.backup_left = BACKUP_TICKS
                self.stopped_ticks = 0
            return left, right, True, self.replan

        scale = (d - self.stop_dist) / (self.slow_dist - self.stop_dist)
        scale = max(MIN_SLOW_SCALE, scale)
        self.stopped, self.replan, self.stopped_ticks = False, False, 0
        return left * scale, right * scale, False, False
