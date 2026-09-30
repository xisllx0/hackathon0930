"""D-4: LiDAR-based stop/slow. Safety overrides drive output.

LiDAR (LDS-01): 360 beams, index 180 = front, 0 = back, 90 = left, 270 = right
(same convention as controllers/tb3_lidar).
"""
import math

ROBOT_RADIUS = 0.105
STOP_DIST = 0.25         # m, front-sector obstacle distance that triggers a stop
SLOW_DIST = 0.50
FRONT_HALF_ANGLE = 25    # beams on each side of index 180
STOP_TICKS_FOR_REPLAN = 15


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
        away once stopped.
        """
        d = self.front_min(ranges)
        forward = (left + right) / 2.0
        if forward <= 0 or d >= self.slow_dist:
            self.stopped, self.replan = False, False
            self.stopped_ticks = 0
            return left, right, False, False

        if d <= self.stop_dist:
            spin = (right - left) / 2.0
            left, right = -spin, spin
            self.stopped = True
            self.stopped_ticks += 1
            self.replan = self.stopped_ticks >= self.replan_ticks
            return left, right, True, self.replan

        scale = (d - self.stop_dist) / (self.slow_dist - self.stop_dist)
        self.stopped, self.replan, self.stopped_ticks = False, False, 0
        return left * scale, right * scale, False, False
