"""D-2: look-ahead path following. World coords in metres/radians, x forward, theta CCW."""
import math

try:  # shared team config
    from config import WHEEL_RADIUS, MAX_WHEEL_SPEED
    try:
        from config import WHEEL_BASE as WHEEL_SEPARATION      # current name (0.177 m measured)
    except ImportError:
        from config import AXLE_LENGTH as WHEEL_SEPARATION     # older name
except ImportError:  # standalone tests without the team config
    WHEEL_RADIUS = 0.033
    WHEEL_SEPARATION = 0.160
    MAX_WHEEL_SPEED = 6.0    # rad/s (motor limit 6.67)

DEFAULT_V = 0.10         # m/s forward speed (0.18 m/s made the robot scrape walls)
LOOKAHEAD = 0.30         # m, path distance ahead of nearest waypoint
ARRIVE_RADIUS = 0.10     # m
TURN_IN_PLACE_ANGLE = math.radians(60)
TURN_SPEED = 0.8         # rad/s angular speed for in-place rotation


def _wheels_from_vw(v, w):
    """(v m/s, w rad/s) -> (left, right) wheel angular velocity in rad/s, clamped."""
    v_r = v + w * WHEEL_SEPARATION / 2.0
    v_l = v - w * WHEEL_SEPARATION / 2.0
    left, right = v_l / WHEEL_RADIUS, v_r / WHEEL_RADIUS
    peak = max(abs(left), abs(right))
    if peak > MAX_WHEEL_SPEED:
        s = MAX_WHEEL_SPEED / peak
        left, right = left * s, right * s
    return left, right


def lookahead_point(pose, path_xy, lookahead=LOOKAHEAD):
    """First waypoint after the one nearest the robot that is >= `lookahead` m away.

    Falls back to the final waypoint. Starting at nearest+1 keeps the target from
    ever being the point behind the robot.
    """
    px, py = pose.x, pose.y
    last = len(path_xy) - 1
    idx = min(range(len(path_xy)), key=lambda i: math.hypot(path_xy[i][0] - px, path_xy[i][1] - py))
    idx = min(idx + 1, last)
    while idx < last and math.hypot(path_xy[idx][0] - px, path_xy[idx][1] - py) < lookahead:
        idx += 1
    return path_xy[idx]


def toward(pose, path_xy, v=DEFAULT_V, lookahead=LOOKAHEAD, arrive_radius=ARRIVE_RADIUS):
    """Follow `path_xy` (list of (x, y) in metres). Returns (left, right, arrived).

    Wheel values are rad/s for Webots velocity mode. `pose` needs .x .y .theta.
    """
    if not path_xy:
        return 0.0, 0.0, True
    if not all(math.isfinite(v) for v in (pose.x, pose.y, pose.theta)):
        return 0.0, 0.0, False          # broken pose from localization: stand still
    gx, gy = path_xy[-1]
    goal_dist = math.hypot(gx - pose.x, gy - pose.y)
    if goal_dist <= arrive_radius:
        return 0.0, 0.0, True

    tx, ty = lookahead_point(pose, path_xy, lookahead)
    dx, dy = tx - pose.x, ty - pose.y
    # target in the robot frame: x_r ahead, y_r left
    c, s = math.cos(pose.theta), math.sin(pose.theta)
    x_r = c * dx + s * dy
    y_r = -s * dx + c * dy
    heading_err = math.atan2(y_r, x_r)

    if abs(heading_err) > TURN_IN_PLACE_ANGLE:
        w = TURN_SPEED if heading_err > 0 else -TURN_SPEED
        left, right = _wheels_from_vw(0.0, w)
        return left, right, False

    d2 = x_r * x_r + y_r * y_r
    curvature = 2.0 * y_r / d2 if d2 > 1e-9 else 0.0
    # slow down in sharp turns and close to the final goal
    v_cmd = v * max(0.3, math.cos(heading_err))
    v_cmd = min(v_cmd, max(0.03, goal_dist))
    left, right = _wheels_from_vw(v_cmd, v_cmd * curvature)
    return left, right, False


class Sweep:
    """Turn once around in place (left turn) so the camera looks in every direction.

    sweep = Sweep(); sweep.start(pose); each step: l, r, done = sweep.step(pose)
    """

    def __init__(self, wheel=2.0, total=2 * math.pi):
        self.wheel = wheel
        self.total = total
        self.acc = 0.0
        self.last = 0.0

    def start(self, pose):
        self.acc = 0.0
        self.last = pose.theta

    def step(self, pose):
        d = (pose.theta - self.last + math.pi) % (2 * math.pi) - math.pi
        self.acc += d
        self.last = pose.theta
        if self.acc >= self.total - 0.1:
            return 0.0, 0.0, True
        return -self.wheel, self.wheel, False
