"""기존 D팀 경로 추종. 바퀴 치수와 속도 한계만 공통 config로 맞춤."""
import math
from config import WHEEL_RADIUS, WHEEL_BASE, MAX_WHEEL_SPEED

WHEEL_SEPARATION = WHEEL_BASE
DEFAULT_V = 0.08
LOOKAHEAD = 0.15
ARRIVE_RADIUS = 0.10
TURN_IN_PLACE_ANGLE = math.radians(60)
TURN_SPEED = 0.8


def _wheels_from_vw(v, w):
    left = (v - w * WHEEL_SEPARATION / 2.0) / WHEEL_RADIUS
    right = (v + w * WHEEL_SEPARATION / 2.0) / WHEEL_RADIUS
    scale = max(1.0, abs(left) / MAX_WHEEL_SPEED,
                abs(right) / MAX_WHEEL_SPEED)
    return left / scale, right / scale


def lookahead_point(pose, path_xy, lookahead=LOOKAHEAD):
    last = len(path_xy) - 1
    idx = min(range(len(path_xy)), key=lambda i: math.hypot(
        path_xy[i][0] - pose.x, path_xy[i][1] - pose.y))
    idx = min(idx + 1, last)
    while idx < last and math.hypot(path_xy[idx][0] - pose.x,
                                     path_xy[idx][1] - pose.y) < lookahead:
        idx += 1
    return path_xy[idx]


def toward(pose, path_xy, v=DEFAULT_V, lookahead=LOOKAHEAD,
           arrive_radius=ARRIVE_RADIUS):
    """(왼쪽 바퀴 rad/s, 오른쪽 바퀴 rad/s, 경로 끝 도착 여부)."""
    if not path_xy:
        return 0.0, 0.0, True
    gx, gy = path_xy[-1]
    goal_dist = math.hypot(gx - pose.x, gy - pose.y)
    if goal_dist <= arrive_radius:
        return 0.0, 0.0, True
    tx, ty = lookahead_point(pose, path_xy, lookahead)
    dx, dy = tx - pose.x, ty - pose.y
    c, s = math.cos(pose.theta), math.sin(pose.theta)
    x_r = c * dx + s * dy
    y_r = -s * dx + c * dy
    heading_err = math.atan2(y_r, x_r)
    if abs(heading_err) > TURN_IN_PLACE_ANGLE:
        w = TURN_SPEED if heading_err > 0 else -TURN_SPEED
        left, right = _wheels_from_vw(0.0, w)
        return left, right, False
    d2 = x_r*x_r + y_r*y_r
    curvature = 2.0*y_r/d2 if d2 > 1e-9 else 0.0
    v_cmd = v * max(0.3, math.cos(heading_err))
    v_cmd = min(v_cmd, max(0.03, goal_dist))
    left, right = _wheels_from_vw(v_cmd, v_cmd*curvature)
    return left, right, False
