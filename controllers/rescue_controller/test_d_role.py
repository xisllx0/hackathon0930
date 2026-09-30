"""Standalone checks for the D-role modules (no Webots needed): python test_d_role.py"""
import math
from collections import namedtuple

from global_planner import astar, plan_to_target, FAIL_OK, FAIL_BLOCKED_START
from drive import toward, _wheels_from_vw, WHEEL_RADIUS, WHEEL_SEPARATION
from safety import SafetyMonitor
from exploration import next_frontier, FREE, UNKNOWN, OCCUPIED

Pose = namedtuple("Pose", "x y theta")


def test_astar():
    g = [[0, 0, 0, 0, 0],
         [0, 1, 1, 1, 0],
         [0, 0, 0, 1, 0],
         [1, 1, 0, 0, 0],
         [0, 0, 0, 1, 0]]
    p = astar(g, (0, 0), (4, 4))
    assert p and p[0] == (0, 0) and p[-1] == (4, 4)
    assert all(not g[r][c] for r, c in p)
    assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in zip(p, p[1:]))
    assert astar(g, (0, 0), (0, 0)) == [(0, 0)]
    assert astar(g, (1, 1), (4, 4)) == []          # start blocked
    assert astar(g, (0, 0), (3, 0)) == []          # goal blocked
    wall = [[0, 1, 0], [0, 1, 0], [0, 1, 0]]
    assert astar(wall, (0, 0), (0, 2)) == []       # unreachable
    path, why = plan_to_target(g, (0, 0), (3, 0))  # blocked goal -> nearest free cell
    assert why == FAIL_OK and not g[path[-1][0]][path[-1][1]]
    assert plan_to_target(g, (1, 1), (4, 4))[1] == FAIL_BLOCKED_START


def simulate(path, pose, steps=6000, dt=0.032):
    for _ in range(steps):
        l, r, done = toward(pose, path)
        if done:
            return pose, True
        v = (l + r) / 2 * WHEEL_RADIUS
        w = (r - l) * WHEEL_RADIUS / WHEEL_SEPARATION
        th = pose.theta + w * dt
        pose = Pose(pose.x + v * math.cos(th) * dt, pose.y + v * math.sin(th) * dt, th)
    return pose, False


def test_drive():
    pose, ok = simulate([(0, 0), (0.5, 0), (1.0, 0)], Pose(0, 0, 0))
    assert ok and abs(pose.x - 1.0) < 0.12 and abs(pose.y) < 0.12
    l, r, _ = toward(Pose(0, 0, 0), [(0, 1.0)])     # goal on the left
    assert r > l
    l, r, _ = toward(Pose(0, 0, 0), [(0, -1.0)])    # goal on the right
    assert l > r
    pose, ok = simulate([(0, 0), (1, 0)], Pose(0, 0, math.pi))   # start facing away
    assert ok
    pose, ok = simulate([(0, 0), (1, 0)], Pose(0, 0, 0))         # (1,0) and back
    pose, ok2 = simulate([(1, 0), (0, 0)], pose)
    assert ok and ok2 and math.hypot(pose.x, pose.y) < 0.12
    assert toward(Pose(0, 0, 0), []) == (0.0, 0.0, True)
    assert max(map(abs, _wheels_from_vw(5.0, 0))) <= 6.0


def test_safety():
    sm = SafetyMonitor()
    clear = [3.0] * 360
    assert sm.safe_wheels(clear, 3, 3)[:2] == (3, 3)
    blocked = [3.0] * 360
    blocked[180] = 0.2
    l, r, stopped, replan = sm.safe_wheels(blocked, 3, 3)
    assert stopped and l == 0 and r == 0
    for _ in range(20):
        l, r, stopped, replan = sm.safe_wheels(blocked, 3, 3)
    assert replan
    l, r, stopped, replan = sm.safe_wheels(clear, 3, 3)   # space opens -> moves again
    assert (l, r) == (3, 3) and not stopped and not replan
    near = [3.0] * 360
    near[180] = 0.4
    l, r, stopped, _ = sm.safe_wheels(near, 3, 3)
    assert 0 < l < 3 and not stopped
    inf = [float("inf")] * 360
    assert sm.safe_wheels(inf, 3, 3)[2] is False


def test_frontier():
    U, F, O = UNKNOWN, FREE, OCCUPIED
    occ = [[F, F, U],
           [F, O, U],
           [F, F, F]]
    blocked = [[c == O for c in row] for row in occ]
    assert next_frontier(occ, blocked, (0, 0)) == (0, 1)
    assert next_frontier(occ, blocked, (0, 0), skip=[(0, 1)]) == (2, 2)
    full = [[F, F], [F, F]]
    assert next_frontier(full, [[0, 0], [0, 0]], (0, 0)) is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
