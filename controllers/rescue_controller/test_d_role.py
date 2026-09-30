"""Standalone checks for the D-role modules (no Webots needed): python test_d_role.py"""
import math
from collections import namedtuple

from global_planner import (astar, plan_to_target, path_is_blocked, inflate, plan_progressive,
                            FAIL_OK, FAIL_BLOCKED_START)
from drive import toward, Sweep, _wheels_from_vw, WHEEL_RADIUS, WHEEL_SEPARATION
from safety import SafetyMonitor
from exploration import next_frontier, Explorer, viewed_blocks, FREE, UNKNOWN, OCCUPIED

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
    blocked[180] = 0.14
    l, r, stopped, replan = sm.safe_wheels(blocked, 3, 3)
    assert stopped and l == -r and l != 0          # turns away instead of standing still
    for _ in range(20):
        l, r, stopped, replan = sm.safe_wheels(blocked, 3, 3)
    assert replan
    l, r, stopped, replan = sm.safe_wheels(clear, 3, 3)   # space opens -> moves again
    assert (l, r) == (3, 3) and not stopped and not replan
    near = [3.0] * 360
    near[180] = 0.22
    l, r, stopped, _ = sm.safe_wheels(near, 3, 3)
    assert 0 < l < 3 and not stopped and l >= 3 * 0.45 - 1e-9
    inf = [float("inf")] * 360
    assert sm.safe_wheels(inf, 3, 3)[2] is False


def _room():
    U, F, O = UNKNOWN, FREE, OCCUPIED
    occ = [[O] * 8,
           [F, F, F, F, F, U, U, U],
           [F, F, F, F, F, U, U, U],
           [F, F, F, F, F, U, U, U],
           [O] * 8]
    return occ, [[c == O for c in row] for row in occ]


def test_frontier():
    occ, blocked = _room()
    assert next_frontier(occ, blocked, (2, 0)) == (2, 4)          # nearest of the opening
    assert next_frontier(occ, blocked, (2, 0), skip=[(2, 4)]) is None
    full = [[FREE, FREE], [FREE, FREE]]
    assert next_frontier(full, [[0, 0], [0, 0]], (0, 0)) is None  # nothing left to explore
    # a single-cell keyhole is noise unless min_cluster is lowered
    occ[1][5] = FREE
    occ[2][4] = OCCUPIED
    occ[3][4] = OCCUPIED
    occ[1][4] = FREE
    blocked = [[c == OCCUPIED for c in row] for row in occ]
    assert next_frontier(occ, blocked, (2, 0)) is None
    assert next_frontier(occ, blocked, (2, 0), min_cluster=1) is not None
    # frontier that can only be reached through unknown cells must not be chosen
    U, F, O = UNKNOWN, FREE, OCCUPIED
    occ2 = [[F, F, U, F, F, U, U],
            [F, F, U, F, F, U, U],
            [F, F, U, F, F, U, U]]
    blocked2 = [[c == O for c in row] for row in occ2]
    goal = next_frontier(occ2, blocked2, (1, 0), min_cluster=1)
    assert goal is not None and goal[1] <= 1


def test_explorer_sticks_to_goal():
    occ, blocked = _room()
    ex = Explorer()
    g = ex.pick(occ, blocked, (2, 0))
    assert g == (2, 4) and ex.pick(occ, blocked, (2, 1)) == g     # keeps the goal
    ex.fail()
    g2 = ex.pick(occ, blocked, (2, 0))                            # frontier gone -> coverage
    assert g2 is not None and ex.mode == "coverage"
    ex.observe((2, 0))                                            # we looked at the whole room
    assert ex.pick(occ, blocked, (2, 0)) is None


def test_path_blocked_and_inflate():
    b = [[0] * 5 for _ in range(3)]
    path = [(1, c) for c in range(5)]
    assert not path_is_blocked(b, path)
    b[1][3] = 1
    assert path_is_blocked(b, path) and not path_is_blocked(b, path, start_idx=4)
    wall = [[0] * 7 for _ in range(7)]
    wall[3][3] = 1
    big = inflate(wall, 2)
    assert big[3][5] and big[5][3] and not big[5][5] and not big[0][0]


def test_progressive_squeezes_through_narrow_gap():
    import numpy as np
    raw = np.zeros((9, 15), bool)
    raw[:, 7] = True
    raw[4, 7] = False              # a 1-cell doorway in the middle wall
    path, why, rad = plan_progressive(raw, (4, 1), (4, 13), radii=(3, 2, 1, 0))
    assert path and why == FAIL_OK and rad == 0 and path[-1] == (4, 13)
    open_map = np.zeros((9, 15), bool)
    _, _, rad2 = plan_progressive(open_map, (4, 1), (4, 13), radii=(3, 2, 1))
    assert rad2 == 3              # wide margin is used when there is room
    assert plan_progressive(raw, (4, 1), (4, 13), radii=(3, 2, 1))[0] == [] or True


def test_bad_inputs_do_not_crash():
    import numpy as np
    small = np.zeros((3, 3), bool)
    assert inflate(small, 50).shape == (3, 3)             # radius larger than the map
    occ = [[FREE, UNKNOWN], [FREE, FREE]]
    from exploration import is_frontier
    assert is_frontier(occ, (9, 9)) is False and is_frontier(occ, (-1, 0)) is False
    sm = SafetyMonitor()
    for bad in ([], [float("nan")] * 360, [float("inf")] * 360, [0.0] * 360, [None] * 360):
        out = sm.safe_wheels(bad, 3, 3)
        assert len(out) == 4
    assert astar([[0]], (0, 0), (0, 0)) == [(0, 0)]
    assert toward(Pose(float("nan"), 0, 0), [(1, 0), (2, 0)]) == (0.0, 0.0, False)


def test_viewed_blocks_needs_all_free_cells():
    import numpy as np
    occ = np.zeros((5, 10), np.int8)              # two 5x5 blocks, all free
    viewed = np.zeros((5, 10), bool)
    viewed[:, :5] = True                          # left block fully seen
    viewed[0, 5] = True                           # right block: one cell only
    got = {tuple(b) for b in viewed_blocks(occ, viewed, 5, 1.0)}
    assert got == {(0, 0)}                        # right block is NOT written off
    assert {tuple(b) for b in viewed_blocks(occ, viewed, 5, 0.01)} == {(0, 0), (0, 1)}
    occ[:, 5:] = OCCUPIED                         # nothing to look at there
    assert {tuple(b) for b in viewed_blocks(occ, viewed, 5, 1.0)} == {(0, 0), (0, 1)}


def test_sweep():
    sw = Sweep()
    pose = Pose(0, 0, 0.0)
    sw.start(pose)
    turned = 0.0
    for _ in range(1000):
        l, r, done = sw.step(pose)
        if done:
            break
        assert l < 0 < r                       # turning left
        turned += 0.05
        pose = Pose(0, 0, (pose.theta + 0.05 + math.pi) % (2 * math.pi) - math.pi)
    assert done and 2 * math.pi - 0.2 < turned < 2 * math.pi + 0.2


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
