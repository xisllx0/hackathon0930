"""D-1/D-3: grid A* (4-neighbour). grid[row, col], path = list of (row, col) cells."""
import heapq
from collections import deque

FAIL_OK = "ok"
FAIL_BLOCKED_START = "start_blocked"
FAIL_BLOCKED_GOAL = "goal_blocked"
FAIL_NO_PATH = "no_path"
FAIL_OUT_OF_MAP = "out_of_map"


def _in_map(blocked, cell):
    return 0 <= cell[0] < len(blocked) and 0 <= cell[1] < len(blocked[0])


def _heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(blocked, start, goal):
    """Return [start, ..., goal] as cell tuples, or [] if impossible.

    blocked[row][col] truthy = wall. A start/goal that is blocked or outside the
    map returns [] without searching.
    """
    start, goal = tuple(start), tuple(goal)
    if not _in_map(blocked, start) or not _in_map(blocked, goal):
        return []
    if blocked[start[0]][start[1]] or blocked[goal[0]][goal[1]]:
        return []
    if start == goal:
        return [start]

    g_score = {start: 0}
    came_from = {}
    closed = set()
    heap = [(_heuristic(start, goal), 0, start)]
    while heap:
        _, g, cur = heapq.heappop(heap)
        if cur in closed:
            continue
        if cur == goal:
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            return path[::-1]
        closed.add(cur)
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nb = (cur[0] + dr, cur[1] + dc)
            if not _in_map(blocked, nb) or blocked[nb[0]][nb[1]] or nb in closed:
                continue
            ng = g + 1
            if ng < g_score.get(nb, float("inf")):
                g_score[nb] = ng
                came_from[nb] = cur
                heapq.heappush(heap, (ng + _heuristic(nb, goal), ng, nb))
    return []


def nearest_free_cell(blocked, cell, max_radius=10):
    """Closest non-blocked cell to `cell` (BFS ring search); None if none within radius."""
    cell = tuple(cell)
    if not _in_map(blocked, cell):
        return None
    if not blocked[cell[0]][cell[1]]:
        return cell
    seen = {cell}
    q = deque([(cell, 0)])
    while q:
        cur, d = q.popleft()
        if d >= max_radius:
            continue
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nb = (cur[0] + dr, cur[1] + dc)
            if nb in seen or not _in_map(blocked, nb):
                continue
            seen.add(nb)
            if not blocked[nb[0]][nb[1]]:
                return nb
            q.append((nb, d + 1))
    return None


def path_is_blocked(blocked, path, start_idx=0, lookahead=60):
    """True if any of the next `lookahead` cells of `path` became blocked.

    Call every few steps: newly seen walls invalidate the old path before the robot
    bumps into them (safety.py only reacts once we are already close).
    """
    for r, c in path[start_idx:start_idx + lookahead]:
        if blocked[r][c]:
            return True
    return False


def inflate(blocked, radius_cells):
    """Grow blocked cells by `radius_cells` (round footprint). Needs numpy.

    Use it when the map builder hands over raw walls: radius_cells ~
    ceil((robot radius 0.105 m + margin) / cell size).
    """
    import numpy as np
    b = np.asarray(blocked, dtype=bool)
    out = b.copy()
    rows, cols = b.shape
    for dr in range(-radius_cells, radius_cells + 1):
        for dc in range(-radius_cells, radius_cells + 1):
            if dr * dr + dc * dc > radius_cells * radius_cells:
                continue
            r0, r1 = max(0, dr), min(rows, rows + dr)
            c0, c1 = max(0, dc), min(cols, cols + dc)
            out[r0 - dr:r1 - dr, c0 - dc:c1 - dc] |= b[r0:r1, c0:c1]
    return out


def plan_to_target(blocked, start, goal, max_radius=10):
    """A* to `goal`; if the goal cell is blocked, go to the nearest free cell instead.

    Returns (path, reason). reason is one of the FAIL_* constants; path == [] unless
    reason == FAIL_OK.
    """
    start, goal = tuple(start), tuple(goal)
    if not _in_map(blocked, start) or not _in_map(blocked, goal):
        return [], FAIL_OUT_OF_MAP
    if blocked[start[0]][start[1]]:
        return [], FAIL_BLOCKED_START
    visit = nearest_free_cell(blocked, goal, max_radius)
    if visit is None:
        return [], FAIL_BLOCKED_GOAL
    path = astar(blocked, start, visit)
    if not path:
        return [], FAIL_NO_PATH
    return path, FAIL_OK


def plan_progressive(raw_blocked, start, goal, radii=(3, 2, 1), extra_blocked=None, max_radius=14):
    """Plan with a wide wall margin first, then squeeze through tighter spots.

    raw_blocked: un-inflated walls. radii: wall margins in cells, widest first
    (e.g. 3 cells x 0.05 m = 0.15 m). extra_blocked: cells that are never allowed
    (e.g. unknown space when only known ground should be used).
    Returns (path, reason, radius_used); path == [] if no margin works. Small rooms
    and corners (where the apples hide) are reached through the tighter margins.
    """
    import numpy as np
    raw = np.asarray(raw_blocked, dtype=bool)
    extra = None if extra_blocked is None else np.asarray(extra_blocked, dtype=bool)
    sr, sc = start
    why = FAIL_NO_PATH
    for rad in radii:
        b = inflate(raw, rad)
        if extra is not None:
            b = b | extra
        n = rad + 1
        b[max(0, sr - n):sr + n + 1, max(0, sc - n):sc + n + 1] = False   # our own footprint is free
        path, why = plan_to_target(b.tolist(), start, goal, max_radius)
        if path:
            return path, why, rad
    return [], why, None
