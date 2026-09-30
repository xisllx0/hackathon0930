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
