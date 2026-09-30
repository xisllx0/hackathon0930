"""D-5: frontier selection. occ[row][col] values: FREE / UNKNOWN / OCCUPIED."""
import math
from collections import deque

FREE = 0
UNKNOWN = -1
OCCUPIED = 1


def _neighbors4(r, c, rows, cols):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield nr, nc


def reachable_cells(blocked, start):
    """BFS over non-blocked cells from start -> {cell: steps}."""
    rows, cols = len(blocked), len(blocked[0])
    start = tuple(start)
    if blocked[start[0]][start[1]]:
        return {}
    dist = {start: 0}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nb in _neighbors4(r, c, rows, cols):
            if nb not in dist and not blocked[nb[0]][nb[1]]:
                dist[nb] = dist[(r, c)] + 1
                q.append(nb)
    return dist


def find_frontiers(occ, blocked):
    """Free, non-blocked cells that touch an UNKNOWN cell."""
    rows, cols = len(occ), len(occ[0])
    out = []
    for r in range(rows):
        for c in range(cols):
            if occ[r][c] != FREE or blocked[r][c]:
                continue
            if any(occ[nr][nc] == UNKNOWN for nr, nc in _neighbors4(r, c, rows, cols)):
                out.append((r, c))
    return out


def next_frontier(occ, blocked, robot_cell, skip=()):
    """Nearest (by path steps) reachable frontier cell, or None when none is left.

    `skip` holds frontier cells to ignore (e.g. ones that already failed to plan).
    """
    dist = reachable_cells(blocked, robot_cell)
    skip = set(map(tuple, skip))
    best, best_d = None, math.inf
    for cell in find_frontiers(occ, blocked):
        if cell in skip or cell == tuple(robot_cell):
            continue
        d = dist.get(cell)
        if d is not None and d < best_d:
            best, best_d = cell, d
    return best
