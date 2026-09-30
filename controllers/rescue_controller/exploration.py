"""D-5: frontier selection. occ[row][col] values: FREE / UNKNOWN / OCCUPIED.

A frontier is a known-free cell that touches an unknown cell. We flood once (BFS)
over known-free cells to get walking distances, group frontier cells into clusters,
ignore tiny clusters (sensor noise / keyholes) and pick the cluster that is cheapest
to reach, with a small bonus for wide openings.
"""
import math
from collections import deque

FREE = 0
UNKNOWN = -1
OCCUPIED = 1

MIN_CLUSTER = 3      # frontier cells in a cluster (>= 3 cells ~ a real opening)
SIZE_BONUS = 0.5     # steps of "discount" per frontier cell in the cluster
SIZE_CAP = 10
SKIP_RADIUS = 3      # cells around a failed goal that are ignored


def _rows(a):
    return a.tolist() if hasattr(a, "tolist") else a


def _neighbors4(r, c, rows, cols):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield nr, nc


def reachable_cells(blocked, start, occ=None):
    """BFS over non-blocked cells from start -> {cell: steps}.

    If `occ` is given, only known-FREE cells are walked (the start cell is always
    allowed), so distances never cut through unexplored space.
    """
    blocked = _rows(blocked)
    occ = _rows(occ) if occ is not None else None
    rows, cols = len(blocked), len(blocked[0])
    start = tuple(start)
    dist = {start: 0}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, rows, cols):
            if (nr, nc) in dist or blocked[nr][nc]:
                continue
            if occ is not None and occ[nr][nc] != FREE:
                continue
            dist[(nr, nc)] = dist[(r, c)] + 1
            q.append((nr, nc))
    return dist


def is_frontier(occ, cell):
    r, c = cell            # works on lists and numpy arrays without copying the map
    if occ[r][c] != FREE:
        return False
    rows, cols = len(occ), len(occ[0])
    return any(occ[nr][nc] == UNKNOWN for nr, nc in _neighbors4(r, c, rows, cols))


def find_frontiers(occ, blocked):
    """Free, non-blocked cells that touch an UNKNOWN cell."""
    occ, blocked = _rows(occ), _rows(blocked)
    rows, cols = len(occ), len(occ[0])
    out = []
    for r in range(rows):
        for c in range(cols):
            if occ[r][c] != FREE or blocked[r][c]:
                continue
            if any(occ[nr][nc] == UNKNOWN for nr, nc in _neighbors4(r, c, rows, cols)):
                out.append((r, c))
    return out


def _clusters(cells):
    """Group cells into 8-connected clusters."""
    left = set(cells)
    out = []
    while left:
        seed = left.pop()
        comp, q = [seed], deque([seed])
        while q:
            r, c = q.popleft()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nb = (r + dr, c + dc)
                    if nb in left:
                        left.discard(nb)
                        comp.append(nb)
                        q.append(nb)
        out.append(comp)
    return out


def next_frontier(occ, blocked, robot_cell, skip=(), min_cluster=MIN_CLUSTER,
                  size_bonus=SIZE_BONUS, skip_radius=SKIP_RADIUS):
    """Best reachable frontier cell, or None when nothing worth visiting is left.

    `skip` holds cells that already failed (path not found / stuck); frontier
    clusters near them are ignored.
    """
    occ, blocked = _rows(occ), _rows(blocked)
    robot_cell = tuple(robot_cell)
    dist = reachable_cells(blocked, robot_cell, occ)
    reach = [f for f in find_frontiers(occ, blocked) if f in dist and f != robot_cell]
    skip = [tuple(s) for s in skip]
    best, best_score = None, math.inf
    for comp in _clusters(reach):
        if len(comp) < min_cluster:
            continue
        rep = min(comp, key=lambda cell: dist[cell])
        if any(max(abs(rep[0] - s[0]), abs(rep[1] - s[1])) <= skip_radius for s in skip):
            continue
        score = dist[rep] - size_bonus * min(len(comp), SIZE_CAP)
        if score < best_score:
            best, best_score = rep, score
    return best


BLOCK = 5            # coverage block = 5x5 cells
SEE_RADIUS_CELLS = 15  # default "we stood here and looked around" radius (cells)


def next_uncovered(occ, blocked, robot_cell, covered_blocks, skip=(), block=BLOCK,
                   skip_radius=SKIP_RADIUS):
    """Nearest reachable known-free cell in a block nobody has looked at yet.

    Used after frontiers run out so corners and side rooms still get a look.
    Returns None when everything reachable has been covered.
    """
    occ, blocked = _rows(occ), _rows(blocked)
    dist = reachable_cells(blocked, tuple(robot_cell), occ)
    skip = [tuple(s) for s in skip]
    best, best_d = None, math.inf
    for cell, d in dist.items():
        if d >= best_d or cell == tuple(robot_cell):
            continue
        if (cell[0] // block, cell[1] // block) in covered_blocks:
            continue
        if any(max(abs(cell[0] - s[0]), abs(cell[1] - s[1])) <= skip_radius for s in skip):
            continue
        best, best_d = cell, d
    return best


class Explorer:
    """Keeps one goal until it is reached, resolved, or marked failed.

    1) frontier goals (unknown space next to known-free space), then
    2) coverage goals (known-free spots nobody has looked at yet).
    Re-picking every step would make the robot dither between goals.

    Coverage bookkeeping (both feed the same set of blocks):
      * add_viewed_blocks(...) - blocks the camera actually saw (line of sight + FOV)
      * observe(cell)          - call after standing at a goal and sweeping 360 degrees
    All sizes are in grid cells, so tune them to the map resolution
    (0.05 m cells: min_cluster=6, skip_radius=6, block=5, see_radius=20).
    """

    def __init__(self, min_cluster=MIN_CLUSTER, skip_radius=SKIP_RADIUS, block=BLOCK,
                 see_radius=SEE_RADIUS_CELLS):
        self.goal = None
        self.mode = None
        self.failed = []
        self.min_cluster = min_cluster
        self.skip_radius = skip_radius
        self.block = block
        self.see_radius = see_radius
        self.covered = set()

    def observe(self, robot_cell):
        br, bc = robot_cell[0] // self.block, robot_cell[1] // self.block
        n = max(1, self.see_radius // self.block)
        for dr in range(-n, n + 1):
            for dc in range(-n, n + 1):
                self.covered.add((br + dr, bc + dc))

    def add_viewed_blocks(self, blocks):
        self.covered.update(map(tuple, blocks))

    def goal_still_valid(self, occ, robot_cell):
        if self.goal is None or self.goal == tuple(robot_cell):
            return False
        if self.mode == "frontier":
            return is_frontier(occ, self.goal)
        return (self.goal[0] // self.block, self.goal[1] // self.block) not in self.covered

    def pick(self, occ, blocked, robot_cell):
        if self.goal_still_valid(occ, robot_cell):
            return self.goal
        self.goal = next_frontier(occ, blocked, robot_cell, self.failed, self.min_cluster,
                                  skip_radius=self.skip_radius)
        self.mode = "frontier"
        if self.goal is None:
            self.goal = next_uncovered(occ, blocked, robot_cell, self.covered, self.failed,
                                       self.block, self.skip_radius)
            self.mode = "coverage"
        return self.goal

    def fail(self):
        if self.goal is not None:
            self.failed.append(self.goal)
        self.goal = None
