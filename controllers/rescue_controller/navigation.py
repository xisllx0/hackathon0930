"""기존 global_planner.py / exploration.py / drive.py 연결용 어댑터.

지도: occ[row][col] = -1 미확인, 0 빈칸, 1 장애물.
좌표: row가 증가하면 월드 y도 증가. grid 원점은 좌하단 셀의 좌하단.
"""
import math

from exploration import next_frontier
from global_planner import astar, FAIL_OK, FAIL_NO_PATH, FAIL_OUT_OF_MAP

UNKNOWN, FREE, OCCUPIED = -1, 0, 1


class Navigator:
    def __init__(self, resolution=0.10, origin=(-10.0, -10.0), clearance=0.23):
        if resolution <= 0:
            raise ValueError('resolution은 양수여야 합니다')
        self.resolution = resolution
        self.origin_x, self.origin_y = origin
        self.clearance = clearance  # 로봇 반지름 + 여유 거리(m)

    def world_to_cell(self, x, y):
        """Odometry / Target의 (x,y)m -> (row,col)."""
        col = math.floor((x - self.origin_x) / self.resolution)
        row = math.floor((y - self.origin_y) / self.resolution)
        return row, col

    def cell_to_world(self, cell):
        """격자 중심 (row,col) -> drive.toward()의 (x,y)m."""
        row, col = cell
        return (self.origin_x + (col + 0.5) * self.resolution,
                self.origin_y + (row + 0.5) * self.resolution)

    def blocked_grid(self, occ, dynamic_cells=()):
        """미탐색 칸/장애물을 통행 불가로 표시하고 로봇 폭만큼 팽창."""
        if not occ or not occ[0] or any(len(row) != len(occ[0]) for row in occ):
            raise ValueError('occ는 비어 있지 않은 직사각형 격자여야 합니다')
        rows, cols = len(occ), len(occ[0])
        obstacles = {(r, c) for r in range(rows) for c in range(cols)
                     if occ[r][c] == OCCUPIED}
        for r, c in dynamic_cells:
            if 0 <= r < rows and 0 <= c < cols:
                obstacles.add((r, c))
        blocked = [[occ[r][c] != FREE for c in range(cols)] for r in range(rows)]
        radius = math.ceil(self.clearance / self.resolution)
        for r, c in obstacles:
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if (dr * dr + dc * dc) * self.resolution**2 > self.clearance**2:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        blocked[nr][nc] = True
        return blocked

    def _valid(self, blocked, cell):
        r, c = cell
        return 0 <= r < len(blocked) and 0 <= c < len(blocked[0])

    def _path_to_world(self, cells):
        return [self.cell_to_world(cell) for cell in cells]

    def to_goal(self, occ, pose, goal_xy, dynamic_cells=(), approach=0.40):
        """사과/복귀 좌표 근처의 갈 수 있는 곳까지 A*.

        반환: (path_xy, reason). path_xy는 drive.toward()에 바로 전달.
        목표 좌표 자체가 사과의 점유 칸이면 approach 이내의 대체 칸을 찾음.
        """
        blocked = self.blocked_grid(occ, dynamic_cells)
        start = self.world_to_cell(pose.x, pose.y)
        goal = self.world_to_cell(*goal_xy)
        if not self._valid(blocked, start) or not self._valid(blocked, goal):
            return [], FAIL_OUT_OF_MAP
        if blocked[start[0]][start[1]]:
            return [], 'start_blocked'
        candidates = []
        radius = math.ceil(approach / self.resolution)
        for r in range(max(0, goal[0]-radius), min(len(blocked),goal[0]+radius+1)):
            for c in range(max(0, goal[1]-radius), min(len(blocked[0]),goal[1]+radius+1)):
                if blocked[r][c]:
                    continue
                x, y = self.cell_to_world((r,c))
                gap = math.hypot(x-goal_xy[0], y-goal_xy[1])
                if gap <= approach:
                    candidates.append((gap,(r,c)))
        # 가장 가까운 칸이 벽 너머일 수 있어 실제로 경로가 있는 후보만 사용.
        for _, cell in sorted(candidates):
            path = astar(blocked, start, cell)
            if path:
                return self._path_to_world(path), FAIL_OK
        return [], FAIL_NO_PATH

    def to_frontier(self, occ, pose, dynamic_cells=(), skip=()):
        """미탐색 영역 경계 중 현재 위치에서 도달 가능한 칸까지 경로."""
        blocked = self.blocked_grid(occ, dynamic_cells)
        start = self.world_to_cell(pose.x, pose.y)
        if not self._valid(blocked,start):
            return [], None, FAIL_OUT_OF_MAP
        if blocked[start[0]][start[1]]:
            return [], None, 'start_blocked'
        local_skip = set(skip)
        while True:
            goal = next_frontier(occ, blocked, start, skip=local_skip)
            if goal is None:
                return [], None, FAIL_NO_PATH
            path = astar(blocked, start, goal)
            if path:
                return self._path_to_world(path), goal, FAIL_OK
            local_skip.add(goal)
