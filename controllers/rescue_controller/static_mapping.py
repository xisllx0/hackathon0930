"""LiDAR 광선으로 갱신하는 간단한 점유 격자 지도."""
import math
from lidar import beam_angle

FREE = 0
UNKNOWN = -1
OCCUPIED = 1


class StaticMapping:
    def __init__(self, size=100, resolution=0.1, max_range=3.5):
        if size <= 0 or resolution <= 0 or max_range <= 0:
            raise ValueError('지도 크기, 해상도, LiDAR 최대 거리는 양수여야 합니다')
        self.size = size
        self.resolution = resolution
        self.max_range = max_range
        self.grid = [[UNKNOWN for _ in range(size)] for _ in range(size)]
        self.origin = size // 2
        self.lo = [[0 for _ in range(size)] for _ in range(size)]   # log-odds: >=3 wall, <=0 seen free
        # navigation.Navigator가 기대하는 월드 좌하단 좌표 (m)
        self.origin_xy = (-self.origin * resolution,
                          -self.origin * resolution)

    def world_to_grid(self, x, y):
        col = math.floor(x / self.resolution) + self.origin
        row = math.floor(y / self.resolution) + self.origin
        return row, col

    def grid_to_world(self, row, col):
        return ((col - self.origin + 0.5) * self.resolution,
                (row - self.origin + 0.5) * self.resolution)

    def _inside(self, row, col):
        return 0 <= row < self.size and 0 <= col < self.size

    def update(self, pose, ranges):
        """광선이 통과한 칸은 FREE, 유효한 끝점은 OCCUPIED로 표시한다."""
        robot_x, robot_y = pose.x, pose.y
        start_r, start_c = self.world_to_grid(robot_x, robot_y)
        hits = set()
        n = len(ranges)
        if n == 0:
            return

        touched = set()
        for i, raw in enumerate(ranges):
            if raw is None or not isinstance(raw, (int, float)) or math.isnan(raw) or raw <= 0:
                continue
            hit = math.isfinite(raw) and raw < self.max_range
            distance = min(raw, self.max_range)
            angle = pose.theta + beam_angle(i, n)
            end_x = robot_x + distance * math.cos(angle)
            end_y = robot_y + distance * math.sin(angle)
            end_r, end_c = self.world_to_grid(end_x, end_y)

            steps = max(abs(end_r - start_r), abs(end_c - start_c), 1)
            count = steps if hit else steps + 1          # open beam: the end cell is free too
            for k in range(count):
                t = k / steps
                r = round(start_r + t * (end_r - start_r))
                c = round(start_c + t * (end_c - start_c))
                if self._inside(r, c):
                    self.lo[r][c] = max(self.lo[r][c] - 1, -4)
                    touched.add((r, c))
            if hit and self._inside(end_r, end_c):
                self.lo[end_r][end_c] = min(self.lo[end_r][end_c] + 3, 8)
                touched.add((end_r, end_c))

        # a cell only becomes wall/free after repeated evidence, so one grazing ray cannot erase a wall
        for r, c in touched:
            v = self.lo[r][c]
            if v >= 3:
                self.grid[r][c] = OCCUPIED
            elif v <= 0:
                self.grid[r][c] = FREE
        if self._inside(start_r, start_c):
            self.lo[start_r][start_c] = min(self.lo[start_r][start_c], -2)
            self.grid[start_r][start_c] = FREE

    def get_occ(self):
        return self.grid

    def get_blocked(self):
        # 아직 관측하지 않은 칸도 경로 계획에서는 통행 불가.
        return [[cell != FREE for cell in row] for row in self.grid]
