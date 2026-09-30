import math

FREE = 0
UNKNOWN = -1
OCCUPIED = 1


class StaticMapping:

    def __init__(self, size=100, resolution=0.1):

        self.size = size
        self.resolution = resolution

        self.grid = [
            [UNKNOWN for _ in range(size)]
            for _ in range(size)
        ]

        self.origin = size // 2

    def world_to_grid(self, x, y):

        col = int(
            x / self.resolution
        ) + self.origin

        row = int(
            y / self.resolution
        ) + self.origin

        return row, col


    def update(self, pose, ranges):

        robot_x = pose.x
        robot_y = pose.y
        theta = pose.theta

        for i, distance in enumerate(ranges):

            if math.isinf(distance):
                continue

            angle = (
                theta
                +
                (i - len(ranges)/2)
                * math.pi * 2 / len(ranges)
            )

            end_x = (
                robot_x
                + distance * math.cos(angle)
            )

            end_y = (
                robot_y
                + distance * math.sin(angle)
            )

            r, c = self.world_to_grid(
                end_x,
                end_y
            )

            if 0 <= r < self.size and 0 <= c < self.size:
                self.grid[r][c] = OCCUPIED


        # 로봇 위치 표시
        r, c = self.world_to_grid(
            robot_x,
            robot_y
        )

        if 0 <= r < self.size and 0 <= c < self.size:
            self.grid[r][c] = FREE


    def get_occ(self):
        return self.grid


    def get_blocked(self):

        return [
            [
                cell == OCCUPIED
                for cell in row
            ]
            for row in self.grid
        ]