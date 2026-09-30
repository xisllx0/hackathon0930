"""탐색 -> 빨간 사과 두 곳 방문 -> 출발점 복귀 상태 관리."""
import math


class Mission:
    def __init__(self):
        self.home_pose = None
        self.state = 'EXPLORE'

    def set_home(self, pose):
        if self.home_pose is None:
            self.home_pose = pose.copy()

    def home_xy(self):
        if self.home_pose is None:
            raise RuntimeError('출발 위치가 아직 저장되지 않았습니다')
        return self.home_pose.x, self.home_pose.y

    def select(self, pose, targets):
        """(상태, 목표 미터 좌표 또는 None)을 반환."""
        if targets.complete:
            hx, hy = self.home_xy()
            if math.hypot(pose.x - hx, pose.y - hy) <= 0.18:
                self.state = 'DONE'
            else:
                self.state = 'RETURN'
            return self.state, (hx, hy)
        target = targets.next_target(pose)
        if target is not None:
            self.state = 'TARGET'
            return self.state, (target.x, target.y)
        self.state = 'EXPLORE'
        return self.state, None
