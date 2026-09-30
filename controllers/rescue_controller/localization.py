"""엔코더 이동량과 compass 방향 변화로 시작점 기준 위치 추정."""
import math
from config import WHEEL_RADIUS
from pose import Pose


def wrap_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class Localization:
    def __init__(self):
        self.pose = Pose()
        self.prev_left = None
        self.prev_right = None
        self.prev_yaw = None

    def update(self, left_encoder, right_encoder, yaw):
        if not all(math.isfinite(v) for v in (left_encoder, right_encoder, yaw)):
            raise ValueError('엔코더 또는 compass yaw가 유효하지 않습니다')
        if self.prev_left is None:
            self.prev_left = left_encoder
            self.prev_right = right_encoder
            self.prev_yaw = yaw
            return self.pose

        dl = (left_encoder - self.prev_left) * WHEEL_RADIUS
        dr = (right_encoder - self.prev_right) * WHEEL_RADIUS
        distance = (dl + dr) / 2.0
        delta_yaw = wrap_angle(yaw - self.prev_yaw)
        middle_yaw = self.pose.yaw + delta_yaw / 2.0
        self.pose.x += distance * math.cos(middle_yaw)
        self.pose.y += distance * math.sin(middle_yaw)
        self.pose.yaw = wrap_angle(self.pose.yaw + delta_yaw)

        self.prev_left = left_encoder
        self.prev_right = right_encoder
        self.prev_yaw = yaw
        return self.pose

    def get_pose(self):
        return self.pose.copy()
