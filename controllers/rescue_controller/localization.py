import math
from pose import Pose

class Localization:
    def __init__(self):
        self.pose = Pose()
        self.prev_left = None
        self.prev_right = None
        self.WHEEL_RADIUS = 0.033
        self.WHEEL_BASE = 0.177

    def update(self, left_encoder, right_encoder, yaw):
        if self.prev_left is None:
            self.prev_left = left_encoder
            self.prev_right = right_encoder
            self.pose.yaw = yaw
            return self.pose

        dl = left_encoder - self.prev_left
        dr = right_encoder - self.prev_right

        self.prev_left = left_encoder
        self.prev_right = right_encoder

        distance = ((dl + dr) / 2) * self.WHEEL_RADIUS

        self.pose.x += distance * math.cos(yaw)
        self.pose.y += distance * math.sin(yaw)
        self.pose.yaw = yaw

        return self.pose

    def get_pose(self):
        return self.pose
