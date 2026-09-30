class Pose:
    def __init__(self, x=0.0, y=0.0, yaw=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw

    @property
    def theta(self):
        """D 팀 drive.py의 pose.theta와 같은 값."""
        return self.yaw

    def copy(self):
        return Pose(self.x, self.y, self.yaw)
