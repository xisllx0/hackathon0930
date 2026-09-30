class Pose:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

    def copy(self):
        p = Pose()
        p.x, p.y, p.yaw = self.x, self.y, self.yaw
        return p
