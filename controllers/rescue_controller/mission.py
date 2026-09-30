class Mission:
    def __init__(self):
        self.home_pose = None

    def set_home(self, pose):
        self.home_pose = pose.copy()
