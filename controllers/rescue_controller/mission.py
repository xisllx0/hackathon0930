class Mission:
    def __init__(self):
        self.home_pose = None

    def set_home(self, pose):
        if self.home_pose is None:
            self.home_pose = pose.copy()

    def home_xy(self):
        if self.home_pose is None:
            raise RuntimeError('출발 위치가 아직 저장되지 않았습니다')
        return self.home_pose.x, self.home_pose.y
