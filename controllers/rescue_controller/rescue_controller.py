from controller import Robot
from robot_io import RobotIO
from localization import Localization
from mission import Mission

robot = Robot()
timestep = int(robot.getBasicTimeStep())

io = RobotIO(robot)
localizer = Localization()
mission = Mission()

home_saved = False

while robot.step(timestep) != -1:
    pose = localizer.update(
        io.get_left_encoder(),
        io.get_right_encoder(),
        io.get_yaw()
    )

    if not home_saved:
        mission.set_home(pose)
        home_saved = True
        print('HOME 저장')

    print(f'x={pose.x:.3f}, y={pose.y:.3f}, yaw={pose.yaw:.3f}')
