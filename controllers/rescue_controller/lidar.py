"""통합용 단일 Webots 루프. 지도/목표 코드가 연결되기 전에는 정지."""
from controller import Robot
from robot_io import RobotIO
from localization import Localization
from mission import Mission
from target_detection import TargetDetector
from config import YOLO_EVERY_STEPS


def main():
    robot = Robot()
    io = RobotIO(robot)
    detector = TargetDetector()
    localizer = Localization()
    mission = Mission()

    if not io.step():
        return
    # 첫 센서 샘플을 기준값으로 잡는다.
    pose = localizer.update(*io.encoder_positions(), io.get_yaw())
    mission.set_home(pose)
    print('HOME 저장', mission.home_xy())
    step_count = 0
    try:
        while io.step():
            pose = localizer.update(*io.encoder_positions(), io.get_yaw())
            ranges = io.lidar_ranges()
            detections = []
            if step_count % YOLO_EVERY_STEPS == 0:
                detections = detector.detect_camera(io.camera)

            # 이후 순서: 지도 갱신 -> 사과 거리 확인/목표 관리 -> 목표 선택
            # -> navigation.py 경로 -> drive.toward -> safety.safe_wheels -> io.wheels
            # 아직 지도/목표 연결 전이므로 모터는 정지시킨다.
            io.stop()
            if detections:
                print('Pose:', pose.x, pose.y, pose.yaw,
                      '빨간 사과 후보:', detections)
            step_count += 1
    finally:
        io.stop()


if __name__ == '__main__':
    main()
