from robot_io import RobotIO
from localization import Odometry
from target_detection import TargetDetector


def main():
    io = RobotIO()
    detector = TargetDetector()  

    # 센서의 첫 측정값 확보
    if not io.step():
        return

    odometry = Odometry(*io.encoder_positions())
    step_count = 0

    try:
        while io.step():  # 프로젝트 전체에서 사용하는 메인 루프 하나
            # 1. 현재 위치 갱신
            pose = odometry.update(*io.encoder_positions())

            # 2. 최신 LiDAR 측정
            ranges = io.lidar_ranges()

            # 3. 사과 탐지: YOLO는 무거우므로 5스텝마다 실행
            detections = []
            if step_count % 5 == 0:
                detections = detector.detect_camera(io.camera)

            # 4. 앞으로 A의 지도 갱신 연결
            # grid.update(pose, ranges)

            # 5. 앞으로 C의 목표 위치·방문 목록 연결
            # for detection in detections:
            #     target_manager.observe(pose, detection)

            # 6. 앞으로 B의 미션과 D의 경로·주행 연결
            # goal = mission.choose_goal(...)
            # path = planner.plan(...)
            # left_speed, right_speed = drive.follow(pose, path)
            # left_speed, right_speed = safety.check(ranges, left_speed, right_speed)
            # io.wheels(left_speed, right_speed)

            # 주행 코드가 연결되기 전까지는 정지
            io.stop()

            if detections:
                print("현재 위치:", pose)
                print("발견한 빨간 사과 후보:", detections)

            step_count += 1
    finally:
        io.stop()


if __name__ == "__main__":
    main()