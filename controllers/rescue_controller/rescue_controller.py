# controllers/rescue_controller/rescue_controller.py
from controller import Robot
from controllers.rescue_controller.target_detection import TargetDetector

robot = Robot()
timestep = int(robot.getBasicTimeStep())

camera = robot.getDevice("camera")
camera.enable(timestep)

left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")

left_motor.setPosition(float("inf"))
right_motor.setPosition(float("inf"))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# YOLO 모델은 반복문 밖에서 한 번만 불러오기
detector = TargetDetector()

step_count = 0

while robot.step(timestep) != -1:
    step_count += 1

    # YOLO가 느릴 수 있으므로 10스텝마다 탐지
    if step_count % 10 == 0:
        detections = detector.detect_camera(camera)

        for target in detections:
            print(
                f"빨간 사과 발견: "
                f"방향={target.bearing:.2f} rad, "
                f"신뢰도={target.confidence:.2f}, "
                f"빨간색 비율={target.red_ratio:.2f}"
            )