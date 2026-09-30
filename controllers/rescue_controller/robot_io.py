# controllers/rescue_controller/robot_io.py
from controller import Robot

from config import (
    CAMERA,
    LIDAR,
    LEFT_MOTOR,
    RIGHT_MOTOR,
    LEFT_ENCODER,
    RIGHT_ENCODER,
    MAX_WHEEL_SPEED,
)


class RobotIO:
    def __init__(self):
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())

        # 장치 가져오기
        self.camera = self.robot.getDevice(CAMERA)
        self.lidar = self.robot.getDevice(LIDAR)
        self.left_motor = self.robot.getDevice(LEFT_MOTOR)
        self.right_motor = self.robot.getDevice(RIGHT_MOTOR)
        self.left_encoder = self.robot.getDevice(LEFT_ENCODER)
        self.right_encoder = self.robot.getDevice(RIGHT_ENCODER)

        # 센서 켜기
        self.camera.enable(self.timestep)
        self.lidar.enable(self.timestep)
        self.left_encoder.enable(self.timestep)
        self.right_encoder.enable(self.timestep)

        # 바퀴를 속도 제어 모드로 설정하고 정지
        self.left_motor.setPosition(float("inf"))
        self.right_motor.setPosition(float("inf"))
        self.wheels(0.0, 0.0)

    def step(self) -> bool:
        return self.robot.step(self.timestep) != -1

    def wheels(self, left: float, right: float) -> None:
        left = max(-MAX_WHEEL_SPEED, min(MAX_WHEEL_SPEED, left))
        right = max(-MAX_WHEEL_SPEED, min(MAX_WHEEL_SPEED, right))

        self.left_motor.setVelocity(left)
        self.right_motor.setVelocity(right)

    def encoder_positions(self) -> tuple[float, float]:
        return (
            self.left_encoder.getValue(),
            self.right_encoder.getValue(),
        )

    def lidar_ranges(self):
        return self.lidar.getRangeImage()

    def camera_image(self):
        return self.camera.getImage()

    def time(self) -> float:
        return self.robot.getTime()

    def stop(self) -> None:
        self.wheels(0.0, 0.0)