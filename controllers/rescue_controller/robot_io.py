import math
from config import (CAMERA, COMPASS, LIDAR, LEFT_MOTOR, RIGHT_MOTOR,
                    MAX_WHEEL_SPEED)


class RobotIO:
    def __init__(self, robot):
        self.robot = robot
        self.timestep = int(robot.getBasicTimeStep())
        self.left_motor = robot.getDevice(LEFT_MOTOR)
        self.right_motor = robot.getDevice(RIGHT_MOTOR)
        self.camera = robot.getDevice(CAMERA)
        self.lidar = robot.getDevice(LIDAR)
        self.compass = robot.getDevice(COMPASS)
        self.left_encoder = self.left_motor.getPositionSensor()
        self.right_encoder = self.right_motor.getPositionSensor()
        for sensor in (self.left_encoder, self.right_encoder,
                       self.camera, self.lidar, self.compass):
            sensor.enable(self.timestep)
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.stop()

    def step(self):
        return self.robot.step(self.timestep) != -1

    def time(self):
        return self.robot.getTime()

    def get_left_encoder(self):
        return self.left_encoder.getValue()

    def get_right_encoder(self):
        return self.right_encoder.getValue()

    def encoder_positions(self):
        return self.get_left_encoder(), self.get_right_encoder()

    def get_yaw(self):
        values = self.compass.getValues()
        return math.atan2(values[0], values[1])

    def lidar_ranges(self):
        return self.lidar.getRangeImage()

    def wheels(self, left, right):
        # 속도 비율을 유지하며 두 바퀴 모두 한계 내로 축소한다.
        peak = max(abs(left), abs(right))
        scale = max(1.0, peak / MAX_WHEEL_SPEED)
        self.left_motor.setVelocity(left / scale)
        self.right_motor.setVelocity(right / scale)

    def stop(self):
        self.wheels(0.0, 0.0)
