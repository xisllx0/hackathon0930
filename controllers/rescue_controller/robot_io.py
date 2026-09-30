import math

class RobotIO:
    def __init__(self, robot):
        timestep = int(robot.getBasicTimeStep())

        self.left_motor = robot.getDevice('left wheel motor')
        self.right_motor = robot.getDevice('right wheel motor')

        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))

        self.left_encoder = self.left_motor.getPositionSensor()
        self.right_encoder = self.right_motor.getPositionSensor()

        self.left_encoder.enable(timestep)
        self.right_encoder.enable(timestep)

        self.compass = robot.getDevice('compass')
        self.compass.enable(timestep)

    def get_left_encoder(self):
        return self.left_encoder.getValue()

    def get_right_encoder(self):
        return self.right_encoder.getValue()

    def get_yaw(self):
        v = self.compass.getValues()
        return math.atan2(v[0], v[1])
