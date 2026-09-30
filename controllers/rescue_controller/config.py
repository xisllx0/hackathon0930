# controllers/rescue_controller/config.py
from pathlib import Path

# Webots 장치 이름
CAMERA = "camera"
LEFT_MOTOR = "left wheel motor"
RIGHT_MOTOR = "right wheel motor"

# 모터 속도 단위: rad/s
MAX_WHEEL_SPEED = 6.0

# 사과 탐지
TARGET_COUNT = 2
YOLO_MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "YOLO"
    / "yolo11n.pt"
)
YOLO_DEVICE = "cpu"
YOLO_CONFIDENCE = 0.1
YOLO_IOU = 0.5
YOLO_CLASSES = [47]  # 제공 코드 기준: apple


LIDAR = "LDS-01"
LEFT_ENCODER = "left wheel sensor"
RIGHT_ENCODER = "right wheel sensor"

# 임의로 설정햇음요 ㅜㅜ
MAX_WHEEL_SPEED = 6.0
WHEEL_RADIUS = 0.033
AXLE_LENGTH = 0.160
TARGET_COUNT = 2