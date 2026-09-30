# controllers/rescue_controller/data_types.py
from dataclasses import dataclass


@dataclass
class Pose:
    x: float
    y: float
    theta: float


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    bearing: float                   # 정면 0, 왼쪽 +, 오른쪽 -
    red_ratio: float
    distance: float | None = None    # 검증된 거리만 기록


@dataclass
class Target:
    x: float
    y: float
    visited: bool = False