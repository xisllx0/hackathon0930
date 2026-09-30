# controllers/rescue_controller/target_detection.py
from pathlib import Path
import math

import cv2
import numpy as np
from ultralytics import YOLO

from data_types import Detection


def camera_to_bgr(camera) -> np.ndarray | None:
    """Webots 카메라 이미지를 OpenCV의 BGR 이미지로 변환한다."""
    image_bytes = camera.getImage()
    if image_bytes is None:
        return None

    width = camera.getWidth()
    height = camera.getHeight()

    frame_bgra = np.frombuffer(
        image_bytes, dtype=np.uint8
    ).reshape((height, width, 4))

    return cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)


def get_red_ratio(
    frame_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> float:
    """사과 탐지 박스 안쪽에서 빨간 픽셀이 차지하는 비율."""
    image_height, image_width = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox

    box_width = x2 - x1
    box_height = y2 - y1
    if box_width <= 0 or box_height <= 0:
        return 0.0

    # 박스 가장자리의 배경을 제외하고 가운데 70%만 확인
    rx1 = max(0, x1 + int(box_width * 0.15))
    rx2 = min(image_width, x2 - int(box_width * 0.15))
    ry1 = max(0, y1 + int(box_height * 0.15))
    ry2 = min(image_height, y2 - int(box_height * 0.15))

    roi = frame_bgr[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return 0.0

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # OpenCV HSV에서는 빨강이 H=0과 H=179 부근에 나뉘어 있다.
    red_low = cv2.inRange(
        hsv,
        np.array([0, 80, 50], dtype=np.uint8),
        np.array([10, 255, 255], dtype=np.uint8),
    )
    red_high = cv2.inRange(
        hsv,
        np.array([170, 80, 50], dtype=np.uint8),
        np.array([179, 255, 255], dtype=np.uint8),
    )

    red_mask = cv2.bitwise_or(red_low, red_high)
    red_pixels = cv2.countNonZero(red_mask)
    total_pixels = roi.shape[0] * roi.shape[1]

    return red_pixels / total_pixels


class TargetDetector:
    def __init__(
        self,
        confidence_threshold: float = 0.25,
        red_threshold: float = 0.15,
    ):
        # 프로젝트/models/YOLO/yolo11n.pt
        project_root = Path(__file__).resolve().parents[2]
        weights_path = project_root / "models" / "YOLO" / "yolo11n.pt"

        if weights_path.exists():
            self.model = YOLO(str(weights_path))
        else:
            # 로컬 파일이 없으면 Ultralytics가 가중치를 받도록 시도
            self.model = YOLO("yolo11n.pt")

        self.confidence_threshold = confidence_threshold
        self.red_threshold = red_threshold

    def detect_bgr(
        self,
        frame_bgr: np.ndarray | None,
        camera_fov: float,
    ) -> list[Detection]:
        """이미지 한 장에서 빨간 사과 후보를 모두 찾는다."""
        if frame_bgr is None:
            return []

        height, width = frame_bgr.shape[:2]
        if width == 0 or height == 0:
            return []

        focal_pixels = width / (2.0 * math.tan(camera_fov / 2.0))

        # COCO 47번 = apple. 색깔은 아래에서 따로 확인한다.
        results = self.model.predict(
            source=frame_bgr,
            conf=self.confidence_threshold,
            iou=0.5,
            classes=[47],
            verbose=False,
        )

        detections: list[Detection] = []

        # max()로 하나만 고르지 않고, 찾은 사과를 전부 검사한다.
        for box in results[0].boxes:
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            bbox = (x1, y1, x2, y2)

            red_ratio = get_red_ratio(frame_bgr, bbox)
            if red_ratio < self.red_threshold:
                continue

            center_x = (x1 + x2) / 2.0

            # 왼쪽에 보이면 bearing 양수, 오른쪽이면 음수
            bearing = math.atan(
                (width / 2.0 - center_x) / focal_pixels
            )

            detections.append(
                Detection(
                    class_name="red_apple",
                    confidence=confidence,
                    bbox=bbox,
                    bearing=bearing,
                    red_ratio=red_ratio,
                    distance=None,
                )
            )

        return detections

    def detect_camera(self, camera) -> list[Detection]:
        """Webots 카메라에서 현재 화면의 빨간 사과 후보를 반환한다."""
        frame_bgr = camera_to_bgr(camera)
        return self.detect_bgr(frame_bgr, camera.getFov())