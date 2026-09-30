# controllers/rescue_controller/target_detection.py

from pathlib import Path
import math

import cv2
import numpy as np
from ultralytics import YOLO

from data_types import Detection


def camera_to_bgr(camera) -> np.ndarray | None:
    """Webots 카메라의 현재 이미지를 OpenCV용 BGR 이미지로 변환한다."""
    image_bytes = camera.getImage()
    if image_bytes is None:
        return None

    width = camera.getWidth()
    height = camera.getHeight()

    frame_bgra = np.frombuffer(
        image_bytes,
        dtype=np.uint8,
    ).reshape((height, width, 4))

    return cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)


def make_red_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """
    빨간 픽셀은 255(흰색), 나머지는 0(검은색)인
    이진 마스크를 만든다.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # HSV에서는 빨강이 색상 범위의 양 끝에 나뉘어 있다.
    red_mask_low = cv2.inRange(
        hsv,
        np.array([0, 80, 60], dtype=np.uint8),
        np.array([10, 255, 255], dtype=np.uint8),
    )

    red_mask_high = cv2.inRange(
        hsv,
        np.array([170, 80, 60], dtype=np.uint8),
        np.array([179, 255, 255], dtype=np.uint8),
    )

    mask = cv2.bitwise_or(red_mask_low, red_mask_high)

    # 작은 빨간 점 형태의 잡음 제거.
    # 사과가 화면에 너무 작게 보이면 이 처리가 사과까지 지울 수 있으므로
    # 그때는 kernel을 (2, 2)로 줄이거나 이 두 줄을 제거한다.
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def get_red_ratio(
    red_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> float:
    """YOLO가 찾은 사과 박스의 가운데에서 빨간색 비율을 계산한다."""
    image_height, image_width = red_mask.shape
    x1, y1, x2, y2 = bbox

    box_width = x2 - x1
    box_height = y2 - y1

    if box_width <= 0 or box_height <= 0:
        return 0.0

    # 박스 가장자리에 들어온 배경을 줄이기 위해 가운데 70%만 검사.
    rx1 = max(0, x1 + int(box_width * 0.15))
    ry1 = max(0, y1 + int(box_height * 0.15))
    rx2 = min(image_width, x2 - int(box_width * 0.15))
    ry2 = min(image_height, y2 - int(box_height * 0.15))

    roi = red_mask[ry1:ry2, rx1:rx2]

    if roi.size == 0:
        return 0.0

    red_pixels = cv2.countNonZero(roi)
    return red_pixels / roi.size


class TargetDetector:
    """현재 카메라 화면에서 빨간 사과 후보를 찾는다."""

    def __init__(
        self,
        confidence_threshold: float = 0.25,
        red_threshold: float = 0.15,
    ):
        # target_detection.py가
        # controllers/rescue_controller/target_detection.py에 있다는 전제.
        weights_path = Path(__file__).resolve().with_name("yolo11n.pt")
        self.model = YOLO(str(weights_path))

        self.confidence_threshold = confidence_threshold
        self.red_threshold = red_threshold

        # 모델에서 apple의 클래스 번호를 찾는다.
        # 기본 COCO 모델에서는 47이지만, 번호를 코드에 고정하지 않는다.
        self.apple_class_id = None

        for class_id, class_name in self.model.names.items():
            if class_name == "apple":
                self.apple_class_id = int(class_id)
                break

        if self.apple_class_id is None:
            raise ValueError(
                "YOLO 모델에 'apple' 클래스가 없습니다. "
                "사용 중인 가중치의 클래스 목록을 확인하세요."
            )

        # 화면 확인이 필요할 때 사용할 수 있는 마지막 이진 마스크.
        self.last_red_mask: np.ndarray | None = None

    def detect_bgr(
        self,
        frame_bgr: np.ndarray | None,
        camera_fov: float,
    ) -> list[Detection]:
        """이미지 한 장에서 빨간 사과 후보를 모두 찾는다."""
        self.last_red_mask = None

        if frame_bgr is None or frame_bgr.size == 0:
            return []

        height, width = frame_bgr.shape[:2]

        if width == 0 or height == 0:
            return []

        if not 0.0 < camera_fov < math.pi:
            raise ValueError(
                f"카메라 FOV 값이 올바르지 않습니다: {camera_fov}"
            )

        # 화면에서 빨간 부분을 표시한 이진 마스크.
        red_mask = make_red_mask(frame_bgr)
        self.last_red_mask = red_mask

        # 화면의 가로 좌표를 로봇 기준 각도로 바꾸는 데 사용.
        focal_pixels = width / (
            2.0 * math.tan(camera_fov / 2.0)
        )

        # YOLO로 사과 후보의 박스를 찾는다.
        results = self.model.predict(
            source=frame_bgr,
            conf=self.confidence_threshold,
            iou=0.5,
            classes=[self.apple_class_id],
            verbose=False,
        )

        detections: list[Detection] = []

        for box in results[0].boxes:
            confidence = float(box.conf[0].item())
            raw_x1, raw_y1, raw_x2, raw_y2 = (
                box.xyxy[0].tolist()
            )

            # 박스가 이미지 경계를 넘어가지 않도록 제한.
            x1 = max(0, min(width, int(raw_x1)))
            y1 = max(0, min(height, int(raw_y1)))
            x2 = max(0, min(width, int(raw_x2)))
            y2 = max(0, min(height, int(raw_y2)))

            if x2 <= x1 or y2 <= y1:
                continue

            bbox = (x1, y1, x2, y2)

            # YOLO가 찾은 사과가 실제로 빨간지 확인.
            red_ratio = get_red_ratio(red_mask, bbox)

            if red_ratio < self.red_threshold:
                continue

            center_x = (x1 + x2) / 2.0

            # 왼쪽: 양수 / 정면: 0 근처 / 오른쪽: 음수.
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
        """Webots 카메라의 현재 화면에서 빨간 사과를 찾는다."""
        frame_bgr = camera_to_bgr(camera)

        return self.detect_bgr(
            frame_bgr,
            camera.getFov(),
        )