from controller import Robot
import cv2
import numpy as np
import math

robot = Robot()
timestep = int(robot.getBasicTimeStep())

camera = robot.getDevice("camera")
camera.enable(timestep)

width = camera.getWidth()
height = camera.getHeight()
fov = camera.getFov()

while robot.step(timestep) != -1:
    image_bytes = camera.getImage()
    if image_bytes is None:
        continue

    frame_bgra = np.frombuffer(image_bytes, np.uint8).reshape(
        (height, width, 4)
    )
    frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)

    # 빨간색은 HSV 색상 범위의 양 끝에 걸쳐 있어서 두 범위를 합친다.
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    red_mask_1 = cv2.inRange(
        hsv,
        np.array([0, 80, 60], dtype=np.uint8),
        np.array([10, 255, 255], dtype=np.uint8),
    )
    red_mask_2 = cv2.inRange(
        hsv,
        np.array([170, 80, 60], dtype=np.uint8),
        np.array([179, 255, 255], dtype=np.uint8),
    )
    mask = cv2.bitwise_or(red_mask_1, red_mask_2)

    # 작은 잡음 제거, 끊어진 빨간 영역 연결
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    detections = []

    # max()를 쓰지 않고 모든 빨간 영역을 확인한다.
    for contour in contours:
        if cv2.contourArea(contour) < 80:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        center_x = x + w / 2
        center_y = y + h / 2

        # 화면 왼쪽이면 양수, 오른쪽이면 음수 (라디안)
        bearing = math.atan(
            ((width / 2 - center_x) / (width / 2))
            * math.tan(fov / 2)
        )

        detections.append({
            "bearing": bearing,
            "distance": None,  # 거리 확인 전에는 임의로 넣지 않음
            "center": (center_x, center_y),
            "bbox": (x, y, x + w, y + h),
        })

        cv2.rectangle(
            frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2
        )
        cv2.putText(
            frame_bgr,
            f"{bearing:+.2f} rad",
            (x, max(y - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    cv2.imshow("Webots Camera", frame_bgr)
    cv2.imshow("Red Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()