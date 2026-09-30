"""LDS-01 스캔 인덱스를 로봇 기준 방향으로 변환한다.

프로젝트의 tb3_lidar 예제: 180=앞, 90=왼쪽, 270=오른쪽.
반환 각도: 앞 0, 왼쪽 +pi/2, 오른쪽 -pi/2.
"""
import math


def beam_angle(index, count):
    if count <= 0:
        raise ValueError('LiDAR 측정 개수는 양수여야 합니다')
    return (count / 2.0 - index) * (2.0 * math.pi / count)
