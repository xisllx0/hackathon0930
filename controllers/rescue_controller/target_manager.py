"""빨간 사과의 거리 확인, 중복 방지, 방문 기록.

Webots Robot이나 별도의 step 루프를 만들지 않는다.
메인 루프에서 YOLO 탐지 결과가 있을 때 observe()를 호출한다.
"""
from dataclasses import dataclass
import math
from statistics import median

from data_types import Target


@dataclass
class _Track:
    target: Target
    observations: int = 1


class TargetManager:
    def __init__(self, match_distance=0.6, visit_distance=0.4,
                 max_distance=3.2, min_observations=2):
        self._tracks = []
        self.match_distance = match_distance
        self.visit_distance = visit_distance
        self.max_distance = max_distance
        self.min_observations = min_observations

    @property
    def targets(self):
        return [track.target for track in self._tracks]

    def _lidar_distance(self, ranges, bearing):
        """프로젝트 LiDAR 규칙: 180 앞, 90 왼쪽, 270 오른쪽."""
        n = len(ranges)
        if n == 0:
            return None
        center = round(n / 2 - bearing * n / (2 * math.pi)) % n
        candidates = []
        for offset in range(-2, 3):
            value = ranges[(center + offset) % n]
            if value is not None and math.isfinite(value) and 0.12 < value <= self.max_distance:
                candidates.append(value)
        # 사과 방향에서 유효한 거리 샘플이 없으면 임의의 거리를 만들지 않는다.
        return float(median(candidates)) if len(candidates) >= 2 else None

    def observe(self, pose, detection, ranges):
        """Detection과 현재 LiDAR를 받아 사과를 등록한다. 등록 시 Target 반환."""
        if detection.class_name != 'red_apple':
            return None
        distance = detection.distance
        if distance is None:
            distance = self._lidar_distance(ranges, detection.bearing)
        if distance is None or not math.isfinite(distance):
            return None
        detection.distance = distance

        heading = pose.theta + detection.bearing
        x = pose.x + distance * math.cos(heading)
        y = pose.y + distance * math.sin(heading)
        closest = min(self._tracks,
                      key=lambda track: math.hypot(track.target.x-x, track.target.y-y),
                      default=None)
        if closest is not None and math.hypot(closest.target.x-x,
                                               closest.target.y-y) <= self.match_distance:
            if not closest.target.visited:
                count = min(closest.observations, 9)
                closest.target.x = (closest.target.x * count + x) / (count + 1)
                closest.target.y = (closest.target.y * count + y) / (count + 1)
            closest.observations += 1
            return closest.target

        # 목표는 두 개지만 오탐을 고려해 후보는 더 받을 수 있다.
        track = _Track(Target(x=x, y=y))
        self._tracks.append(track)
        return track.target

    def next_target(self, pose):
        candidates = [track.target for track in self._tracks
                      if not track.target.visited
                      and track.observations >= self.min_observations]
        return min(candidates,
                   key=lambda target: math.hypot(target.x-pose.x,
                                                 target.y-pose.y),
                   default=None)

    def check_visit(self, pose):
        """충분히 관측한 사과에 가까이 갔을 때에만 방문 처리."""
        for track in self._tracks:
            target = track.target
            if (not target.visited
                    and track.observations >= self.min_observations
                    and math.hypot(target.x-pose.x,
                                   target.y-pose.y) <= self.visit_distance):
                target.visited = True
                return target
        return None

    @property
    def visited_count(self):
        return sum(target.visited for target in self.targets)

    @property
    def complete(self):
        return self.visited_count >= 2
