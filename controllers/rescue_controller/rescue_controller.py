"""Webots 컨트롤러. 센서, 탐색, 방문, 복귀를 하나의 step 루프에서 실행."""
import math
from controller import Robot

from config import YOLO_EVERY_STEPS
from robot_io import RobotIO
from localization import Localization
from mission import Mission
from target_detection import TargetDetector
from target_manager import TargetManager
from static_mapping import StaticMapping
from navigation import Navigator
from exploration import Explorer
from global_planner import astar
from drive import toward, Sweep
from safety import SafetyMonitor


def main():
    robot = Robot()
    io = RobotIO(robot)
    detector = TargetDetector()
    localizer = Localization()
    mission = Mission()
    targets = TargetManager(visit_distance=0.42)
    mapping = StaticMapping()
    navigator = Navigator(mapping.resolution, mapping.origin_xy, clearance=0.23)
    explorer = Explorer()
    sweep = Sweep()
    safety = SafetyMonitor()

    if not io.step():
        return
    pose = localizer.update(*io.encoder_positions(), io.get_yaw())
    mission.set_home(pose)
    print('[MISSION] HOME 저장', mission.home_xy())

    path_xy = []
    goal_key = None
    need_replan = True
    sweeping = False
    step_count = 0
    last_state = None

    try:
        while io.step():
            pose = localizer.update(*io.encoder_positions(), io.get_yaw())
            ranges = io.lidar_ranges()
            mapping.update(pose, ranges)

            if step_count % YOLO_EVERY_STEPS == 0:
                for detection in detector.detect_camera(io.camera):
                    found = targets.observe(pose, detection, ranges)
                    if found is not None and step_count % 40 == 0:
                        print('[TARGET] 관측:', found)

            visited = targets.check_visit(pose)
            if visited is not None:
                print(f'[TARGET] 방문 {targets.visited_count}/2:', visited)
                need_replan = True

            state, goal_xy = mission.select(pose, targets)
            if state != last_state:
                print('[MISSION]', state)
                last_state = state
                need_replan = True
            if state == 'DONE':
                io.stop()
                print('[MISSION] 사과 두 곳 방문 후 출발점 복귀 완료')
                break

            if state != 'EXPLORE':
                sweeping = False
                key = (state, round(goal_xy[0], 1), round(goal_xy[1], 1))
            else:
                key = ('EXPLORE', explorer.goal)
            if key != goal_key:
                goal_key = key
                need_replan = True

            if sweeping and state == 'EXPLORE':
                left, right, finished = sweep.step(pose)
                if finished:
                    explorer.observe(mapping.world_to_grid(pose.x, pose.y))
                    explorer.fail()  # 다음 탐색 지점을 새로 고른다.
                    sweeping = False
                    need_replan = True
                    left = right = 0.0
            else:
                if need_replan or step_count % 12 == 0:
                    occ = mapping.get_occ()
                    if state == 'EXPLORE':
                        blocked = navigator.blocked_grid(occ)
                        start = navigator.world_to_cell(pose.x, pose.y)
                        if navigator._valid(blocked, start) and not blocked[start[0]][start[1]]:
                            frontier = explorer.pick(occ, blocked, start)
                            cells = astar(blocked, start, frontier) if frontier else []
                            path_xy = [navigator.cell_to_world(cell) for cell in cells]
                            if frontier is not None and not path_xy:
                                explorer.fail()
                        else:
                            path_xy = []
                    else:
                        approach = 0.30 if state == 'TARGET' else 0.15
                        path_xy, reason = navigator.to_goal(
                            occ, pose, goal_xy, approach=approach)
                        if not path_xy and step_count % 60 == 0:
                            print('[PATH] 경로 없음:', state, reason)
                    need_replan = False

                if path_xy:
                    left, right, arrived = toward(pose, path_xy)
                    if arrived and state == 'EXPLORE':
                        sweep.start(pose)
                        sweeping = True
                        left, right = 0.0, 0.0
                    elif arrived and state == 'TARGET':
                        # 목표 표면까지 충분히 접근하면 다음 루프에서 방문 여부 확인.
                        # 거리 측정 오차가 있으면 목표를 다시 관측할 수 있게 회전.
                        left, right = -0.7, 0.7
                    elif arrived:
                        left = right = 0.0
                else:
                    # 아직 탐색 경계가 없거나 경로가 막혔을 때 카메라 방향을 바꾼다.
                    left, right = -0.7, 0.7

            left, right, stopped, replan = safety.safe_wheels(ranges, left, right)
            if replan:
                need_replan = True
                if state == 'EXPLORE' and not sweeping:
                    explorer.fail()
            io.wheels(left, right)
            step_count += 1
    finally:
        io.stop()


if __name__ == '__main__':
    main()
