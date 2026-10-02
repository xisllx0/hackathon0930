"""D-role integration test on the real Webots robot (temporary, not for submission).

Stand-ins for the other roles:
  * pose  <- Supervisor ground truth      (B's localization later)
  * map   <- tiny LiDAR mapper below      (A's mapping later)
  * goals <- coordinates below            (C's detection later)
Everything the D role owns (planner, drive, safety, exploration) is the real code.

Phases: 0 LiDAR direction check, 1 (1 m, 0) round trip, 2 frontier exploration,
3 visit each red apple and return to the start.
"""
import json
import math
import os
import sys
import time

import numpy as np
from controller import Supervisor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "rescue_controller"))
from global_planner import plan_to_target, plan_progressive, path_is_blocked, inflate  # noqa: E402
from drive import toward, Sweep  # noqa: E402
from safety import SafetyMonitor  # noqa: E402
from exploration import Explorer, FREE, UNKNOWN, OCCUPIED  # noqa: E402

APPLES = [(-5.34, -10.54), (-12.02, -3.02)]   # test-only ground truth of the two red apples
RES = 0.05
X0, Y0 = -14.0, -15.0
W, H = 320, 420
INFLATE_CELLS = 3                              # 3 cells x 0.05 m = 0.15 m from every wall
MAP_EVERY = 4                                  # steps between LiDAR map updates
EXPLORE_TIME = float(os.environ.get("D_EXPLORE_T", 1500))
LEG_TIME = float(os.environ.get("D_LEG_T", 500))

robot = Supervisor()
dt = int(robot.getBasicTimeStep())
node = robot.getSelf()
lm = robot.getDevice("left wheel motor")
rm = robot.getDevice("right wheel motor")
lm.setPosition(float("inf"))
rm.setPosition(float("inf"))
lm.setVelocity(0.0)
rm.setVelocity(0.0)
lidar = robot.getDevice("LDS-01")
lidar.enable(dt)
camera = robot.getDevice("camera")
camera.enable(dt)
CAM_FOV = camera.getFov()
apple_seen = {}          # apple index -> (sim time, distance, bearing error)
false_hits = 0
found = []               # apple targets the robot itself estimated: {"x","y","n","visited"}
det_stats = {"detections": 0, "false_positive": 0, "seen_true": set()}
detector = None
if os.environ.get("D_DETECTOR", "1") == "1":
    try:
        from target_detection import TargetDetector  # teammates' YOLO + red filter
        detector = TargetDetector()
        print("DT: using teammates' TargetDetector (YOLO)", flush=True)
    except Exception as ex_:                            # cv2/ultralytics missing etc.
        print("DT: TargetDetector unavailable, using simple red-pixel check:", repr(ex_), flush=True)
DET_EVERY = 8
APPLE_DIAMETER = 0.05

L = np.zeros((H, W), np.int16)      # log-odds
seen = np.zeros((H, W), bool)
viewed = np.zeros((H, W), bool)     # cells the camera could have looked at (FOV cone, LOS, <= VIEW_RANGE)
STEPS = np.arange(0.12, 3.4, 0.04)
VIEW_RANGE = 2.0                   # m: an apple (5 cm) is only reliably seen this close
IDX = np.arange(360)

traj = []
OBSERVER = None
stats = {"stops": 0, "replans": 0, "stuck": 0, "near_collision": 0, "path_len_m": 0.0}
log = []
sign = 1


class Pose:
    __slots__ = ("x", "y", "theta")

    def __init__(self, x, y, theta):
        self.x, self.y, self.theta = x, y, theta


def say(*a):
    msg = " ".join(str(x) for x in a)
    print("DT:", msg, flush=True)
    log.append(msg)
    with open(os.path.join(HERE, "d_test_log.txt"), "a", encoding="utf-8") as f:
        f.write("%.1f %s" % (robot.getTime(), msg) + chr(10))


def gt_pose():
    p = node.getPosition()
    o = node.getOrientation()
    return Pose(p[0], p[1], math.atan2(o[3], o[0]))


def cell_of(x, y):
    return int((y - Y0) / RES), int((x - X0) / RES)


def xy_of(cell):
    return X0 + (cell[1] + 0.5) * RES, Y0 + (cell[0] + 0.5) * RES


def update_map(pose, ranges):
    r = np.asarray(ranges, dtype=np.float32)
    hit = np.isfinite(r) & (r < 3.45) & (r > 0.12)
    rr = np.where(hit, r, 3.4)
    ang = pose.theta + sign * np.deg2rad(IDX - 180)
    ca, sa = np.cos(ang), np.sin(ang)
    lim = np.where(hit, rr - 0.08, 3.4)
    m = STEPS[None, :] <= lim[:, None]
    fx = pose.x + STEPS[None, :] * ca[:, None]
    fy = pose.y + STEPS[None, :] * sa[:, None]
    c = ((fx - X0) / RES).astype(int)
    row = ((fy - Y0) / RES).astype(int)
    ok = m & (c >= 0) & (c < W) & (row >= 0) & (row < H)
    cone = np.abs(sign * (IDX - 180)) <= 30
    okv = ok & cone[:, None] & (STEPS[None, :] <= VIEW_RANGE)
    viewed[row[okv], c[okv]] = True
    flat = np.unique(row[ok] * W + c[ok])
    L.reshape(-1)[flat] = np.maximum(L.reshape(-1)[flat] - 1, -6)
    seen.reshape(-1)[flat] = True
    hx = pose.x + rr[hit] * ca[hit]
    hy = pose.y + rr[hit] * sa[hit]
    hc = ((hx - X0) / RES).astype(int)
    hr = ((hy - Y0) / RES).astype(int)
    ok = (hc >= 0) & (hc < W) & (hr >= 0) & (hr < H)
    hflat = np.unique(hr[ok] * W + hc[ok])
    L.reshape(-1)[hflat] = np.minimum(L.reshape(-1)[hflat] + 3, 8)
    seen.reshape(-1)[hflat] = True


def occ_grid():
    o = np.full((H, W), UNKNOWN, np.int8)
    o[seen & (L <= 0)] = FREE
    o[L >= 3] = OCCUPIED
    # LiDAR rays fan out (1 deg apart): beyond ~3 m they leave 1-2 cell gaps that would
    # look like frontiers. Close gaps that have free cells on both sides.
    for _ in range(2):
        fr = o == FREE
        un = o == UNKNOWN
        gap = np.zeros_like(fr)
        gap[:, 1:-1] |= fr[:, :-2] & fr[:, 2:]
        gap[1:-1, :] |= fr[:-2, :] & fr[2:, :]
        o[un & gap] = FREE
    return o


def blocked_grids(occ, pose, rad=2):
    walls = inflate(occ == OCCUPIED, rad)
    r, c = cell_of(pose.x, pose.y)
    walls[max(0, r - 4):r + 5, max(0, c - 4):c + 5] = False       # never trap ourselves
    known_only = walls | (occ != FREE)
    known_only[max(0, r - 4):r + 5, max(0, c - 4):c + 5] = False
    return walls, known_only


t_start_wall = time.time()


def tick(state):
    """Advance one step; refresh the map now and then. Returns (pose, ranges)."""
    if robot.step(dt) == -1:
        raise SystemExit
    state["n"] += 1
    pose = gt_pose()
    ranges = lidar.getRangeImage()
    if state["n"] % MAP_EVERY == 0:
        update_map(pose, ranges)
    if state["n"] % 8 == 0:
        traj.append((pose.x, pose.y))
    if state["n"] % DET_EVERY == 0:
        if detector is not None:
            process_detections(pose)
        else:
            process_oracle(pose, ranges)
    if OBSERVER is not None and state["n"] % 16 == 0:
        OBSERVER()
    return pose, ranges


def dump(tag, goal=None):
    p = gt_pose()
    np.savez(os.path.join(HERE, "dump_%s.npz" % tag), occ=occ_grid(), traj=np.array(traj),
             pose=np.array([p.x, p.y, p.theta]),
             goal=np.array(goal if goal is not None else [np.nan, np.nan]))


def do_sweep(state):
    """Turn once around in place so the camera sees every direction."""
    sw = Sweep()
    pose, _ = tick(state)
    sw.start(pose)
    for _ in range(900):
        pose, _ = tick(state)
        l, r, done = sw.step(pose)
        if done:
            break
        lm.setVelocity(l)
        rm.setVelocity(r)
    stop_motors()


def sync_viewed(ex):
    mask = viewed.reshape(H // 5, 5, W // 5, 5).any(axis=(1, 3))
    ex.add_viewed_blocks(np.argwhere(mask).tolist())


def red_detect():
    """numpy-only stand-in for C's detector: red pixels -> (pixel count, bearing rad, left +)."""
    w, h = camera.getWidth(), camera.getHeight()
    img = camera.getImage()
    if img is None:
        return 0, 0.0
    a = np.frombuffer(img, np.uint8).reshape(h, w, 4)
    b, g, r = a[..., 0].astype(np.int16), a[..., 1].astype(np.int16), a[..., 2].astype(np.int16)
    mask = (r > 120) & (g < 60) & (b < 60)
    n = int(mask.sum())
    if n < 12:
        return 0, 0.0
    cx = np.nonzero(mask)[1].mean()
    return n, (w / 2 - cx) / (w / 2) * (CAM_FOV / 2)


def check_camera(pose):
    global false_hits
    n, bearing = red_detect()
    if n == 0:
        return
    best = None
    for i, (ax, ay) in enumerate(APPLES):
        dist = math.hypot(ax - pose.x, ay - pose.y)
        want = math.atan2(ay - pose.y, ax - pose.x) - pose.theta
        want = (want + math.pi) % (2 * math.pi) - math.pi
        err = abs(want - bearing)
        if dist < 5.0 and err < 0.12 and (best is None or dist < best[1]):
            best = (i, dist, err)
    if best is None:
        false_hits += 1
        return
    i, dist, err = best
    if i not in apple_seen:
        apple_seen[i] = (round(robot.getTime(), 1), round(dist, 2), round(err, 3))
        say("  CAMERA sees apple", i + 1, "at", round(dist, 2), "m, time", round(robot.getTime(), 1))


def recover(state, safety):
    """Back up a little, then turn ~70 degrees toward the more open side."""
    for _ in range(25):
        pose, ranges = tick(state)
        if safety.rear_min(ranges) < 0.2:
            break
        lm.setVelocity(-1.5)
        rm.setVelocity(-1.5)
    stop_motors()
    pose, ranges = tick(state)
    left_open = SafetyMonitor._side_open(ranges, (60, 120))
    right_open = SafetyMonitor._side_open(ranges, (240, 300))
    spin_to(state, pose.theta + (1 if left_open >= right_open else -1) * math.radians(70))


def leg(state, safety, goal_cell, radius, timeout, known_only=False, tries=4):
    """drive_to with recovery: when stuck, back off/turn and try again."""
    st = "timeout"
    for _ in range(tries):
        st = drive_to(state, safety, goal_cell, known_only, radius, timeout)
        if st in ("stuck", "no_path:no_path"):
            recover(state, safety)
            continue
        break
    return st


class _Det:
    def __init__(self, bbox, bearing, confidence):
        self.bbox, self.bearing, self.confidence = bbox, bearing, confidence


def process_red_blob(pose):
    """Simple stand-in detector (no cv2/YOLO): biggest red blob -> bbox width + bearing."""
    w, h = camera.getWidth(), camera.getHeight()
    img = camera.getImage()
    if img is None:
        return
    a = np.frombuffer(img, np.uint8).reshape(h, w, 4)
    b, g, r = a[..., 0].astype(np.int16), a[..., 1].astype(np.int16), a[..., 2].astype(np.int16)
    mask = (r > 120) & (g < 60) & (b < 60)
    if int(mask.sum()) < 12:
        return
    cols = np.nonzero(mask.any(axis=0))[0]
    c0, c1 = int(cols.min()), int(cols.max())
    bearing = (w / 2 - (c0 + c1) / 2) / (w / 2) * (CAM_FOV / 2)
    _register(pose, [_Det((c0, 0, c1 + 1, 0), bearing, 1.0)])


def process_oracle(pose, ranges):
    """Stand-in for a perfect camera (tests D only): a real apple inside the 60 deg field of view,
    within 3.5 m and not hidden behind a wall (LiDAR line of sight) is 'detected'."""
    f_px = camera.getWidth() / (2 * math.tan(CAM_FOV / 2))
    dets = []
    for ax, ay in APPLES:
        dx, dy = ax - pose.x, ay - pose.y
        dist = math.hypot(dx, dy)
        rel = (math.atan2(dy, dx) - pose.theta + math.pi) % (2 * math.pi) - math.pi
        if dist > 3.5 or abs(rel) > CAM_FOV / 2 - 0.05:
            continue
        r = ranges[int(round(180 - math.degrees(rel))) % 360]     # index runs clockwise, 180 = front
        if r == r and r < dist - 0.10:
            continue                                               # a wall is in the way
        wpx = max(2, int(APPLE_DIAMETER * f_px / dist))
        cx = camera.getWidth() / 2 - rel / (CAM_FOV / 2) * camera.getWidth() / 2
        dets.append(_Det((int(cx - wpx / 2), 0, int(cx + wpx / 2), 0), rel, 1.0))
    if dets:
        _register(pose, dets)


def process_detections(pose):
    """Run the teammates' detector and turn each hit into a world-frame apple estimate."""
    dets = detector.detect_camera(camera)
    if not dets:
        return
    _register(pose, dets)


def _register(pose, dets):
    f_px = camera.getWidth() / (2 * math.tan(CAM_FOV / 2))
    for det in dets:
        x1, y1, x2, y2 = det.bbox
        wpx = max(1, x2 - x1)
        dist = APPLE_DIAMETER * f_px / wpx                  # pinhole range from the box width
        if not (0.15 < dist < 6.0):
            continue
        ang = pose.theta + det.bearing
        ex_, ey_ = pose.x + dist * math.cos(ang), pose.y + dist * math.sin(ang)
        det_stats["detections"] += 1
        # score against the real apples (the robot never sees these numbers)
        errs = [math.hypot(ex_ - ax, ey_ - ay) for ax, ay in APPLES]
        k = int(np.argmin(errs))
        if errs[k] < 1.0:
            det_stats["seen_true"].add(k + 1)
        else:
            det_stats["false_positive"] += 1
        for f in found:                                     # merge with a known target
            if math.hypot(f["x"] - ex_, f["y"] - ey_) < 0.8:
                f["x"] = (f["x"] * f["n"] + ex_) / (f["n"] + 1)
                f["y"] = (f["y"] * f["n"] + ey_) / (f["n"] + 1)
                f["n"] += 1
                break
        else:
            found.append({"x": ex_, "y": ey_, "n": 1, "visited": False, "t": robot.getTime()})
            say("  DETECTED new apple estimate", (round(ex_, 2), round(ey_, 2)), "range", round(dist, 2),
                "m conf", round(det.confidence, 2), "| nearest true apple err", round(errs[k], 2), "m")


def stop_motors():
    lm.setVelocity(0.0)
    rm.setVelocity(0.0)


def spin_to(state, target_theta, tol=0.05):
    for _ in range(400):
        pose, _ = tick(state)
        err = (target_theta - pose.theta + math.pi) % (2 * math.pi) - math.pi
        if abs(err) < tol:
            break
        w = 1.5 if err > 0 else -1.5
        lm.setVelocity(-w)
        rm.setVelocity(w)
    stop_motors()


def drive_to(state, safety, goal_cell, known_only, radius, timeout, still_wanted=None):
    """Plan to goal_cell and follow it. Returns a status string."""
    t0 = robot.getTime()
    path = None
    path_rad = INFLATE_CELLS
    path_xy = None
    last_replan = -99.0
    check_t = robot.getTime()
    check_p = gt_pose()
    last_xy = None
    while True:
        pose, ranges = tick(state)
        now = robot.getTime()
        if now - t0 > timeout:
            stop_motors()
            return "timeout"
        gx, gy = xy_of(goal_cell)
        if math.hypot(gx - pose.x, gy - pose.y) <= radius:
            stop_motors()
            return "arrived"
        if still_wanted is not None and state["n"] % 8 == 0 and not still_wanted():
            stop_motors()
            return "resolved"
        # stuck detection: barely moved over 10 s
        if now - check_t >= 10.0:
            if math.hypot(pose.x - check_p.x, pose.y - check_p.y) < 0.15:
                stats["stuck"] += 1
                stop_motors()
                say("  stuck at", round(pose.x, 2), round(pose.y, 2), "goal", xy_of(goal_cell),
                    "front_min", round(safety.front_min(ranges), 2), "stopped", safety.stopped)
                return "stuck"
            check_t, check_p = now, pose
        need = path is None or (now - last_replan > 2.0)
        if need:
            occ = occ_grid()
            raw = occ == OCCUPIED
            extra = (occ != FREE) if known_only else None
            rc = cell_of(pose.x, pose.y)
            replan_now = path is None or safety.replan
            if not replan_now:
                bb = inflate(raw, path_rad)
                if extra is not None:
                    bb = bb | extra
                n_ = path_rad + 1
                bb[max(0, rc[0] - n_):rc[0] + n_ + 1, max(0, rc[1] - n_):rc[1] + n_ + 1] = False
                replan_now = path_is_blocked(bb.tolist(), path[state_idx(path, pose):])
            if replan_now:
                path, why, used = plan_progressive(raw, rc, goal_cell, (3, 2), extra, 14)
                last_replan = now
                stats["replans"] += 1
                if not path:
                    stop_motors()
                    say("  no_path at", round(pose.x, 2), round(pose.y, 2), "goal", xy_of(goal_cell), why)
                    return "no_path:" + why
                path_rad = used
                path_xy = [xy_of(c) for c in path]
                safety.replan = False
            else:
                last_replan = now
        left, right, arrived = toward(pose, path_xy)
        left, right, stopped, replan = safety.safe_wheels(ranges, left, right)
        if stopped:
            stats["stops"] += 1
        if replan:
            path = None      # force a fresh plan next step (obstacle just appeared / mapped)
        front = safety.front_min(ranges)
        if front < 0.14:
            stats["near_collision"] += 1
        lm.setVelocity(left)
        rm.setVelocity(right)
        if last_xy is not None:
            stats["path_len_m"] += math.hypot(pose.x - last_xy[0], pose.y - last_xy[1])
        last_xy = (pose.x, pose.y)


def state_idx(path, pose):
    rc = cell_of(pose.x, pose.y)
    best, bd = 0, 1e9
    for i, c in enumerate(path):
        d = abs(c[0] - rc[0]) + abs(c[1] - rc[1])
        if d < bd:
            best, bd = i, d
    return best


def coverage():
    return int(((seen & (L <= 0)).sum()))


def mission_run(state, safety, start, start_cell, result):
    """Explore until an apple is spotted -> visit it -> keep exploring until both are visited -> go home."""
    global OBSERVER
    ex = Explorer(min_cluster=6, skip_radius=6, block=5, see_radius=5)
    OBSERVER = lambda: sync_viewed(ex)
    t0 = robot.getTime()
    visits = []
    MISSION_TIME = float(os.environ.get("D_MISSION_T", 1500))
    while robot.getTime() - t0 < MISSION_TIME and len(visits) < len(APPLES):
        target = next((f for f in found if not f["visited"]), None)
        if target is not None:
            tx, ty = target["x"], target["y"]
            st = leg(state, safety, cell_of(tx, ty), 0.35, 250)
            target["visited"] = True
            p = gt_pose()
            true_d = min(math.hypot(p.x - ax, p.y - ay) for ax, ay in APPLES)
            visits.append({"status": st, "dist_to_nearest_true_apple_m": round(true_d, 2), "t": round(robot.getTime(), 1)})
            say("  VISIT", len(visits), st, "| robot to nearest true apple", round(true_d, 2), "m",
                "| est", (round(tx, 2), round(ty, 2)))
            do_sweep(state)
            continue
        pose, _ = tick(state)
        occ = occ_grid()
        walls, kn = blocked_grids(occ, pose, 2)
        goal = ex.pick(occ, kn, cell_of(pose.x, pose.y))
        if goal is None:
            say("  exploration exhausted")
            break
        mode = ex.mode
        st = drive_to(state, safety, goal, True, 0.2, 120,
                      still_wanted=lambda: (not any(not f["visited"] for f in found)) and
                      ex.goal_still_valid(occ_grid(), cell_of(gt_pose().x, gt_pose().y)))
        if st.startswith("no_path") or st in ("stuck", "timeout"):
            if st == "stuck":
                recover(state, safety)
            ex.fail()
        elif st == "arrived":
            do_sweep(state)
            ex.observe(cell_of(gt_pose().x, gt_pose().y))
        say("  goal", mode, st, "found", len(found), "visited", len(visits))
    t_home = robot.getTime()
    hs = leg(state, safety, start_cell, 0.2, 500)
    p = gt_pose()
    home_err = math.hypot(p.x - start.x, p.y - start.y)
    say("  RETURN", hs, "dist_to_start_m", round(home_err, 2), "time_s", round(robot.getTime() - t_home, 1))
    ok_visits = sum(1 for v in visits if v["dist_to_nearest_true_apple_m"] < 0.6)
    result["mission"] = {"apples_estimated": len(found), "true_apples_detected": sorted(det_stats["seen_true"]),
                         "false_positive_detections": det_stats["false_positive"], "detections": det_stats["detections"],
                         "visits": visits, "visits_within_0.6m_of_true_apple": ok_visits,
                         "return_status": hs, "return_err_m": home_err,
                         "sim_time_s": round(robot.getTime(), 1)}
    say("MISSION SUMMARY: true apples detected", sorted(det_stats["seen_true"]), "| visits near a true apple", ok_visits,
        "/", len(APPLES), "| false positives", det_stats["false_positive"], "| returned", hs, round(home_err, 2), "m")


def main():
    global sign, OBSERVER
    state = {"n": 0}
    safety = SafetyMonitor()
    for _ in range(6):
        tick(state)
    start = gt_pose()
    start_cell = cell_of(start.x, start.y)
    say("start", round(start.x, 2), round(start.y, 2), round(start.theta, 2))

    # ---- phase 0: which way do LiDAR indices run? (turn left 60 deg, correlate scans)
    before = np.nan_to_num(np.array(lidar.getRangeImage()), posinf=3.5, neginf=3.5)
    spin_to(state, start.theta + math.radians(60))
    for _ in range(3):
        tick(state)
    after = np.nan_to_num(np.array(lidar.getRangeImage()), posinf=3.5, neginf=3.5)
    turned = math.degrees(gt_pose().theta - start.theta)
    turned = (turned + 180) % 360 - 180
    best_k, best_e = 0, 1e9
    for k in range(-150, 151):
        e = np.abs(np.roll(after, -k) - before).mean()
        if e < best_e:
            best_k, best_e = k, e
    # a left turn by D deg moves a fixed feature from index i to i-D if index runs CCW
    sign = 1 if best_k < 0 else -1
    say("phase0 turned_deg", round(turned, 1), "index_shift", best_k, "-> index runs",
        "CCW (left)" if sign == 1 else "CW", "match_err", round(best_e, 3))
    L[:] = 0                       # discard the map built with the unknown direction
    seen[:] = False
    # go back to the original heading so later phases start clean
    spin_to(state, start.theta)
    for _ in range(3):
        tick(state)

    result = {"lidar_index_direction": "CCW" if sign == 1 else "CW", "phases": {}}

    # ---- phase 1: temporary goal 1 m ahead of the start, then back
    ahead = (start.x + math.cos(start.theta) * 1.0, start.y + math.sin(start.theta) * 1.0)
    g = cell_of(*ahead)
    t = robot.getTime()
    s1 = drive_to(state, safety, g, False, 0.12, 60)
    p = gt_pose()
    e_out = math.hypot(p.x - ahead[0], p.y - ahead[1])
    s2 = drive_to(state, safety, start_cell, False, 0.15, 60)
    p = gt_pose()
    e_back = math.hypot(p.x - start.x, p.y - start.y)
    say("phase1 out", s1, "err_m", round(e_out, 3), "| back", s2, "err_m", round(e_back, 3),
        "time_s", round(robot.getTime() - t, 1))
    result["phases"]["round_trip_1m"] = {"out": s1, "out_err_m": e_out, "back": s2, "back_err_m": e_back}

    if os.environ.get("D_MISSION", "1") == "1":
        mission_run(state, safety, start, start_cell, result)
        result["stats"] = stats
        json.dump(result, open(os.path.join(HERE, "result.json"), "w", encoding="utf-8"), indent=2, default=str)
        say("stats", stats)
        say("DONE")
        stop_motors()
        if os.environ.get("D_GUI"):
            robot.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
            return
        robot.simulationQuit(0)
        return

    # ---- phase 2: frontier exploration
    ex = Explorer(min_cluster=6, skip_radius=6, block=5, see_radius=5)
    OBSERVER = lambda: sync_viewed(ex)
    t0 = robot.getTime()
    picks, ended = 0, "time"
    last_dump = robot.getTime()
    cov = [coverage()]
    while robot.getTime() - t0 < EXPLORE_TIME:
        pose, _ = tick(state)
        occ = occ_grid()
        walls, kn = blocked_grids(occ, pose, 2)
        goal = ex.pick(occ, kn, cell_of(pose.x, pose.y))
        if goal is None:
            ended = "nothing_left_to_explore"
            break
        picks += 1
        mode = ex.mode
        st = drive_to(state, safety, goal, True, 0.2, 120,
                      still_wanted=lambda: ex.goal_still_valid(
                          occ_grid(), cell_of(gt_pose().x, gt_pose().y)))
        if st.startswith("no_path") or st in ("stuck", "timeout"):
            if st == "stuck":
                recover(state, safety)
            ex.fail()
        elif st == "arrived":
            do_sweep(state)                      # look around 360 degrees at every goal
            ex.observe(cell_of(gt_pose().x, gt_pose().y))
        if robot.getTime() - last_dump > 250:
            dump("t%d" % int(robot.getTime()))
            last_dump = robot.getTime()
        cov.append(coverage())
        say("  goal", picks, mode, st, "free_cells", cov[-1], "looked_blocks", len(ex.covered))
    say("phase2 explore ended", ended, "goals_picked", picks, "free_cells", coverage(),
        "sim_s", round(robot.getTime() - t0, 1))
    dump("after_explore")
    result["phases"]["explore"] = {"ended": ended, "goals": picks, "free_cells_over_time": cov[::max(1, len(cov) // 12)]}

    # ---- phase 3: visit each apple (near it), come back to the start
    result["phases"]["apples"] = []
    for k, (ax, ay) in enumerate(APPLES, 1):
        t = robot.getTime()
        g = cell_of(ax, ay)
        v = leg(state, safety, g, 0.35, LEG_TIME)
        p = gt_pose()
        d_apple = math.hypot(p.x - ax, p.y - ay)
        b = leg(state, safety, start_cell, 0.2, LEG_TIME)
        p = gt_pose()
        d_start = math.hypot(p.x - start.x, p.y - start.y)
        say("phase3 apple", k, "visit", v, "dist_m", round(d_apple, 2), "| return", b,
            "dist_to_start_m", round(d_start, 2), "time_s", round(robot.getTime() - t, 1))
        dump("apple%d" % k, (ax, ay))
        result["phases"]["apples"].append({"apple": k, "visit": v, "dist_to_apple_m": d_apple,
                                           "return": b, "dist_to_start_m": d_start})

    result["camera_saw_apples"] = {str(k + 1): v for k, v in apple_seen.items()}
    result["camera_false_hits"] = false_hits
    result["stats"] = stats
    result["wall_clock_s"] = round(time.time() - t_start_wall, 1)
    json.dump(result, open(os.path.join(HERE, "result.json"), "w", encoding="utf-8"), indent=2)
    say("stats", stats)
    say("DONE")
    stop_motors()
    if os.environ.get("D_GUI"):
        robot.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)   # keep the window open
        return
    robot.simulationQuit(0)


def is_front(cell):
    from exploration import is_frontier
    return is_frontier(occ_grid(), cell)


try:
    main()
except SystemExit:
    pass
except Exception as e:  # keep the failure visible in the log
    import traceback
    say("ERROR", repr(e))
    traceback.print_exc()
    json.dump({"error": repr(e), "log": log}, open(os.path.join(HERE, "result.json"), "w"), indent=2)
    robot.simulationQuit(1)
