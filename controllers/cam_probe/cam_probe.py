"""Camera probe: put the robot at known spots, save camera frames, run the teammates' detector."""
import json
import math
import os
import random
import sys

import cv2
from controller import Supervisor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "rescue_controller"))
from target_detection import TargetDetector, camera_to_bgr  # noqa: E402

robot = Supervisor()
dt = int(robot.getBasicTimeStep())
node = robot.getSelf()
tf, rf = node.getField("translation"), node.getField("rotation")
cam = robot.getDevice("camera")
cam.enable(dt)
det = TargetDetector()
APPLES = [(-5.34, -10.54), (-12.02, -3.02)]
rows = []
os.makedirs(os.path.join(HERE, "frames"), exist_ok=True)


def snap(x, y, yaw, tag):
    tf.setSFVec3f([x, y, 0.0])
    rf.setSFRotation([0, 0, 1, yaw])
    node.resetPhysics()
    for _ in range(4):
        robot.step(dt)
    p = node.getPosition()
    if abs(p[0] - x) > 0.03 or abs(p[1] - y) > 0.03:
        return None                                   # pose is inside furniture/wall
    img = camera_to_bgr(cam)
    if img is None:
        return None
    cv2.imwrite(os.path.join(HERE, "frames", tag + ".png"), img)
    return det.detect_camera(cam)


# positives: facing each real apple from several distances/angles
for ai, (ax, ay) in enumerate(APPLES):
    for d in (0.4, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0):
        for a in range(0, 360, 45):
            x, y = ax + d * math.cos(math.radians(a)), ay + d * math.sin(math.radians(a))
            yaw = math.atan2(ay - y, ax - x)
            tag = "pos_a%d_d%03d_ang%03d" % (ai + 1, int(d * 100), a)
            res = snap(x, y, yaw, tag)
            if res is None:
                continue
            rows.append({"tag": tag, "kind": "pos", "apple": ai + 1, "dist": d, "yolo_hits": len(res),
                         "yolo": [(round(r.confidence, 2), round(r.red_ratio, 2)) for r in res]})

# negatives: random spots/headings around the apartment (the apple is usually not in view)
random.seed(3)
n = 0
while n < 70:
    x, y = random.uniform(-12.5, 0.3), random.uniform(-13.0, 4.0)
    yaw = random.uniform(-math.pi, math.pi)
    tag = "neg_%03d" % n
    res = snap(x, y, yaw, tag)
    if res is None:
        continue
    n += 1
    rows.append({"tag": tag, "kind": "neg", "x": round(x, 2), "y": round(y, 2), "yaw": round(yaw, 2),
                 "yolo_hits": len(res), "yolo": [(round(r.confidence, 2), round(r.red_ratio, 2)) for r in res]})

json.dump(rows, open(os.path.join(HERE, "probe_result.json"), "w"), indent=1)
print("PROBE_DONE", len(rows))
robot.simulationQuit(0)
