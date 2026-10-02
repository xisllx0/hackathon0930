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


# decoys: red things that are NOT targets (fire extinguisher by the start, apple on a shelf)
DECOYS = [("ext", -0.44, -8.16), ("shelf", -10.19, -1.62)]
for name, dx, dy in DECOYS:
    for d in (0.7, 1.5, 2.5, 4.0):
        for a in range(0, 360, 45):
            x, y = dx + d * math.cos(math.radians(a)), dy + d * math.sin(math.radians(a))
            yaw = math.atan2(dy - y, dx - x)
            tag = "decoy_%s_d%03d_ang%03d" % (name, int(d * 100), a)
            res = snap(x, y, yaw, tag)
            if res is None:
                continue
            rows.append({"tag": tag, "kind": "decoy", "decoy": name, "dist": d, "yolo_hits": len(res)})
json.dump(rows, open(os.path.join(HERE, "probe_result.json"), "w"), indent=1)
print("PROBE_DONE", len(rows))
robot.simulationQuit(0)
