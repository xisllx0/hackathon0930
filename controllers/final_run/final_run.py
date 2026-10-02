"""Runs the real merged controller unchanged and traces what safety/drive/explorer/A* do (log file)."""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "rescue_controller"))


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


log = open(os.path.join(HERE, "final_run_log.txt"), "w", encoding="utf-8")
sys.stdout = Tee(log, sys.__stdout__)
sys.stderr = Tee(log, sys.__stderr__)

import rescue_controller as rc  # noqa: E402
import safety  # noqa: E402
import exploration  # noqa: E402

N = {"sw": 0, "tw": 0, "astar": 0, "pick": 0, "fail": 0, "spin_no_path": 0}
last = {"pose": None}


def T(msg):
    print("TR:", msg, flush=True)


_sw = safety.SafetyMonitor.safe_wheels


def sw(self, ranges, l, r):
    out = _sw(self, ranges, l, r)
    N["sw"] += 1
    if N["sw"] % 25 == 0 or (out[2] and N["sw"] % 5 == 0):
        p = last["pose"]
        T("step %d pose=%s front=%.2f in=(%.2f,%.2f) out=(%.2f,%.2f) stopped=%s replan=%s" % (
            N["sw"], None if p is None else "(%.2f,%.2f,%.0fdeg)" % (p[0], p[1], p[2] * 57.3),
            self.front_min(ranges), l, r, out[0], out[1], out[2], out[3]))
    return out


safety.SafetyMonitor.safe_wheels = sw

_toward = rc.toward


def toward(pose, path_xy, *a, **k):
    last["pose"] = (pose.x, pose.y, pose.theta)
    N["tw"] += 1
    out = _toward(pose, path_xy, *a, **k)
    if N["tw"] % 60 == 0:
        T("toward path_len=%d goal=%s arrived=%s" % (len(path_xy), path_xy[-1] if path_xy else None, out[2]))
    return out


rc.toward = toward

_astar = rc.astar


def astar(blocked, start, goal, *a, **k):
    N["astar"] += 1
    out = _astar(blocked, start, goal, *a, **k)
    if not out:
        T("A* found NO path start=%s goal=%s" % (tuple(start), goal))
    return out


rc.astar = astar

_pick = exploration.Explorer.pick


def pick(self, occ, blocked, cell):
    g = _pick(self, occ, blocked, cell)
    N["pick"] += 1
    if N["pick"] % 5 == 0 or g is None:
        T("explorer.pick -> %s mode=%s failed=%d covered=%d robot_cell=%s" % (g, self.mode, len(self.failed), len(self.covered), cell))
    return g


exploration.Explorer.pick = pick

_fail = exploration.Explorer.fail


def fail(self):
    N["fail"] += 1
    if N["fail"] % 3 == 0:
        T("explorer.fail() total=%d (goal dropped)" % N["fail"])
    return _fail(self)


exploration.Explorer.fail = fail

rc.main()
