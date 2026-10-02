"""Live occupancy-grid viewer (OpenCV window) and PNG snapshot.

Optional helper: every call is wrapped so a drawing problem can never stop the robot loop.
Colours: unknown = gray, free = white, wall = black; robot = orange, path = blue,
start = green, apple candidates = red.
"""
import math

import numpy as np

try:
    import cv2
except Exception:                      # opencv missing -> viewer silently disabled
    cv2 = None

UNKNOWN, FREE, OCCUPIED = -1, 0, 1


class MapViewer:
    def __init__(self, scale=4, margin=12, title="Map  (gray=unknown  white=free  black=wall)"):
        self.scale = scale
        self.margin = margin
        self.title = title
        self.window_ok = cv2 is not None
        self.last_image = None

    def render(self, mapping, pose=None, path_xy=None, targets=(), home=(0.0, 0.0)):
        grid = np.asarray(mapping.grid, dtype=np.int8)
        rows, cols = grid.shape
        s = self.scale
        img = np.full((rows, cols, 3), 128, np.uint8)
        img[grid == FREE] = (255, 255, 255)
        img[grid == OCCUPIED] = (0, 0, 0)
        canvas = cv2.resize(img, (cols * s, rows * s), interpolation=cv2.INTER_NEAREST)

        def px(x, y):
            r, c = mapping.world_to_grid(x, y)
            return int(c * s + s // 2), int(r * s + s // 2)

        if path_xy and len(path_xy) > 1:
            pts = np.array([px(x, y) for x, y in path_xy], np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], False, (255, 90, 0), 2)
        cv2.circle(canvas, px(*home), 7, (0, 170, 0), 2)
        for tx, ty in targets:
            cv2.circle(canvas, px(tx, ty), 8, (0, 0, 230), -1)
        if pose is not None:
            cx, cy = px(pose.x, pose.y)
            cv2.circle(canvas, (cx, cy), 7, (0, 140, 255), -1)
            hx = int(cx + 14 * math.cos(pose.theta))
            hy = int(cy + 14 * math.sin(pose.theta))
            cv2.line(canvas, (cx, cy), (hx, hy), (0, 60, 200), 2)

        known = np.argwhere(grid != UNKNOWN)
        if len(known):                                   # crop to the explored area
            (r0, c0), (r1, c1) = known.min(0), known.max(0)
            m = self.margin
            r0, c0 = max(0, r0 - m), max(0, c0 - m)
            r1, c1 = min(rows, r1 + m), min(cols, c1 + m)
            canvas = canvas[r0 * s:r1 * s, c0 * s:c1 * s]
        canvas = cv2.flip(canvas, 0)                     # +y (forward/left) is up
        cv2.putText(canvas, self.title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1)
        self.last_image = canvas
        return canvas

    def show(self, mapping, pose=None, path_xy=None, targets=(), home=(0.0, 0.0)):
        try:
            img = self.render(mapping, pose, path_xy, targets, home)
            if self.window_ok:
                cv2.imshow("Occupancy map", img)
                cv2.waitKey(1)
        except Exception:
            self.window_ok = False                       # never let drawing break the robot

    def save(self, path):
        try:
            if self.last_image is not None:
                cv2.imwrite(path, self.last_image)
        except Exception:
            pass
