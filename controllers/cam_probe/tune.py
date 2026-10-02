import json
import math
import os

import cv2
import numpy as np

rows = {r['tag']: r for r in json.load(open('probe_result.json'))}
FOV = 1.0472
FPX = 640 / (2 * math.tan(FOV / 2))


def red_blobs(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (0, 150, 70), (8, 255, 255)) | cv2.inRange(hsv, (172, 150, 70), (179, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        a = cv2.contourArea(c)
        if a < 12:
            continue
        x, y, w, h = cv2.boundingRect(c)
        per = cv2.arcLength(c, True)
        circ = 4 * np.pi * a / (per * per + 1e-9)
        out.append(dict(area=a, x=x, y=y, w=w, h=h, circ=round(circ, 2), cy=y + h / 2))
    return sorted(out, key=lambda b: -b['area'])


print("focal length (px) =", round(FPX, 1), "-> distance = 0.05 * f / width_px")
print("\nTRUE apple blobs (frames where the apple is in view):")
errs = []
cys = []
for tag, r in rows.items():
    if r['kind'] != 'pos':
        continue
    img = cv2.imread(os.path.join('frames', tag + '.png'))
    b = red_blobs(img)
    if not b or b[0]['area'] < 25:
        continue
    o = b[0]
    est = 0.05 * FPX / max(o['w'], 1)
    errs.append((r['dist'], est))
    cys.append(o['cy'])
    if r['dist'] in (0.4, 1.0, 2.0, 3.0):
        print("  true %.1f m | w=%3d h=%3d aspect=%.2f circ=%.2f cy=%3d | pinhole dist=%.2f m" % (
            r['dist'], o['w'], o['h'], o['w'] / max(o['h'], 1), o['circ'], o['cy'], est))
rel = [abs(e - d) / d for d, e in errs]
print("\nframes:", len(errs), "| median distance error: %.0f%%" % (100 * float(np.median(rel))),
      "| cy (blob center row) min/median/max = %d / %d / %d (image height 480, center 240)" % (
          min(cys), float(np.median(cys)), max(cys)))
