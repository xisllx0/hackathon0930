import json, os, sys
import cv2, numpy as np
rows = {r['tag']: r for r in json.load(open('probe_result.json'))}

def red_blobs(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (0, 150, 70), (8, 255, 255)) | cv2.inRange(hsv, (172, 150, 70), (179, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        a = cv2.contourArea(c)
        if a < 12: continue
        x, y, w, h = cv2.boundingRect(c)
        per = cv2.arcLength(c, True)
        circ = 4 * np.pi * a / (per * per + 1e-9)
        out.append(dict(area=a, x=x, y=y, w=w, h=h, circ=round(circ, 2), cy=y + h / 2))
    return sorted(out, key=lambda b: -b['area'])

res = []
for tag, r in rows.items():
    img = cv2.imread(os.path.join('frames', tag + '.png'))
    res.append((tag, r, red_blobs(img)))

pos = [(t, r, b) for t, r, b in res if r['kind'] == 'pos']
neg = [(t, r, b) for t, r, b in res if r['kind'] == 'neg']
vis = [(t, r, b) for t, r, b in pos if b and b[0]['area'] >= 25]
print("positive frames:", len(pos), "| with a clearly visible red blob (apple in view):", len(vis))
print("\nOn frames where the apple IS in view (by color):")
from collections import defaultdict
agg = defaultdict(lambda: [0, 0, 0])
for t, r, b in vis:
    agg[r['dist']][0] += 1
    agg[r['dist']][1] += 1 if r['yolo_hits'] else 0
    agg[r['dist']][2] += 1
for d in sorted(agg):
    n, y, _ = agg[d]
    ex = [b[0] for t, r, b in vis if r['dist'] == d][:1]
    print("  %.1f m: apple in view in %2d frames | YOLO found %2d (%.0f%%) | color blob w~%s px" % (d, n, y, 100 * y / n, ex[0]['w'] if ex else '-'))
print("\nnegative frames with a red blob (potential false positives): %d / %d" % (sum(1 for t, r, b in neg if b and b[0]['area'] >= 25), len(neg)))
for t, r, b in neg:
    if b and b[0]['area'] >= 25:
        print("   ", t, "pos=(%s,%s)" % (r['x'], r['y']), "blob:", {k: b[0][k] for k in ('area', 'w', 'h', 'cy', 'circ')}, "yolo:", r['yolo_hits'])
