import json, math, os, sys
import cv2, numpy as np
sys.path.insert(0, '../rescue_controller')
from target_detection import TargetDetector
det = TargetDetector()
rows = json.load(open('probe_result.json'))
FOV = 1.0472
tp = fn = fp = 0
per = {}
neg_fp = []
errs = []
for r in rows:
    img = cv2.imread(os.path.join('frames', r['tag'] + '.png'))
    dets = det.detect_bgr(img, FOV)
    if r['kind'] == 'pos':
        # ground truth "apple in view" = a big round red blob exists (same test as before)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, (0,150,70),(8,255,255)) | cv2.inRange(hsv,(172,150,70),(179,255,255))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        cs,_ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        big = [c for c in cs if cv2.contourArea(c) >= 25]
        if not big: continue
        d = r['dist']; per.setdefault(d, [0, 0, 0]); per[d][0] += 1
        if r['yolo_hits']: per[d][1] += 1
        if dets:
            per[d][2] += 1
            for x in dets:
                if x.distance: errs.append((d, x.distance))
    else:
        if dets: neg_fp.append((r['tag'], [round(x.distance, 2) if x.distance else None for x in dets]))
print("apple in view -> old YOLO-only vs new (YOLO + colour fallback):")
tot = [0, 0, 0]
for d in sorted(per):
    n, y, c = per[d]; tot[0] += n; tot[1] += y; tot[2] += c
    print("  %.1f m: %2d frames | old %2d (%3.0f%%) | new %2d (%3.0f%%)" % (d, n, y, 100*y/n, c, 100*c/n))
print("  TOTAL : %d frames | old %d (%.0f%%) | new %d (%.0f%%)" % (tot[0], tot[1], 100*tot[1]/tot[0], tot[2], 100*tot[2]/tot[0]))
print("false detections on 70 negative frames:", len(neg_fp), neg_fp[:4])
rel = [abs(e - d) / d for d, e in errs]
print("distance from bbox width: median error %.0f%% over %d detections" % (100 * float(np.median(rel)), len(rel)))
