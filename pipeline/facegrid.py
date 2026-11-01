"""facegrid.py out.jpg clip... -> 2 face crops per clip (YuNet), 4 clips per row"""
import sys, cv2, numpy as np
out = sys.argv[1]; det = None; rows = []
for n in sys.argv[2:]:
    cap = cv2.VideoCapture(f"/home/iaadmin/tg_comfy_bot/outputs/v2/clips/{n}.mp4"); N = int(cap.get(7)); tiles = []
    for t in (0.3, 0.7):
        cap.set(1, int(N * t)); ok, fr = cap.read(); H, W = fr.shape[:2]
        if det is None: det = cv2.FaceDetectorYN.create("/home/iaadmin/tg_comfy_bot/outputs/v2/models/yunet.onnx", "", (W, H), 0.6)
        det.setInputSize((W, H)); _, r = det.detect(fr)
        if r is not None:
            x, y, w, h = [int(v) for v in r[0][:4]]; s = int(h * 1.3); cx, cy = x + w // 2, y + h // 2
            crop = fr[max(0, cy - s // 2):cy + s // 2, max(0, cx - s // 2):cx + s // 2]
        else: crop = fr
        t2 = cv2.resize(crop, (260, 260)); cv2.putText(t2, n[:14], (4, 18), 0, 0.5, (0, 255, 255), 1); tiles.append(t2)
    rows.append(np.hstack(tiles))
pad = lambda r: r + [np.zeros_like(rows[0])] * (4 - len(r))
cv2.imwrite(out, np.vstack([np.hstack(pad(rows[i:i + 4])) for i in range(0, len(rows), 4)]))
