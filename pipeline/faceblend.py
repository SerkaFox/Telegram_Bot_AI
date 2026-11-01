"""faceblend.py orig.mp4 locked.mp4 out.mp4 — use the face-locked frames only where faces are frontal; profile frames keep the
original (ReActor smears profiles like plasticine). Per-frame weight from YuNet yaw, smoothed for soft transitions."""
import sys, cv2, numpy as np, subprocess
o, l, out = sys.argv[1:4]
co, cl = cv2.VideoCapture(o), cv2.VideoCapture(l); fps = co.get(5)
fo, fl = [], []
while True:
    a, x = co.read(); b, y = cl.read()
    if not (a and b): break
    fo.append(x); fl.append(y)
H, W = fo[0].shape[:2]
det = cv2.FaceDetectorYN.create("/home/iaadmin/tg_comfy_bot/outputs/v2/models/yunet.onnx", "", (W, H), 0.5)
def yaw(fr):
    _, r = det.detect(fr)
    if r is None: return 1.0
    ys = []
    for f in r[:2]:
        rex, rey, lex, ley, nx, ny = f[4], f[5], f[6], f[7], f[8], f[9]
        ed = max(1.0, abs(lex - rex)); ys.append(abs(nx - (rex + lex) / 2) / ed)
    return max(ys) if len(r) >= 2 else 1.0           # one face lost = someone in profile
w = np.array([1.0 if yaw(f) < 0.22 else 0.0 for f in fo])
k = int(fps * 0.35); w = np.convolve(np.pad(w, k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), mode="valid")
print("frontal share", round(float(w.mean()), 2))
p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
                      "-c:v", "libx264", "-crf", "15", "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
for a, b, t in zip(fo, fl, w): p.stdin.write((b.astype(np.float32) * t + a.astype(np.float32) * (1 - t)).astype(np.uint8).tobytes())
p.stdin.close(); p.wait(); print("SAVED", out)
