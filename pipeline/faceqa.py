"""Full-res face crops of talk clips at 4 moments -> one sheet per clip (uses insightface via ReActor's env if available, else centre-top crop)."""
import sys, subprocess, pathlib, numpy as np, cv2
S = pathlib.Path("/tmp/claude-1000/-home-iaadmin-tg-comfy-bot/606374ee-87ad-47e3-8a98-2f3268dde32c/scratchpad/face"); S.mkdir(exist_ok=True)
casc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
for name in sys.argv[1:]:
    f = f"/home/iaadmin/tg_comfy_bot/outputs/v2/clips/{name}.mp4"
    cap = cv2.VideoCapture(f); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); tiles = []; sizes = []
    for t in np.linspace(0.15, 0.9, 4):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * t)); ok, fr = cap.read()
        if not ok: continue
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY); faces = casc.detectMultiScale(g, 1.1, 5, minSize=(40, 40))
        if len(faces):
            x, y, w, h = max(faces, key=lambda r: r[2] * r[3]); sizes.append(h)
            cx, cy, s = x + w // 2, y + h // 2, int(h * 1.6)
        else:
            cx, cy, s = fr.shape[1] // 2, fr.shape[0] // 3, 300
        crop = fr[max(0, cy - s // 2):cy + s // 2, max(0, cx - s // 2):cx + s // 2]
        tiles.append(cv2.resize(crop, (360, 360)))
    cv2.imwrite(str(S / f"{name}.jpg"), np.hstack(tiles))
    print(name, "face px (of 960):", sizes, S / f"{name}.jpg")
