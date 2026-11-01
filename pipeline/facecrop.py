"""facecrop.py in out [target] — crop a still to 16:9 around the largest face so the face is ~target of frame height.
Prints the face fraction before cropping. Copies unchanged if the face is already big enough or none is found."""
import sys, cv2, shutil
src, dst = sys.argv[1], sys.argv[2]; target = float(sys.argv[3]) if len(sys.argv) > 3 else 0.45
im = cv2.imread(src); H, W = im.shape[:2]
det = cv2.FaceDetectorYN.create("/home/iaadmin/tg_comfy_bot/outputs/v2/models/yunet.onnx", "", (W, H), 0.7)
_, res = det.detect(im)
faces = [tuple(int(v) for v in r[:4]) for r in (res if res is not None else []) if r[3] > 0.04 * H]
if not len(faces): shutil.copy(src, dst); print("noface"); sys.exit()
x, y, w, h = max(faces, key=lambda r: r[2] * r[3]); frac = h / H
if frac >= target * 0.9: shutil.copy(src, dst); print(f"ok {frac:.2f}"); sys.exit()
ch = int(h / target); cw = int(ch * 16 / 9)
if cw > W: cw = W; ch = int(cw * 9 / 16)
cx = x + w // 2; cy = y + int(h * 0.75)            # face slightly above centre (room for shoulders)
x0 = min(max(0, cx - cw // 2), W - cw); y0 = min(max(0, cy - ch // 2), H - ch)
crop = im[y0:y0 + ch, x0:x0 + cw]
cv2.imwrite(dst, cv2.resize(crop, (max(1664, cw), int(max(1664, cw) * 9 / 16)), interpolation=cv2.INTER_LANCZOS4))
print(f"crop {frac:.2f}->{h / ch:.2f}")
