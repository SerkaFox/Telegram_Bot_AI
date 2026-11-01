#!/usr/bin/env python3
"""Single-image Qwen edit: change only the presenter's outfit, keep face/pose/room. args: src_input_name dst_id"""
import sys, pathlib
sys.path.insert(0, "/home/iaadmin/tg_comfy_bot/outputs")
import make_fullbody as fb, batch_meatze as bm
src, dst_id = sys.argv[1], sys.argv[2]
pos = ("Change ONLY her clothes: she now wears a navy business suit jacket over a white blouse. Keep her face, hair, "
       "expression, pose, framing and the entire background exactly unchanged. Do not add any text or objects. "
       "Slight closed-mouth smile. Photorealistic.")
neg = "text, letters, posters, new objects, changed room, cardigan, skirt, toothy smile, deformed hands"
g = fb.wf(src, src, "female", "", "", 9191, 1152, 640)
g["68"]["inputs"].update({"prompt": pos}); g["69"]["inputs"].update({"prompt": neg})
for n in ("68", "69"): g[n]["inputs"].pop("image2", None)
g.pop("42", None)
dst = pathlib.Path(f"/home/iaadmin/tg_comfy_bot/outputs/stills/lib_{dst_id}.jpg")
pid = fb.post(g); res = fb.wait(pid)
img = [im for o in res["outputs"].values() for im in o.get("images", [])][-1]
bm.fetch(img, dst)
(pathlib.Path("/home/iaadmin/ComfyUI/input") / f"meatze_lib_{dst_id}.jpg").write_bytes(dst.read_bytes())
print("SAVED", dst)
