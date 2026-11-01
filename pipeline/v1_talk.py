import sys; sys.path.insert(0, "/home/iaadmin/tg_comfy_bot/outputs")
import make_talk2 as mt
V = "/home/iaadmin/tg_comfy_bot/outputs/v2/"; T = V + "vo_v1/trim/"
JOBS = [
 ("v1_tk_m1", V + "clips/v1_walk.mp4", T + "v1_m1.wav", True, 1),
 ("v1_tk_w12", V + "codex1/woman_ai.png", T + "v1_w12.wav", False, 0),
 ("v1_tk_m2", V + "codex1/man_b.png", T + "v1_m2.wav", False, 0),
 ("v1_tk_w4", V + "codex1/woman_ai.png", T + "v1_w4.wav", False, 0),
 ("v1_tk_m4", V + "codex1/man_a.png", T + "v1_m4.wav", False, 0),
]
only = set(sys.argv[1:])
for name, src, wav, video, ss in JOBS:
    if only and name not in only: continue
    if (mt.CLIPS / f"{name}.mp4").exists(): print("skip", name); continue
    mt.run(name, src, wav, video=video, start_step=ss, seed=7)
print("V1 TALK DONE", flush=True)
