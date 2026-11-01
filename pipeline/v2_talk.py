import sys, pathlib; sys.path.insert(0, "/home/iaadmin/tg_comfy_bot/outputs")
import make_talk2 as mt
V = "/home/iaadmin/tg_comfy_bot/outputs/v2/"; C3 = V + "codex3/"; T = V + "vo_v2/trim/"
VOX = ("A person is being interviewed on a city street and answers naturally, talking to the interviewer standing just to the left of the camera. "
       "Natural lively facial expressions, eyebrows and head move with the words, natural blinking, small nods, relaxed. "
       "Clear articulate lip movement, correct lip sync. Realistic documentary look, natural real-time speed, slight handheld camera.")
REP = ("A friendly TV reporter on a city street talks warmly to the camera holding a microphone, natural lively expressions, "
       "gentle head movement, natural blinking, light closed-mouth smile between phrases, correct lip sync, realistic, natural real-time speed.")
JOBS = [
 ("v2_tk_w1", C3 + "rep_woman_street.png", "v2_w1", REP),
 ("v2_tk_boy", C3 + "rep_woman_ask.png", "v2_boy", VOX),
 ("v2_tk_girl", C3 + "vp_girl.png", "v2_girl", VOX),
 ("v2_tk_man46", C3 + "vp_man46.png", "v2_man46", VOX),
 ("v2_tk_woman35", C3 + "vp_woman35.png", "v2_woman35", VOX),
 ("v2_tk_w2", C3 + "rep_woman_street.png", "v2_w2", REP),
 ("v2_tk_man52", C3 + "vp_man52.png", "v2_man52", VOX),
 ("v2_tk_woman41", C3 + "vp_woman41.png", "v2_woman41", VOX),
 ("v2_tk_man28", C3 + "vp_man28.png", "v2_man28", VOX),
 ("v2_tk_woman58", C3 + "vp_woman58.png", "v2_woman58", VOX),
 ("v2_tk_m1", C3 + "rep_man_lobby.png", "v2_m1", mt.POS),
 ("v2_tk_m2", V + "codex1/man_b.png", "v2_m2", mt.POS),
 ("v2_tk_m3", C3 + "rep_man_lobby.png", "v2_m3", mt.POS),
]
only = set(sys.argv[1:])
for name, src, wav, prompt in JOBS:
    if only and name not in only: continue
    if (mt.CLIPS / f"{name}.mp4").exists(): print("skip", name); continue
    if not pathlib.Path(src).exists(): print("MISSING", src, flush=True); continue
    mt.run(name, src, T + wav + ".wav", prompt=prompt, seed=11)
print("V2 TALK DONE", flush=True)
