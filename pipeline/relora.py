"""Regenerate presenter talk clips made before the distill-lora recipe (ghost blinks / grid teeth). Writes <name>_fix.mp4."""
import sys, pathlib; sys.path.insert(0, "/home/iaadmin/tg_comfy_bot/outputs"); sys.path.insert(0, "/home/iaadmin/tg_comfy_bot/outputs/v2")
import make_talk2 as mt
from jobs import JOBS, C1, C3, C5, tr, REP
V = "/home/iaadmin/tg_comfy_bot/outputs/v2/"
NEUTRAL = ("A calm, serious, professional presenter talks to the camera in a neutral friendly tone, looking into the lens. Natural head movement, "
           "eyebrows move with the words, natural blinking. Lips relaxed, no smiling while talking, never a grin. Correct lip sync, natural real-time speed.")
T = {j[1]: (j[2], j[3], j[4]) for j in JOBS if j[0] == "t"}
SPEC = {   # name: (still, wav, prompt, crop)
 "v3_tk_w1": (C5 + "w_tv_calm.png", tr("v3", "v3_w1"), NEUTRAL, 0.45),
 "v3_tk_m1": (C5 + "m_desk_mess.png", tr("v3", "v3_m1"), mt.POS, 0.45),
 "v3_tk_m2": (*T["v3_tk_m2"][:2], mt.POS, 0.45),
 "v1_tk_m1": (C1 + "man_c.png", V + "vo_v1/trim/v1_m1.wav", mt.POS, 0.45),
 "v1_tk_w12": (C1 + "woman_ai.png", V + "vo_v1/trim/v1_w12.wav", NEUTRAL, 0.45),
 "v1_tk_m2": (C1 + "man_b.png", V + "vo_v1/trim/v1_m2.wav", mt.POS, 0.45),
 "v1_tk_w4": (C1 + "woman_ai.png", V + "vo_v1/trim/v1_w4.wav", NEUTRAL, 0.45),
 "v1_tk_m4": (C1 + "man_a.png", V + "vo_v1/trim/v1_m4.wav", mt.POS, 0.45),
 "v2_tk_w1": (C3 + "rep_woman_street.png", V + "vo_v2/trim/v2_w1.wav", REP + " Calm, only a slight closed-mouth smile, no teeth.", 0.45),
 "v2_tk_w2": (C3 + "rep_woman_street.png", V + "vo_v2/trim/v2_w2.wav", REP + " Calm, only a slight closed-mouth smile, no teeth.", 0.45),
 "v2_tk_m1": (C3 + "rep_man_lobby.png", V + "vo_v2/trim/v2_m1.wav", mt.POS, 0.45),
 "v2_tk_m2": (C1 + "man_b.png", V + "vo_v2/trim/v2_m2.wav", mt.POS, 0.45),
 "v2_tk_m3": (C3 + "rep_man_lobby.png", V + "vo_v2/trim/v2_m3.wav", mt.POS, 0.45),
 "v4_tk_m1": (*T["v4_tk_m1"][:2], mt.POS, 0.45),
 "v4_tk_w1": (*T["v4_tk_w1"][:2], NEUTRAL, 0.45),
 "v4_tk_m2": (*T["v4_tk_m2"][:2], mt.POS, 0.45),
 "v4_tk_w2": (*T["v4_tk_w2"][:2], NEUTRAL, 0.45),
}
only = sys.argv[1:] or list(SPEC)
for n in only:
    still, wav, prompt, crop = SPEC[n]
    if mt.run(n + "_fix", still, wav, prompt=prompt, seed=200 + only.index(n), crop=crop): print("FIXED", n, flush=True)
print("RELORA DONE", flush=True)
