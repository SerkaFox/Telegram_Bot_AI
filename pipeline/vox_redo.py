"""Vox-pop people come alive: Wan body/head/handheld motion first, then InfiniteTalk V2V lip-sync (distill lora). -> v2_tk_<x>_fix.mp4"""
import sys, pathlib; sys.path.insert(0, "/home/iaadmin/tg_comfy_bot/outputs"); sys.path.insert(0, "/home/iaadmin/tg_comfy_bot/outputs/v2")
import make_motion2 as mm, make_talk2 as mt
from jobs import C3, T2, VOX
BODY = ("Real-time natural speed. The {who} is being interviewed on the street: talks with the interviewer just left of the camera, natural relaxed body language, "
        "small hand gestures, slight shoulder and head movement, shifts weight, glances away and back while thinking. Slight natural handheld camera movement, "
        "people walking in the background.")
P = [("boy", "rep_woman_ask", "young man in the hoodie"), ("man46", "vp_man46", "man"), ("woman35", "vp_woman35", "woman"),
     ("man52", "vp_man52", "man"), ("woman41", "vp_woman41", "woman"), ("man28", "vp_man28", "man"), ("woman58", "vp_woman58", "woman")]
import shutil
shutil.copy("/home/iaadmin/tg_comfy_bot/outputs/v2/clips/vox_girl_v2v.mp4", "/home/iaadmin/tg_comfy_bot/outputs/v2/clips/v2_tk_girl_fix.mp4")
for i, (k, still, who) in enumerate(P):
    mo = mm.run(f"vox_{k}_body", C3 + still + ".png", 5, BODY.format(who=who), cfg_raw=3.0, seed=300 + i)
    if mo and mt.run(f"v2_tk_{k}_fix", str(mo), T2 + f"v2_{k}.wav", video=True, start_step=1, prompt=VOX, seed=310 + i):
        print("FIXED", k, flush=True)
print("VOX DONE", flush=True)
