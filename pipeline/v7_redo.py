import sys, time, pathlib, subprocess; sys.path.insert(0, "/home/iaadmin/tg_comfy_bot/outputs"); sys.path.insert(0, "/home/iaadmin/tg_comfy_bot/outputs/v2")
import make_talk2 as mt, make_motion2 as mm
from jobs import tr, RT
C10 = "/home/iaadmin/tg_comfy_bot/outputs/v2/codex10/"
subprocess.run(["python3", "/home/iaadmin/tg_comfy_bot/outputs/v2/facelock.py", "/home/iaadmin/tg_comfy_bot/outputs/v2/clips/v7_both.mp4", "v7_both_lock"])
mt.run("v7_tk_m1_fix", C10 + "rec_man_look.png", tr("v7", "v7_m1"), seed=111,
       prompt="A man standing at a reception desk turns to the woman next to him (off-frame) and asks a short curious question, natural head movement, correct lip sync, natural real-time speed.")
mt.run("v7_tk_m2_fix", C10 + "rec_man_cam.png", tr("v7", "v7_m2"), seed=112)
mt.run("v7_tk_w3_fix", C10 + "rec_woman_cam.png", tr("v7", "v7_w3"), seed=113,
       prompt="A calm, serious, professional presenter behind a reception desk talks to the camera in a neutral friendly tone. Natural head movement, natural blinking. Lips relaxed, no smiling while talking. Correct lip sync, natural real-time speed.")
while not pathlib.Path(C10 + "robot_wave.png").exists(): time.sleep(20)
mm.run("v7_robot_wave", C10 + "robot_wave.png", 4, RT + "The white robot waves its raised hand at the camera in a friendly way, tilts its head, its blue eyes glow softly. Static camera.", cfg_raw=3.0, seed=114)
print("V7 REDO DONE", flush=True)
