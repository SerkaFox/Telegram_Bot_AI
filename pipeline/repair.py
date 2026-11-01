"""Repair pass: clips that were previously trimmed around defects are regenerated whole (user rule: fix, don't cut).
New clip goes to clips/<name>.mp4 only after it is rendered; the old one is kept in clips/rejected/<name>_v1.mp4."""
import sys, pathlib; sys.path.insert(0, "/home/iaadmin/tg_comfy_bot/outputs"); sys.path.insert(0, "/home/iaadmin/tg_comfy_bot/outputs/v2")
import make_motion2 as mm
from jobs import C1, C2, C5, C6, KIT, RT
STATIC = " Static locked-off camera, no camera movement. Everyone stays in place; people never pass through desks or furniture; the room does not change."
JOBS = [
 ("v3_desk_wide", C5 + "m_desk_mess.png", 3, RT + "The man at the messy desk glances at the piles of paper, then at the camera, and gives a small helpless shrug with an amused look. "
  "His mouth stays firmly CLOSED the whole time, lips together, he does not speak." + STATIC, 81),
 ("cls_p7", C6 + "cls_p7.png", 4, RT + "The teacher at the whiteboard gestures toward the diagram; the students stay seated at their desks, look up at her, then type on their keyboards." + STATIC, 82),
 ("cls_b6", C6 + "cls_b6.png", 4, RT + "The teacher stands still in the aisle and looks at a student's screen, nodding; the students stay seated and type and click." + STATIC, 83),
 ("v1_r_home", C2 + "robot_home.png", 4, RT + "The white robot keeps bending over the open dishwasher and places two plates into the rack one by one; its body shape stays the same; "
  "the father and child set the table in the background. Static camera.", 84),
 ("v1_screen", C1 + "monitor_close.png", 4, RT + "The camera pushes in steadily toward the monitor. The woman on the screen keeps looking straight into the camera with a calm, confident, "
  "professional expression and only a slight closed-mouth smile; she blinks naturally and nods once. She never looks down.", 85),
 ("v1_power", "/home/iaadmin/tg_comfy_bot/outputs/v2/codex8/power_off.png", 3, RT + "The finger presses the power button; after the press the dark monitor screen switches on and lights up with a soft blue glow. "
  "The hand then moves away. Static close-up.", 86),
 ("son_b2", KIT + "B_Bagatza/b2.jpeg", 3, "The computer monitors in the empty classroom switch on one after another, each screen lighting up with a soft blue glow. "
  "The keyboards stay black, the desks and chairs do not change at all. Static camera with a very slight slow push-in. No people.", 87),
]
only = set(sys.argv[1:])
for name, still, sec, prompt, seed in JOBS:
    if only and name not in only: continue
    tmp = name + "_fix"
    if mm.run(tmp, still, sec, prompt, cfg_raw=3.0, seed=seed):
        print("FIXED", name, flush=True)
print("REPAIR DONE", flush=True)
