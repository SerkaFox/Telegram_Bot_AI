import sys, pathlib; sys.path.insert(0, "/home/iaadmin/tg_comfy_bot/outputs"); sys.path.insert(0, "/home/iaadmin/tg_comfy_bot/outputs/v2")
import make_motion2 as mm
from jobs import RT
C9 = "/home/iaadmin/tg_comfy_bot/outputs/v2/codex9/"
ST = " Static camera. Everyone stays in place, the monitor does not move."
JOBS = [("v5_help", C9 + "help_fixed.png", 4, RT + "The teacher points at the screen and explains; the woman nods, understands and smiles slightly, then moves the mouse." + ST, 93),
        ("v5_team", C9 + "team_fixed.png", 4, RT + "The classmate points at the chart on the screen; the three look at the screen together and smile naturally." + ST, 94),
        ("v3_r_web", C9 + "web_fixed.png", 4, RT + "The developer points at the screen and talks; the white robot turns its head to look at the screen and types." + ST, 95),
        ("v3_r_mkt", C9 + "mkt_fixed.png", 4, RT + "The woman points at a tile on the screen and smiles; the robot nods and moves its hand toward the screen." + ST, 96)]
for name, still, sec, prompt, seed in JOBS:
    while not pathlib.Path(still).exists(): import time; time.sleep(30)
    if mm.run(name + "_fix", still, sec, prompt, cfg_raw=3.0, seed=seed): print("FIXED", name, flush=True)
print("MONITORS DONE", flush=True)
