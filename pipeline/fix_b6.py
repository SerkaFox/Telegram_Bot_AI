import sys; sys.path.insert(0, "/home/iaadmin/tg_comfy_bot/outputs"); sys.path.insert(0, "/home/iaadmin/tg_comfy_bot/outputs/v2")
import make_motion2 as mm
from jobs import RT
mm.run("cls_b6_fix", "/home/iaadmin/tg_comfy_bot/outputs/v2/codex11/cls_b6_fixed.png", 4,
       RT + "The teacher stands in the aisle and looks at the student's screen, nodding; the students stay seated, type and click. "
       "Static locked-off camera. Everyone stays in place; nobody passes through desks; feet stay on the floor; the room does not change.", cfg_raw=3.0, seed=401)
