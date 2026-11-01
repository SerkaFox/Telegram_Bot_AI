import sys; sys.path.insert(0, "/home/iaadmin/tg_comfy_bot/outputs"); sys.path.insert(0, "/home/iaadmin/tg_comfy_bot/outputs/v2")
import make_motion2 as mm
from jobs import RT
mm.run("v7_robot_wave_fix", "/home/iaadmin/tg_comfy_bot/outputs/v2/codex10/robot_wave_wide.png", 4,
       RT + "The white robot waves its raised hand side to side at the camera in a friendly way and tilts its head slightly; its blue eyes glow softly. Static camera, the whole head stays in frame.", cfg_raw=3.0, seed=501)
