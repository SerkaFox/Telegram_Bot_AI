import sys; sys.path.insert(0, "/home/iaadmin/tg_comfy_bot/outputs"); sys.path.insert(0, "/home/iaadmin/tg_comfy_bot/outputs/v2")
import make_motion2 as mm
from jobs import RT
V = "/home/iaadmin/tg_comfy_bot/outputs/v2/crops/"
mm.run("v5_kitchen_b", V + "v5_kitchen_end.png", 3, RT + "The woman keeps looking at the laptop, takes a sip of coffee and puts the mug down, thoughtful. Soft morning light, static camera.", cfg_raw=3.0, seed=91)
mm.run("v5_door_b", V + "v5_door_end.png", 3, RT + "The woman at the reception desk listens and nods while the receptionist explains politely; both stay in place. Static camera.", cfg_raw=3.0, seed=92)
