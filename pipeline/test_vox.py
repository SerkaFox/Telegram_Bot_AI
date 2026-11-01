import sys; sys.path.insert(0, "/home/iaadmin/tg_comfy_bot/outputs"); sys.path.insert(0, "/home/iaadmin/tg_comfy_bot/outputs/v2")
import make_motion2 as mm, make_talk2 as mt
from jobs import C3, T2, VOX
mo = mm.run("vox_girl_body", C3 + "vp_girl.png", 5, "Real-time natural speed. The young woman is being interviewed on the street: she talks with the interviewer just left of the camera, "
            "natural relaxed body language, small hand gestures, slight shoulder and head movement, shifts her weight, glances away and back while thinking. "
            "Slight natural handheld camera movement, people walking in the background.", cfg_raw=3.0, seed=101)
if mo:
    mt.run("vox_girl_v2v", str(mo), T2 + "v2_girl.wav", video=True, start_step=1, prompt=VOX, seed=102)
