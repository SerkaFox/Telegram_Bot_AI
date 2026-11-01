import sys; sys.path.insert(0, "/home/iaadmin/tg_comfy_bot/outputs"); sys.path.insert(0, "/home/iaadmin/tg_comfy_bot/outputs/v2")
import make_talk2 as mt
from jobs import REP, VOX, C3, T2
mt.run("v2_tk_w2", C3 + "rep_woman_street.png", T2 + "v2_w2.wav", prompt=REP + " Calm, only a slight closed-mouth smile, no teeth.", seed=31)
mt.run("v2_tk_man52", C3 + "vp_man52.png", T2 + "v2_man52.wav", prompt=VOX, seed=32)
