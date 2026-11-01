import sys; sys.path.insert(0, "/home/iaadmin/tg_comfy_bot/outputs"); sys.path.insert(0, "/home/iaadmin/tg_comfy_bot/outputs/v2")
import make_talk2 as mt
from jobs import C5, tr
mt.run("v3_tk_m1", C5 + "m_desk_mess.png", tr("v3", "v3_m1"), crop=0.30, seed=71,
       prompt="A man sitting at a desk covered with messy piles of paper talks to the camera with an amused, self-deprecating look and a small shrug, lively natural expressions, light closed-mouth smile between phrases. Correct lip sync, natural real-time speed.")
