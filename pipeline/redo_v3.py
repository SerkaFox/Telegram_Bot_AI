import sys; sys.path.insert(0, "/home/iaadmin/tg_comfy_bot/outputs"); sys.path.insert(0, "/home/iaadmin/tg_comfy_bot/outputs/v2")
import make_talk2 as mt, make_motion2 as mm
from jobs import C5, tr, RT
mt.run("v3_tk_w1", C5 + "w_tv_green.png", tr("v3", "v3_w1"), prompt="A calm professional presenter in a classroom talks warmly to the camera, natural lively expressions, eyebrows move with the words, gentle nods, only a slight closed-mouth smile between phrases, never a wide toothy grin. Correct lip sync, natural real-time speed.", seed=51)
mm.run("v3_desk_wide", C5 + "m_desk_mess.png", 3, RT + "The man at the messy desk looks at the piles of paper, then at the camera, and shrugs with an amused smile; his mouth stays closed, he does not speak. Static camera.", cfg_raw=3.0)
