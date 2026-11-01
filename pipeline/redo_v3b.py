import sys; sys.path.insert(0, "/home/iaadmin/tg_comfy_bot/outputs"); sys.path.insert(0, "/home/iaadmin/tg_comfy_bot/outputs/v2")
import make_talk2 as mt
from jobs import C5, tr
mt.run("v3_tk_w1", C5 + "w_tv_calm.png", tr("v3", "v3_w1"), prompt="A calm, serious, professional TV presenter talks to the camera in a neutral friendly tone. Natural head movement, eyebrows move with the words, natural blinking. Lips relaxed, no smiling while talking, neutral mouth between phrases. Correct lip sync, natural real-time speed.", seed=61)
