import sys; sys.path.insert(0, "/home/iaadmin/tg_comfy_bot/outputs")
import make_talk2 as mt
mt.run("v1_tk_m1b", "/home/iaadmin/tg_comfy_bot/outputs/v2/clips/v1_walk.mp4", "/home/iaadmin/tg_comfy_bot/outputs/v2/vo_v1/trim/v1_m1.wav", video=True, start_step=2, seed=7)
