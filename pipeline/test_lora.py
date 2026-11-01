import sys; sys.path.insert(0, "/home/iaadmin/tg_comfy_bot/outputs")
import make_talk2 as mt
V = "/home/iaadmin/tg_comfy_bot/outputs/v2/"
mt.run("test_lx2v", V + "codex1/woman_ai.png", V + "vo_v1/trim/v1_w4.wav", seed=7, lora="lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors")
