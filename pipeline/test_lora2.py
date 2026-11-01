import sys; sys.path.insert(0, "/home/iaadmin/tg_comfy_bot/outputs")
import make_talk2 as mt
V = "/home/iaadmin/tg_comfy_bot/outputs/v2/"; L = "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"
NEUTRAL = ("A calm, serious, professional TV presenter talks to the camera in a neutral friendly tone, looking straight into the lens. "
           "Natural head movement, eyebrows move with the words, natural blinking. Lips relaxed, no smiling while talking, "
           "neutral mouth between phrases, never a grin. Correct lip sync, natural real-time speed.")
mt.run("test_lx2v_neutral", V + "codex1/woman_ai.png", V + "vo_v1/trim/v1_w4.wav", seed=7, lora=L, prompt=NEUTRAL)
mt.LORA_STRENGTH = 0.7
mt.run("test_lx2v_neutral07", V + "codex1/woman_ai.png", V + "vo_v1/trim/v1_w4.wav", seed=7, lora=L, prompt=NEUTRAL)
