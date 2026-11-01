#!/usr/bin/env python3
"""Clean SFW Wan 2.2 I2V (SVI FASTMOVE dual H/L + lightx2v) — animates a still into motion:
walking, waving, gesturing, with camera movement. No audio (voice-over added at assembly).
Reuses the exact models the bot's workflow_video.json runs on this 12GB card."""
import json, time, urllib.request, urllib.parse, sys, pathlib

COMFY = "http://127.0.0.1:8188"
CLIPS = pathlib.Path("/home/iaadmin/tg_comfy_bot/outputs/clips"); CLIPS.mkdir(parents=True, exist_ok=True)
NEG = ("palms toward the camera, hands reaching toward the camera, stop gesture, raised hands, toothy smile, showing teeth, open-mouth smile, laughing, seductive, flirtatious, sensual, provocative, exaggerated smile, wide grin, grinning, smirk, giggling, "
       "hip sway, swaying body, dancing, twisting, leaning, bending over, posing, model pose, head tilt, "
       "overacting, 色调艳丽，过曝，细节模糊不清，字幕，画面，整体发灰，最差质量，低质量，"
       "deformed hands, extra fingers, extra limbs, duplicated person, distorted face, warped furniture, text, watermark")

def wf(still_name, prompt, seconds, seed, w, h):
    length = int(seconds) * 16 + 1
    return {
        "371": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf"}},
        "372": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf"}},
        "151": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["371", 0], "lora_name": "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors", "strength_model": 1.0}},
        "154": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["372", 0], "lora_name": "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors", "strength_model": 1.0}},
        "364": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}},
        "357": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
        "97": {"class_type": "LoadImage", "inputs": {"image": still_name}},
        "361": {"class_type": "ImageResize+", "inputs": {"width": w, "height": h, "interpolation": "lanczos",
                "method": "keep proportion", "condition": "always", "multiple_of": 64, "image": ["97", 0]}},
        "93": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["364", 0]}},
        "360": {"class_type": "CLIPTextEncode", "inputs": {"text": NEG, "clip": ["364", 0]}},
        "358": {"class_type": "WanImageToVideo", "inputs": {"width": w, "height": h, "length": length, "batch_size": 1,
                "positive": ["93", 0], "negative": ["360", 0], "vae": ["357", 0], "start_image": ["361", 0]}},
        "141": {"class_type": "WanMoeKSampler", "inputs": {"boundary": 0.9, "seed": seed, "steps": 8,
                "cfg_high_noise": 1, "cfg_low_noise": 1, "sampler_name": "euler", "scheduler": "simple",
                "sigma_shift": 8, "denoise": 1, "model_high_noise": ["151", 0], "model_low_noise": ["154", 0],
                "positive": ["358", 0], "negative": ["358", 1], "latent_image": ["358", 2]}},
        "363": {"class_type": "VAEDecode", "inputs": {"samples": ["141", 0], "vae": ["357", 0]}},
        "380": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "2xNomosUni_span_multijpg.safetensors"}},
        "381": {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["380", 0], "image": ["363", 0]}},
        "314": {"class_type": "VHS_VideoCombine", "inputs": {"images": ["381", 0], "frame_rate": 16, "loop_count": 0,
                "filename_prefix": "meatze_motion", "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 19,
                "save_metadata": False, "trim_to_audio": False, "pingpong": False, "save_output": True}},
    }

def post(w):
    data = json.dumps({"prompt": w}).encode()
    return json.load(urllib.request.urlopen(urllib.request.Request(COMFY + "/prompt", data=data, headers={"Content-Type": "application/json"})))["prompt_id"]

def wait(pid, t=5400):
    t0 = time.time()
    while time.time() - t0 < t:
        try:
            h = json.load(urllib.request.urlopen(COMFY + f"/history/{pid}"))
            if pid in h: return h[pid]
        except Exception: pass
        time.sleep(6)

if __name__ == "__main__":
    # args: out_name still_name seconds "prompt" [w h]
    out_name, still_name, seconds, prompt = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    w = int(sys.argv[5]) if len(sys.argv) > 5 else 832
    h = int(sys.argv[6]) if len(sys.argv) > 6 else 480
    graph = wf(still_name, prompt, seconds, seed=12345, w=w, h=h)
    pid = post(graph)
    print("submitted", out_name, pid, flush=True)
    res = wait(pid)
    if not res or res.get("status", {}).get("status_str") != "success":
        print("FAILED", res.get("status") if res else "timeout")
        import sys as _s; _s.exit(1)
    vid = None
    for o in res.get("outputs", {}).values():
        for v in o.get("gifs", []) + o.get("videos", []): vid = v
    url = COMFY + "/view?" + urllib.parse.urlencode({"filename": vid["filename"], "subfolder": vid.get("subfolder", ""), "type": vid.get("type", "output")})
    dst = CLIPS / f"{out_name}.mp4"; dst.write_bytes(urllib.request.urlopen(url).read())
    print("SAVED", dst)
