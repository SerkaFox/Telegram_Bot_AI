#!/usr/bin/env python3
"""Wan 2.2 I2V v2 — anti slow-motion + smooth output.
- 3-stage sampling: first steps on the HIGH model WITHOUT the lightx2v lora at real CFG
  (restores motion speed that the distill lora kills), then HIGH+lora, then LOW+lora.
- RIFE x3 at native res (16 -> 48 fps), then 2x model upscale; saved at 48 fps.
- Optional last frame (first->last frame transition, e.g. camera push into a monitor).
args: out_name still seconds "prompt" [w h] [--last img] [--seed n] [--fast]"""
import json, time, urllib.request, urllib.parse, sys, pathlib, argparse

COMFY = "http://127.0.0.1:8188"
CLIPS = pathlib.Path("/home/iaadmin/tg_comfy_bot/outputs/v2/clips"); CLIPS.mkdir(parents=True, exist_ok=True)
NEG = ("slow motion, frozen, static, still image, stiff, robotic, "
       "toothy smile, showing teeth, open-mouth smile, laughing, seductive, flirtatious, coy, shy, looking down, flirty head tilt, wide grin, grinning, "
       "hip sway, swaying body, dancing, posing, model pose, palms toward the camera, stop gesture, overacting, "
       "色调艳丽，过曝，细节模糊不清，字幕，画面，整体发灰，最差质量，低质量，"
       "deformed hands, extra fingers, extra limbs, duplicated person, distorted face, warped furniture, text, watermark")

def wf(still, prompt, seconds, seed, w, h, last=None, steps=8, s_raw=2, s_high=4, cfg_raw=3.5, rife=3):
    length = int(seconds) * 16 + 1
    g = {
        "371": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf"}},
        "372": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf"}},
        "151": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["371", 0], "lora_name": "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors", "strength_model": 1.0}},
        "154": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["372", 0], "lora_name": "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors", "strength_model": 1.0}},
        "m0": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["371", 0], "shift": 8.0}},
        "m1": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["151", 0], "shift": 8.0}},
        "m2": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["154", 0], "shift": 8.0}},
        "364": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}},
        "357": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
        "97": {"class_type": "LoadImage", "inputs": {"image": still}},
        "361": {"class_type": "ImageResize+", "inputs": {"width": w, "height": h, "interpolation": "lanczos",
                "method": "fill / crop", "condition": "always", "multiple_of": 16, "image": ["97", 0]}},
        "93": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["364", 0]}},
        "360": {"class_type": "CLIPTextEncode", "inputs": {"text": NEG, "clip": ["364", 0]}},
    }
    if last:
        g["98"] = {"class_type": "LoadImage", "inputs": {"image": last}}
        g["362"] = {"class_type": "ImageResize+", "inputs": {"width": w, "height": h, "interpolation": "lanczos",
                    "method": "fill / crop", "condition": "always", "multiple_of": 16, "image": ["98", 0]}}
        g["358"] = {"class_type": "WanFirstLastFrameToVideo", "inputs": {"width": w, "height": h, "length": length, "batch_size": 1,
                    "positive": ["93", 0], "negative": ["360", 0], "vae": ["357", 0],
                    "start_image": ["361", 0], "end_image": ["362", 0]}}
    else:
        g["358"] = {"class_type": "WanImageToVideo", "inputs": {"width": w, "height": h, "length": length, "batch_size": 1,
                    "positive": ["93", 0], "negative": ["360", 0], "vae": ["357", 0], "start_image": ["361", 0]}}
    ks = lambda model, cfg, a, b, noise, left, lat: {"class_type": "KSamplerAdvanced", "inputs": {
        "model": [model, 0], "add_noise": "enable" if noise else "disable", "noise_seed": seed, "steps": steps, "cfg": cfg,
        "sampler_name": "euler", "scheduler": "simple", "positive": ["358", 0], "negative": ["358", 1],
        "latent_image": lat, "start_at_step": a, "end_at_step": b,
        "return_with_leftover_noise": "enable" if left else "disable"}}
    if s_raw:
        g["k0"] = ks("m0", cfg_raw, 0, s_raw, True, True, ["358", 2])
        g["k1"] = ks("m1", 1.0, s_raw, s_high, False, True, ["k0", 0])
    else:
        g["k1"] = ks("m1", 1.0, 0, s_high, True, True, ["358", 2])
    g["k2"] = ks("m2", 1.0, s_high, 10000, False, False, ["k1", 0])
    g["363"] = {"class_type": "VAEDecode", "inputs": {"samples": ["k2", 0], "vae": ["357", 0]}}
    g["rf"] = {"class_type": "RIFE VFI", "inputs": {"frames": ["363", 0], "ckpt_name": "rife49.pth", "clear_cache_after_n_frames": 10,
               "multiplier": rife, "fast_mode": True, "ensemble": True, "scale_factor": 1.0, "dtype": "float32",
               "torch_compile": False, "batch_size": 1}}
    g["380"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "2xNomosUni_span_multijpg.safetensors"}}
    g["381"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["380", 0], "image": ["rf", 0]}}
    g["314"] = {"class_type": "VHS_VideoCombine", "inputs": {"images": ["381", 0], "frame_rate": 16 * rife, "loop_count": 0,
                "filename_prefix": "meatze_v2", "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 17,
                "save_metadata": False, "trim_to_audio": False, "pingpong": False, "save_output": True}}
    return g

def post(w):
    data = json.dumps({"prompt": w}).encode()
    return json.load(urllib.request.urlopen(urllib.request.Request(COMFY + "/prompt", data=data, headers={"Content-Type": "application/json"})))["prompt_id"]

def wait(pid, t=7200):
    t0 = time.time()
    while time.time() - t0 < t:
        try:
            h = json.load(urllib.request.urlopen(COMFY + f"/history/{pid}"))
            if pid in h: return h[pid]
        except Exception: pass
        time.sleep(6)

def upload(path):
    """Copy a local image into ComfyUI/input and return its name."""
    p = pathlib.Path(path)
    dst = pathlib.Path("/home/iaadmin/ComfyUI/input") / p.name
    if p.resolve() != dst.resolve(): dst.write_bytes(p.read_bytes())
    return p.name

def run(out_name, still, seconds, prompt, w=832, h=480, last=None, seed=12345, **kw):
    pid = post(wf(upload(still), prompt, seconds, seed, w, h, last=upload(last) if last else None, **kw))
    print("submitted", out_name, pid, flush=True)
    res = wait(pid)
    if not res or res.get("status", {}).get("status_str") != "success":
        print("FAILED", out_name, (res or {}).get("status", "timeout"), flush=True); return None
    vid = None
    for o in res.get("outputs", {}).values():
        for v in o.get("gifs", []) + o.get("videos", []): vid = v
    url = COMFY + "/view?" + urllib.parse.urlencode({"filename": vid["filename"], "subfolder": vid.get("subfolder", ""), "type": vid.get("type", "output")})
    dst = CLIPS / f"{out_name}.mp4"; dst.write_bytes(urllib.request.urlopen(url).read())
    print("SAVED", dst, flush=True); return dst

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("still"); ap.add_argument("seconds", type=int); ap.add_argument("prompt")
    ap.add_argument("w", type=int, nargs="?", default=832); ap.add_argument("h", type=int, nargs="?", default=480)
    ap.add_argument("--last"); ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--fast", action="store_true", help="old 2-stage behaviour (no raw-CFG steps)")
    a = ap.parse_args()
    kw = {"s_raw": 0} if a.fast else {}
    sys.exit(0 if run(a.out, a.still, a.seconds, a.prompt, a.w, a.h, a.last, a.seed, **kw) else 1)
