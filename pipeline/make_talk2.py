#!/usr/bin/env python3
"""Talking shot v2 (InfiniteTalk): expressive, not robotic; still OR moving video as source; RIFE x2 -> 50 fps.
- Image mode: still -> lip-sync with lively head/eyebrow/shoulder motion.
- V2V mode (--video): dubs lip-sync onto an existing motion clip (presenter walks / sits / points while talking).
args: out_name source audio [--video] [--prompt "..."] [--seed n] [--denoise-start k]"""
import json, time, urllib.request, urllib.parse, sys, pathlib, subprocess, argparse

COMFY = "http://127.0.0.1:8188"
INPUT = pathlib.Path("/home/iaadmin/ComfyUI/input")
CLIPS = pathlib.Path("/home/iaadmin/tg_comfy_bot/outputs/v2/clips"); CLIPS.mkdir(parents=True, exist_ok=True)
W, H, FPS = 832, 480, 25.0
LORA_STRENGTH = 0.7
DISTILL = "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"   # without it 4 steps leave ghost blinks / grid teeth
POS = ("A calm, friendly, professional presenter talks to the camera in a natural conversational tone, looking into the lens. "
       "Natural head movement, eyebrows move with the words, gentle nods, natural blinking, relaxed shoulders. "
       "Lips relaxed while talking, only a light closed-mouth smile between phrases, never a wide toothy grin. "
       "Clear articulate lip movement, correct lip sync. Realistic, cinematic, natural real-time speed.")
NEG = ("stiff, frozen face, robotic, staring, intense stare, expressionless, still body, slow motion, "
       "toothy grin, laughing, exaggerated smile, deformed hands, extra fingers, bright tones, overexposed, "
       "blurred face, subtitles, text, watermark, worst quality, low quality, cartoon, duplicated person, distorted room, nudity")

def build(src, audio, frames, video=False, prompt=POS, seed=1, start_step=0, rife=2, lora=None):
    g = {
        "3": {"class_type": "LoadAudio", "inputs": {"audio": audio}},
        "4": {"class_type": "Wav2VecModelLoader", "inputs": {"model": "wav2vec2-chinese-base_fp16.safetensors", "base_precision": "fp16", "load_device": "main_device"}},
        "5": {"class_type": "MultiTalkWav2VecEmbeds", "inputs": {"wav2vec_model": ["4", 0], "audio_1": ["3", 0], "normalize_loudness": True,
              "num_frames": frames, "fps": FPS, "audio_scale": 1.15, "audio_cfg_scale": 1.0, "multi_audio_type": "para"}},
        "6": {"class_type": "WanVideoVAELoader", "inputs": {"model_name": "Wan2_1_VAE_bf16.safetensors", "precision": "bf16", "use_cpu_cache": True, "verbose": False}},
        "7": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": "clip_vision_h.safetensors"}},
        "10": {"class_type": "WanVideoLoraSelect", "inputs": {"lora": lora or "nsfw_wan_14b_sex.safetensors", "strength": (LORA_STRENGTH if lora else 0.0), "low_mem_load": True, "merge_loras": False}},
        "11": {"class_type": "MultiTalkModelLoader", "inputs": {"model": "WanVideo/InfiniteTalk/Wan2_1-InfiniteTalk_Single_Q4_K_M.gguf"}},
        "12": {"class_type": "WanVideoBlockSwap", "inputs": {"blocks_to_swap": 36, "offload_img_emb": True, "offload_txt_emb": True,
               "use_non_blocking": True, "vace_blocks_to_swap": 0, "prefetch_blocks": 1, "block_swap_debug": False}},
        "13": {"class_type": "WanVideoModelLoader", "inputs": {"model": "WanVideo/wan2.1-i2v-14b-480p-Q4_K_M.gguf", "base_precision": "fp16_fast",
               "quantization": "disabled", "load_device": "offload_device", "attention_mode": "sageattn", "block_swap_args": ["12", 0],
               "lora": ["10", 0], "multitalk_model": ["11", 0], "rms_norm_function": "default"}},
        "14": {"class_type": "WanVideoTextEncodeCached", "inputs": {"model_name": "umt5-xxl-enc-bf16.safetensors", "precision": "bf16",
               "positive_prompt": prompt, "negative_prompt": NEG, "quantization": "disabled", "use_disk_cache": True, "device": "cpu"}},
    }
    if video:
        g["1"] = {"class_type": "VHS_LoadVideo", "inputs": {"video": src, "force_rate": FPS, "custom_width": W, "custom_height": H,
                  "frame_load_cap": 0, "skip_first_frames": 0, "select_every_nth": 1, "format": "Wan"}}
        g["2"] = {"class_type": "ImageResizeKJv2", "inputs": {"image": ["1", 0], "width": W, "height": H, "upscale_method": "lanczos",
                  "keep_proportion": "crop", "pad_color": "0, 0, 0", "crop_position": "center", "divisible_by": 16, "device": "cpu"}}
        g["2f"] = {"class_type": "GetImageRangeFromBatch", "inputs": {"images": ["2", 0], "start_index": 0, "num_frames": 1}}
        g["enc"] = {"class_type": "WanVideoEncode", "inputs": {"vae": ["6", 0], "image": ["2", 0], "enable_vae_tiling": True,
                    "tile_x": 272, "tile_y": 272, "tile_stride_x": 144, "tile_stride_y": 128}}
        first = ["2f", 0]
    else:
        g["1"] = {"class_type": "LoadImage", "inputs": {"image": src}}
        g["2"] = {"class_type": "ImageResizeKJv2", "inputs": {"image": ["1", 0], "width": W, "height": H, "upscale_method": "lanczos",
                  "keep_proportion": "crop", "pad_color": "0, 0, 0", "crop_position": "center", "divisible_by": 16, "device": "cpu"}}
        first = ["2", 0]
    g["8"] = {"class_type": "WanVideoClipVisionEncode", "inputs": {"clip_vision": ["7", 0], "image_1": first, "strength_1": 1.0, "strength_2": 1.0,
              "crop": "center", "combine_embeds": "average", "force_offload": True, "tiles": 0, "ratio": 0.5}}
    g["9"] = {"class_type": "WanVideoImageToVideoMultiTalk", "inputs": {"vae": ["6", 0], "width": W, "height": H, "frame_window_size": 81,
              "motion_frame": 9, "force_offload": True, "colormatch": "disabled", "start_image": first, "tiled_vae": True,
              "clip_embeds": ["8", 0], "mode": "infinitetalk", "output_path": ""}}
    sampler = {"model": ["13", 0], "image_embeds": ["9", 0], "steps": 4, "cfg": 1.0, "shift": 11.0,
               "seed": seed, "force_offload": True, "scheduler": "dpm++_sde", "riflex_freq_index": 0, "text_embeds": ["14", 0],
               "multitalk_embeds": ["5", 0], "rope_function": "comfy", "start_step": start_step, "end_step": -1, "add_noise_to_samples": True}
    if video: sampler["samples"] = ["enc", 0]
    g["15"] = {"class_type": "WanVideoSampler", "inputs": sampler}
    g["16"] = {"class_type": "WanVideoPassImagesFromSamples", "inputs": {"samples": ["15", 0]}}
    g["rf"] = {"class_type": "RIFE VFI", "inputs": {"frames": ["16", 0], "ckpt_name": "rife49.pth", "clear_cache_after_n_frames": 10,
               "multiplier": rife, "fast_mode": True, "ensemble": True, "scale_factor": 1.0, "dtype": "float32",
               "torch_compile": False, "batch_size": 1}}
    g["18"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "2xNomosUni_span_multijpg.safetensors"}}
    g["19"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["18", 0], "image": ["rf", 0]}}
    g["17"] = {"class_type": "VHS_VideoCombine", "inputs": {"images": ["19", 0], "audio": ["5", 1], "frame_rate": FPS * rife, "loop_count": 0,
               "filename_prefix": "meatze_talk2", "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 17, "save_metadata": False,
               "trim_to_audio": True, "pingpong": False, "save_output": True}}
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

def to_input(path):
    p = pathlib.Path(path); dst = INPUT / p.name
    if p.resolve() != dst.resolve(): dst.write_bytes(p.read_bytes())
    return p.name

def facecrop(src, target=0.45):
    tag = "" if target == 0.45 else f"_{int(target * 100)}"
    """InfiniteTalk breaks small faces (garbled eyes / grid teeth) -> talk only on close-ups: crop the still around the face."""
    dst = CLIPS.parent / "crops" / (pathlib.Path(src).stem + tag + ".png"); dst.parent.mkdir(exist_ok=True)
    if not dst.exists() or dst.stat().st_mtime < pathlib.Path(src).stat().st_mtime:
        r = subprocess.run(["/home/iaadmin/ComfyUI/venv/bin/python", str(CLIPS.parent / "facecrop.py"), str(src), str(dst), str(target)],
                           capture_output=True, text=True)
        print("facecrop", pathlib.Path(src).name, r.stdout.strip(), flush=True)
    return dst

def run(out_name, src, audio, video=False, prompt=POS, seed=1, start_step=0, crop=0.45, lora=DISTILL):
    if not video: src = facecrop(src, crop)
    wav = pathlib.Path(audio)
    dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nk=1:nw=1", str(wav)]).decode())
    frames = int(round(dur * FPS)) + 2
    pid = post(build(to_input(src), to_input(audio), frames, video, prompt, seed, start_step, lora=lora))
    print("submitted", out_name, pid, "frames", frames, flush=True)
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
    ap.add_argument("out"); ap.add_argument("src"); ap.add_argument("audio")
    ap.add_argument("--video", action="store_true"); ap.add_argument("--prompt", default=POS)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--start-step", type=int, default=0)
    a = ap.parse_args()
    sys.exit(0 if run(a.out, a.src, a.audio, a.video, a.prompt, a.seed, a.start_step) else 1)
