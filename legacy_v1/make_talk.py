#!/usr/bin/env python3
"""Talking close-up: chest-up still + voice -> InfiniteTalk lip-sync, 832x480 -> x2 upscale 1664x960.
Same proven recipe as the batch. args: out_name still_name audio_name_6s"""
import json, time, urllib.request, urllib.parse, sys, pathlib, subprocess

COMFY = "http://127.0.0.1:8188"
CLIPS = pathlib.Path("/home/iaadmin/tg_comfy_bot/outputs/clips"); CLIPS.mkdir(parents=True, exist_ok=True)
W, H, FPS = 832, 480, 25.0
POS = ("A presenter speaks calmly and warmly to the camera inside a professional training centre. Very subtle "
       "natural motion: gentle head movement, natural blinking, slight weight shift. Hands stay out of frame. "
       "Realistic, cinematic, restrained professional expression, correct lip sync.")
NEG = ("raising hands, hand gestures, arms, fingers, deformed hands, waving, bright tones, overexposed, static, "
       "blurred face, subtitles, text, watermark, worst quality, low quality, cartoon, exaggerated smile, "
       "duplicated person, distorted room, warped monitors, nudity")

def build(still, audio, frames, seed=1):
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": still}},
        "2": {"class_type": "ImageResizeKJv2", "inputs": {"image": ["1", 0], "width": W, "height": H, "upscale_method": "lanczos",
              "keep_proportion": "crop", "pad_color": "0, 0, 0", "crop_position": "center", "divisible_by": 16, "device": "cpu"}},
        "3": {"class_type": "LoadAudio", "inputs": {"audio": audio}},
        "4": {"class_type": "Wav2VecModelLoader", "inputs": {"model": "wav2vec2-chinese-base_fp16.safetensors", "base_precision": "fp16", "load_device": "main_device"}},
        "5": {"class_type": "MultiTalkWav2VecEmbeds", "inputs": {"wav2vec_model": ["4", 0], "audio_1": ["3", 0], "normalize_loudness": True,
              "num_frames": frames, "fps": FPS, "audio_scale": 1.0, "audio_cfg_scale": 1.0, "multi_audio_type": "para"}},
        "6": {"class_type": "WanVideoVAELoader", "inputs": {"model_name": "Wan2_1_VAE_bf16.safetensors", "precision": "bf16", "use_cpu_cache": True, "verbose": False}},
        "7": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": "clip_vision_h.safetensors"}},
        "8": {"class_type": "WanVideoClipVisionEncode", "inputs": {"clip_vision": ["7", 0], "image_1": ["2", 0], "strength_1": 1.0, "strength_2": 1.0,
              "crop": "center", "combine_embeds": "average", "force_offload": True, "tiles": 0, "ratio": 0.5}},
        "9": {"class_type": "WanVideoImageToVideoMultiTalk", "inputs": {"vae": ["6", 0], "width": W, "height": H, "frame_window_size": 81,
              "motion_frame": 9, "force_offload": True, "colormatch": "disabled", "start_image": ["2", 0], "tiled_vae": True,
              "clip_embeds": ["8", 0], "mode": "infinitetalk", "output_path": ""}},
        "10": {"class_type": "WanVideoLoraSelect", "inputs": {"lora": "nsfw_wan_14b_sex.safetensors", "strength": 0.0, "low_mem_load": True, "merge_loras": False}},
        "11": {"class_type": "MultiTalkModelLoader", "inputs": {"model": "WanVideo/InfiniteTalk/Wan2_1-InfiniteTalk_Single_Q4_K_M.gguf"}},
        "12": {"class_type": "WanVideoBlockSwap", "inputs": {"blocks_to_swap": 36, "offload_img_emb": True, "offload_txt_emb": True,
               "use_non_blocking": True, "vace_blocks_to_swap": 0, "prefetch_blocks": 1, "block_swap_debug": False}},
        "13": {"class_type": "WanVideoModelLoader", "inputs": {"model": "WanVideo/wan2.1-i2v-14b-480p-Q4_K_M.gguf", "base_precision": "fp16_fast",
               "quantization": "disabled", "load_device": "offload_device", "attention_mode": "sageattn", "block_swap_args": ["12", 0],
               "lora": ["10", 0], "multitalk_model": ["11", 0], "rms_norm_function": "default"}},
        "14": {"class_type": "WanVideoTextEncodeCached", "inputs": {"model_name": "umt5-xxl-enc-bf16.safetensors", "precision": "bf16",
               "positive_prompt": POS, "negative_prompt": NEG, "quantization": "disabled", "use_disk_cache": True, "device": "cpu"}},
        "15": {"class_type": "WanVideoSampler", "inputs": {"model": ["13", 0], "image_embeds": ["9", 0], "steps": 4, "cfg": 1.0, "shift": 11.0,
               "seed": seed, "force_offload": True, "scheduler": "dpm++_sde", "riflex_freq_index": 0, "text_embeds": ["14", 0],
               "multitalk_embeds": ["5", 0], "rope_function": "comfy", "start_step": 0, "end_step": -1, "add_noise_to_samples": True}},
        "16": {"class_type": "WanVideoPassImagesFromSamples", "inputs": {"samples": ["15", 0]}},
        "18": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "2xNomosUni_span_multijpg.safetensors"}},
        "19": {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["18", 0], "image": ["16", 0]}},
        "17": {"class_type": "VHS_VideoCombine", "inputs": {"images": ["19", 0], "audio": ["5", 1], "frame_rate": FPS, "loop_count": 0,
               "filename_prefix": "meatze_talk", "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 19, "save_metadata": False,
               "trim_to_audio": False, "pingpong": False, "save_output": True}},
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
    out_name, still, audio = sys.argv[1], sys.argv[2], sys.argv[3]
    wav = pathlib.Path("/home/iaadmin/ComfyUI/input") / audio
    try:
        dur = float(subprocess.check_output(["ffprobe","-v","error","-show_entries","format=duration","-of","default=nk=1:nw=1",str(wav)]).decode())
    except Exception:
        dur = 6.0
    frames = max(150, int(round(dur * FPS)) + 2)
    graph = build(still, audio, frames)
    pid = post(graph)
    print("submitted", out_name, pid, "frames", frames, flush=True)
    res = wait(pid)
    if not res or res.get("status", {}).get("status_str") != "success":
        print("FAILED", res.get("status") if res else "timeout"); sys.exit(1)
    vid = None
    for o in res.get("outputs", {}).values():
        for v in o.get("gifs", []) + o.get("videos", []): vid = v
    url = COMFY + "/view?" + urllib.parse.urlencode({"filename": vid["filename"], "subfolder": vid.get("subfolder", ""), "type": vid.get("type", "output")})
    dst = CLIPS / f"{out_name}.mp4"; dst.write_bytes(urllib.request.urlopen(url).read())
    print("SAVED", dst)
