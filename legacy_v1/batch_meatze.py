#!/usr/bin/env python3
"""MEATZE overnight batch: for each scene compose a still (Qwen-Image-Edit two-ref: presenter+room,
ReActor face-lock) then animate it speaking via InfiniteTalk, QA it, and push to Telegram.
Runs sequentially so a single job owns VRAM; queues politely behind the fox-worker.
Per-scene try/except + one retry so one failure can't abort the night."""
import json, time, urllib.request, urllib.parse, subprocess, pathlib, sys, traceback

COMFY = "http://127.0.0.1:8188"
ROOT = pathlib.Path("/home/iaadmin/tg_comfy_bot")
OUT = ROOT / "outputs"
CLIPS = OUT / "clips"; CLIPS.mkdir(parents=True, exist_ok=True)
STILLS = OUT / "stills"; STILLS.mkdir(parents=True, exist_ok=True)
CIN = pathlib.Path("/home/iaadmin/ComfyUI/input")
LOG = OUT / "batch_log.txt"
SCENES = json.load(open(OUT / "scenes.json"))

TOKEN = subprocess.check_output(
    ["grep", "-oP", r"TELEGRAM_BOT_TOKEN=\K\S+", "/etc/systemd/system/tg-comfy-bot.service"]).decode().strip()
CHAT = "517188056"
W, H, FPS, FRAMES = 832, 480, 25.0, 150  # 480p native base; upscaled x2 -> 1664x960

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")

def tg_video(path, caption):
    try:
        subprocess.run(["curl", "-s", f"https://api.telegram.org/bot{TOKEN}/sendVideo",
            "-F", f"chat_id={CHAT}", "-F", f"video=@{path}", "-F", f"caption={caption}"],
            capture_output=True, timeout=180)
    except Exception as e:
        log(f"  telegram send failed: {e}")

def tg_msg(text):
    try:
        subprocess.run(["curl", "-s", f"https://api.telegram.org/bot{TOKEN}/sendMessage",
            "--data-urlencode", f"chat_id={CHAT}", "--data-urlencode", f"text={text}"],
            capture_output=True, timeout=60)
    except Exception: pass

def post(wf):
    data = json.dumps({"prompt": wf}).encode()
    req = urllib.request.Request(COMFY + "/prompt", data=data, headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))["prompt_id"]

def wait(pid, timeout=5400):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            h = json.load(urllib.request.urlopen(COMFY + f"/history/{pid}"))
            if pid in h: return h[pid]
        except Exception: pass
        time.sleep(6)
    return None

def fetch(item, dst):
    url = COMFY + "/view?" + urllib.parse.urlencode(
        {"filename": item["filename"], "subfolder": item.get("subfolder", ""), "type": item.get("type", "output")})
    dst.write_bytes(urllib.request.urlopen(url).read())

# ---------- still compose ----------
def reactor(chain, src, gender):
    return {"class_type": "ReActorFaceSwap", "inputs": {
        "enabled": True, "input_image": chain, "source_image": [src, 0],
        "swap_model": "inswapper_128.onnx", "facedetection": "retinaface_resnet50",
        "face_restore_model": "GFPGANv1.3.pth", "face_restore_visibility": 0.7,
        "codeformer_weight": 0.5, "detect_gender_input": gender, "detect_gender_source": gender,
        "input_faces_index": "0", "source_faces_index": "0", "console_log_level": 1}}

def still_wf(presenter_img, room_img, gender, wardrobe, seed):
    pos = (f"A photorealistic professional head-and-shoulders portrait of the {'woman' if gender=='female' else 'man'} "
           f"from the first image, standing in the room shown in the second image. Framed from the top of the head "
           f"down to the upper chest only; hands and arms are NOT visible, below the frame. Facing the camera with a "
           f"calm, friendly, restrained expression, mouth closed. Wearing {wardrobe}. Keep the exact face from the "
           f"first image. Keep the room EXACTLY as in the second image: same furniture, computers, windows, walls, "
           f"signage and perspective, nicely blurred behind the presenter. Realistic human scale, natural indoor "
           f"lighting, soft shadow, physically inside the room, not pasted on top.")
    neg = ("hands, arms, fingers, gesturing, cartoon, illustration, 3d render, extra people, second person, "
           "duplicated face, deformed face, distorted monitors, warped furniture, changed room layout, invented "
           "walls, floating, cutout, flat sticker, text, watermark, nudity, oversaturated")
    return {
        "72": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image"}},
        "71": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "73": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_edit_2509_fp8_e4m3fn.safetensors", "weight_dtype": "default"}},
        "74": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["73", 0], "lora_name": "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors", "strength_model": 1.0}},
        "67": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["74", 0], "shift": 3.1}},
        "41": {"class_type": "LoadImage", "inputs": {"image": presenter_img}},
        "42": {"class_type": "LoadImage", "inputs": {"image": room_img}},
        "68": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["72", 0], "prompt": pos, "vae": ["71", 0], "image1": ["41", 0], "image2": ["42", 0]}},
        "69": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["72", 0], "prompt": neg, "vae": ["71", 0], "image1": ["41", 0], "image2": ["42", 0]}},
        "66": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 1152, "height": 640, "batch_size": 1}},
        "65": {"class_type": "KSampler", "inputs": {"model": ["67", 0], "seed": seed, "steps": 4, "cfg": 1.0,
                "sampler_name": "euler", "scheduler": "simple", "positive": ["68", 0], "negative": ["69", 0],
                "latent_image": ["66", 0], "denoise": 1}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["65", 0], "vae": ["71", 0]}},
        "80": reactor(["8", 0], "41", gender),
        "128": {"class_type": "SaveImage", "inputs": {"images": ["80", 0], "filename_prefix": "meatze_batch_still"}},
    }

# ---------- infinitetalk ----------
def video_wf(still_img, audio_file, seed, frames):
    pos = ("A presenter speaks calmly and warmly to the camera inside a professional training centre. "
           "Very subtle natural motion: gentle head movement, natural blinking, a slight weight shift only. "
           "Hands stay down, out of frame. Realistic, cinematic, restrained professional expression, correct lip sync.")
    neg = ("raising hands, hand gestures, arms, fingers, deformed hands, waving, bright tones, overexposed, static, "
           "blurred face, subtitles, text, watermark, worst quality, low quality, cartoon, exaggerated smile, "
           "duplicated person, distorted room, warped monitors, nudity")
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": still_img}},
        "2": {"class_type": "ImageResizeKJv2", "inputs": {"image": ["1", 0], "width": W, "height": H,
              "upscale_method": "lanczos", "keep_proportion": "crop", "pad_color": "0, 0, 0",
              "crop_position": "center", "divisible_by": 16, "device": "cpu"}},
        "3": {"class_type": "LoadAudio", "inputs": {"audio": audio_file}},
        "4": {"class_type": "Wav2VecModelLoader", "inputs": {"model": "wav2vec2-chinese-base_fp16.safetensors",
              "base_precision": "fp16", "load_device": "main_device"}},
        "5": {"class_type": "MultiTalkWav2VecEmbeds", "inputs": {"wav2vec_model": ["4", 0], "audio_1": ["3", 0],
              "normalize_loudness": True, "num_frames": frames, "fps": FPS, "audio_scale": 1.0,
              "audio_cfg_scale": 1.0, "multi_audio_type": "para"}},
        "6": {"class_type": "WanVideoVAELoader", "inputs": {"model_name": "Wan2_1_VAE_bf16.safetensors",
              "precision": "bf16", "use_cpu_cache": True, "verbose": False}},
        "7": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": "clip_vision_h.safetensors"}},
        "8": {"class_type": "WanVideoClipVisionEncode", "inputs": {"clip_vision": ["7", 0], "image_1": ["2", 0],
              "strength_1": 1.0, "strength_2": 1.0, "crop": "center", "combine_embeds": "average",
              "force_offload": True, "tiles": 0, "ratio": 0.5}},
        "9": {"class_type": "WanVideoImageToVideoMultiTalk", "inputs": {"vae": ["6", 0], "width": W, "height": H,
              "frame_window_size": 81, "motion_frame": 9, "force_offload": True, "colormatch": "disabled",
              "start_image": ["2", 0], "tiled_vae": True, "clip_embeds": ["8", 0], "mode": "infinitetalk", "output_path": ""}},
        "10": {"class_type": "WanVideoLoraSelect", "inputs": {"lora": "nsfw_wan_14b_sex.safetensors",
               "strength": 0.0, "low_mem_load": True, "merge_loras": False}},
        "11": {"class_type": "MultiTalkModelLoader", "inputs": {"model": "WanVideo/InfiniteTalk/Wan2_1-InfiniteTalk_Single_Q4_K_M.gguf"}},
        "12": {"class_type": "WanVideoBlockSwap", "inputs": {"blocks_to_swap": 36, "offload_img_emb": True,
               "offload_txt_emb": True, "use_non_blocking": True, "vace_blocks_to_swap": 0, "prefetch_blocks": 1, "block_swap_debug": False}},
        "13": {"class_type": "WanVideoModelLoader", "inputs": {"model": "WanVideo/wan2.1-i2v-14b-480p-Q4_K_M.gguf",
               "base_precision": "fp16_fast", "quantization": "disabled", "load_device": "offload_device",
               "attention_mode": "sageattn", "block_swap_args": ["12", 0], "lora": ["10", 0],
               "multitalk_model": ["11", 0], "rms_norm_function": "default"}},
        "14": {"class_type": "WanVideoTextEncodeCached", "inputs": {"model_name": "umt5-xxl-enc-bf16.safetensors",
               "precision": "bf16", "positive_prompt": pos, "negative_prompt": neg,
               "quantization": "disabled", "use_disk_cache": True, "device": "cpu"}},
        "15": {"class_type": "WanVideoSampler", "inputs": {"model": ["13", 0], "image_embeds": ["9", 0],
               "steps": 4, "cfg": 1.0, "shift": 11.0, "seed": seed, "force_offload": True, "scheduler": "dpm++_sde",
               "riflex_freq_index": 0, "text_embeds": ["14", 0], "multitalk_embeds": ["5", 0], "rope_function": "comfy",
               "start_step": 0, "end_step": -1, "add_noise_to_samples": True}},
        "16": {"class_type": "WanVideoPassImagesFromSamples", "inputs": {"samples": ["15", 0]}},
        "18": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "2xNomosUni_span_multijpg.safetensors"}},
        "19": {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["18", 0], "image": ["16", 0]}},
        "17": {"class_type": "VHS_VideoCombine", "inputs": {"images": ["19", 0], "audio": ["5", 1], "frame_rate": FPS,
               "loop_count": 0, "filename_prefix": "meatze_batch", "format": "video/h264-mp4", "pix_fmt": "yuv420p",
               "crf": 19, "save_metadata": False, "trim_to_audio": False, "pingpong": False, "save_output": True}},
    }

def qa(mp4):
    try:
        out = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries",
            "format=duration:stream=codec_type", "-of", "json", str(mp4)]).decode()
        j = json.loads(out)
        types = {s["codec_type"] for s in j.get("streams", [])}
        dur = float(j["format"]["duration"])
        return ("video" in types and "audio" in types and dur > 3.0), dur, types
    except Exception as e:
        return False, 0, str(e)

def do_still(s):
    presenter = "meatze_woman.jpg" if s["gender"] == "female" else "meatze_man.jpg"
    room = f"meatze_room_{s['room']}.jpg"
    wf = still_wf(presenter, room, s["gender"], s["wardrobe"], seed=hash(s["id"]) % 100000)
    pid = post(wf)
    res = wait(pid, timeout=1200)
    if not res or res.get("status", {}).get("status_str") != "success":
        raise RuntimeError(f"still failed: {res.get('status') if res else 'timeout'}")
    img = None
    for o in res.get("outputs", {}).values():
        for im in o.get("images", []): img = im
    if not img: raise RuntimeError("still: no image output")
    dst = STILLS / f"{s['id']}_still.jpg"; fetch(img, dst)
    cin = CIN / f"meatze_still_{s['id']}.jpg"; cin.write_bytes(dst.read_bytes())
    return f"meatze_still_{s['id']}.jpg"

def do_video(s, still_name):
    audio = f"meatze_voice_{s['id']}_6s.wav"
    wav = pathlib.Path("/home/iaadmin/tg_comfy_bot/outputs/voices") / f"{s['id']}_6s.wav"
    try:
        dur = float(subprocess.check_output(["ffprobe","-v","error","-show_entries","format=duration","-of","default=nk=1:nw=1",str(wav)]).decode().strip())
    except Exception:
        dur = 6.0
    frames = max(150, int(round(dur * FPS)) + 2)
    log(f"  audio {dur:.2f}s -> {frames} frames")
    wf = video_wf(still_name, audio, seed=1, frames=frames)
    pid = post(wf)
    res = wait(pid, timeout=5400)
    if not res or res.get("status", {}).get("status_str") != "success":
        raise RuntimeError(f"video failed: {res.get('status') if res else 'timeout'}")
    vid = None
    for o in res.get("outputs", {}).values():
        for v in o.get("gifs", []) + o.get("videos", []): vid = v
    if not vid: raise RuntimeError("video: no output")
    dst = CLIPS / f"{s['id']}_{s['presenter']}_{s['room']}.mp4"; fetch(vid, dst)
    return dst

def run_scene(s):
    log(f"=== {s['id']} ({s['presenter']}, {s['center']}, room {s['room']}) ===")
    for attempt in (1, 2):
        try:
            still = do_still(s); log(f"  still ok ({still})")
            mp4 = do_video(s, still); ok, dur, types = qa(mp4)
            log(f"  video ok ({mp4.name}) qa_pass={ok} dur={dur:.1f} streams={types}")
            if not ok and attempt == 1:
                log("  QA failed, retry once"); continue
            cap = (f"🎬 MEATZE {s['id'].upper()} — {s['center']}\n"
                   f"{'👩 Ведущая' if s['gender']=='female' else '👨 Ведущий'}: «{s['line_es']}»\n"
                   f"{dur:.1f}с, липсинк+движение, реальная комната.")
            tg_video(mp4, cap)
            return True
        except Exception as e:
            log(f"  attempt {attempt} error: {e}")
            if attempt == 2:
                log(f"  SCENE {s['id']} FAILED after retry\n{traceback.format_exc()}")
                tg_msg(f"⚠️ {s['id']} не удался: {e}")
                return False
    return False

if __name__ == "__main__":
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    todo = [s for s in SCENES if (not only or s["id"] in only)]
    log(f"BATCH START: {len(todo)} scenes")
    tg_msg(f"🌙 Ночной батч MEATZE: {len(todo)} сцен. Результаты будут приходить сюда по мере готовности.")
    done = 0
    for s in todo:
        if run_scene(s): done += 1
    log(f"BATCH DONE: {done}/{len(todo)} succeeded")
    tg_msg(f"✅ Батч завершён: {done}/{len(todo)} роликов готово. Утром соберу черновой монтаж + отчёт.")
