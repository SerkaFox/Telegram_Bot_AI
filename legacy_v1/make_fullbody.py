#!/usr/bin/env python3
"""Compose a FULL-BODY presenter still at a given room (reception etc.) via Qwen-Image-Edit two-ref
+ ReActor. For dynamic Wan-I2V shots (walk / wave / show). Standing, head-to-feet, natural pose."""
import json, time, urllib.request, urllib.parse, sys, pathlib

COMFY = "http://127.0.0.1:8188"
OUT = pathlib.Path("/home/iaadmin/tg_comfy_bot/outputs/stills"); OUT.mkdir(parents=True, exist_ok=True)
CIN = pathlib.Path("/home/iaadmin/ComfyUI/input")

def reactor(chain, src, gender):
    return {"class_type": "ReActorFaceSwap", "inputs": {
        "enabled": True, "input_image": chain, "source_image": [src, 0],
        "swap_model": "inswapper_128.onnx", "facedetection": "retinaface_resnet50",
        "face_restore_model": "GFPGANv1.3.pth", "face_restore_visibility": 0.7,
        "codeformer_weight": 0.5, "detect_gender_input": gender, "detect_gender_source": gender,
        "input_faces_index": "0", "source_faces_index": "0", "console_log_level": 1}}

def wf(presenter, room, gender, wardrobe, pose, seed, w=768, h=1024):
    who = "woman" if gender == "female" else "man"
    pos = (f"A photorealistic FULL-BODY photograph of the {who} from the first image, standing in the room shown "
           f"in the second image, {pose}. Head-to-knees / full figure visible, correct human proportions and scale, "
           f"feet on the floor. Facing the camera, calm serious professional expression with only a slight closed-mouth smile (no teeth). Wearing {wardrobe}. "
           f"Keep the exact face from the first image. Keep the room EXACTLY as in the second image: same furniture, "
           f"reception desk, signage, walls, lighting and perspective. Do NOT add any posters, signs, screens, furniture or decorations that are not in the second image. Natural indoor lighting, realistic contact "
           f"shadow, physically present in the room, not pasted on top, photographic.")
    neg = ("toothy grin, wide smile, posters, invented signage, fake text on walls, cropped head, cut off legs, deformed hands, extra fingers, extra limbs, cartoon, illustration, 3d render, "
           "extra people, duplicated face, distorted furniture, changed room layout, floating, cutout, flat sticker, "
           "text, watermark, nudity, oversaturated")
    return {
        "72": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image"}},
        "71": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "73": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_edit_2509_fp8_e4m3fn.safetensors", "weight_dtype": "default"}},
        "74": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["73", 0], "lora_name": "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors", "strength_model": 1.0}},
        "67": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["74", 0], "shift": 3.1}},
        "41": {"class_type": "LoadImage", "inputs": {"image": presenter}},
        "42": {"class_type": "LoadImage", "inputs": {"image": room}},
        "68": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["72", 0], "prompt": pos, "vae": ["71", 0], "image1": ["41", 0], "image2": ["42", 0]}},
        "69": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["72", 0], "prompt": neg, "vae": ["71", 0], "image1": ["41", 0], "image2": ["42", 0]}},
        "66": {"class_type": "EmptySD3LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "65": {"class_type": "KSampler", "inputs": {"model": ["67", 0], "seed": seed, "steps": 4, "cfg": 1.0,
                "sampler_name": "euler", "scheduler": "simple", "positive": ["68", 0], "negative": ["69", 0],
                "latent_image": ["66", 0], "denoise": 1}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["65", 0], "vae": ["71", 0]}},
        "80": reactor(["8", 0], "41", gender),
        "128": {"class_type": "SaveImage", "inputs": {"images": ["80", 0], "filename_prefix": "meatze_fb"}},
    }

def post(w):
    data = json.dumps({"prompt": w}).encode()
    return json.load(urllib.request.urlopen(urllib.request.Request(COMFY + "/prompt", data=data, headers={"Content-Type": "application/json"})))["prompt_id"]

def wait(pid, t=1200):
    t0 = time.time()
    while time.time() - t0 < t:
        try:
            h = json.load(urllib.request.urlopen(COMFY + f"/history/{pid}"))
            if pid in h: return h[pid]
        except Exception: pass
        time.sleep(5)

if __name__ == "__main__":
    # args: id presenter_img room_img gender "wardrobe" "pose" [w h]
    sid, presenter, room, gender, wardrobe, pose = sys.argv[1:7]
    w = int(sys.argv[7]) if len(sys.argv) > 7 else 768
    h = int(sys.argv[8]) if len(sys.argv) > 8 else 1024
    graph = wf(presenter, room, gender, wardrobe, pose, seed=abs(hash(sid)) % 100000, w=w, h=h)
    pid = post(graph)
    print("submitted", sid, pid, flush=True)
    res = wait(pid)
    if not res or res.get("status", {}).get("status_str") != "success":
        print("FAILED", res.get("status") if res else "timeout"); sys.exit(1)
    img = None
    for o in res.get("outputs", {}).values():
        for im in o.get("images", []): img = im
    url = COMFY + "/view?" + urllib.parse.urlencode({"filename": img["filename"], "subfolder": img.get("subfolder", ""), "type": img.get("type", "output")})
    dst = OUT / f"{sid}_fullbody.jpg"; dst.write_bytes(urllib.request.urlopen(url).read())
    (CIN / f"meatze_fb_{sid}.jpg").write_bytes(dst.read_bytes())
    print("SAVED", dst)
