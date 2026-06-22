import asyncio
import io
import json
import logging
import os
import re
import copy
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import requests
from deep_translator import GoogleTranslator
from PIL import Image, ImageOps
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, InputFile, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ============================================================
# CONFIG
# ============================================================
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
COMFY_BASE = os.getenv("COMFY_BASE", "http://127.0.0.1:8188").rstrip("/")

WORKFLOW_VIDEO = os.getenv("COMFY_WORKFLOW_VIDEO", "./workflow_video.json")
WORKFLOW_LTX_SULPHUR = os.getenv("COMFY_WORKFLOW_LTX_SULPHUR", "./LTX2.3_2.json")
WORKFLOW_MOPMIX = os.getenv("COMFY_WORKFLOW_MOPMIX", "./workflow_mopmix.json")
WORKFLOW_LTX_EROS = os.getenv("COMFY_WORKFLOW_LTX_EROS", "./workflow_ltx_eros.json")
WORKFLOW_MOPMIX_DUO = os.getenv("COMFY_WORKFLOW_MOPMIX_DUO", "./workflow_mopmix_duo.json")

TMP_DIR = Path(os.getenv("BOT_TMP_DIR", "./tmp_bot"))
TMP_DIR.mkdir(parents=True, exist_ok=True)

COMFY_INPUT_DIR = Path(os.getenv("COMFY_INPUT_DIR", "/home/iaadmin/ComfyUI/input"))
COMFY_OUTPUT_DIR = Path(os.getenv("COMFY_OUTPUT_DIR", "/home/iaadmin/ComfyUI/output"))

MAX_CAPTION = 1000

ALLOWED_USER_IDS = {
    int(x.strip()) for x in os.getenv("ALLOWED_USER_IDS", "").split(",") if x.strip().isdigit()
}

DEFAULT_SECONDS = int(os.getenv("DEFAULT_SECONDS", "8"))
MIN_SECONDS = int(os.getenv("MIN_SECONDS", "2"))
MAX_SECONDS = int(os.getenv("MAX_SECONDS", "12"))
DEFAULT_QUALITY = os.getenv("DEFAULT_QUALITY", "medium").strip().lower()
QUALITY_PRESETS = {
    "low": {"max_side": 480, "video_fps": 16},
    "medium": {"max_side": 640, "video_fps": 16},
    "high": {"max_side": 768, "video_fps": 16},
}
ROUND_TO = int(os.getenv("ROUND_TO", "64"))

# LTX Sulphur (LTX2.3_2.json) target generation resolution per quality level.
LTX_SULPHUR_QUALITY = {
    "low": (576, 324),
    "medium": (832, 468),
    "high": (1152, 648),
}

# MopMix (workflow_mopmix.json) SDXL bucket resolution per quality level.
MOPMIX_RESOLUTIONS = {
    "low": "768x1280 (0.6)",
    "medium": "832x1216 (0.68)",
    "high": "896x1152 (0.78)",
}
# img2img strength for mopmix: 0 keeps the uploaded photo untouched, 1 regenerates it
# from scratch. 0.6 redraws per the prompt while keeping the photo's pose/composition.
MOPMIX_DENOISE = float(os.getenv("MOPMIX_DENOISE", "0.6"))
# Duo's starting image is a rough side-by-side composite of two unrelated photos, not one
# coherent photo, so it needs more repainting than single-photo MopMix to merge into one scene.
MOPMIX_DUO_DENOISE = float(os.getenv("MOPMIX_DUO_DENOISE", "0.85"))

# LTX Eros (workflow_ltx_eros.json) target generation resolution per quality level
# (width, height fed to the MultiImageLoader; portrait orientation by default).
LTX_EROS_QUALITY = {
    "low": (288, 512),
    "medium": (416, 736),
    "high": (576, 1024),
}
# Original workflow's 4 keyframe insert points (seconds) at its default 12s duration;
# scaled proportionally to whatever duration the job actually requests.
LTX_EROS_KEYFRAME_SECONDS = [0, 3, 6, 10]
LTX_EROS_KEYFRAME_DEFAULT_SECONDS = 12
# The workflow's own "ablit-norms-biproj-fp8mixed" gemma text encoder fails to load on
# this ComfyUI install ('Linear' object has no attribute 'weight' on partial GPU load of
# this particular fp8 quantization scheme); substitute the gemma encoder already used
# successfully by LTX Sulphur instead.
LTX_EROS_CLIP_NAME1 = os.getenv("LTX_EROS_CLIP_NAME1", "gemma_3_12B_it_fp8_e4m3fn.safetensors")

# ReActor face-swap model/quality settings shared by mopmix_duo's two swap passes.
REACTOR_SWAP_MODEL = os.getenv("REACTOR_SWAP_MODEL", "inswapper_128.onnx")
REACTOR_FACE_DETECTION = os.getenv("REACTOR_FACE_DETECTION", "retinaface_resnet50")
REACTOR_FACE_RESTORE_MODEL = os.getenv("REACTOR_FACE_RESTORE_MODEL", "GFPGANv1.3.pth")

# Image (Qwen-Image-Edit) instruction-based photo editor: keeps the original photo intact
# except for whatever the prompt asks to change (clothes, body, background, add/remove
# someone), unlike img2img-from-noise which redraws everything.
IMAGE_EDIT_QWEN_UNET = os.getenv("IMAGE_EDIT_QWEN_UNET", "qwen_image_edit_2509_fp8_e4m3fn.safetensors")
IMAGE_EDIT_QWEN_CLIP = os.getenv("IMAGE_EDIT_QWEN_CLIP", "qwen_2.5_vl_7b_fp8_scaled.safetensors")
IMAGE_EDIT_QWEN_VAE = os.getenv("IMAGE_EDIT_QWEN_VAE", "qwen_image_vae.safetensors")
IMAGE_EDIT_STEPS = int(os.getenv("IMAGE_EDIT_STEPS", "20"))
IMAGE_EDIT_CFG = float(os.getenv("IMAGE_EDIT_CFG", "4"))
IMAGE_EDIT_SHIFT = float(os.getenv("IMAGE_EDIT_SHIFT", "3.1"))
IMAGE_EDIT_QUALITY = {
    "low": (1024, 1024),
    "medium": (1328, 1328),
    "high": (1536, 1536),
}

# Local Ollama instance: expands a short user idea into a detailed, chronologically-ordered
# scenario (runs on CPU, doesn't compete with the GPU video pipeline for VRAM).
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_SCENARIO_MODEL = os.getenv(
    "OLLAMA_SCENARIO_MODEL", "hf.co/mradermacher/Qwen2.5-32B-Instruct-abliterated-v2-i1-GGUF:Q4_K_M"
)
OLLAMA_SCENARIO_TIMEOUT = int(os.getenv("OLLAMA_SCENARIO_TIMEOUT", "180"))
# "🎰 Рулетка": with repeat >= 2, re-roll a fresh scenario variation every N jobs in the batch
# instead of reusing one prompt for the whole batch.
ROULETTE_GROUP_SIZE = int(os.getenv("ROULETTE_GROUP_SIZE", "2"))

# "🎙 Дубляж": replaces the native voice in LTX Sulphur/Eros's generated audio with a
# reference voice via OpenVoice V2 tone-color conversion. Runs on the full mixed audio
# (no Demucs vocal separation) because Demucs is trained on music and mis-routes non-speech
# vocalizations (moans) into the "background" stem, leaving the original voice audible there.
DUB_VOICE_ENABLED_MODES = {"ltx_sulphur", "ltx_eros"}
VOICES_DIR = Path(os.getenv("VOICES_DIR", "./voices"))
DEFAULT_VOICE_NAME = os.getenv("DEFAULT_VOICE_NAME", "tati")
OPENVOICE_DIR = Path(os.getenv("OPENVOICE_DIR", "./third_party/OpenVoice"))
OPENVOICE_TAU = float(os.getenv("OPENVOICE_TAU", "0.3"))
VOICE_FILE_EXTENSIONS = (".mp3", ".wav", ".ogg", ".m4a", ".oga", ".flac")


def list_voice_names() -> list[str]:
    if not VOICES_DIR.exists():
        return []
    return sorted(p.stem for p in VOICES_DIR.iterdir() if p.suffix.lower() in VOICE_FILE_EXTENSIONS)


def voice_path(name: str) -> Path | None:
    if not VOICES_DIR.exists():
        return None
    for ext in VOICE_FILE_EXTENSIONS:
        p = VOICES_DIR / f"{name}{ext}"
        if p.exists():
            return p
    return None


def next_voice_name() -> str:
    existing = set(list_voice_names())
    i = 2
    while f"voice{i}" in existing:
        i += 1
    return f"voice{i}"
OLLAMA_SCENARIO_SYSTEM_PROMPT = (
    "Ты — сценарист для генерации коротких 12-секундных видео нейросетью. "
    "Видео склеивается из 3-4 смысловых блоков, каждый блок — одно предложение, и получает "
    "пропорциональную (по длине текста) долю из 12 секунд. Поэтому пиши РОВНО 3-4 предложения, "
    "не больше: каждое предложение — один цельный, законченный эпизод-кадр (не дробить одно "
    "действие на несколько предложений), без нумерации и списков, обычным связным текстом на "
    "русском языке. Сценарий должен идти строго в хронологическом порядке без скачков: каждое "
    "следующее предложение — плавное продолжение и нарастание предыдущего, а не отдельная не "
    "связанная сцена. ПЕРВОЕ предложение — ОБЯЗАТЕЛЬНО подробное описание внешности и одежды "
    "персонажа (а не общая фраза вроде \"она лежит на кровати\"): конкретные детали тела "
    "(грудь, бёдра, фигура) и конкретная одежда/бельё (например чулки, мини-юбка, расстёгнутая "
    "рубашка — придумай уместные по сцене детали, не повторяй один и тот же набор каждый раз). "
    "Только после этого, во 2-4 предложениях — обстановка → действие → развитие/кульминация. "
    "В одном из первых предложений обязательно дай короткую реплику персонажа в кавычках "
    "(1 фраза, прямая речь) — это озвучивается отдельно, поэтому реплика должна быть короткой "
    "и однозначной. Без цензуры, explicit-контент разрешён и ожидается. Пиши только сам "
    "сценарий, без пояснений."
)

# Modes whose final ComfyUI output is a video, vs a still image.
VIDEO_MODES = {"video", "ltx_sulphur", "ltx_eros"}
# Modes that take a single uploaded photo into st["video_source"] (video modes, plus
# mopmix which runs img2img off of it, plus image which edits it directly).
SINGLE_PHOTO_MODES = VIDEO_MODES | {"mopmix", "image"}
# Modes that take two uploaded photos into st["duo_photos"].
DUO_PHOTO_MODES = {"mopmix_duo"}
# Modes whose workflow renders silent video and needs the MMAudio postprocess pass.
SILENT_VIDEO_MODES = {"video"}

REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "300"))
POLL_SECONDS = float(os.getenv("POLL_SECONDS", "2"))

MEDIA_LIBRARY_LIMIT = int(os.getenv("MEDIA_LIBRARY_LIMIT", "10"))
VIDEO_AUDIO = os.getenv("VIDEO_AUDIO", "1").strip().lower() not in {"0", "false", "no", "off"}
VIDEO_AUDIO_MODEL = os.getenv("VIDEO_AUDIO_MODEL", "mmaudio_large_44k_nsfw_gold_8.5k_final_fp16.safetensors")
VIDEO_AUDIO_VAE = os.getenv("VIDEO_AUDIO_VAE", "mmaudio_vae_44k_fp16.safetensors")
VIDEO_AUDIO_SYNCHFORMER = os.getenv("VIDEO_AUDIO_SYNCHFORMER", "mmaudio_synchformer_fp16.safetensors")
VIDEO_AUDIO_CLIP = os.getenv("VIDEO_AUDIO_CLIP", "apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors")
VIDEO_AUDIO_STEPS = int(os.getenv("VIDEO_AUDIO_STEPS", "25"))
VIDEO_AUDIO_CFG = float(os.getenv("VIDEO_AUDIO_CFG", "4.5"))
VIDEO_AUDIO_TIMEOUT = int(os.getenv("VIDEO_AUDIO_TIMEOUT", "900"))
VIDEO_AUDIO_NEGATIVE_PROMPT = os.getenv("VIDEO_AUDIO_NEGATIVE_PROMPT", "")
VIDEO_AUDIO_LOAD_FPS = int(os.getenv("VIDEO_AUDIO_LOAD_FPS", "25"))
VIDEO_NO_TEXT_PROMPT = "No subtitles, no captions, no on-screen text, no speech bubbles, no written words, no labels."
VIDEO_NO_TEXT_NEGATIVE = "subtitles, captions, on-screen text, text overlay, speech bubbles, written words, labels, watermark"
VIDEO_NO_LOOP_PROMPT = "Continuous non-looping motion, no replay, no boomerang, no ping-pong motion, no reset to the first frame."
VIDEO_NO_LOOP_NEGATIVE = "loop, looping, replay, boomerang, ping-pong motion, reverse playback, return to first frame, reset to starting pose"
VIDEO_TTS = os.getenv("VIDEO_TTS", "0").strip().lower() not in {"0", "false", "no", "off"}
EDGE_TTS_BIN = os.getenv("EDGE_TTS_BIN", "/home/iaadmin/miniconda3/bin/edge-tts")
VIDEO_TTS_VOICE_RU = os.getenv("VIDEO_TTS_VOICE_RU", "ru-RU-SvetlanaNeural")
VIDEO_TTS_VOICE_EN = os.getenv("VIDEO_TTS_VOICE_EN", "en-US-AvaNeural")
VIDEO_TTS_RATE = os.getenv("VIDEO_TTS_RATE", "+0%")
VIDEO_TTS_VOLUME = os.getenv("VIDEO_TTS_VOLUME", "+20%")
VIDEO_TTS_DELAY_MS = int(os.getenv("VIDEO_TTS_DELAY_MS", "500"))
VIDEO_TTS_BG_VOLUME = float(os.getenv("VIDEO_TTS_BG_VOLUME", "0.65"))
VIDEO_TTS_SPEECH_VOLUME = float(os.getenv("VIDEO_TTS_SPEECH_VOLUME", "1.25"))
VIDEO_MAX_LORAS = int(os.getenv("VIDEO_MAX_LORAS", "8"))
VIDEO_LORA_STRENGTH_DEFAULT = float(os.getenv("VIDEO_LORA_STRENGTH_DEFAULT", "0.35"))
VIDEO_LORA_STRENGTH_MULTIPLIER = float(os.getenv("VIDEO_LORA_STRENGTH_MULTIPLIER", "2.0"))
VIDEO_LORA_STRENGTH_MAX = float(os.getenv("VIDEO_LORA_STRENGTH_MAX", "1.0"))
VIDEO_LORA_OPTIONS = [
    {"key": "lightx2v", "label": "LightX2V", "high": "Wan2.2-I2V-A14B-Moe-Distill-Lightx2v_high.safetensors", "low": "Wan2.2-I2V-A14B-Moe-Distill-Lightx2v_low.safetensors", "strength": 0.25},
    {"key": "bounce", "label": "Bounce", "high": "BounceHighWan2_2.safetensors", "low": "BounceLowWan2_2.safetensors", "strength": 0.35},
    {"key": "dr34m", "label": "DR34M I2V", "high": "DR34MJOB_I2V_14b_HighNoise.safetensors", "low": "DR34MJOB_I2V_14b_LowNoise.safetensors", "strength": 0.35},
    {"key": "dreamlay", "label": "Dreamlay I2V", "high": "DR34ML4Y_I2V_14B_HIGH_V2.safetensors", "low": "DR34ML4Y_I2V_14B_LOW_V2.safetensors", "strength": 0.35},
    {"key": "hands_body", "label": "Hands/Body", "high": "HIGH_hands_trace_body.safetensors", "low": "LOW_hands_trace_body.safetensors", "strength": 0.30},
    {"key": "pen_insert", "label": "Pen Insert", "high": "PenInsert_high_noise.safetensors", "low": "PenInsert_low_noise.safetensors", "strength": 0.35},
    {"key": "pubic_hair", "label": "Pubic Hair", "high": "PubicHair_wan22_high_e40.safetensors", "low": "PubicHair_wan22_low_e50.safetensors", "strength": 0.30},
    {"key": "smooth_anim", "label": "Smooth Animation", "high": "SmoothXXXAnimation_High.safetensors", "low": "SmoothXXXAnimation_Low.safetensors", "strength": 0.25},
    {"key": "handjob", "label": "Handjob", "high": "WAN-2.2-I2V-Handjob-HIGH-v1.safetensors", "low": "WAN-2.2-I2V-Handjob-LOW-v1.safetensors", "strength": 0.35},
    {"key": "handjob_combo", "label": "Handjob+Blowjob", "high": "WAN-2.2-I2V-HandjobBlowjobCombo-HIGH-v1.safetensors", "low": "WAN-2.2-I2V-HandjobBlowjobCombo-LOW-v1.safetensors", "strength": 0.35},
    {"key": "teasing", "label": "Sensual Teasing", "high": "WAN-2.2-I2V-SensualTeasingBlowjob-HIGH-v1.safetensors", "low": "WAN-2.2-I2V-SensualTeasingBlowjob-LOW-v1.safetensors", "strength": 0.35},
    {"key": "breast_play", "label": "Breast Play", "high": "Wan2.2_BreastPlay-v1-HighNoise-I2V_T2V.safetensors", "low": "Wan2.2_BreastPlay-v1-LowNoise-I2V_T2V.safetensors", "strength": 0.35},
    {"key": "cum_v2", "label": "Cum V2", "high": "Wan22_CumV2_High.safetensors", "low": "Wan22_CumV2_Low.safetensors", "strength": 0.35},
    {"key": "ffgo", "label": "FFGO", "high": "Wan22_FFGO-LoRA-HIGH_bf16.safetensors", "low": "Wan22_FFGO-LoRA-LOW_bf16.safetensors", "strength": 0.35},
    {"key": "deepthroat", "label": "Deepthroat", "high": "jfj-deepthroat-W22-I2V-HN.safetensors", "low": "jfj-deepthroat-W22-I2V-LN.safetensors", "strength": 0.35},
    {"key": "pov_contact", "label": "POV Contact", "high": "povintimatecontact_WAN22_I2V_high_noise.safetensors", "low": "povintimatecontact_WAN22_I2V_low_noise.safetensors", "strength": 0.35},
]

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("tg-comfy-bot")

# ============================================================
# STATE / QUEUE
# ============================================================
GEN_QUEUE: asyncio.Queue = asyncio.Queue()
WORKER_STARTED = False
JOB_SEQ = 0
ACTIVE_PROMPTS: dict[str, dict] = {}

# Live progress status message per chat, edited in place while a job is generating so the
# user doesn't think the bot/server hung. Rolling average duration per mode (seconds) used
# for the ETA estimate, seeded with rough defaults and refined after every completed job.
CHAT_STATUS: dict[int, dict[str, Any]] = {}
MODE_AVG_DURATION: dict[str, float] = {
    "video": 240.0,
    "ltx_sulphur": 230.0,
    "ltx_eros": 230.0,
    "mopmix": 60.0,
    "mopmix_duo": 90.0,
    "image": 30.0,
}
STATUS_UPDATE_INTERVAL = 8.0


@dataclass
class Job:
    job_id: int
    chat_id: int
    mode: str
    prompt: str
    seconds: int
    max_side: int
    video_fps: int
    seed: int
    video_source: dict | None
    duo_photos: list[dict]
    video_loras: list[str]
    quality: str
    batch_index: int
    batch_total: int
    dub_voice: bool
    dub_voice_name: str


# ============================================================
# HELPERS
# ============================================================
def quality_label(max_side: int) -> str:
    for name, preset in QUALITY_PRESETS.items():
        if preset["max_side"] == max_side:
            return name
    return f"custom-{max_side}"


def quality_preset(name: str) -> dict[str, int] | None:
    return QUALITY_PRESETS.get((name or "").strip().lower())


def apply_quality(st: dict[str, Any], name: str) -> bool:
    preset = quality_preset(name)
    if not preset:
        return False
    st["quality"] = name
    st["max_side"] = int(preset["max_side"])
    st["video_fps"] = int(preset["video_fps"])
    refresh_all_sizes(st)
    return True


def quality_status(st: dict[str, Any]) -> str:
    quality = (st.get("quality") or "medium").strip().lower()
    mode = st.get("mode")
    if mode == "ltx_sulphur":
        w, h = LTX_SULPHUR_QUALITY.get(quality, LTX_SULPHUR_QUALITY["medium"])
        return f"{quality} ({w}x{h})"
    if mode == "ltx_eros":
        w, h = LTX_EROS_QUALITY.get(quality, LTX_EROS_QUALITY["medium"])
        return f"{quality} ({w}x{h})"
    if mode in {"mopmix", "mopmix_duo"}:
        return f"{quality} ({MOPMIX_RESOLUTIONS.get(quality, MOPMIX_RESOLUTIONS['medium'])})"
    if mode == "image":
        w, h = IMAGE_EDIT_QUALITY.get(quality, IMAGE_EDIT_QUALITY["medium"])
        return f"{quality} ({w}x{h})"
    label = quality_label(int(st.get("max_side") or 0))
    fps = int(st.get("video_fps") or 0)
    fps_text = f", {fps} fps" if fps else ""
    return f"{label} ({st['max_side']} px{fps_text})"


def make_seed() -> int:
    return int.from_bytes(os.urandom(8), "big") & ((1 << 53) - 1)


def round_to_multiple(v: int, m: int) -> int:
    return max(m, int((v // m) * m))


def short_preview(s: str, limit: int = 80) -> str:
    s = (s or "").strip()
    if not s:
        return "—"
    if len(s) <= limit:
        return s
    return s[: limit - 3] + "..."


def fit_size_keep_aspect(width: int, height: int, max_side: int, multiple: int = ROUND_TO) -> tuple[int, int]:
    w, h = int(width), int(height)
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        w = max(1, int(w * scale))
        h = max(1, int(h * scale))
    w = round_to_multiple(w, multiple)
    h = round_to_multiple(h, multiple)
    return max(multiple, w), max(multiple, h)


def fit_to_pixel_budget(src_width: int, src_height: int, target_pixels: int, multiple: int = ROUND_TO) -> tuple[int, int]:
    """Scale (src_width, src_height) to roughly target_pixels total pixels, keeping the
    source's own aspect ratio. Used so LTX Sulphur/Eros quality presets control output
    cost (resolution) without forcing every photo into one fixed orientation/box, which
    was cropping faces out of photos whose aspect ratio didn't match the preset."""
    src_width = max(1, int(src_width))
    src_height = max(1, int(src_height))
    scale = (target_pixels / (src_width * src_height)) ** 0.5
    w = round_to_multiple(src_width * scale, multiple)
    h = round_to_multiple(src_height * scale, multiple)
    return max(multiple, w), max(multiple, h)


def build_duo_composite(path_a: str, path_b: str, *, height: int = 1536, half_width: int = 1075) -> Image.Image:
    """Side-by-side composite of two source photos, used as the img2img starting latent for
    MopMix Duo so body/hair/age/clothing come from the real photos instead of being imagined
    by the model from scratch (which a pure txt2img + final face-swap can't fix)."""
    canvas = Image.new("RGB", (half_width * 2, height))
    for i, path in enumerate((path_a, path_b)):
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            fitted = ImageOps.fit(im, (half_width, height), method=Image.LANCZOS, centering=(0.5, 0.3))
        canvas.paste(fitted, (i * half_width, 0))
    return canvas


def save_bytes(path: Path, blob: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)


def run_cmd(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    timeout: int | None = None,
    env: dict[str, str] | None = None,
) -> None:
    p = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        timeout=timeout,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed ({p.returncode}): {' '.join(cmd)}\n\nSTDERR:\n{p.stderr}"
        )


def gif_to_mp4(input_path: Path, output_path: Path) -> None:
    run_cmd([
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-movflags", "+faststart",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ])


def mux_video_with_audio(video_path: Path, audio_source_path: Path, output_path: Path) -> None:
    run_cmd([
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-i", str(audio_source_path),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        "-movflags", "+faststart",
        str(output_path),
    ])


def extract_speech_text(prompt: str) -> str:
    text = (prompt or "").strip()
    if not text:
        return ""

    quoted_after_speech = re.findall(
        r"(?:говорит|говоря|сказала|скажет|произносит|шепчет|says|said|speaks|whispers)[^\n\"«”]{0,40}[\"«“](.{1,220}?)[\"»”]",
        text,
        flags=re.IGNORECASE,
    )
    if quoted_after_speech:
        return clean_speech_text(". ".join(quoted_after_speech))

    quoted = re.findall(r"[\"«“](.{1,180}?)[\"»”]", text)
    if quoted:
        return clean_speech_text(". ".join(quoted[:2]))

    match = re.search(
        r"(?:говорит|говоря|сказала|скажет|произносит|шепчет|says|said|speaks|whispers)\s*[:\-—]?\s*(.{1,180}?)(?=$|[.!?;\n])",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return clean_speech_text(match.group(1))
    return ""


def clean_speech_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip(" \t\r\n'\"«»“”.,;:-—")
    text = re.sub(r"\b(и|and)\s+(улыбается|смотрит|looks|smiles).*$", "", text, flags=re.IGNORECASE).strip()
    return text[:220]


def tts_voice_for_text(text: str) -> str:
    if re.search(r"[А-Яа-яЁё]", text or ""):
        return VIDEO_TTS_VOICE_RU
    return VIDEO_TTS_VOICE_EN


def synthesize_speech(text: str, output_path: Path) -> None:
    run_cmd([
        EDGE_TTS_BIN,
        "--voice", tts_voice_for_text(text),
        "--text", text,
        "--rate", VIDEO_TTS_RATE,
        "--volume", VIDEO_TTS_VOLUME,
        "--write-media", str(output_path),
    ])


def video_has_audio(video_path: Path) -> bool:
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=index", "-of", "csv=p=0", str(video_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return p.returncode == 0 and bool(p.stdout.strip())


def mix_video_with_tts(video_path: Path, speech_path: Path, output_path: Path) -> None:
    delay = max(0, int(VIDEO_TTS_DELAY_MS))
    if video_has_audio(video_path):
        filter_complex = (
            f"[0:a]volume={VIDEO_TTS_BG_VOLUME}[a0];"
            f"[1:a]adelay={delay}|{delay},volume={VIDEO_TTS_SPEECH_VOLUME}[a1];"
            "[a0][a1]amix=inputs=2:duration=first:dropout_transition=0[a]"
        )
    else:
        filter_complex = f"[1:a]adelay={delay}|{delay},volume={VIDEO_TTS_SPEECH_VOLUME}[a]"

    run_cmd([
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-i", str(speech_path),
        "-filter_complex", filter_complex,
        "-map", "0:v:0",
        "-map", "[a]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-movflags", "+faststart",
        str(output_path),
    ])


# ============================================================
# ACCESS CONTROL
# ============================================================
def allowed(update: Update) -> bool:
    if not ALLOWED_USER_IDS:
        return True
    user = update.effective_user
    return bool(user and user.id in ALLOWED_USER_IDS)


async def reject_if_needed(update: Update) -> bool:
    if allowed(update):
        return False
    if update.message:
        await update.message.reply_text("Нет доступа.")
    elif update.callback_query:
        await update.callback_query.answer("Нет доступа.", show_alert=True)
    return True


# ============================================================
# USER STATE
# ============================================================
def blank_media() -> dict[str, Any]:
    return {
        "path": None,
        "name": None,
        "orig_width": None,
        "orig_height": None,
        "fit_width": None,
        "fit_height": None,
    }


# Per-mode duration ceiling overrides (RTX 5070, 12GB VRAM - the latent upscale step in
# LTX Sulphur/Eros can already use ~92% of the card at 12s; raise per-mode only after
# testing on this hardware, not by guessing).
MODE_MAX_SECONDS = {
    "ltx_sulphur": int(os.getenv("MAX_SECONDS_LTX_SULPHUR", str(MAX_SECONDS))),
    "ltx_eros": int(os.getenv("MAX_SECONDS_LTX_EROS", str(MAX_SECONDS))),
}


def mode_max_seconds(mode: str) -> int:
    return MODE_MAX_SECONDS.get(mode, MAX_SECONDS)


def clamp_seconds(value: int, mode: str) -> int:
    return max(MIN_SECONDS, min(mode_max_seconds(mode), int(value)))


def initial_state() -> dict[str, Any]:
    return {
        "mode": "video",
        "prompt": "",
        "repeat": 1,
        "seconds": DEFAULT_SECONDS,
        "quality": DEFAULT_QUALITY if DEFAULT_QUALITY in QUALITY_PRESETS else "medium",
        "max_side": QUALITY_PRESETS.get(DEFAULT_QUALITY, QUALITY_PRESETS["medium"])["max_side"],
        "video_fps": QUALITY_PRESETS.get(DEFAULT_QUALITY, QUALITY_PRESETS["medium"])["video_fps"],
        "video_source": blank_media(),
        "duo_photos": [blank_media(), blank_media()],
        "video_loras": [],
        "roulette": False,
        "dub_voice": False,
        "dub_voice_name": DEFAULT_VOICE_NAME,
    }


def get_state(context: ContextTypes.DEFAULT_TYPE) -> dict[str, Any]:
    if "job_state" not in context.user_data:
        context.user_data["job_state"] = initial_state()
    st = context.user_data["job_state"]
    if st.get("mode") in {"director", "talk"}:
        st["mode"] = "video"
    if "video_loras" not in st:
        st["video_loras"] = []
    if "duo_photos" not in st:
        st["duo_photos"] = [blank_media(), blank_media()]
    if "roulette" not in st:
        st["roulette"] = False
    if "dub_voice" not in st:
        st["dub_voice"] = False
    if "dub_voice_name" not in st:
        st["dub_voice_name"] = DEFAULT_VOICE_NAME
    if "quality" not in st:
        st["quality"] = quality_label(int(st.get("max_side") or QUALITY_PRESETS["medium"]["max_side"]))
    preset = quality_preset(st.get("quality")) or QUALITY_PRESETS["medium"]
    if "video_fps" not in st or int(st.get("video_fps") or 0) > 60:
        st["video_fps"] = int(preset["video_fps"])
    st.pop("director_loras", None)
    return st


def reset_state(context: ContextTypes.DEFAULT_TYPE) -> dict[str, Any]:
    context.user_data["job_state"] = initial_state()
    return context.user_data["job_state"]


def refresh_media_size(media: dict[str, Any], max_side: int) -> None:
    if not media or not media.get("path"):
        return
    width = media.get("orig_width")
    height = media.get("orig_height")
    if not width or not height:
        try:
            with Image.open(media["path"]) as img:
                width, height = img.size
            media["orig_width"] = width
            media["orig_height"] = height
        except Exception:
            return
    media["fit_width"], media["fit_height"] = fit_size_keep_aspect(int(width), int(height), int(max_side))


def refresh_all_sizes(st: dict[str, Any]) -> None:
    max_side = int(st.get("max_side") or QUALITY_PRESETS["medium"]["max_side"])
    refresh_media_size(st.get("video_source") or {}, max_side)


def get_media_library(context: ContextTypes.DEFAULT_TYPE) -> list[dict[str, Any]]:
    if "media_library" not in context.user_data:
        context.user_data["media_library"] = []
    return context.user_data["media_library"]


def remember_media(context: ContextTypes.DEFAULT_TYPE, media: dict[str, Any]) -> None:
    library = get_media_library(context)
    path = media.get("path")
    if not path:
        return
    library[:] = [x for x in library if x.get("path") != path]
    library.insert(0, copy.deepcopy(media))
    del library[MEDIA_LIBRARY_LIMIT:]


def rebuild_media_library_from_disk(context: ContextTypes.DEFAULT_TYPE, user_id: int, max_side: int) -> list[dict[str, Any]]:
    paths = sorted(
        TMP_DIR.glob(f"tg_{user_id}_*.jpg"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    library = get_media_library(context)
    library[:] = [x for x in library if Path(x.get("path", "")).exists()]
    seen_paths = {x.get("path") for x in library}

    for path in paths:
        if str(path) in seen_paths:
            continue
        try:
            with Image.open(path) as img:
                width, height = img.size
        except Exception:
            continue

        fit_w, fit_h = fit_size_keep_aspect(width, height, max_side)
        library.append(
            {
                "path": str(path),
                "name": path.name,
                "orig_width": width,
                "orig_height": height,
                "fit_width": fit_w,
                "fit_height": fit_h,
            }
        )
        seen_paths.add(str(path))
        if len(library) >= MEDIA_LIBRARY_LIMIT:
            break

    del library[MEDIA_LIBRARY_LIMIT:]
    return library


# ============================================================
# UI TRACKING
# ============================================================
def get_ui_state(context: ContextTypes.DEFAULT_TYPE) -> dict[str, Any]:
    if "ui_state" not in context.user_data:
        context.user_data["ui_state"] = {"last_ui_message_id": None}
    return context.user_data["ui_state"]


async def delete_last_ui_message(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> None:
    ui = get_ui_state(context)
    msg_id = ui.get("last_ui_message_id")
    if not msg_id:
        return
    try:
        await context.bot.delete_message(chat_id=chat_id, message_id=msg_id)
    except Exception:
        pass
    ui["last_ui_message_id"] = None


async def send_ui_message(target_message, context: ContextTypes.DEFAULT_TYPE, text: str, reply_markup=None) -> None:
    chat_id = target_message.chat_id
    await delete_last_ui_message(context, chat_id)
    msg = await target_message.reply_text(text, reply_markup=reply_markup)
    get_ui_state(context)["last_ui_message_id"] = msg.message_id


async def replace_ui_message_from_callback(query, context: ContextTypes.DEFAULT_TYPE, text: str, reply_markup=None) -> None:
    ui = get_ui_state(context)
    old_id = ui.get("last_ui_message_id")

    try:
        await query.message.delete()
    except Exception:
        pass

    if old_id == query.message.message_id:
        ui["last_ui_message_id"] = None

    msg = await context.bot.send_message(
        chat_id=query.message.chat_id,
        text=text,
        reply_markup=reply_markup,
    )
    ui["last_ui_message_id"] = msg.message_id


# ============================================================
# UI
# ============================================================
def main_keyboard(st: dict[str, Any] | None = None) -> InlineKeyboardMarkup:
    roulette_on = bool(st.get("roulette")) if st else False
    dub_voice_on = bool(st.get("dub_voice")) if st else False
    roulette_label = f"🎰 Рулетка: {'✅ ВКЛ' if roulette_on else '⬜ выкл'}"
    dub_voice_label = f"🎙 Дубляж: {'✅ ВКЛ' if dub_voice_on else '⬜ выкл'}"
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("🎬 Video", callback_data="mode:video"),
                InlineKeyboardButton("✏️ Edit Photo", callback_data="mode:image"),
            ],
            [
                InlineKeyboardButton("🧪 LTX Sulphur", callback_data="mode:ltx_sulphur"),
                InlineKeyboardButton("🔥 LTX Eros", callback_data="mode:ltx_eros"),
            ],
            [
                InlineKeyboardButton("🎨 MopMix", callback_data="mode:mopmix"),
                InlineKeyboardButton("👯 MopMix Duo", callback_data="mode:mopmix_duo"),
            ],
            [
                InlineKeyboardButton("L", callback_data="quality:low"),
                InlineKeyboardButton("M", callback_data="quality:medium"),
                InlineKeyboardButton("H", callback_data="quality:high"),
                InlineKeyboardButton("➖2s", callback_data="sec:-2"),
                InlineKeyboardButton("➕2s", callback_data="sec:+2"),
            ],
            [
                InlineKeyboardButton("1x", callback_data="repeat:1"),
                InlineKeyboardButton("10x", callback_data="repeat:10"),
                InlineKeyboardButton("30x", callback_data="repeat:30"),
            ],
            [
                InlineKeyboardButton("📷 Recent photos", callback_data="media:list"),
                InlineKeyboardButton("🎚 LoRA", callback_data="lora:list"),
                InlineKeyboardButton("🧹 Reset", callback_data="do:reset"),
            ],
            [
                InlineKeyboardButton("✨ Развить идею", callback_data="do:expand"),
            ],
            [
                InlineKeyboardButton(roulette_label, callback_data="do:roulette"),
                InlineKeyboardButton(dub_voice_label, callback_data="do:dubvoice"),
            ],
            [
                InlineKeyboardButton(f"🎙 Голос: {st.get('dub_voice_name', DEFAULT_VOICE_NAME) if st else DEFAULT_VOICE_NAME}", callback_data="voice:list"),
            ],
            [
                InlineKeyboardButton("⛔🚮 Stop + Clear", callback_data="queue:stopclear"),
                InlineKeyboardButton("🚀 Generate", callback_data="do:go"),
            ],
        ]
    )


def media_preview_caption(media: dict[str, Any], index: int, total: int) -> str:
    return (
        f"Фото {index + 1}/{total}\n"
        f"{media.get('orig_width')}×{media.get('orig_height')} → "
        f"{media.get('fit_width')}×{media.get('fit_height')}"
    )


def media_preview_keyboard(index: int, total: int) -> InlineKeyboardMarkup:
    prev_idx = (index - 1) % total
    next_idx = (index + 1) % total
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("←", callback_data=f"media:page:{prev_idx}"),
                InlineKeyboardButton(f"{index + 1}/{total}", callback_data="noop"),
                InlineKeyboardButton("→", callback_data=f"media:page:{next_idx}"),
            ],
            [InlineKeyboardButton("Use this photo", callback_data=f"media:select:{index}")],
            [InlineKeyboardButton("↩️ Back", callback_data="show:status")],
        ]
    )


def video_lora_by_key(key: str) -> dict[str, Any] | None:
    for opt in VIDEO_LORA_OPTIONS:
        if opt["key"] == key:
            return opt
    return None


def selected_lora_labels(st: dict[str, Any], limit: int = 4) -> str:
    selected = st.get("video_loras") or []
    if not selected:
        return "none"
    labels = [(video_lora_by_key(key) or {"label": key})["label"] for key in selected]
    if len(labels) > limit:
        return ", ".join(labels[:limit]) + f" +{len(labels) - limit}"
    return ", ".join(labels)


def lora_keyboard(st: dict[str, Any]) -> InlineKeyboardMarkup:
    selected = set(st.get("video_loras") or [])
    rows = []
    for i, opt in enumerate(VIDEO_LORA_OPTIONS):
        mark = "✓" if opt["key"] in selected else "○"
        rows.append([InlineKeyboardButton(f"{mark} {opt['label']}", callback_data=f"lora:toggle:{i}")])
    rows.append([InlineKeyboardButton("Clear", callback_data="lora:clear"), InlineKeyboardButton("↩️ Back", callback_data="show:status")])
    return InlineKeyboardMarkup(rows)


def lora_text(st: dict[str, Any]) -> str:
    selected = st.get("video_loras") or []
    lines = [
        "LoRA для обычного video",
        "",
        f"Выбрано: {len(selected)}/{VIDEO_MAX_LORAS}",
    ]
    if selected:
        for key in selected:
            opt = video_lora_by_key(key) or {"label": key, "strength": VIDEO_LORA_STRENGTH_DEFAULT}
            lines.append(f"• {opt['label']} ({effective_lora_strength(opt):.2f})")
    else:
        lines.append("• none")
    return "\n".join(lines)


async def send_media_preview_message(target_message, context: ContextTypes.DEFAULT_TYPE, index: int = 0) -> None:
    chat_id = target_message.chat_id
    await delete_last_ui_message(context, chat_id)

    library = get_media_library(context)
    if not library:
        msg = await target_message.reply_text(
            "Пока нет сохранённых фото. Пришли фото один раз, потом его можно будет выбрать здесь.",
            reply_markup=main_keyboard(get_state(context)),
        )
    else:
        index = max(0, min(index, len(library) - 1))
        media = library[index]
        with Path(media["path"]).open("rb") as f:
            msg = await target_message.reply_photo(
                photo=InputFile(f, filename=media.get("name") or Path(media["path"]).name),
                caption=media_preview_caption(media, index, len(library)),
                reply_markup=media_preview_keyboard(index, len(library)),
            )

    get_ui_state(context)["last_ui_message_id"] = msg.message_id


async def replace_media_preview_from_callback(query, context: ContextTypes.DEFAULT_TYPE, index: int = 0) -> None:
    ui = get_ui_state(context)
    old_id = ui.get("last_ui_message_id")

    try:
        await query.message.delete()
    except Exception:
        pass

    if old_id == query.message.message_id:
        ui["last_ui_message_id"] = None

    library = get_media_library(context)
    if not library:
        msg = await context.bot.send_message(
            chat_id=query.message.chat_id,
            text="Пока нет сохранённых фото. Пришли фото один раз, потом его можно будет выбрать здесь.",
            reply_markup=main_keyboard(get_state(context)),
        )
    else:
        index = max(0, min(index, len(library) - 1))
        media = library[index]
        with Path(media["path"]).open("rb") as f:
            msg = await context.bot.send_photo(
                chat_id=query.message.chat_id,
                photo=InputFile(f, filename=media.get("name") or Path(media["path"]).name),
                caption=media_preview_caption(media, index, len(library)),
                reply_markup=media_preview_keyboard(index, len(library)),
            )

    ui["last_ui_message_id"] = msg.message_id


def media_line(m: dict[str, Any]) -> str:
    if not m.get("name"):
        return "нет"
    return (
        f"{m.get('name')} ({m.get('orig_width')}×{m.get('orig_height')} → "
        f"{m.get('fit_width')}×{m.get('fit_height')})"
    )


def help_text(st: dict[str, Any]) -> str:
    prompt_preview = short_preview(st.get("prompt") or "—", 180)

    return (
        "Бот готов.\n\n"
        "Режимы:\n"
        "• video: 1 фото + промт + секунды + звук MMAudio\n"
        "• ltx_sulphur: 1 фото + промт + секунды, видео+звук нативно (LTX2.3 Sulphur)\n"
        "• ltx_eros: 1 фото + промт + секунды, видео+звук нативно (LTX2.3 10Eros)\n"
        "• image: 1 фото + промт-редактирование (поменять одежду/тело/фон/добавить или убрать кого-то, Qwen-Image-Edit)\n"
        "• mopmix: 1 фото + промт + качество → картинка img2img (SDXL + face detailer)\n"
        "• mopmix_duo: 2 фото (лица) + промт → сцена с обоими лицами (face swap)\n\n"
        "Команды:\n"
        "/video — обычный photo → video\n"
        "/ltxsulphur — photo → video+audio (LTX2.3 Sulphur)\n"
        "/ltxeros — photo → video+audio (LTX2.3 10Eros)\n"
        "/image — photo → редактирование фото по промту\n"
        "/mopmix — photo → img2img картинка (MopMix BigASP 2.5)\n"
        "/mopmixduo — 2 фото → сцена с обоими лицами (face swap)\n"
        "/prompt текст — сохранить промт\n"
        f"/seconds 8 — video/ltx_sulphur/ltx_eros до {MAX_SECONDS} сек\n"
        "/quality low|medium|high\n"
        "/repeat 1\n"
        "/loras — выбрать LoRA для обычного video\n"
        "/photos — выбрать базовое фото из последних загруженных\n"
        "/go — генерация\n"
        "/reset\n\n"
        "Текущее состояние:\n"
        f"• mode: {st['mode']}\n"
        f"• quality: {quality_status(st)}\n"
        f"• seconds: {st['seconds']}\n"
        f"• repeat: {st.get('repeat', 1)}\n"
        f"• рулетка: {'on' if st.get('roulette') else 'off'}\n"
        f"• дубляж голоса: {'on' if st.get('dub_voice') else 'off'}\n"
        f"• video LoRA: {selected_lora_labels(st)}\n"
        f"• video audio: {'on' if VIDEO_AUDIO else 'off'}\n"
        f"• video TTS: {'on' if VIDEO_TTS else 'off'}\n"
        f"• video source: {media_line(st['video_source'])}\n"
        f"• duo фото A: {media_line(st['duo_photos'][0])}\n"
        f"• duo фото B: {media_line(st['duo_photos'][1])}\n"
        f"• prompt: {prompt_preview}"
    )


# ============================================================
# COMFY API
# ============================================================
def load_workflow(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def upload_image_to_comfy(local_path: str, filename: str) -> str:
    with open(local_path, "rb") as f:
        files = {"image": (filename, f, "application/octet-stream")}
        data = {"overwrite": "true", "type": "input"}
        r = requests.post(f"{COMFY_BASE}/upload/image", files=files, data=data, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json().get("name", filename)


def queue_prompt(prompt: dict[str, Any], client_id: str) -> str:
    r = requests.post(
        f"{COMFY_BASE}/prompt",
        json={"prompt": prompt, "client_id": client_id},
        timeout=REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    data = r.json()
    if "prompt_id" not in data:
        raise RuntimeError(f"ComfyUI prompt error: {data}")
    return data["prompt_id"]


def get_history(prompt_id: str) -> dict[str, Any]:
    r = requests.get(f"{COMFY_BASE}/history/{prompt_id}", timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


def fetch_file(filename: str, subfolder: str = "", file_type: str = "output") -> bytes:
    r = requests.get(
        f"{COMFY_BASE}/view",
        params={"filename": filename, "subfolder": subfolder, "type": file_type},
        timeout=REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    return r.content


def get_queue_state() -> dict[str, Any]:
    r = requests.get(f"{COMFY_BASE}/queue", timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


def interrupt_current() -> dict[str, Any]:
    r = requests.post(f"{COMFY_BASE}/interrupt", timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    try:
        return r.json()
    except Exception:
        return {"ok": True}


def clear_comfy_queue() -> dict[str, Any]:
    r = requests.post(
        f"{COMFY_BASE}/queue",
        json={"clear": True},
        timeout=REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    try:
        return r.json()
    except Exception:
        return {"cleared": True}


def clear_local_queue() -> int:
    cleared = 0
    while True:
        try:
            GEN_QUEUE.get_nowait()
            GEN_QUEUE.task_done()
            cleared += 1
        except asyncio.QueueEmpty:
            break
    return cleared


def clear_active_prompts() -> int:
    n = len(ACTIVE_PROMPTS)
    ACTIVE_PROMPTS.clear()
    return n


def delete_comfy_result_file(filename: str, subfolder: str = "") -> None:
    try:
        p = (COMFY_OUTPUT_DIR / subfolder / filename).resolve()
        if p.is_file():
            p.unlink()
    except Exception as e:
        log.warning("Cannot delete result file: %s | %s", filename, e)


# ============================================================
# WORKFLOW PATCHERS
# ============================================================
def clear_power_lora_node(wf: dict[str, Any], node_id: str) -> dict[str, Any]:
    node = wf.get(node_id)
    if not node:
        return {}
    inputs = node.setdefault("inputs", {})
    for key, value in list(inputs.items()):
        if key.startswith("lora_") and isinstance(value, dict):
            value["on"] = False
    return inputs


def effective_lora_strength(opt: dict[str, Any]) -> float:
    base = float(opt.get("strength", VIDEO_LORA_STRENGTH_DEFAULT))
    return min(VIDEO_LORA_STRENGTH_MAX, base * VIDEO_LORA_STRENGTH_MULTIPLIER)


def apply_video_loras(wf: dict[str, Any], selected_loras: list[str]) -> None:
    high_inputs = clear_power_lora_node(wf, "152")
    low_inputs = clear_power_lora_node(wf, "155")

    valid = []
    for key in selected_loras:
        opt = video_lora_by_key(key)
        if opt and opt["key"] not in [x["key"] for x in valid]:
            valid.append(opt)

    if not valid:
        return

    high_prev: list[Any] = ["371", 0]
    low_prev: list[Any] = ["372", 0]
    for index, opt in enumerate(valid[:VIDEO_MAX_LORAS], start=1):
        strength = effective_lora_strength(opt)
        high_id = f"tg_high_lora_{index}"
        low_id = f"tg_low_lora_{index}"
        wf[high_id] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": high_prev,
                "lora_name": opt["high"],
                "strength_model": strength,
            },
        }
        wf[low_id] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": low_prev,
                "lora_name": opt["low"],
                "strength_model": strength,
            },
        }
        high_prev = [high_id, 0]
        low_prev = [low_id, 0]

        if high_inputs:
            high_inputs[f"lora_{index}"] = {"on": True, "lora": opt["high"], "strength": strength}
        if low_inputs:
            low_inputs[f"lora_{index}"] = {"on": True, "lora": opt["low"], "strength": strength}

    wf["141"]["inputs"]["model_high_noise"] = high_prev
    wf["141"]["inputs"]["model_low_noise"] = low_prev


def patch_video_workflow(
    wf: dict[str, Any],
    *,
    prompt: str,
    image_name: str,
    width: int,
    height: int,
    seconds: int,
    video_fps: int,
    seed: int,
    selected_loras: list[str] | None = None,
    continuity_prompt: str = "",
    continuity_negative: str = "",
) -> dict[str, Any]:
    wf = json.loads(json.dumps(wf))
    prompt_parts = [prompt, VIDEO_NO_TEXT_PROMPT, VIDEO_NO_LOOP_PROMPT]
    if continuity_prompt:
        prompt_parts.append(continuity_prompt)
    wf["93"]["inputs"]["text"] = "\n\n".join(x for x in prompt_parts if x)
    negative_text = wf["373:360"]["inputs"].get("text", "")
    if VIDEO_NO_TEXT_NEGATIVE not in negative_text:
        wf["373:360"]["inputs"]["text"] = f"{negative_text}, {VIDEO_NO_TEXT_NEGATIVE}"
        negative_text = wf["373:360"]["inputs"]["text"]
    if VIDEO_NO_LOOP_NEGATIVE not in negative_text:
        wf["373:360"]["inputs"]["text"] = f"{negative_text}, {VIDEO_NO_LOOP_NEGATIVE}"
        negative_text = wf["373:360"]["inputs"]["text"]
    if continuity_negative and continuity_negative not in negative_text:
        wf["373:360"]["inputs"]["text"] = f"{negative_text}, {continuity_negative}"
    wf["385"]["inputs"]["image"] = image_name
    wf["164"]["inputs"]["value"] = int(width)
    wf["165"]["inputs"]["value"] = int(height)
    fps = max(1, int(video_fps))
    frame_count = max(1, int(seconds) * fps + 1)
    wf["243"]["inputs"]["value"] = int(seconds)
    wf["373:359"]["inputs"]["value"] = str(frame_count)
    wf["314"]["inputs"]["frame_rate"] = fps
    wf["141"]["inputs"]["seed"] = int(seed)
    apply_video_loras(wf, selected_loras or [])
    return wf


def build_image_edit_workflow(
    *,
    image_name: str,
    prompt: str,
    width: int,
    height: int,
    seed: int,
) -> dict[str, Any]:
    """Qwen-Image-Edit: conditions on the source photo via cross-attention and edits only
    what the prompt asks for, instead of redrawing the whole image like img2img-from-noise."""
    return {
        "72": {"class_type": "CLIPLoader", "inputs": {"clip_name": IMAGE_EDIT_QWEN_CLIP, "type": "qwen_image"}},
        "71": {"class_type": "VAELoader", "inputs": {"vae_name": IMAGE_EDIT_QWEN_VAE}},
        "73": {"class_type": "UNETLoader", "inputs": {"unet_name": IMAGE_EDIT_QWEN_UNET, "weight_dtype": "default"}},
        "67": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["73", 0], "shift": IMAGE_EDIT_SHIFT}},
        "41": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "68": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {"clip": ["72", 0], "prompt": prompt, "vae": ["71", 0], "image1": ["41", 0]},
        },
        "69": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {"clip": ["72", 0], "prompt": "", "vae": ["71", 0], "image1": ["41", 0]},
        },
        "66": {"class_type": "EmptySD3LatentImage", "inputs": {"width": int(width), "height": int(height), "batch_size": 1}},
        "65": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["67", 0],
                "seed": int(seed),
                "steps": IMAGE_EDIT_STEPS,
                "cfg": IMAGE_EDIT_CFG,
                "sampler_name": "euler",
                "scheduler": "simple",
                "positive": ["68", 0],
                "negative": ["69", 0],
                "latent_image": ["66", 0],
                "denoise": 1,
            },
        },
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["65", 0], "vae": ["71", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "tg_image_edit"}},
    }


def patch_ltx_sulphur_workflow(
    wf: dict[str, Any],
    *,
    prompt: str,
    image_name: str,
    width: int,
    height: int,
    seconds: int,
    seed: int,
) -> dict[str, Any]:
    wf = json.loads(json.dumps(wf))
    wf["28"]["inputs"]["text"] = prompt
    wf["15"]["inputs"]["image"] = image_name
    wf["19"]["inputs"]["Xi"] = int(width)
    wf["19"]["inputs"]["Xf"] = int(width)
    wf["181"]["inputs"]["Xi"] = int(height)
    wf["181"]["inputs"]["Xf"] = int(height)
    wf["18"]["inputs"]["Xi"] = int(seconds)
    wf["18"]["inputs"]["Xf"] = int(seconds)
    wf["125"]["inputs"]["seed"] = int(seed) % 1_125_899_906_842_624
    if "163" in wf:
        cleanup_inputs = wf["163"]["inputs"]
        if "anything" in cleanup_inputs:
            cleanup_inputs["input"] = cleanup_inputs.pop("anything")
        cleanup_inputs["cleanup_mode"] = "Cache Only"
    return wf


FLORENCE2_MODEL = os.getenv("FLORENCE2_CAPTION_MODEL", "microsoft/Florence-2-large")
CAPTION_TIMEOUT = int(os.getenv("CAPTION_TIMEOUT", "120"))
CAPTION_SCENE_KEYWORDS = (
    "background", "standing in", "sitting in", "kitchen", "street", "wall",
    "room", "building", "indoor", "outdoor", "in front of", "taken in", "setting",
)


def strip_scene_details(caption: str) -> str:
    """Florence's caption mixes appearance with background/location ('standing in a
    kitchen'), which then competes with the actual requested scene in the final prompt.
    Strip location/background clauses, keep clothing/hair/age clauses."""
    caption = re.sub(
        r"^(the image (is|shows)\s*)?(a |an )?(selfie|portrait|photo|picture)\s+of\s+",
        "", caption, flags=re.IGNORECASE,
    )
    sentences = re.split(r"(?<=[.!?])\s+", caption)
    kept_sentences = []
    for sentence in sentences:
        clauses = re.split(r",\s+| and (?=\w)", sentence)
        kept_clauses = [c for c in clauses if not any(kw in c.lower() for kw in CAPTION_SCENE_KEYWORDS)]
        if kept_clauses:
            kept_sentences.append(", ".join(kept_clauses))
    return ". ".join(s.strip(" .") for s in kept_sentences if s.strip(" ."))


def build_caption_workflow(image_name: str) -> dict[str, Any]:
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "2": {
            "class_type": "DownloadAndLoadFlorence2Model",
            "inputs": {"model": FLORENCE2_MODEL, "precision": "fp16", "attention": "sdpa"},
        },
        "3": {
            "class_type": "Florence2Run",
            "inputs": {
                "image": ["1", 0],
                "florence2_model": ["2", 0],
                "text_input": "",
                "task": "more_detailed_caption",
                "fill_mask": True,
                "keep_model_loaded": False,
                "max_new_tokens": 256,
                "num_beams": 3,
                "do_sample": True,
                "seed": 1,
            },
        },
        "4": {"class_type": "ShowText|pysssss", "inputs": {"text": ["3", 2]}},
    }


def caption_photo(local_path: str, name_hint: str) -> str:
    """Describe a photo's subject (age impression, body type, hair, clothing) via Florence-2,
    so MopMix Duo's prompt can fight the checkpoint's own bias (which skews young/athletic)
    instead of relying on img2img denoise alone to carry those traits through."""
    try:
        uploaded = upload_image_to_comfy(local_path, name_hint)
        prompt_id = queue_prompt(build_caption_workflow(uploaded), str(uuid.uuid4()))
        deadline = time.time() + CAPTION_TIMEOUT
        while time.time() < deadline:
            history = get_history(prompt_id)
            item = history.get(prompt_id)
            if item and item.get("outputs"):
                texts = item["outputs"].get("4", {}).get("text") or []
                if texts:
                    return strip_scene_details(str(texts[0]).strip())
                return ""
            time.sleep(POLL_SECONDS)
    except Exception:
        log.warning("Photo captioning failed, continuing without it", exc_info=True)
    return ""


def expand_idea_with_ollama(idea: str, photo_caption: str = "") -> str:
    prompt = f"{OLLAMA_SCENARIO_SYSTEM_PROMPT}"
    if photo_caption:
        # Florence-2's caption comes back in English; Qwen2.5 (heavily Chinese+English-tuned)
        # sometimes drifts into Chinese mid-generation if fed English text inside an otherwise
        # Russian prompt, so translate it first to keep the whole prompt one language.
        photo_caption = translate_to_russian(photo_caption)
        prompt += (
            f"\n\nНа референс-фото, с которого делается видео, видно: {photo_caption}\n"
            "ПЕРВОЕ предложение обязано описывать именно то, что на фото (внешность, одежда) — "
            "бери эти детали из описания фото, а не выдумывай свои. Не меняй внешность/одежду "
            "персонажа между предложениями (без этого видео-модель «теряет» лицо и причёску, "
            "переключаясь на других людей)."
        )
    prompt += f"\n\nИдея пользователя: {idea}"
    r = requests.post(
        f"{OLLAMA_BASE}/api/generate",
        json={
            "model": OLLAMA_SCENARIO_MODEL,
            "prompt": prompt,
            "stream": False,
            # Force CPU-only: Ollama was offloading this 32B model onto the GPU by default
            # (~9.7GB VRAM, observed via `ollama ps`), starving ComfyUI's video generation
            # and causing it to OOM in the text encoder within seconds of starting. The two
            # services share one physical GPU even though they're separate processes.
            "options": {"num_gpu": 0},
            "keep_alive": 0,
        },
        timeout=OLLAMA_SCENARIO_TIMEOUT,
    )
    r.raise_for_status()
    text = (r.json().get("response") or "").strip()
    return text or idea


def translate_to_english(text: str) -> str:
    """MopMix's SDXL checkpoint uses a plain CLIPTextEncode (English-only training data);
    non-English prompts produce near-random conditioning instead of following the request.
    LTX Sulphur/Eros use LLM-based multilingual text encoders and don't need this."""
    try:
        translated = GoogleTranslator(source="auto", target="en").translate(text)
        return translated or text
    except Exception:
        log.warning("Prompt translation failed, using original text", exc_info=True)
        return text


def translate_to_russian(text: str) -> str:
    """Qwen2.5 is heavily Chinese+English-tuned; feeding it English photo-caption text inside
    an otherwise-Russian prompt sometimes makes it drift into Chinese mid-generation instead of
    staying in Russian. Translate the caption first so the whole prompt is one language."""
    try:
        translated = GoogleTranslator(source="auto", target="ru").translate(text)
        return translated or text
    except Exception:
        log.warning("Caption translation failed, using original text", exc_info=True)
        return text


def patch_mopmix_workflow(
    wf: dict[str, Any],
    *,
    prompt: str,
    resolution: str,
    image_name: str,
    seed: int,
    denoise: float = MOPMIX_DENOISE,
) -> dict[str, Any]:
    wf = json.loads(json.dumps(wf))
    wf["109"]["inputs"]["text"] = prompt
    wf["18"]["inputs"]["resolution"] = resolution
    wf["300"]["inputs"]["image"] = image_name

    # img2img: node 168 (high-noise KSamplerAdvanced) now starts from the photo's encoded
    # latent (node 302) instead of an empty one; partially skip its own step schedule so
    # the photo survives instead of being fully redrawn from noise.
    steps = int(wf["168"]["inputs"]["steps"])
    wf["168"]["inputs"]["start_at_step"] = max(0, min(steps - 1, round(steps * (1 - float(denoise)))))

    seed = int(seed)
    wf["168"]["inputs"]["noise_seed"] = seed
    wf["169"]["inputs"]["noise_seed"] = seed
    wf["170"]["inputs"]["seed"] = seed
    wf["193:137"]["inputs"]["seed"] = seed + 1
    wf["194:160"]["inputs"]["seed"] = seed + 2
    return wf


def patch_mopmix_duo_workflow(
    wf: dict[str, Any],
    *,
    prompt: str,
    resolution: str,
    image_name_a: str,
    image_name_b: str,
    composite_image_name: str,
    seed: int,
    denoise: float = MOPMIX_DUO_DENOISE,
) -> dict[str, Any]:
    wf = json.loads(json.dumps(wf))
    wf["109"]["inputs"]["text"] = prompt
    wf["18"]["inputs"]["resolution"] = resolution
    wf["300"]["inputs"]["image"] = composite_image_name
    wf["400"]["inputs"]["image"] = image_name_a
    wf["401"]["inputs"]["image"] = image_name_b

    # img2img from the two-photo composite (node 300/301/302) instead of an empty latent, so
    # body/hair/age/clothing come from the real photos; ReActor below only fine-tunes faces.
    steps = int(wf["168"]["inputs"]["steps"])
    wf["168"]["inputs"]["start_at_step"] = max(0, min(steps - 1, round(steps * (1 - float(denoise)))))

    for node_id in ("402", "403"):
        inputs = wf[node_id]["inputs"]
        inputs["swap_model"] = REACTOR_SWAP_MODEL
        inputs["facedetection"] = REACTOR_FACE_DETECTION
        inputs["face_restore_model"] = REACTOR_FACE_RESTORE_MODEL

    seed = int(seed)
    wf["168"]["inputs"]["noise_seed"] = seed
    wf["169"]["inputs"]["noise_seed"] = seed
    wf["170"]["inputs"]["seed"] = seed
    wf["193:137"]["inputs"]["seed"] = seed + 1
    wf["194:160"]["inputs"]["seed"] = seed + 2
    return wf


def split_prompt_into_timeline_segments(prompt: str, max_frames: int) -> tuple[list[str], list[int]]:
    """Break a free-form prompt into sentence-level segments and proportionally distribute
    the video's frame budget across them in the order they were written, so an event
    described early in the prompt actually happens early in the video instead of every
    sentence competing for the same time window (PromptRelayEncodeTimeline otherwise treats
    a single-segment prompt as applying uniformly across the whole duration)."""
    parts = [p.strip() for p in re.split(r"(?<=[.!?…])\s+", prompt.strip()) if p.strip()]
    if len(parts) <= 1:
        return [prompt.strip()], [max_frames]

    weights = [max(1, len(p)) for p in parts]
    total_weight = sum(weights)
    lengths = [max(1, round(max_frames * w / total_weight)) for w in weights]
    lengths[-1] = max(1, lengths[-1] + (max_frames - sum(lengths)))
    return parts, lengths


def patch_ltx_eros_workflow(
    wf: dict[str, Any],
    *,
    prompt: str,
    image_name: str,
    width: int,
    height: int,
    seconds: int,
    seed: int,
) -> dict[str, Any]:
    wf = json.loads(json.dumps(wf))

    wf["990"]["inputs"]["ckpt_name"] = "10Eros_v1.2_fp8mixed_learned.safetensors"
    wf["988"]["inputs"]["lora_name"] = "ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors"
    wf["971"]["inputs"]["clip_name1"] = LTX_EROS_CLIP_NAME1

    wf["791"]["inputs"]["Xi"] = int(width)
    wf["791"]["inputs"]["Xf"] = int(width)
    wf["792"]["inputs"]["Xi"] = int(height)
    wf["792"]["inputs"]["Xf"] = int(height)
    wf["796"]["inputs"]["Xi"] = int(seconds)
    wf["796"]["inputs"]["Xf"] = int(seconds)

    wf["1053"]["inputs"]["image_paths"] = "\n".join([image_name] * 4)

    fps = 24
    max_frames = int(seconds) * fps + 1
    segments, segment_lengths = split_prompt_into_timeline_segments(prompt, max_frames)
    wf["1048"]["inputs"]["global_prompt"] = prompt
    wf["1048"]["inputs"]["max_frames"] = max_frames
    wf["1048"]["inputs"]["timeline_data"] = json.dumps(
        {
            "segments": [
                {"prompt": seg, "length": length, "color": "#4f8edc"}
                for seg, length in zip(segments, segment_lengths)
            ]
        }
    )
    wf["1048"]["inputs"]["local_prompts"] = " | ".join(segments)
    wf["1048"]["inputs"]["segment_lengths"] = ", ".join(str(n) for n in segment_lengths)

    # Only one distinct source photo is available (duplicated into all 4 MultiImageLoader
    # slots), so re-inserting it as a keyframe partway through/near the end just re-anchors
    # the video to the static photo and produces a visible snap-back/jump-cut. Keep only the
    # first keyframe (the actual img2img starting frame) active; disable the rest so the
    # video is driven continuously by the prompt instead of repeatedly reverting to the photo.
    keyframe_seconds = [
        min(int(seconds), round(p * int(seconds) / LTX_EROS_KEYFRAME_DEFAULT_SECONDS))
        for p in LTX_EROS_KEYFRAME_SECONDS
    ]
    for node_id in ("1132", "889:1054", "906:1059"):
        for i, sec in enumerate(keyframe_seconds, start=1):
            wf[node_id]["inputs"][f"insert_second_{i}"] = sec
            if i > 1:
                wf[node_id]["inputs"][f"strength_{i}"] = 0.0

    wf["524"]["inputs"]["seed"] = int(seed) % 1_125_899_906_842_624
    return wf


def pick_first_result_from_outputs(outputs, preferred_node=None):
    if preferred_node and preferred_node in outputs:
        node = outputs[preferred_node]
        for key in ("videos", "images", "gifs"):
            arr = node.get(key, [])
            if arr:
                return arr[0]

    for node in outputs.values():
        for key in ("videos", "images", "gifs"):
            arr = node.get(key, [])
            if arr:
                return arr[0]

    raise RuntimeError("No result files found")


def pick_required_result_from_outputs(outputs, preferred_node: str, keys: tuple[str, ...]) -> dict[str, Any] | None:
    node = outputs.get(preferred_node) or {}
    for key in keys:
        arr = node.get(key, [])
        if arr:
            return arr[0]
    return None


async def pick_result_from_history(prompt_id: str, mode: str) -> dict[str, Any] | None:
    history = await asyncio.to_thread(get_history, prompt_id)
    item = history.get(prompt_id)
    if not item:
        return None

    preferred_nodes = {
        "video": "314",
        "ltx_sulphur": "61",
        "ltx_eros": "1135:597",
        "image": "9",
        "mopmix": "128",
        "mopmix_duo": "128",
    }
    preferred_node = preferred_nodes.get(mode, "9")

    outputs = item.get("outputs") or {}
    result = pick_required_result_from_outputs(outputs, preferred_node, ("videos", "images", "gifs"))
    if result is not None:
        return result

    status = item.get("status") or {}
    if status.get("status_str") == "error" or status.get("completed"):
        # The prompt finished (or errored) without ever producing the expected output node -
        # e.g. a mid-generation crash (OOM, etc). Don't silently fall back to some unrelated
        # node's leftover/partial file; surface a real failure instead.
        raise RuntimeError(f"ComfyUI prompt {prompt_id} finished without node {preferred_node} (status={status.get('status_str')})")

    return None


def clear_directory_safe(path: Path, max_age_sec=120):
    now = time.time()
    for f in path.glob("*"):
        try:
            if f.is_file() and now - f.stat().st_mtime > max_age_sec:
                f.unlink()
        except Exception:
            pass


def clear_directory(path: Path):
    for f in path.glob("*"):
        try:
            if f.is_file():
                f.unlink()
        except Exception as e:
            log.warning("Cannot delete %s: %s", f, e)


def is_system_idle() -> bool:
    return GEN_QUEUE.qsize() == 0 and len(ACTIVE_PROMPTS) == 0



def build_video_audio_workflow(
    *,
    video_name: str,
    prompt: str,
    seed: int,
) -> dict[str, Any]:
    return {
        "1": {
            "class_type": "VHS_LoadVideo",
            "inputs": {
                "video": video_name,
                "force_rate": int(VIDEO_AUDIO_LOAD_FPS),
                "custom_width": 0,
                "custom_height": 0,
                "frame_load_cap": 0,
                "skip_first_frames": 0,
                "select_every_nth": 1,
                "format": "None",
            },
        },
        "2": {
            "class_type": "VHS_VideoInfo",
            "inputs": {
                "video_info": ["1", 3],
            },
        },
        "3": {
            "class_type": "MMAudioModelLoader",
            "inputs": {
                "mmaudio_model": VIDEO_AUDIO_MODEL,
                "base_precision": "fp16",
            },
        },
        "4": {
            "class_type": "MMAudioFeatureUtilsLoader",
            "inputs": {
                "vae_model": VIDEO_AUDIO_VAE,
                "synchformer_model": VIDEO_AUDIO_SYNCHFORMER,
                "clip_model": VIDEO_AUDIO_CLIP,
                "mode": "44k",
                "precision": "fp16",
            },
        },
        "5": {
            "class_type": "MMAudioSampler",
            "inputs": {
                "mmaudio_model": ["3", 0],
                "feature_utils": ["4", 0],
                "images": ["1", 0],
                "duration": ["2", 7],
                "steps": VIDEO_AUDIO_STEPS,
                "cfg": VIDEO_AUDIO_CFG,
                "seed": int(seed),
                "prompt": prompt,
                "negative_prompt": VIDEO_AUDIO_NEGATIVE_PROMPT,
                "mask_away_clip": False,
                "force_offload": True,
            },
        },
        "6": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["1", 0],
                "audio": ["5", 0],
                "frame_rate": ["2", 5],
                "loop_count": 0,
                "filename_prefix": "tg_video_audio",
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 19,
                "save_metadata": True,
                "trim_to_audio": False,
                "pingpong": False,
                "save_output": True,
            },
        },
    }


async def wait_for_result_from_prompt(
    prompt_id: str,
    *,
    preferred_node: str,
    timeout: int,
) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        history = await asyncio.to_thread(get_history, prompt_id)
        item = history.get(prompt_id)
        if item and item.get("outputs"):
            return pick_first_result_from_outputs(item["outputs"], preferred_node=preferred_node)
        await asyncio.sleep(POLL_SECONDS)
    raise TimeoutError(f"Timed out waiting for prompt_id={prompt_id}")


async def run_video_audio_postprocess(blob: bytes, meta: dict[str, Any], filename: str) -> tuple[bytes, str] | None:
    input_name = f"tg_mmaudio_{uuid.uuid4().hex}.mp4"
    input_path = COMFY_INPUT_DIR / input_name
    save_bytes(input_path, blob)

    try:
        wf = build_video_audio_workflow(
            video_name=input_name,
            prompt=meta.get("prompt") or "",
            seed=make_seed(),
        )
        prompt_id = await asyncio.to_thread(queue_prompt, wf, str(uuid.uuid4()))
        result = await wait_for_result_from_prompt(
            prompt_id,
            preferred_node="6",
            timeout=VIDEO_AUDIO_TIMEOUT,
        )
        audio_blob = await asyncio.to_thread(
            fetch_file,
            result["filename"],
            result.get("subfolder", ""),
            result.get("type", "output"),
        )
        audio_name = result.get("filename") or filename
        await asyncio.to_thread(
            delete_comfy_result_file,
            result["filename"],
            result.get("subfolder", ""),
        )
        return audio_blob, audio_name
    finally:
        try:
            input_path.unlink(missing_ok=True)
        except Exception:
            pass


_openvoice_converter = None
_openvoice_target_se_cache: dict[str, Any] = {}


def _get_openvoice_converter():
    global _openvoice_converter
    if _openvoice_converter is None:
        if str(OPENVOICE_DIR) not in sys.path:
            sys.path.insert(0, str(OPENVOICE_DIR))
        from openvoice.api import ToneColorConverter

        # CPU, not CUDA: the bot and ComfyUI are separate processes sharing one 12GB GPU.
        # Once loaded this stays cached (and resident in VRAM if put there) for the bot's
        # whole uptime, permanently shrinking ComfyUI's headroom - same class of bug as the
        # Ollama scenario model defaulting to GPU and starving video generation of VRAM.
        conv = ToneColorConverter(str(OPENVOICE_DIR / "checkpoints_v2/converter/config.json"), device="cpu")
        conv.load_ckpt(str(OPENVOICE_DIR / "checkpoints_v2/converter/checkpoint.pth"))
        _openvoice_converter = conv
    return _openvoice_converter


def _get_dub_target_se(voice_name: str):
    if voice_name not in _openvoice_target_se_cache:
        path = voice_path(voice_name)
        if not path:
            raise RuntimeError(f"Voice '{voice_name}' not found in {VOICES_DIR}")
        conv = _get_openvoice_converter()
        from openvoice import se_extractor

        se, _ = se_extractor.get_se(str(path), conv, vad=False)
        _openvoice_target_se_cache[voice_name] = se
    return _openvoice_target_se_cache[voice_name]


def dub_voice_in_video(video_path: Path, output_path: Path, voice_name: str) -> None:
    conv = _get_openvoice_converter()
    target_se = _get_dub_target_se(voice_name)
    from openvoice import se_extractor

    extracted_path = video_path.with_name(video_path.stem + "_extracted.wav")
    converted_path = video_path.with_name(video_path.stem + "_converted.wav")
    try:
        run_cmd(["ffmpeg", "-y", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le",
                  "-ar", "44100", "-ac", "2", str(extracted_path)])
        source_se, _ = se_extractor.get_se(str(extracted_path), conv, vad=False)
        conv.convert(
            audio_src_path=str(extracted_path),
            src_se=source_se,
            tgt_se=target_se,
            output_path=str(converted_path),
            tau=OPENVOICE_TAU,
        )
        run_cmd(["ffmpeg", "-y", "-i", str(video_path), "-i", str(converted_path),
                  "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
                  "-shortest", str(output_path)])
    finally:
        extracted_path.unlink(missing_ok=True)
        converted_path.unlink(missing_ok=True)


async def run_voice_dub_postprocess(blob: bytes, meta: dict[str, Any], filename: str) -> tuple[bytes, str] | None:
    voice_name = meta.get("dub_voice_name") or DEFAULT_VOICE_NAME
    if not voice_path(voice_name):
        log.warning("Voice dub sample '%s' not found in %s, skipping", voice_name, VOICES_DIR)
        return None

    video_path = TMP_DIR / f"tg_dub_{uuid.uuid4().hex}.mp4"
    output_path = TMP_DIR / f"tg_dub_out_{uuid.uuid4().hex}.mp4"
    try:
        save_bytes(video_path, blob)
        await asyncio.to_thread(dub_voice_in_video, video_path, output_path, voice_name)
        return output_path.read_bytes(), output_path.name
    finally:
        video_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)


async def run_video_tts_postprocess(blob: bytes, meta: dict[str, Any], filename: str) -> tuple[bytes, str] | None:
    speech_text = extract_speech_text(meta.get("prompt") or "")
    if not speech_text:
        return None

    video_path = TMP_DIR / f"tg_tts_video_{uuid.uuid4().hex}.mp4"
    speech_path = TMP_DIR / f"tg_tts_speech_{uuid.uuid4().hex}.mp3"
    mixed_path = TMP_DIR / f"tg_tts_mixed_{uuid.uuid4().hex}.mp4"
    try:
        save_bytes(video_path, blob)
        await asyncio.to_thread(synthesize_speech, speech_text, speech_path)
        await asyncio.to_thread(mix_video_with_tts, video_path, speech_path, mixed_path)
        return mixed_path.read_bytes(), mixed_path.name
    finally:
        for path in (video_path, speech_path, mixed_path):
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass


MODE_DISPLAY_NAMES = {
    "video": "Video",
    "image": "Edit Photo",
    "ltx_sulphur": "LTX Sulphur",
    "ltx_eros": "LTX Eros",
    "mopmix": "MopMix",
    "mopmix_duo": "MopMix Duo",
}


def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"~{seconds} сек"
    minutes, sec = divmod(seconds, 60)
    return f"~{minutes} мин {sec} сек" if sec else f"~{minutes} мин"


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds} сек"
    minutes, sec = divmod(seconds, 60)
    return f"{minutes} мин {sec} сек" if sec else f"{minutes} мин"


def job_status_text(meta: dict[str, Any], *, started: bool) -> str:
    mode_label = MODE_DISPLAY_NAMES.get(meta.get("mode"), meta.get("mode") or "")
    batch_index = meta.get("batch_index", 1)
    batch_total = meta.get("batch_total", 1)
    position = f" {batch_index}/{batch_total}" if batch_total > 1 else ""
    header = f"🔄 Генерация{position}: {mode_label}"
    avg = MODE_AVG_DURATION.get(meta.get("mode"), 120.0)
    if not started:
        return f"{header}\nОжидаемое время: {format_eta(avg)}"
    elapsed = time.time() - meta.get("started_at", time.time())
    remaining = max(0.0, avg - elapsed)
    return f"{header}\nПрошло: {format_eta(elapsed)} · Осталось: {format_eta(remaining)}"


async def update_chat_status(bot, chat_id: int, text: str) -> None:
    entry = CHAT_STATUS.setdefault(chat_id, {})
    if entry.get("message_id"):
        try:
            await bot.edit_message_text(chat_id=chat_id, message_id=entry["message_id"], text=text)
            return
        except Exception as e:
            if "not modified" in str(e).lower():
                return
            entry["message_id"] = None
    try:
        msg = await bot.send_message(chat_id=chat_id, text=text)
        entry["message_id"] = msg.message_id
    except Exception:
        log.exception("Failed to send status message to chat %s", chat_id)


async def delete_chat_status_message(bot, chat_id: int) -> None:
    """Drop the tracked status message without replacing it, so the next update_chat_status
    call sends a fresh message instead of editing one stuck above newer chat content (Telegram
    can't move a message - only delete+resend keeps the status pinned to the bottom)."""
    entry = CHAT_STATUS.get(chat_id)
    if not entry:
        return
    msg_id = entry.pop("message_id", None)
    if msg_id:
        try:
            await bot.delete_message(chat_id=chat_id, message_id=msg_id)
        except Exception:
            pass


async def cleanup_chat_status(bot, chat_id: int) -> None:
    """Remove leftover artifacts from the previous generation cycle (the status message and
    the post-completion menu) before starting a new one, so the chat doesn't accumulate trace
    messages and the new status lands at the bottom instead of editing a stale, scrolled-away one."""
    await delete_chat_status_message(bot, chat_id)
    entry = CHAT_STATUS.get(chat_id)
    if not entry:
        return
    menu_message_id = entry.pop("menu_message_id", None)
    if menu_message_id:
        try:
            await bot.delete_message(chat_id=chat_id, message_id=menu_message_id)
        except Exception:
            pass


async def finish_chat_status(bot, chat_id: int, text: str, *, show_menu: bool) -> None:
    await update_chat_status(bot, chat_id, text)
    if show_menu:
        entry = CHAT_STATUS.setdefault(chat_id, {})
        old_menu_id = entry.pop("menu_message_id", None)
        if old_menu_id:
            try:
                await bot.delete_message(chat_id=chat_id, message_id=old_menu_id)
            except Exception:
                pass
        try:
            msg = await bot.send_message(chat_id=chat_id, text="Готово! Выбери режим:", reply_markup=main_keyboard())
            entry["menu_message_id"] = msg.message_id
        except Exception:
            log.exception("Failed to send menu to chat %s", chat_id)


def workflow_info_line(meta: dict[str, Any]) -> str:
    label = MODE_DISPLAY_NAMES.get(meta.get("mode"), meta.get("mode") or "")
    parts = [label]
    seconds = meta.get("seconds")
    if seconds:
        parts.append(f"{seconds}с")
    quality = meta.get("quality")
    if quality:
        parts.append(str(quality))
    started_at = meta.get("started_at")
    if started_at:
        parts.append(format_duration(time.time() - started_at))
    return " · ".join(p for p in parts if p)


async def send_result(app: Application, meta: dict[str, Any], result: dict[str, Any]) -> None:
    blob = await asyncio.to_thread(
        fetch_file,
        result["filename"],
        result.get("subfolder", ""),
        result.get("type", "output"),
    )

    filename = result.get("filename", "result.bin")

    if (
        VIDEO_AUDIO
        and meta.get("mode") in SILENT_VIDEO_MODES
        and filename.lower().endswith((".mp4", ".mov", ".webm"))
    ):
        try:
            processed = await run_video_audio_postprocess(blob, meta, filename)
            if processed:
                blob, filename = processed
                log.info("MMAudio postprocess applied for video job #%s", meta.get("job_id"))
        except Exception:
            log.exception("MMAudio postprocess failed; sending original silent video")

    if (
        VIDEO_TTS
        and meta.get("mode") in SILENT_VIDEO_MODES
        and filename.lower().endswith((".mp4", ".mov", ".webm"))
    ):
        try:
            processed = await run_video_tts_postprocess(blob, meta, filename)
            if processed:
                blob, filename = processed
                log.info("TTS speech postprocess applied for video job #%s", meta.get("job_id"))
        except Exception:
            log.exception("TTS speech postprocess failed; sending video without TTS speech")

    if (
        meta.get("dub_voice")
        and meta.get("mode") in DUB_VOICE_ENABLED_MODES
        and filename.lower().endswith((".mp4", ".mov", ".webm"))
    ):
        try:
            processed = await run_voice_dub_postprocess(blob, meta, filename)
            if processed:
                blob, filename = processed
                log.info("Voice dub postprocess applied for job #%s", meta.get("job_id"))
        except Exception:
            log.exception("Voice dub postprocess failed; sending video with original voice")

    prompt_text = (meta.get("prompt") or "").strip()

    caption = prompt_text or ""
    info_line = workflow_info_line(meta)
    if info_line:
        caption = f"{caption}\n\n{info_line}" if caption else info_line
    caption = caption[:MAX_CAPTION]

    input_audio_name = meta.get("input_audio_name")
    if input_audio_name:
        try:
            (COMFY_INPUT_DIR / input_audio_name).unlink(missing_ok=True)
        except Exception:
            pass

    bio = io.BytesIO(blob)
    bio.name = filename

    if filename.lower().endswith(".gif"):
        gif_path = TMP_DIR / filename
        mp4_path = TMP_DIR / (filename + ".mp4")

        save_bytes(gif_path, blob)
        await asyncio.to_thread(gif_to_mp4, gif_path, mp4_path)

        with mp4_path.open("rb") as f:
            await app.bot.send_video(
                chat_id=meta["chat_id"],
                video=InputFile(f, filename=mp4_path.name),
                caption=caption,
            )
        await asyncio.to_thread(clear_directory_safe, COMFY_OUTPUT_DIR)
        return

    if meta["mode"] in VIDEO_MODES:
        if not filename.lower().endswith((".mp4", ".mov", ".webm", ".gif")):
            filename += ".mp4"
            bio.name = filename

        await app.bot.send_video(
            chat_id=meta["chat_id"],
            video=InputFile(bio, filename=filename),
            caption=caption,
        )
    else:
        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            filename += ".png"
            bio.name = filename

        await app.bot.send_photo(
            chat_id=meta["chat_id"],
            photo=InputFile(bio, filename=filename),
            caption=caption,
        )

    await asyncio.to_thread(clear_directory_safe, COMFY_OUTPUT_DIR)
    if is_system_idle():
        await asyncio.to_thread(clear_directory_safe, COMFY_INPUT_DIR)

    await asyncio.to_thread(
        delete_comfy_result_file,
        result["filename"],
        result.get("subfolder", ""),
    )


# ============================================================
# MONITOR / WORKER
# ============================================================
async def monitor_loop(app: Application) -> None:
    log.info("Monitor loop started")

    while True:
        done_ids = []

        for prompt_id, meta in list(ACTIVE_PROMPTS.items()):
            try:
                if "started_at" in meta:
                    now = time.time()
                    if now - meta.get("last_status_edit", 0.0) >= STATUS_UPDATE_INTERVAL:
                        meta["last_status_edit"] = now
                        await update_chat_status(app.bot, meta["chat_id"], job_status_text(meta, started=True))

                try:
                    result = await pick_result_from_history(prompt_id, meta["mode"])
                except Exception as e:
                    log.exception("Prompt %s failed", prompt_id)
                    done_ids.append(prompt_id)
                    try:
                        await app.bot.send_message(meta["chat_id"], f"⚠️ Генерация не удалась: {e}")
                    except Exception:
                        pass
                    if meta.get("batch_index", 1) == meta.get("batch_total", 1):
                        await finish_chat_status(app.bot, meta["chat_id"], "⚠️ Завершено с ошибкой.", show_menu=True)
                    else:
                        await delete_chat_status_message(app.bot, meta["chat_id"])
                    continue

                if result is None:
                    continue
                # Drop the status message before delivering the photo/video so it doesn't end
                # up stuck above the new content; it gets recreated fresh at the bottom below.
                await delete_chat_status_message(app.bot, meta["chat_id"])
                await send_result(app, meta, result)
                done_ids.append(prompt_id)

                if "started_at" in meta:
                    duration = time.time() - meta["started_at"]
                    prev_avg = MODE_AVG_DURATION.get(meta["mode"], duration)
                    MODE_AVG_DURATION[meta["mode"]] = 0.7 * prev_avg + 0.3 * duration

                if meta.get("batch_index", 1) == meta.get("batch_total", 1):
                    mode_label = MODE_DISPLAY_NAMES.get(meta["mode"], meta["mode"])
                    await finish_chat_status(app.bot, meta["chat_id"], f"✅ Готово: {mode_label}", show_menu=True)

            except Exception:
                log.exception("Monitor error for %s", prompt_id)

        for prompt_id in done_ids:
            ACTIVE_PROMPTS.pop(prompt_id, None)

        await asyncio.sleep(POLL_SECONDS)


async def submit_worker_loop(app: Application) -> None:
    global WORKER_STARTED
    if WORKER_STARTED:
        return

    WORKER_STARTED = True
    log.info("Submit worker started")

    while True:
        while ACTIVE_PROMPTS:
            await asyncio.sleep(POLL_SECONDS)

        job = await GEN_QUEUE.get()
        await update_chat_status(
            app.bot, job.chat_id,
            job_status_text(
                {"mode": job.mode, "batch_index": job.batch_index, "batch_total": job.batch_total},
                started=False,
            ),
        )
        try:
            if job.mode == "video":
                await submit_video_job(app, job)
            elif job.mode == "ltx_sulphur":
                await submit_ltx_sulphur_job(app, job)
            elif job.mode == "ltx_eros":
                await submit_ltx_eros_job(app, job)
            elif job.mode == "image":
                await submit_image_job(app, job)
            elif job.mode == "mopmix":
                await submit_mopmix_job(app, job)
            elif job.mode == "mopmix_duo":
                await submit_mopmix_duo_job(app, job)
            else:
                raise RuntimeError(f"Unknown job mode: {job.mode}")

            for meta in ACTIVE_PROMPTS.values():
                if meta.get("job_id") == job.job_id:
                    meta["batch_index"] = job.batch_index
                    meta["batch_total"] = job.batch_total
                    meta["started_at"] = time.time()
                    meta["last_status_edit"] = 0.0
                    break
        except Exception as e:
            log.exception("Submit failed")
            try:
                await app.bot.send_message(job.chat_id, f"Ошибка отправки задачи #{job.job_id}: {e}")
            except Exception:
                pass
            if job.batch_index == job.batch_total:
                await finish_chat_status(app.bot, job.chat_id, "⚠️ Завершено с ошибкой.", show_menu=True)
        finally:
            GEN_QUEUE.task_done()


# ============================================================
# COMMANDS
# ============================================================
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_if_needed(update):
        return
    await send_ui_message(update.message, context, help_text(get_state(context)), reply_markup=main_keyboard(get_state(context)))


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_if_needed(update):
        return
    await send_ui_message(update.message, context, help_text(get_state(context)), reply_markup=main_keyboard(get_state(context)))


async def reset_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_if_needed(update):
        return
    st = reset_state(context)
    await send_ui_message(update.message, context, "Состояние очищено.\n\n" + help_text(st), reply_markup=main_keyboard(st))


async def stop_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_if_needed(update):
        return

    try:
        await asyncio.to_thread(interrupt_current)
        active_cleared = clear_active_prompts()
        await send_ui_message(
            update.message,
            context,
            f"Текущая генерация остановлена.\nСброшено active prompt: {active_cleared}",
            reply_markup=main_keyboard(get_state(context)),
        )
    except Exception as e:
        await send_ui_message(update.message, context, f"Ошибка остановки: {e}", reply_markup=main_keyboard(get_state(context)))


async def clearqueue_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_if_needed(update):
        return

    try:
        comfy_resp = await asyncio.to_thread(clear_comfy_queue)
        local_cleared = clear_local_queue()
        await send_ui_message(
            update.message,
            context,
            f"Очередь очищена.\n• локальных задач удалено: {local_cleared}\n• ответ ComfyUI: {comfy_resp}",
            reply_markup=main_keyboard(get_state(context)),
        )
    except Exception as e:
        await send_ui_message(update.message, context, f"Ошибка очистки очереди: {e}", reply_markup=main_keyboard(get_state(context)))


async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_if_needed(update):
        return

    st = get_state(context)
    try:
        q = await asyncio.to_thread(get_queue_state)
        running = len(q.get("queue_running", []) or [])
        pending = len(q.get("queue_pending", []) or [])
        comfy_info = (
            f"\n\nСервер:\n"
            f"• local queue: {GEN_QUEUE.qsize()}\n"
            f"• comfy running: {running}\n"
            f"• comfy pending: {pending}\n"
            f"• active prompts tracked: {len(ACTIVE_PROMPTS)}"
        )
    except Exception as e:
        comfy_info = f"\n\nСервер:\n• local queue: {GEN_QUEUE.qsize()}\n• comfy status error: {e}"

    await send_ui_message(update.message, context, help_text(st) + comfy_info, reply_markup=main_keyboard(st))


async def video_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_if_needed(update):
        return
    st = get_state(context)
    st["mode"] = "video"
    st["seconds"] = clamp_seconds(st["seconds"], st["mode"])
    await send_ui_message(update.message, context, "Режим: photo → video", reply_markup=main_keyboard(st))


async def ltx_sulphur_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_if_needed(update):
        return
    st = get_state(context)
    st["mode"] = "ltx_sulphur"
    st["seconds"] = clamp_seconds(st["seconds"], st["mode"])
    await send_ui_message(update.message, context, "Режим: photo → video+audio (LTX2.3 Sulphur)", reply_markup=main_keyboard(st))


async def ltx_eros_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_if_needed(update):
        return
    st = get_state(context)
    st["mode"] = "ltx_eros"
    st["seconds"] = clamp_seconds(st["seconds"], st["mode"])
    await send_ui_message(update.message, context, "Режим: photo → video+audio (LTX2.3 10Eros)", reply_markup=main_keyboard(st))


async def image_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_if_needed(update):
        return
    st = get_state(context)
    st["mode"] = "image"
    st["seconds"] = clamp_seconds(st["seconds"], st["mode"])
    await send_ui_message(update.message, context, "Режим: photo → image", reply_markup=main_keyboard(st))


async def mopmix_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_if_needed(update):
        return
    st = get_state(context)
    st["mode"] = "mopmix"
    await send_ui_message(update.message, context, "Режим: photo → img2img картинка (MopMix BigASP 2.5)", reply_markup=main_keyboard(st))


async def mopmix_duo_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_if_needed(update):
        return
    st = get_state(context)
    st["mode"] = "mopmix_duo"
    await send_ui_message(update.message, context, "Режим: 2 фото → сцена с обоими лицами (face swap)", reply_markup=main_keyboard(st))


async def loras_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_if_needed(update):
        return
    st = get_state(context)
    await send_ui_message(update.message, context, lora_text(st), reply_markup=lora_keyboard(st))


async def photos_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_if_needed(update):
        return

    user_id = update.effective_user.id if update.effective_user else 0
    st = get_state(context)
    library = rebuild_media_library_from_disk(context, user_id, st["max_side"])
    if not library:
        await send_ui_message(
            update.message,
            context,
            "Пока нет сохранённых фото. Пришли фото один раз, потом его можно будет выбрать здесь.",
            reply_markup=main_keyboard(st),
        )
        return

    await send_media_preview_message(update.message, context, index=0)


async def prompt_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_if_needed(update):
        return

    text = " ".join(context.args).strip()
    if not text:
        await send_ui_message(update.message, context, "Используй: /prompt camera slowly zooms in", reply_markup=main_keyboard(get_state(context)))
        return

    st = get_state(context)
    st["prompt"] = text
    await send_ui_message(update.message, context, "Промт сохранён.", reply_markup=main_keyboard(st))


async def seconds_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_if_needed(update):
        return

    st = get_state(context)
    if not context.args:
        await send_ui_message(
            update.message,
            context,
            f"Сейчас: {st['seconds']} сек. Максимум для режима {st['mode']}: {mode_max_seconds(st['mode'])} сек.",
            reply_markup=main_keyboard(st),
        )
        return

    try:
        sec = int(context.args[0])
    except Exception:
        await send_ui_message(update.message, context, "Пример: /seconds 8", reply_markup=main_keyboard(st))
        return

    st["seconds"] = clamp_seconds(sec, st["mode"])
    await send_ui_message(
        update.message,
        context,
        f"Длина видео: {st['seconds']} сек. Максимум для режима {st['mode']}: {mode_max_seconds(st['mode'])} сек.",
        reply_markup=main_keyboard(st),
    )


async def quality_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_if_needed(update):
        return

    st = get_state(context)
    if not context.args:
        await send_ui_message(
            update.message,
            context,
            f"Сейчас качество: {quality_status(st)}. Используй /quality low | medium | high",
            reply_markup=main_keyboard(st),
        )
        return

    name = (context.args[0] or "").strip().lower()
    if not apply_quality(st, name):
        await send_ui_message(update.message, context, "Используй: /quality low | medium | high", reply_markup=main_keyboard(st))
        return

    await send_ui_message(update.message, context, f"Качество: {quality_status(st)}.", reply_markup=main_keyboard(st))


async def repeat_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_if_needed(update):
        return

    st = get_state(context)
    if not context.args:
        await send_ui_message(update.message, context, f"Сейчас repeat: {st['repeat']}. Пример: /repeat 4", reply_markup=main_keyboard(st))
        return

    try:
        n = int(context.args[0])
    except Exception:
        await send_ui_message(update.message, context, "Пример: /repeat 4", reply_markup=main_keyboard(st))
        return

    n = max(1, min(200, n))
    st["repeat"] = n

    await send_ui_message(update.message, context, f"Количество запусков: {n}", reply_markup=main_keyboard(st))


async def go_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_if_needed(update):
        return
    await enqueue_generation(update, context)


# ============================================================
# PHOTO / TEXT INPUT
# ============================================================
async def voice_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_if_needed(update):
        return

    msg = update.message
    if msg.voice:
        tg_file = await context.bot.get_file(msg.voice.file_id)
        ext = ".ogg"
    elif msg.audio:
        tg_file = await context.bot.get_file(msg.audio.file_id)
        name = msg.audio.file_name or ""
        ext = Path(name).suffix.lower() if Path(name).suffix else ".mp3"
    else:
        return

    path = TMP_DIR / f"tg_voice_upload_{uuid.uuid4().hex}{ext}"
    await tg_file.download_to_drive(custom_path=str(path))
    context.user_data["pending_voice_upload"] = str(path)

    keyboard = InlineKeyboardMarkup(
        [
            [InlineKeyboardButton(f"🔄 Заменить «{DEFAULT_VOICE_NAME}»", callback_data="voice:replace")],
            [InlineKeyboardButton("➕ Добавить как новый голос", callback_data="voice:add")],
            [InlineKeyboardButton("❌ Отмена", callback_data="voice:cancel")],
        ]
    )
    await send_ui_message(msg, context, "🎙 Получил голос. Что сделать?", reply_markup=keyboard)


async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_if_needed(update):
        return

    st = get_state(context)
    photo = update.message.photo[-1]
    tg_file = await context.bot.get_file(photo.file_id)

    file_name = f"tg_{update.effective_user.id}_{uuid.uuid4().hex}.jpg"
    path = TMP_DIR / file_name
    await tg_file.download_to_drive(custom_path=str(path))

    with Image.open(path) as img:
        width, height = img.size

    fit_w, fit_h = fit_size_keep_aspect(width, height, st["max_side"])
    media = {
        "path": str(path),
        "name": file_name,
        "orig_width": width,
        "orig_height": height,
        "fit_width": fit_w,
        "fit_height": fit_h,
    }
    remember_media(context, media)

    if st["mode"] in SINGLE_PHOTO_MODES:
        st["video_source"] = media
        await send_ui_message(
            update.message,
            context,
            f"Фото для {st['mode']} сохранено: {width}×{height} → {fit_w}×{fit_h}",
            reply_markup=main_keyboard(st),
        )
        return

    duo = st["duo_photos"]
    slot = 0 if not duo[0].get("path") else 1
    duo[slot] = media
    await send_ui_message(
        update.message,
        context,
        f"Фото лица {'A' if slot == 0 else 'B'} сохранено: {width}×{height} → {fit_w}×{fit_h}",
        reply_markup=main_keyboard(st),
    )


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_if_needed(update):
        return

    text = (update.message.text or "").strip()
    if not text or text.startswith("/"):
        return

    st = get_state(context)
    st["prompt"] = text

    await send_ui_message(update.message, context, "Текст принят как промт.", reply_markup=main_keyboard(st))


async def run_expand_flow(
    context: ContextTypes.DEFAULT_TYPE,
    st: dict[str, Any],
    chat_id: int,
    idea: str,
    photo_caption: str | None = None,
) -> None:
    """Caption the attached photo (if any and not already known from a previous round),
    expand the idea via Ollama, then ask the user to use/retry/cancel the result instead of
    silently overwriting the prompt - so it's never unclear whether a generated scenario was
    actually applied."""
    status_msg = None
    if photo_caption is None:
        photo_caption = ""
        photo = st.get("video_source") if st["mode"] in SINGLE_PHOTO_MODES else None
        if photo and photo.get("path"):
            status_msg = await context.bot.send_message(chat_id, "🖼 Смотрю на фото...")
            photo_caption = await asyncio.to_thread(caption_photo, photo["path"], photo["name"])

    if status_msg is None:
        status_msg = await context.bot.send_message(chat_id, "✨ Развиваю идею...")
    else:
        try:
            await context.bot.edit_message_text(chat_id=chat_id, message_id=status_msg.message_id, text="✨ Развиваю идею...")
        except Exception:
            pass

    try:
        expanded = await asyncio.to_thread(expand_idea_with_ollama, idea, photo_caption)
    except Exception as e:
        log.exception("Ollama scenario expansion failed")
        try:
            await context.bot.delete_message(chat_id=chat_id, message_id=status_msg.message_id)
        except Exception:
            pass
        await context.bot.send_message(chat_id, f"Не получилось развить идею: {e}", reply_markup=main_keyboard(st))
        return

    try:
        await context.bot.delete_message(chat_id=chat_id, message_id=status_msg.message_id)
    except Exception:
        pass

    st["pending_expansion"] = {"idea": idea, "photo_caption": photo_caption, "expanded": expanded}
    keyboard = InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("✅ Использовать", callback_data="expand:use")],
            [InlineKeyboardButton("🔁 Другой вариант", callback_data="expand:retry")],
            [InlineKeyboardButton("❌ Отмена", callback_data="expand:cancel")],
        ]
    )
    await context.bot.send_message(chat_id, f"Новый промт:\n\n{expanded}", reply_markup=keyboard)


# ============================================================
# BUTTONS
# ============================================================
async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_if_needed(update):
        return

    query = update.callback_query
    await query.answer()
    st = get_state(context)
    data = query.data or ""

    if data == "noop":
        return

    if data.startswith("mode:"):
        st["mode"] = data.split(":", 1)[1]
        st["seconds"] = clamp_seconds(st["seconds"], st["mode"])
        await replace_ui_message_from_callback(query, context, f"Режим: {st['mode']}", reply_markup=main_keyboard(st))
        return

    if data.startswith("quality:"):
        name = data.split(":", 1)[1]
        if apply_quality(st, name):
            await replace_ui_message_from_callback(query, context, f"Качество: {quality_status(st)}.", reply_markup=main_keyboard(st))
        return

    if data.startswith("sec:"):
        delta = data.split(":", 1)[1]
        step = -2 if delta == "-2" else 2
        st["seconds"] = clamp_seconds(st["seconds"] + step, st["mode"])
        await replace_ui_message_from_callback(
            query,
            context,
            f"Длина видео: {st['seconds']} сек. Максимум для режима {st['mode']}: {mode_max_seconds(st['mode'])} сек.",
            reply_markup=main_keyboard(st),
        )
        return

    if data.startswith("repeat:"):
        try:
            repeat = int(data.split(":", 1)[1])
        except Exception:
            repeat = 1
        st["repeat"] = max(1, min(200, repeat))
        await replace_ui_message_from_callback(
            query,
            context,
            "Количество запусков: {}".format(st["repeat"]),
            reply_markup=main_keyboard(st),
        )
        return

    if data == "lora:list":
        await replace_ui_message_from_callback(query, context, lora_text(st), reply_markup=lora_keyboard(st))
        return

    if data.startswith("lora:toggle:"):
        try:
            idx = int(data.rsplit(":", 1)[1])
        except Exception:
            idx = -1
        if 0 <= idx < len(VIDEO_LORA_OPTIONS):
            key = VIDEO_LORA_OPTIONS[idx]["key"]
            selected = list(st.get("video_loras") or [])
            if key in selected:
                selected.remove(key)
            elif len(selected) < VIDEO_MAX_LORAS:
                selected.append(key)
            st["video_loras"] = selected
        await replace_ui_message_from_callback(query, context, lora_text(st), reply_markup=lora_keyboard(st))
        return

    if data == "lora:clear":
        st["video_loras"] = []
        await replace_ui_message_from_callback(query, context, lora_text(st), reply_markup=lora_keyboard(st))
        return

    if data == "media:list":
        user_id = update.effective_user.id if update.effective_user else 0
        library = rebuild_media_library_from_disk(context, user_id, st["max_side"])
        if not library:
            await replace_ui_message_from_callback(
                query,
                context,
                "Пока нет сохранённых фото. Пришли фото один раз, потом его можно будет выбрать здесь.",
                reply_markup=main_keyboard(st),
            )
            return
        await replace_media_preview_from_callback(query, context, index=0)
        return

    if data.startswith("media:page:"):
        user_id = update.effective_user.id if update.effective_user else 0
        rebuild_media_library_from_disk(context, user_id, st["max_side"])
        try:
            idx = int(data.rsplit(":", 1)[1])
        except Exception:
            idx = 0
        await replace_media_preview_from_callback(query, context, index=idx)
        return

    if data.startswith("media:select:"):
        try:
            idx = int(data.rsplit(":", 1)[1])
        except Exception:
            idx = -1

        library = get_media_library(context)
        if idx < 0 or idx >= len(library):
            await replace_ui_message_from_callback(query, context, "Это фото уже недоступно.", reply_markup=main_keyboard(st))
            return

        media = copy.deepcopy(library[idx])
        if not Path(media.get("path", "")).exists():
            await replace_ui_message_from_callback(query, context, "Файл этого фото уже удалён с диска. Пришли его заново.", reply_markup=main_keyboard(st))
            return

        if st["mode"] in DUO_PHOTO_MODES:
            duo = st["duo_photos"]
            slot = 0 if not duo[0].get("path") else 1
            duo[slot] = media
            await replace_ui_message_from_callback(
                query, context, f"Фото лица {'A' if slot == 0 else 'B'} выбрано: {media_line(media)}", reply_markup=main_keyboard(st)
            )
            return

        st["video_source"] = media
        await replace_ui_message_from_callback(query, context, f"Базовое фото выбрано: {media_line(media)}", reply_markup=main_keyboard(st))
        return

    if data == "queue:stopclear":
        try:
            await asyncio.to_thread(interrupt_current)
            active_cleared = clear_active_prompts()
            comfy_resp = await asyncio.to_thread(clear_comfy_queue)
            local_cleared = clear_local_queue()
            await replace_ui_message_from_callback(
                query,
                context,
                f"Остановлено и очищено.\n• active prompt сброшено: {active_cleared}\n"
                f"• локальных задач удалено: {local_cleared}\n• ответ ComfyUI: {comfy_resp}",
                reply_markup=main_keyboard(st),
            )
        except Exception as e:
            await replace_ui_message_from_callback(query, context, f"Ошибка очистки очереди: {e}", reply_markup=main_keyboard(st))
        return

    if data == "show:status":
        try:
            q = await asyncio.to_thread(get_queue_state)
            running = len(q.get("queue_running", []) or [])
            pending = len(q.get("queue_pending", []) or [])
            comfy_info = (
                f"\n\nСервер:\n"
                f"• local queue: {GEN_QUEUE.qsize()}\n"
                f"• comfy running: {running}\n"
                f"• comfy pending: {pending}\n"
                f"• active prompts tracked: {len(ACTIVE_PROMPTS)}"
            )
        except Exception as e:
            comfy_info = f"\n\nСервер:\n• local queue: {GEN_QUEUE.qsize()}\n• comfy status error: {e}"

        await replace_ui_message_from_callback(query, context, help_text(st) + comfy_info, reply_markup=main_keyboard(st))
        return

    if data == "do:reset":
        st = reset_state(context)
        await replace_ui_message_from_callback(query, context, "Состояние очищено.\n\n" + help_text(st), reply_markup=main_keyboard(st))
        return

    if data == "do:expand":
        idea = (st.get("prompt") or "").strip()
        if not idea:
            await replace_ui_message_from_callback(query, context, "Сначала напиши идею промтом.", reply_markup=main_keyboard(st))
            return
        if st["mode"] in SINGLE_PHOTO_MODES and not st["video_source"].get("path"):
            await replace_ui_message_from_callback(
                query, context, f"Сначала пришли фото для {st['mode']}, потом разворачивай идею — иначе Ollama сочинит внешность от себя.",
                reply_markup=main_keyboard(st),
            )
            return
        if st["mode"] in DUO_PHOTO_MODES and not all(x.get("path") for x in st["duo_photos"]):
            await replace_ui_message_from_callback(
                query, context, "Сначала пришли оба фото (лицо A и лицо B), потом разворачивай идею.",
                reply_markup=main_keyboard(st),
            )
            return
        try:
            await query.message.delete()
        except Exception:
            pass
        await run_expand_flow(context, st, query.message.chat_id, idea)
        return

    if data == "expand:retry":
        pending = st.get("pending_expansion")
        if not pending:
            await replace_ui_message_from_callback(query, context, "Нечего повторять — начни заново через «✨ Развить идею».", reply_markup=main_keyboard(st))
            return
        try:
            await query.message.delete()
        except Exception:
            pass
        await run_expand_flow(context, st, query.message.chat_id, pending["idea"], photo_caption=pending["photo_caption"])
        return

    if data == "expand:use":
        pending = st.get("pending_expansion")
        if not pending:
            await replace_ui_message_from_callback(query, context, "Нечего применять — начни заново через «✨ Развить идею».", reply_markup=main_keyboard(st))
            return
        st["prompt"] = pending["expanded"]
        st.pop("pending_expansion", None)
        await replace_ui_message_from_callback(query, context, f"✅ Промт применён:\n\n{st['prompt']}", reply_markup=main_keyboard(st))
        return

    if data == "expand:cancel":
        st.pop("pending_expansion", None)
        await replace_ui_message_from_callback(query, context, "Отменено, промт не изменён.", reply_markup=main_keyboard(st))
        return

    if data == "voice:replace":
        pending_path = context.user_data.pop("pending_voice_upload", None)
        if not pending_path or not Path(pending_path).exists():
            await replace_ui_message_from_callback(query, context, "Нечего применять — пришли голос заново.", reply_markup=main_keyboard(st))
            return
        VOICES_DIR.mkdir(parents=True, exist_ok=True)
        for ext in VOICE_FILE_EXTENSIONS:
            (VOICES_DIR / f"{DEFAULT_VOICE_NAME}{ext}").unlink(missing_ok=True)
        dest = VOICES_DIR / f"{DEFAULT_VOICE_NAME}{Path(pending_path).suffix}"
        shutil.move(pending_path, dest)
        _openvoice_target_se_cache.pop(DEFAULT_VOICE_NAME, None)
        st["dub_voice_name"] = DEFAULT_VOICE_NAME
        await replace_ui_message_from_callback(query, context, f"✅ Голос «{DEFAULT_VOICE_NAME}» заменён.", reply_markup=main_keyboard(st))
        return

    if data == "voice:add":
        pending_path = context.user_data.pop("pending_voice_upload", None)
        if not pending_path or not Path(pending_path).exists():
            await replace_ui_message_from_callback(query, context, "Нечего применять — пришли голос заново.", reply_markup=main_keyboard(st))
            return
        VOICES_DIR.mkdir(parents=True, exist_ok=True)
        name = next_voice_name()
        dest = VOICES_DIR / f"{name}{Path(pending_path).suffix}"
        shutil.move(pending_path, dest)
        st["dub_voice_name"] = name
        await replace_ui_message_from_callback(
            query, context, f"✅ Добавлен новый голос «{name}» и выбран как активный для дубляжа.",
            reply_markup=main_keyboard(st),
        )
        return

    if data == "voice:cancel":
        pending_path = context.user_data.pop("pending_voice_upload", None)
        if pending_path:
            Path(pending_path).unlink(missing_ok=True)
        await replace_ui_message_from_callback(query, context, "Отменено.", reply_markup=main_keyboard(st))
        return

    if data == "voice:list":
        names = list_voice_names()
        if not names:
            await replace_ui_message_from_callback(query, context, "Банк голосов пуст — пришли голосовое сообщение боту.", reply_markup=main_keyboard(st))
            return
        rows = [
            [InlineKeyboardButton(f"{'✅ ' if n == st.get('dub_voice_name') else ''}{n}", callback_data=f"voice:pick:{n}")]
            for n in names
        ]
        rows.append([InlineKeyboardButton("↩️ Back", callback_data="show:status")])
        await replace_ui_message_from_callback(query, context, "Выбери голос для дубляжа:", reply_markup=InlineKeyboardMarkup(rows))
        return

    if data.startswith("voice:pick:"):
        name = data.split(":", 2)[2]
        if not voice_path(name):
            await replace_ui_message_from_callback(query, context, "Этот голос больше не найден.", reply_markup=main_keyboard(st))
            return
        st["dub_voice_name"] = name
        await replace_ui_message_from_callback(query, context, f"🎙 Активный голос для дубляжа: «{name}».", reply_markup=main_keyboard(st))
        return

    if data == "do:roulette":
        st["roulette"] = not st.get("roulette")
        state_text = "включена" if st["roulette"] else "выключена"
        await replace_ui_message_from_callback(
            query,
            context,
            f"🎰 Рулетка {state_text}.\n"
            f"При repeat ≥ 2 каждые {ROULETTE_GROUP_SIZE} видео промт будет переписываться заново "
            f"через Ollama (та же идея, другие детали) — на 10x получится 5 вариаций, на 30x — 15.",
            reply_markup=main_keyboard(st),
        )
        return

    if data == "do:dubvoice":
        st["dub_voice"] = not st.get("dub_voice")
        state_text = "включён" if st["dub_voice"] else "выключен"
        await replace_ui_message_from_callback(
            query,
            context,
            f"🎙 Дубляж голоса {state_text}.\n"
            f"Работает только для LTX Sulphur/LTX Eros — после генерации голос в дорожке "
            f"заменяется на голос из {DUB_VOICE_SAMPLE_PATH}, остальные звуки (шлепки/стоны) сохраняются.",
            reply_markup=main_keyboard(st),
        )
        return

    if data == "do:go":
        ui = get_ui_state(context)
        if ui.get("last_ui_message_id") == query.message.message_id:
            ui["last_ui_message_id"] = None
        try:
            await query.message.delete()
        except Exception:
            pass
        await enqueue_generation(update, context)
        return


# ============================================================
# QUEUE / JOBS
# ============================================================
async def enqueue_generation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global JOB_SEQ

    st = get_state(context)
    target = update.message if update.message else update.callback_query.message

    if not st.get("prompt"):
        await send_ui_message(target, context, "Сначала задай промт.", reply_markup=main_keyboard(st))
        return

    if st["mode"] in SINGLE_PHOTO_MODES:
        if not st["video_source"].get("path"):
            await send_ui_message(target, context, f"Для {st['mode']} сначала пришли фото.", reply_markup=main_keyboard(st))
            return
    elif st["mode"] in DUO_PHOTO_MODES:
        if not all(x.get("path") for x in st["duo_photos"]):
            await send_ui_message(target, context, "Для mopmix_duo пришли два фото (лицо A и лицо B).", reply_markup=main_keyboard(st))
            return
    else:
        await send_ui_message(target, context, f"Неизвестный режим: {st['mode']}", reply_markup=main_keyboard(st))
        return

    repeat = max(1, int(st.get("repeat", 1)))
    first_job_id = JOB_SEQ + 1
    last_job_id = first_job_id + repeat - 1
    roulette = bool(st.get("roulette")) and repeat > 1

    ui = get_ui_state(context)
    ui["last_ui_message_id"] = None
    await cleanup_chat_status(context.bot, target.chat_id)
    status_text = f"🕐 В очереди: {repeat} задач(а). ID: #{first_job_id}–#{last_job_id}."
    if roulette:
        status_text += f"\n🎰 Рулетка включена: вариации каждые {ROULETTE_GROUP_SIZE} видео."
    await update_chat_status(context.bot, target.chat_id, status_text)

    base_idea = st["prompt"]
    current_prompt = st["prompt"]
    variations = 1

    roulette_photo_caption = ""
    if roulette:
        photo = st.get("video_source") if st["mode"] in SINGLE_PHOTO_MODES else None
        if photo and photo.get("path"):
            roulette_photo_caption = await asyncio.to_thread(caption_photo, photo["path"], photo["name"])

    for i in range(repeat):
        if roulette and i % ROULETTE_GROUP_SIZE == 0:
            if i > 0:
                try:
                    current_prompt = await asyncio.to_thread(
                        expand_idea_with_ollama, base_idea, roulette_photo_caption
                    )
                    variations += 1
                except Exception:
                    log.exception("Roulette re-roll failed, reusing previous prompt")

        JOB_SEQ += 1
        job = Job(
            job_id=JOB_SEQ,
            chat_id=target.chat_id,
            mode=st["mode"],
            prompt=current_prompt,
            seconds=int(st["seconds"]),
            max_side=int(st["max_side"]),
            video_fps=int(st.get("video_fps") or QUALITY_PRESETS["medium"]["video_fps"]),
            seed=make_seed(),
            video_source=copy.deepcopy(st["video_source"]),
            duo_photos=copy.deepcopy(st["duo_photos"]),
            video_loras=list(st.get("video_loras") or []),
            quality=(st.get("quality") or "medium").strip().lower(),
            batch_index=i + 1,
            batch_total=repeat,
            dub_voice=bool(st.get("dub_voice")),
            dub_voice_name=st.get("dub_voice_name") or DEFAULT_VOICE_NAME,
        )
        await GEN_QUEUE.put(job)

    if roulette:
        try:
            await context.bot.send_message(
                target.chat_id, f"🎰 Рулетка: подготовлено {variations} вариаций промта по {ROULETTE_GROUP_SIZE} видео."
            )
        except Exception:
            log.exception("Failed to send roulette summary")


async def submit_image_job(app: Application, job: Job) -> None:
    src = job.video_source
    if not src or not src.get("path"):
        raise RuntimeError("Для image сначала пришли фото.")

    width, height = IMAGE_EDIT_QUALITY.get(job.quality, IMAGE_EDIT_QUALITY["medium"])
    target_w, target_h = fit_to_pixel_budget(src["orig_width"], src["orig_height"], width * height)

    uploaded_name = await asyncio.to_thread(upload_image_to_comfy, src["path"], src["name"])
    wf = build_image_edit_workflow(
        image_name=uploaded_name,
        prompt=job.prompt,
        width=target_w,
        height=target_h,
        seed=job.seed,
    )

    prompt_id = await asyncio.to_thread(queue_prompt, wf, str(uuid.uuid4()))

    ACTIVE_PROMPTS[prompt_id] = {
        "job_id": job.job_id,
        "chat_id": job.chat_id,
        "mode": "image",
        "preferred_node": "9",
        "quality": job.quality,
        "prompt": job.prompt,
    }


async def submit_video_job(app: Application, job: Job) -> None:
    src = job.video_source
    if not src or not src.get("path"):
        raise RuntimeError("Для video сначала пришли фото.")

    wf = await asyncio.to_thread(load_workflow, WORKFLOW_VIDEO)
    uploaded_name = await asyncio.to_thread(upload_image_to_comfy, src["path"], src["name"])

    wf = await asyncio.to_thread(
        patch_video_workflow,
        wf,
        prompt=job.prompt,
        image_name=uploaded_name,
        width=src["fit_width"],
        height=src["fit_height"],
        seconds=job.seconds,
        video_fps=job.video_fps,
        seed=job.seed,
        selected_loras=job.video_loras,
    )

    prompt_id = await asyncio.to_thread(queue_prompt, wf, str(uuid.uuid4()))

    ACTIVE_PROMPTS[prompt_id] = {
        "job_id": job.job_id,
        "chat_id": job.chat_id,
        "mode": "video",
        "preferred_node": "314",
        "seconds": job.seconds,
        "width": src["fit_width"],
        "height": src["fit_height"],
        "video_fps": job.video_fps,
        "quality": job.quality,
        "prompt": job.prompt,
        "video_loras": job.video_loras,
    }


async def submit_ltx_sulphur_job(app: Application, job: Job) -> None:
    src = job.video_source
    if not src or not src.get("path"):
        raise RuntimeError("Для ltx_sulphur сначала пришли фото.")

    preset_w, preset_h = LTX_SULPHUR_QUALITY.get(job.quality, LTX_SULPHUR_QUALITY["medium"])
    width, height = fit_to_pixel_budget(src["orig_width"], src["orig_height"], preset_w * preset_h)

    wf = await asyncio.to_thread(load_workflow, WORKFLOW_LTX_SULPHUR)
    uploaded_name = await asyncio.to_thread(upload_image_to_comfy, src["path"], src["name"])

    wf = await asyncio.to_thread(
        patch_ltx_sulphur_workflow,
        wf,
        prompt=job.prompt,
        image_name=uploaded_name,
        width=width,
        height=height,
        seconds=job.seconds,
        seed=job.seed,
    )

    prompt_id = await asyncio.to_thread(queue_prompt, wf, str(uuid.uuid4()))

    ACTIVE_PROMPTS[prompt_id] = {
        "job_id": job.job_id,
        "chat_id": job.chat_id,
        "mode": "ltx_sulphur",
        "preferred_node": "61",
        "seconds": job.seconds,
        "width": width,
        "height": height,
        "quality": job.quality,
        "prompt": job.prompt,
        "dub_voice": job.dub_voice,
        "dub_voice_name": job.dub_voice_name,
    }


async def submit_ltx_eros_job(app: Application, job: Job) -> None:
    src = job.video_source
    if not src or not src.get("path"):
        raise RuntimeError("Для ltx_eros сначала пришли фото.")

    preset_w, preset_h = LTX_EROS_QUALITY.get(job.quality, LTX_EROS_QUALITY["medium"])
    width, height = fit_to_pixel_budget(src["orig_width"], src["orig_height"], preset_w * preset_h)

    wf = await asyncio.to_thread(load_workflow, WORKFLOW_LTX_EROS)
    uploaded_name = await asyncio.to_thread(upload_image_to_comfy, src["path"], src["name"])

    wf = await asyncio.to_thread(
        patch_ltx_eros_workflow,
        wf,
        prompt=job.prompt,
        image_name=uploaded_name,
        width=width,
        height=height,
        seconds=job.seconds,
        seed=job.seed,
    )

    prompt_id = await asyncio.to_thread(queue_prompt, wf, str(uuid.uuid4()))

    ACTIVE_PROMPTS[prompt_id] = {
        "job_id": job.job_id,
        "chat_id": job.chat_id,
        "mode": "ltx_eros",
        "preferred_node": "1135:597",
        "seconds": job.seconds,
        "width": width,
        "height": height,
        "quality": job.quality,
        "prompt": job.prompt,
        "dub_voice": job.dub_voice,
        "dub_voice_name": job.dub_voice_name,
    }


async def submit_mopmix_job(app: Application, job: Job) -> None:
    src = job.video_source
    if not src or not src.get("path"):
        raise RuntimeError("Для mopmix сначала пришли фото.")

    resolution = MOPMIX_RESOLUTIONS.get(job.quality, MOPMIX_RESOLUTIONS["medium"])

    wf = await asyncio.to_thread(load_workflow, WORKFLOW_MOPMIX)
    uploaded_name = await asyncio.to_thread(upload_image_to_comfy, src["path"], src["name"])
    translated_prompt = await asyncio.to_thread(translate_to_english, job.prompt)

    wf = await asyncio.to_thread(
        patch_mopmix_workflow,
        wf,
        prompt=translated_prompt,
        resolution=resolution,
        image_name=uploaded_name,
        seed=job.seed,
    )

    prompt_id = await asyncio.to_thread(queue_prompt, wf, str(uuid.uuid4()))

    ACTIVE_PROMPTS[prompt_id] = {
        "job_id": job.job_id,
        "chat_id": job.chat_id,
        "mode": "mopmix",
        "preferred_node": "128",
        "quality": job.quality,
        "prompt": job.prompt,
    }


async def submit_mopmix_duo_job(app: Application, job: Job) -> None:
    photo_a, photo_b = job.duo_photos
    if not photo_a.get("path") or not photo_b.get("path"):
        raise RuntimeError("Для mopmix_duo нужны два фото.")

    resolution = MOPMIX_RESOLUTIONS.get(job.quality, MOPMIX_RESOLUTIONS["medium"])

    wf = await asyncio.to_thread(load_workflow, WORKFLOW_MOPMIX_DUO)
    uploaded_a = await asyncio.to_thread(upload_image_to_comfy, photo_a["path"], photo_a["name"])
    uploaded_b = await asyncio.to_thread(upload_image_to_comfy, photo_b["path"], photo_b["name"])
    translated_prompt = await asyncio.to_thread(translate_to_english, job.prompt)

    caption_a, caption_b = await asyncio.gather(
        asyncio.to_thread(caption_photo, photo_a["path"], photo_a["name"]),
        asyncio.to_thread(caption_photo, photo_b["path"], photo_b["name"]),
    )
    appearance_notes = []
    if caption_a:
        appearance_notes.append(f"Person A appearance (match exactly): {caption_a}")
    if caption_b:
        appearance_notes.append(f"Person B appearance (match exactly): {caption_b}")
    if appearance_notes:
        translated_prompt = translated_prompt + ". " + ". ".join(appearance_notes)

    composite = await asyncio.to_thread(build_duo_composite, photo_a["path"], photo_b["path"])
    composite_path = TMP_DIR / f"tg_duo_composite_{uuid.uuid4().hex}.jpg"
    await asyncio.to_thread(composite.save, composite_path, "JPEG", quality=95)
    try:
        uploaded_composite = await asyncio.to_thread(upload_image_to_comfy, str(composite_path), composite_path.name)
    finally:
        composite_path.unlink(missing_ok=True)

    wf = await asyncio.to_thread(
        patch_mopmix_duo_workflow,
        wf,
        prompt=translated_prompt,
        resolution=resolution,
        image_name_a=uploaded_a,
        image_name_b=uploaded_b,
        composite_image_name=uploaded_composite,
        seed=job.seed,
    )

    prompt_id = await asyncio.to_thread(queue_prompt, wf, str(uuid.uuid4()))

    ACTIVE_PROMPTS[prompt_id] = {
        "job_id": job.job_id,
        "chat_id": job.chat_id,
        "mode": "mopmix_duo",
        "preferred_node": "128",
        "quality": job.quality,
        "prompt": job.prompt,
    }


# ============================================================
# ERROR HANDLER / STARTUP
# ============================================================
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    log.exception("Unhandled exception", exc_info=context.error)


async def post_init(app: Application) -> None:
    asyncio.create_task(submit_worker_loop(app))
    asyncio.create_task(monitor_loop(app))
    log.info("Bot initialized")


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is empty")

    if not Path(WORKFLOW_VIDEO).exists():
        raise RuntimeError(f"Workflow file not found: {WORKFLOW_VIDEO}")
    if not Path(WORKFLOW_LTX_SULPHUR).exists():
        raise RuntimeError(f"Workflow file not found: {WORKFLOW_LTX_SULPHUR}")
    if not Path(WORKFLOW_MOPMIX).exists():
        raise RuntimeError(f"Workflow file not found: {WORKFLOW_MOPMIX}")
    if not Path(WORKFLOW_LTX_EROS).exists():
        raise RuntimeError(f"Workflow file not found: {WORKFLOW_LTX_EROS}")
    if not Path(WORKFLOW_MOPMIX_DUO).exists():
        raise RuntimeError(f"Workflow file not found: {WORKFLOW_MOPMIX_DUO}")
    app = Application.builder().token(BOT_TOKEN).post_init(post_init).build()

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("reset", reset_cmd))
    app.add_handler(CommandHandler("status", status_cmd))

    app.add_handler(CommandHandler("video", video_cmd))
    app.add_handler(CommandHandler("ltxsulphur", ltx_sulphur_cmd))
    app.add_handler(CommandHandler("ltxeros", ltx_eros_cmd))
    app.add_handler(CommandHandler("image", image_cmd))
    app.add_handler(CommandHandler("mopmix", mopmix_cmd))
    app.add_handler(CommandHandler("mopmixduo", mopmix_duo_cmd))
    app.add_handler(CommandHandler("photos", photos_cmd))
    app.add_handler(CommandHandler("loras", loras_cmd))
    app.add_handler(CommandHandler("prompt", prompt_cmd))
    app.add_handler(CommandHandler("seconds", seconds_cmd))
    app.add_handler(CommandHandler("quality", quality_cmd))
    app.add_handler(CommandHandler("repeat", repeat_cmd))
    app.add_handler(CommandHandler("stop", stop_cmd))
    app.add_handler(CommandHandler("clearqueue", clearqueue_cmd))
    app.add_handler(CommandHandler("go", go_cmd))

    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(filters.PHOTO, photo_handler))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, voice_message_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))
    app.add_error_handler(error_handler)

    log.info("Starting polling")
    app.run_polling()


if __name__ == "__main__":
    main()
