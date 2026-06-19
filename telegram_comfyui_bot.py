import asyncio
import io
import json
import logging
import os
import re
import copy
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import requests
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
WORKFLOW_IMAGE = os.getenv("COMFY_WORKFLOW_IMAGE", "./workflow_image.json")
WORKFLOW_MULTITALK = os.getenv("COMFY_WORKFLOW_MULTITALK", "./workflow_multitalk.json")

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
TALK_MAX_SECONDS = int(os.getenv("TALK_MAX_SECONDS", "8"))
DEFAULT_QUALITY = os.getenv("DEFAULT_QUALITY", "medium").strip().lower()
QUALITY_PRESETS = {
    "low": {"max_side": 480, "video_fps": 16},
    "medium": {"max_side": 640, "video_fps": 16},
    "high": {"max_side": 768, "video_fps": 16},
}
ROUND_TO = int(os.getenv("ROUND_TO", "64"))

REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "300"))
POLL_SECONDS = float(os.getenv("POLL_SECONDS", "2"))

MAX_REFS_FOR_IMAGE = 3
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
VIDEO_TTS = os.getenv("VIDEO_TTS", "0").strip().lower() not in {"0", "false", "no", "off"}
EDGE_TTS_BIN = os.getenv("EDGE_TTS_BIN", "/home/iaadmin/miniconda3/bin/edge-tts")
VIDEO_TTS_VOICE_RU = os.getenv("VIDEO_TTS_VOICE_RU", "ru-RU-SvetlanaNeural")
VIDEO_TTS_VOICE_EN = os.getenv("VIDEO_TTS_VOICE_EN", "en-US-AvaNeural")
VIDEO_TTS_RATE = os.getenv("VIDEO_TTS_RATE", "+0%")
VIDEO_TTS_VOLUME = os.getenv("VIDEO_TTS_VOLUME", "+20%")
VIDEO_TTS_DELAY_MS = int(os.getenv("VIDEO_TTS_DELAY_MS", "500"))
VIDEO_TTS_BG_VOLUME = float(os.getenv("VIDEO_TTS_BG_VOLUME", "0.65"))
VIDEO_TTS_SPEECH_VOLUME = float(os.getenv("VIDEO_TTS_SPEECH_VOLUME", "1.25"))
MULTITALK_FPS = int(os.getenv("MULTITALK_FPS", "25"))
MULTITALK_STEPS = int(os.getenv("MULTITALK_STEPS", "4"))
MULTITALK_CFG = float(os.getenv("MULTITALK_CFG", "1.0"))
MULTITALK_SHIFT = float(os.getenv("MULTITALK_SHIFT", "11.0"))
MULTITALK_AUDIO_SCALE = float(os.getenv("MULTITALK_AUDIO_SCALE", "1.0"))
MULTITALK_AUDIO_CFG_SCALE = float(os.getenv("MULTITALK_AUDIO_CFG_SCALE", "1.0"))
MULTITALK_LORA = os.getenv("MULTITALK_LORA", "nsfw_wan_14b_sex.safetensors")
MULTITALK_LORA_STRENGTH = float(os.getenv("MULTITALK_LORA_STRENGTH", "0.65"))
VIDEO_MAX_LORAS = int(os.getenv("VIDEO_MAX_LORAS", "8"))
VIDEO_LORA_STRENGTH_DEFAULT = float(os.getenv("VIDEO_LORA_STRENGTH_DEFAULT", "0.35"))
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
    image_refs: list[dict]
    video_loras: list[str]


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


def save_bytes(path: Path, blob: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)


def run_cmd(cmd: list[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
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


def mode_max_seconds(mode: str) -> int:
    if mode == "talk":
        return TALK_MAX_SECONDS
    return MAX_SECONDS


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
        "image_refs": [blank_media(), blank_media(), blank_media()],
        "video_loras": [],
    }


def get_state(context: ContextTypes.DEFAULT_TYPE) -> dict[str, Any]:
    if "job_state" not in context.user_data:
        context.user_data["job_state"] = initial_state()
    st = context.user_data["job_state"]
    if st.get("mode") == "director":
        st["mode"] = "video"
    if "video_loras" not in st:
        st["video_loras"] = []
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
    for media in st.get("image_refs") or []:
        refresh_media_size(media, max_side)


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
def main_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("🎬 Video", callback_data="mode:video"),
                InlineKeyboardButton("🗣 Talk", callback_data="mode:talk"),
                InlineKeyboardButton("🖼 Image", callback_data="mode:image"),
            ],
            [
                InlineKeyboardButton("L", callback_data="quality:low"),
                InlineKeyboardButton("M", callback_data="quality:medium"),
                InlineKeyboardButton("H", callback_data="quality:high"),
            ],
            [
                InlineKeyboardButton("➖2s", callback_data="sec:-2"),
                InlineKeyboardButton("➕2s", callback_data="sec:+2"),
            ],
            [
                InlineKeyboardButton("1x", callback_data="repeat:1"),
                InlineKeyboardButton("10x", callback_data="repeat:10"),
                InlineKeyboardButton("30x", callback_data="repeat:30"),
            ],
            [
                InlineKeyboardButton("📋 Status", callback_data="show:status"),
                InlineKeyboardButton("📷 Recent photos", callback_data="media:list"),
                InlineKeyboardButton("🎚 LoRA", callback_data="lora:list"),
            ],
            [
                InlineKeyboardButton("🧹 Reset", callback_data="do:reset"),
            ],
            [
                InlineKeyboardButton("⛔ Stop", callback_data="queue:stop"),
                InlineKeyboardButton("🚮 Clear queue", callback_data="queue:clear"),
            ],
            [
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
            lines.append(f"• {opt['label']} ({opt.get('strength', VIDEO_LORA_STRENGTH_DEFAULT):.2f})")
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
            reply_markup=main_keyboard(),
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
            reply_markup=main_keyboard(),
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
    refs = st["image_refs"]

    return (
        "Бот готов.\n\n"
        "Режимы:\n"
        "• video: 1 фото + промт + секунды + звук MMAudio\n"
        "• talk: 1 фото + реплика из промта + lip-sync через InfiniteTalk\n"
        "• image: до 3 фото + промт\n\n"
        "Команды:\n"
        "/video — обычный photo → video\n"
        "/talk — experimental photo → talking video\n"
        "/image — photo → image\n"
        "/prompt текст — сохранить промт\n"
        f"/seconds 8 — video до {MAX_SECONDS} сек, talk до {TALK_MAX_SECONDS} сек\n"
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
        f"• video LoRA: {selected_lora_labels(st)}\n"
        f"• video audio: {'on' if VIDEO_AUDIO else 'off'}\n"
        f"• video TTS: {'on' if VIDEO_TTS else 'off'}\n"
        f"• talk LoRA: {MULTITALK_LORA} ({MULTITALK_LORA_STRENGTH:.2f})\n"
        f"• video source: {media_line(st['video_source'])}\n"
        f"• image ref #1: {media_line(refs[0])}\n"
        f"• image ref #2: {media_line(refs[1])}\n"
        f"• image ref #3: {media_line(refs[2])}\n"
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


def apply_video_loras(wf: dict[str, Any], selected_loras: list[str]) -> None:
    high_inputs = clear_power_lora_node(wf, "152")
    low_inputs = clear_power_lora_node(wf, "155")
    if not high_inputs or not low_inputs:
        return

    valid = []
    for key in selected_loras:
        opt = video_lora_by_key(key)
        if opt and opt["key"] not in [x["key"] for x in valid]:
            valid.append(opt)

    for index, opt in enumerate(valid[:VIDEO_MAX_LORAS], start=1):
        strength = float(opt.get("strength", VIDEO_LORA_STRENGTH_DEFAULT))
        high_inputs[f"lora_{index}"] = {
            "on": True,
            "lora": opt["high"],
            "strength": strength,
        }
        low_inputs[f"lora_{index}"] = {
            "on": True,
            "lora": opt["low"],
            "strength": strength,
        }


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
) -> dict[str, Any]:
    wf = json.loads(json.dumps(wf))
    wf["93"]["inputs"]["text"] = f"{prompt}\n\n{VIDEO_NO_TEXT_PROMPT}"
    negative_text = wf["373:360"]["inputs"].get("text", "")
    if VIDEO_NO_TEXT_NEGATIVE not in negative_text:
        wf["373:360"]["inputs"]["text"] = f"{negative_text}, {VIDEO_NO_TEXT_NEGATIVE}"
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


def patch_image_workflow(
    wf: dict[str, Any],
    *,
    prompt: str,
    image_names: list[str],
    seed: int,
) -> dict[str, Any]:
    wf = json.loads(json.dumps(wf))

    wf["435"]["inputs"]["value"] = prompt
    wf["433:3"]["inputs"]["seed"] = int(seed)

    if len(image_names) >= 1:
        wf["78"]["inputs"]["image"] = image_names[0]
    if len(image_names) >= 2:
        wf["436"]["inputs"]["image"] = image_names[1]
    if len(image_names) >= 3:
        wf["437"]["inputs"]["image"] = image_names[2]

    return wf


def patch_multitalk_workflow(
    wf: dict[str, Any],
    *,
    prompt: str,
    image_name: str,
    audio_name: str,
    width: int,
    height: int,
    seconds: int,
    seed: int,
) -> dict[str, Any]:
    wf = json.loads(json.dumps(wf))
    fps = max(1, int(MULTITALK_FPS))
    frame_count = max(1, int(seconds) * fps + 1)

    wf["1"]["inputs"]["image"] = image_name
    wf["2"]["inputs"]["width"] = int(width)
    wf["2"]["inputs"]["height"] = int(height)
    wf["3"]["inputs"]["audio"] = audio_name
    wf["5"]["inputs"]["num_frames"] = frame_count
    wf["5"]["inputs"]["fps"] = float(fps)
    wf["5"]["inputs"]["audio_scale"] = float(MULTITALK_AUDIO_SCALE)
    wf["5"]["inputs"]["audio_cfg_scale"] = float(MULTITALK_AUDIO_CFG_SCALE)
    wf["9"]["inputs"]["width"] = int(width)
    wf["9"]["inputs"]["height"] = int(height)
    wf["9"]["inputs"]["frame_window_size"] = min(81, max(17, ((frame_count - 1) // 4) * 4 + 1))
    wf["10"]["inputs"]["lora"] = MULTITALK_LORA
    wf["10"]["inputs"]["strength"] = float(MULTITALK_LORA_STRENGTH)
    wf["14"]["inputs"]["positive_prompt"] = f"{prompt}\n\n{VIDEO_NO_TEXT_PROMPT}"
    negative_text = wf["14"]["inputs"].get("negative_prompt", "")
    if VIDEO_NO_TEXT_NEGATIVE not in negative_text:
        wf["14"]["inputs"]["negative_prompt"] = f"{negative_text}, {VIDEO_NO_TEXT_NEGATIVE}"
    wf["15"]["inputs"]["steps"] = int(MULTITALK_STEPS)
    wf["15"]["inputs"]["cfg"] = float(MULTITALK_CFG)
    wf["15"]["inputs"]["shift"] = float(MULTITALK_SHIFT)
    wf["15"]["inputs"]["seed"] = int(seed)
    wf["17"]["inputs"]["frame_rate"] = float(fps)
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
    if not item or not item.get("outputs"):
        return None

    outputs = item["outputs"]
    required_outputs = {
        "talk": ("17", ("videos", "gifs")),
    }
    if mode in required_outputs:
        preferred_node, keys = required_outputs[mode]
        return pick_required_result_from_outputs(outputs, preferred_node, keys)

    preferred_nodes = {
        "video": "314",
        "image": "60",
    }
    preferred_node = preferred_nodes.get(mode, "60")

    return pick_first_result_from_outputs(outputs, preferred_node=preferred_node)


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
        and meta.get("mode") == "video"
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
        and meta.get("mode") == "video"
        and filename.lower().endswith((".mp4", ".mov", ".webm"))
    ):
        try:
            processed = await run_video_tts_postprocess(blob, meta, filename)
            if processed:
                blob, filename = processed
                log.info("TTS speech postprocess applied for video job #%s", meta.get("job_id"))
        except Exception:
            log.exception("TTS speech postprocess failed; sending video without TTS speech")

    prompt_text = (meta.get("prompt") or "").strip()

    caption = "Готово."
    if prompt_text:
        caption = f"Готово.\n\n{prompt_text}"[:MAX_CAPTION]

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

    if meta["mode"] in {"video", "talk"}:
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
                history = await asyncio.to_thread(get_history, prompt_id)
                item = history.get(prompt_id)
                if not item or not item.get("outputs"):
                    continue

                result = await pick_result_from_history(prompt_id, meta["mode"])
                if result is None:
                    continue
                await send_result(app, meta, result)
                done_ids.append(prompt_id)

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
        try:
            if job.mode == "video":
                await submit_video_job(app, job)
            elif job.mode == "talk":
                await submit_multitalk_job(app, job)
            elif job.mode == "image":
                await submit_image_job(app, job)
            else:
                raise RuntimeError(f"Unknown job mode: {job.mode}")
        except Exception as e:
            log.exception("Submit failed")
            try:
                await app.bot.send_message(job.chat_id, f"Ошибка отправки задачи #{job.job_id}: {e}")
            except Exception:
                pass
        finally:
            GEN_QUEUE.task_done()


# ============================================================
# COMMANDS
# ============================================================
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_if_needed(update):
        return
    await send_ui_message(update.message, context, help_text(get_state(context)), reply_markup=main_keyboard())


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_if_needed(update):
        return
    await send_ui_message(update.message, context, help_text(get_state(context)), reply_markup=main_keyboard())


async def reset_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_if_needed(update):
        return
    st = reset_state(context)
    await send_ui_message(update.message, context, "Состояние очищено.\n\n" + help_text(st), reply_markup=main_keyboard())


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
            reply_markup=main_keyboard(),
        )
    except Exception as e:
        await send_ui_message(update.message, context, f"Ошибка остановки: {e}", reply_markup=main_keyboard())


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
            reply_markup=main_keyboard(),
        )
    except Exception as e:
        await send_ui_message(update.message, context, f"Ошибка очистки очереди: {e}", reply_markup=main_keyboard())


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

    await send_ui_message(update.message, context, help_text(st) + comfy_info, reply_markup=main_keyboard())


async def video_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_if_needed(update):
        return
    st = get_state(context)
    st["mode"] = "video"
    st["seconds"] = clamp_seconds(st["seconds"], st["mode"])
    await send_ui_message(update.message, context, "Режим: photo → video", reply_markup=main_keyboard())


async def talk_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_if_needed(update):
        return
    st = get_state(context)
    st["mode"] = "talk"
    st["seconds"] = clamp_seconds(st["seconds"], st["mode"])
    await send_ui_message(update.message, context, "Режим: photo → talking video", reply_markup=main_keyboard())


async def image_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_if_needed(update):
        return
    st = get_state(context)
    st["mode"] = "image"
    st["seconds"] = clamp_seconds(st["seconds"], st["mode"])
    await send_ui_message(update.message, context, "Режим: photo → image", reply_markup=main_keyboard())


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
            reply_markup=main_keyboard(),
        )
        return

    await send_media_preview_message(update.message, context, index=0)


async def prompt_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_if_needed(update):
        return

    text = " ".join(context.args).strip()
    if not text:
        await send_ui_message(update.message, context, "Используй: /prompt camera slowly zooms in", reply_markup=main_keyboard())
        return

    st = get_state(context)
    st["prompt"] = text
    await send_ui_message(update.message, context, "Промт сохранён.", reply_markup=main_keyboard())


async def seconds_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_if_needed(update):
        return

    st = get_state(context)
    if not context.args:
        await send_ui_message(
            update.message,
            context,
            f"Сейчас: {st['seconds']} сек. Максимум для режима {st['mode']}: {mode_max_seconds(st['mode'])} сек.",
            reply_markup=main_keyboard(),
        )
        return

    try:
        sec = int(context.args[0])
    except Exception:
        await send_ui_message(update.message, context, "Пример: /seconds 8", reply_markup=main_keyboard())
        return

    st["seconds"] = clamp_seconds(sec, st["mode"])
    await send_ui_message(
        update.message,
        context,
        f"Длина видео: {st['seconds']} сек. Максимум для режима {st['mode']}: {mode_max_seconds(st['mode'])} сек.",
        reply_markup=main_keyboard(),
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
            reply_markup=main_keyboard(),
        )
        return

    name = (context.args[0] or "").strip().lower()
    if not apply_quality(st, name):
        await send_ui_message(update.message, context, "Используй: /quality low | medium | high", reply_markup=main_keyboard())
        return

    await send_ui_message(update.message, context, f"Качество: {quality_status(st)}.", reply_markup=main_keyboard())


async def repeat_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_if_needed(update):
        return

    st = get_state(context)
    if not context.args:
        await send_ui_message(update.message, context, f"Сейчас repeat: {st['repeat']}. Пример: /repeat 4", reply_markup=main_keyboard())
        return

    try:
        n = int(context.args[0])
    except Exception:
        await send_ui_message(update.message, context, "Пример: /repeat 4", reply_markup=main_keyboard())
        return

    n = max(1, min(200, n))
    st["repeat"] = n

    await send_ui_message(update.message, context, f"Количество запусков: {n}", reply_markup=main_keyboard())


async def go_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_if_needed(update):
        return
    await enqueue_generation(update, context)


# ============================================================
# PHOTO / TEXT INPUT
# ============================================================
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

    if st["mode"] in {"video", "talk"}:
        st["video_source"] = media
        await send_ui_message(
            update.message,
            context,
            f"Фото для {st['mode']} сохранено: {width}×{height} → {fit_w}×{fit_h}",
            reply_markup=main_keyboard(),
        )
        return

    placed = False
    for i in range(MAX_REFS_FOR_IMAGE):
        if not st["image_refs"][i].get("path"):
            st["image_refs"][i] = media
            await send_ui_message(
                update.message,
                context,
                f"Фото #{i + 1} для image сохранено: {width}×{height} → {fit_w}×{fit_h}",
                reply_markup=main_keyboard(),
            )
            placed = True
            break

    if not placed:
        st["image_refs"][MAX_REFS_FOR_IMAGE - 1] = media
        await send_ui_message(
            update.message,
            context,
            "Слоты image уже заполнены. Я заменил 3-е фото новым.",
            reply_markup=main_keyboard(),
        )


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_if_needed(update):
        return

    text = (update.message.text or "").strip()
    if not text or text.startswith("/"):
        return

    st = get_state(context)
    st["prompt"] = text

    await send_ui_message(update.message, context, "Текст принят как промт.", reply_markup=main_keyboard())


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
        await replace_ui_message_from_callback(query, context, f"Режим: {st['mode']}", reply_markup=main_keyboard())
        return

    if data.startswith("quality:"):
        name = data.split(":", 1)[1]
        if apply_quality(st, name):
            await replace_ui_message_from_callback(query, context, f"Качество: {quality_status(st)}.", reply_markup=main_keyboard())
        return

    if data.startswith("sec:"):
        delta = data.split(":", 1)[1]
        step = -2 if delta == "-2" else 2
        st["seconds"] = clamp_seconds(st["seconds"] + step, st["mode"])
        await replace_ui_message_from_callback(
            query,
            context,
            f"Длина видео: {st['seconds']} сек. Максимум для режима {st['mode']}: {mode_max_seconds(st['mode'])} сек.",
            reply_markup=main_keyboard(),
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
            reply_markup=main_keyboard(),
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
                reply_markup=main_keyboard(),
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
            await replace_ui_message_from_callback(query, context, "Это фото уже недоступно.", reply_markup=main_keyboard())
            return

        media = copy.deepcopy(library[idx])
        if not Path(media.get("path", "")).exists():
            await replace_ui_message_from_callback(query, context, "Файл этого фото уже удалён с диска. Пришли его заново.", reply_markup=main_keyboard())
            return

        st["video_source"] = media
        await replace_ui_message_from_callback(query, context, f"Базовое фото выбрано: {media_line(media)}", reply_markup=main_keyboard())
        return

    if data == "queue:stop":
        try:
            await asyncio.to_thread(interrupt_current)
            active_cleared = clear_active_prompts()
            await replace_ui_message_from_callback(query, context, f"Текущая генерация остановлена.\nСброшено active prompt: {active_cleared}", reply_markup=main_keyboard())
        except Exception as e:
            await replace_ui_message_from_callback(query, context, f"Ошибка остановки: {e}", reply_markup=main_keyboard())
        return

    if data == "queue:clear":
        try:
            comfy_resp = await asyncio.to_thread(clear_comfy_queue)
            local_cleared = clear_local_queue()
            await replace_ui_message_from_callback(
                query,
                context,
                f"Очередь очищена.\n• локальных задач удалено: {local_cleared}\n• ответ ComfyUI: {comfy_resp}",
                reply_markup=main_keyboard(),
            )
        except Exception as e:
            await replace_ui_message_from_callback(query, context, f"Ошибка очистки очереди: {e}", reply_markup=main_keyboard())
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

        await replace_ui_message_from_callback(query, context, help_text(st) + comfy_info, reply_markup=main_keyboard())
        return

    if data == "do:reset":
        st = reset_state(context)
        await replace_ui_message_from_callback(query, context, "Состояние очищено.\n\n" + help_text(st), reply_markup=main_keyboard())
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
        await send_ui_message(target, context, "Сначала задай промт.", reply_markup=main_keyboard())
        return

    if st["mode"] in {"video", "talk"}:
        if not st["video_source"].get("path"):
            await send_ui_message(target, context, f"Для {st['mode']} сначала пришли фото.", reply_markup=main_keyboard())
            return
        if st["mode"] == "talk" and not extract_speech_text(st["prompt"]):
            await send_ui_message(target, context, "Для talk нужна реплика в промте: например, говорит: привет", reply_markup=main_keyboard())
            return
    elif st["mode"] == "image":
        if not any(x.get("path") for x in st["image_refs"]):
            await send_ui_message(target, context, "Для image пришли хотя бы одно фото.", reply_markup=main_keyboard())
            return
    else:
        await send_ui_message(target, context, f"Неизвестный режим: {st['mode']}", reply_markup=main_keyboard())
        return

    repeat = max(1, int(st.get("repeat", 1)))
    first_job_id = JOB_SEQ + 1

    for _ in range(repeat):
        JOB_SEQ += 1
        job = Job(
            job_id=JOB_SEQ,
            chat_id=target.chat_id,
            mode=st["mode"],
            prompt=st["prompt"],
            seconds=int(st["seconds"]),
            max_side=int(st["max_side"]),
            video_fps=int(st.get("video_fps") or QUALITY_PRESETS["medium"]["video_fps"]),
            seed=make_seed(),
            video_source=copy.deepcopy(st["video_source"]),
            image_refs=copy.deepcopy(st["image_refs"]),
            video_loras=list(st.get("video_loras") or []),
        )
        await GEN_QUEUE.put(job)

    await send_ui_message(
        target,
        context,
        f"Добавлено задач: {repeat}. ID: #{first_job_id}–#{JOB_SEQ}. Сейчас в очереди: {GEN_QUEUE.qsize()}",
        reply_markup=main_keyboard(),
    )


async def submit_image_job(app: Application, job: Job) -> None:
    refs = [x for x in job.image_refs if x.get("path")]
    if not refs:
        raise RuntimeError("Для image нужно хотя бы одно фото.")

    wf = await asyncio.to_thread(load_workflow, WORKFLOW_IMAGE)

    uploaded_names = []
    for media in refs[:MAX_REFS_FOR_IMAGE]:
        name = await asyncio.to_thread(upload_image_to_comfy, media["path"], media["name"])
        uploaded_names.append(name)

    while len(uploaded_names) < 3:
        uploaded_names.append(uploaded_names[-1])

    wf = await asyncio.to_thread(
        patch_image_workflow,
        wf,
        prompt=job.prompt,
        image_names=uploaded_names,
        seed=job.seed,
    )

    prompt_id = await asyncio.to_thread(queue_prompt, wf, str(uuid.uuid4()))

    ACTIVE_PROMPTS[prompt_id] = {
        "job_id": job.job_id,
        "chat_id": job.chat_id,
        "mode": "image",
        "preferred_node": "60",
        "refs_count": len(refs),
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
        "prompt": job.prompt,
        "video_loras": job.video_loras,
    }


async def submit_multitalk_job(app: Application, job: Job) -> None:
    src = job.video_source
    if not src or not src.get("path"):
        raise RuntimeError("Для talk сначала пришли фото.")

    speech_text = extract_speech_text(job.prompt)
    if not speech_text:
        raise RuntimeError("Для talk нужна реплика в промте: например, говорит: привет")

    wf = await asyncio.to_thread(load_workflow, WORKFLOW_MULTITALK)
    uploaded_name = await asyncio.to_thread(upload_image_to_comfy, src["path"], src["name"])

    audio_name = f"tg_multitalk_{uuid.uuid4().hex}.mp3"
    audio_path = COMFY_INPUT_DIR / audio_name
    await asyncio.to_thread(synthesize_speech, speech_text, audio_path)

    wf = await asyncio.to_thread(
        patch_multitalk_workflow,
        wf,
        prompt=job.prompt,
        image_name=uploaded_name,
        audio_name=audio_name,
        width=src["fit_width"],
        height=src["fit_height"],
        seconds=clamp_seconds(job.seconds, "talk"),
        seed=job.seed,
    )

    prompt_id = await asyncio.to_thread(queue_prompt, wf, str(uuid.uuid4()))

    ACTIVE_PROMPTS[prompt_id] = {
        "job_id": job.job_id,
        "chat_id": job.chat_id,
        "mode": "talk",
        "preferred_node": "17",
        "seconds": clamp_seconds(job.seconds, "talk"),
        "width": src["fit_width"],
        "height": src["fit_height"],
        "video_fps": MULTITALK_FPS,
        "prompt": job.prompt,
        "speech_text": speech_text,
        "input_audio_name": audio_name,
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
    if not Path(WORKFLOW_IMAGE).exists():
        raise RuntimeError(f"Workflow file not found: {WORKFLOW_IMAGE}")
    if not Path(WORKFLOW_MULTITALK).exists():
        raise RuntimeError(f"Workflow file not found: {WORKFLOW_MULTITALK}")
    app = Application.builder().token(BOT_TOKEN).post_init(post_init).build()

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("reset", reset_cmd))
    app.add_handler(CommandHandler("status", status_cmd))

    app.add_handler(CommandHandler("video", video_cmd))
    app.add_handler(CommandHandler("talk", talk_cmd))
    app.add_handler(CommandHandler("image", image_cmd))
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
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))
    app.add_error_handler(error_handler)

    log.info("Starting polling")
    app.run_polling()


if __name__ == "__main__":
    main()
