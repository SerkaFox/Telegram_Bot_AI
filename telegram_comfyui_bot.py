import asyncio
import io
import json
import logging
import os
import copy
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
from PIL import Image
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

# Новый workflow LTX Director
WORKFLOW_DIRECTOR = os.getenv("COMFY_WORKFLOW_DIRECTOR", "./New.json")

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
MAX_DIRECTOR_SECONDS = int(os.getenv("MAX_DIRECTOR_SECONDS", "60"))

DEFAULT_QUALITY = os.getenv("DEFAULT_QUALITY", "medium").strip().lower()
QUALITY_PRESETS = {
    "low": 640,
    "medium": 768,
    "high": 1024,
}
ROUND_TO = int(os.getenv("ROUND_TO", "64"))

REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "300"))
POLL_SECONDS = float(os.getenv("POLL_SECONDS", "2"))

MAX_REFS_FOR_IMAGE = 3
DIRECTOR_FPS = int(os.getenv("DIRECTOR_FPS", "24"))
DIRECTOR_ANCHOR_SECONDS = int(os.getenv("DIRECTOR_ANCHOR_SECONDS", "8"))
MEDIA_LIBRARY_LIMIT = int(os.getenv("MEDIA_LIBRARY_LIMIT", "10"))
DIRECTOR_FACE_SWAP = os.getenv("DIRECTOR_FACE_SWAP", "0").strip().lower() not in {"0", "false", "no", "off"}
DIRECTOR_FACE_SWAP_MODEL = os.getenv("DIRECTOR_FACE_SWAP_MODEL", "inswapper_128.onnx")
DIRECTOR_FACE_ANALYSIS_MODEL = os.getenv("DIRECTOR_FACE_ANALYSIS_MODEL", "buffalo_l")
DIRECTOR_FACE_SWAP_INDICES = os.getenv("DIRECTOR_FACE_SWAP_INDICES", "0")
DIRECTOR_FACE_SWAP_TIMEOUT = int(os.getenv("DIRECTOR_FACE_SWAP_TIMEOUT", "900"))
DIRECTOR_ROPE_SIMILARITY = float(os.getenv("DIRECTOR_ROPE_SIMILARITY", "65"))
DIRECTOR_ROPE_DETECTION = float(os.getenv("DIRECTOR_ROPE_DETECTION", "0.5"))
DIRECTOR_ROPE_MATCHING = os.getenv("DIRECTOR_ROPE_MATCHING", "0").strip() or "0"
DIRECTOR_MAX_LORAS = int(os.getenv("DIRECTOR_MAX_LORAS", "13"))
DIRECTOR_LORA_STRENGTH_DEFAULT = float(os.getenv("DIRECTOR_LORA_STRENGTH_DEFAULT", "0.35"))
DIRECTOR_LORA_OPTIONS = [
    {"label": "LTX 2.3 Distill 384", "file": "ltx-2.3-22b-distilled-lora-384.safetensors", "strength": 0.25},
    {"label": "LTX 2.3 Dynamic", "file": "ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors", "strength": 0.25},
    {"label": "Dreamlay LTX V2", "file": "DR34ML4Y_LTXXX_V2.safetensors", "strength": 0.30},
    {"label": "Sex Thrust", "file": "LTX2-i2v-SexThrust.safetensors", "strength": 0.25},
    {"label": "Orgasm", "file": "LTX-2.3 - Orgasm.safetensors", "strength": 0.30},
    {"label": "Passionate Kissing", "file": "LTX-2 - Passionate Kissing.safetensors", "strength": 0.30},
    {"label": "Motion 7K", "file": "LTX2_SS_Motion_7K.safetensors", "strength": 0.25},
    {"label": "Best Breasts", "file": "LTX2_BestBreasts_lora_V2_step_06000.safetensors", "strength": 0.25},
    {"label": "Animation", "file": "cr3ampi3_animation_i2v_ltx2_v1.0.safetensors", "strength": 0.25},
    {"label": "Doggy Mission", "file": "doggy_mission_3d_ltx2_v1.0.safetensors", "strength": 0.25},
    {"label": "NSFW Merge", "file": "ltx2-phr00tmerge-nsfw-v62.safetensors", "strength": 0.25},
    {"label": "Riding Backshot", "file": "nsfw_riding_backshot_frontshot_ltx23_v1.0.safetensors", "strength": 0.25},
    {"label": "Jiggle", "file": "LTX-2 - Jiggle Tits.safetensors", "strength": 0.20},
]

DIRECTOR_FACELOCK_PROMPT = (
    "Continuity lock: keep exactly the same identity as the reference image, same face, facial proportions, eye shape, "
    "nose, mouth, jawline, hair, body shape, skin tone, outfit state, lighting, camera angle, and background. "
    "The face must stay stable between frames with no morphing or identity drift. Preserve natural anatomy, stable body "
    "physics, coherent hands, five fingers per hand, and consistent limb count."
)
DIRECTOR_NEGATIVE_PROMPT = (
    "face morphing, face shifting, face changing, identity drift, different face, inconsistent identity, deformed face, "
    "asymmetric facial features, blurry face, flickering face, unnatural blinking, frozen face, plastic skin, over-smoothed "
    "skin, body morphing, deformed body, broken anatomy, extra limbs, missing limbs, extra fingers, fused fingers, bad hands, "
    "warped hands, distorted torso, broken joints, unstable body proportions"
)

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
    seed: int
    video_source: dict | None
    image_refs: list[dict]
    director_loras: list[str]


# ============================================================
# HELPERS
# ============================================================
def quality_label(max_side: int) -> str:
    for name, size in QUALITY_PRESETS.items():
        if size == max_side:
            return name
    return f"custom-{max_side}"


def make_seed() -> int:
    return int.from_bytes(os.urandom(8), "big") & ((1 << 53) - 1)


def round_to_multiple(v: int, m: int) -> int:
    return max(m, int(round(v / m) * m))


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
    if mode == "director":
        return MAX_DIRECTOR_SECONDS
    return MAX_SECONDS


def clamp_seconds(value: int, mode: str) -> int:
    return max(MIN_SECONDS, min(mode_max_seconds(mode), int(value)))


def initial_state() -> dict[str, Any]:
    return {
        "mode": "video",
        "prompt": "",
        "repeat": 1,
        "seconds": DEFAULT_SECONDS,
        "max_side": QUALITY_PRESETS.get(DEFAULT_QUALITY, 768),
        "video_source": blank_media(),
        "image_refs": [blank_media(), blank_media(), blank_media()],
        "director_loras": [],
    }


def get_state(context: ContextTypes.DEFAULT_TYPE) -> dict[str, Any]:
    if "job_state" not in context.user_data:
        context.user_data["job_state"] = initial_state()
    return context.user_data["job_state"]


def reset_state(context: ContextTypes.DEFAULT_TYPE) -> dict[str, Any]:
    context.user_data["job_state"] = initial_state()
    return context.user_data["job_state"]


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
                InlineKeyboardButton("🖼 Image", callback_data="mode:image"),
            ],
            [
                InlineKeyboardButton("🎞 LTX Director", callback_data="mode:director"),
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


def director_lora_by_file(filename: str) -> dict[str, Any] | None:
    for opt in DIRECTOR_LORA_OPTIONS:
        if opt["file"] == filename:
            return opt
    return None


def selected_lora_labels(st: dict[str, Any], limit: int = 4) -> str:
    selected = st.get("director_loras") or []
    if not selected:
        return "none"
    labels = [(director_lora_by_file(name) or {"label": name})["label"] for name in selected]
    if len(labels) > limit:
        return ", ".join(labels[:limit]) + f" +{len(labels) - limit}"
    return ", ".join(labels)


def lora_keyboard(st: dict[str, Any]) -> InlineKeyboardMarkup:
    selected = set(st.get("director_loras") or [])
    rows = []
    for i, opt in enumerate(DIRECTOR_LORA_OPTIONS):
        mark = "✓" if opt["file"] in selected else "○"
        rows.append([InlineKeyboardButton(f"{mark} {opt['label']}", callback_data=f"lora:toggle:{i}")])
    rows.append([InlineKeyboardButton("Clear", callback_data="lora:clear"), InlineKeyboardButton("↩️ Back", callback_data="show:status")])
    return InlineKeyboardMarkup(rows)


def lora_text(st: dict[str, Any]) -> str:
    selected = st.get("director_loras") or []
    lines = [
        "LoRA для LTX Director",
        "",
        f"Выбрано: {len(selected)}/{DIRECTOR_MAX_LORAS}",
    ]
    if selected:
        for name in selected:
            opt = director_lora_by_file(name) or {"label": name, "strength": DIRECTOR_LORA_STRENGTH_DEFAULT}
            lines.append(f"• {opt['label']} ({opt.get('strength', DIRECTOR_LORA_STRENGTH_DEFAULT):.2f})")
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
        "• video: 1 фото + промт + секунды\n"
        "• director: New.json / LTX Director, 1 фото + промт + звук\n"
        "• image: до 3 фото + промт\n\n"
        "Команды:\n"
        "/video — обычный photo → video\n"
        "/director — новый LTX Director / New.json\n"
        "/image — photo → image\n"
        "/prompt текст — сохранить промт\n"
        f"/seconds 8 — video до {MAX_SECONDS} сек, director до {MAX_DIRECTOR_SECONDS} сек\n"
        "/quality low|medium|high\n"
        "/repeat 1\n"
        "/loras — выбрать LoRA для Director\n"
        "/photos — выбрать базовое фото из последних загруженных\n"
        "/go — генерация\n"
        "/reset\n\n"
        "Как тестировать Director:\n"
        "1) /director\n"
        "2) /seconds 8\n"
        "3) пришли фото\n"
        "4) отправь промт обычным текстом или через /prompt\n"
        "5) /go\n\n"
        "Текущее состояние:\n"
        f"• mode: {st['mode']}\n"
        f"• quality: {quality_label(st['max_side'])} ({st['max_side']} px)\n"
        f"• seconds: {st['seconds']}\n"
        f"• repeat: {st.get('repeat', 1)}\n"
        f"• director LoRA: {selected_lora_labels(st)}\n"
        f"• video/director source: {media_line(st['video_source'])}\n"
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
def patch_video_workflow(
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
    wf["93"]["inputs"]["text"] = prompt
    wf["385"]["inputs"]["image"] = image_name
    wf["164"]["inputs"]["value"] = int(width)
    wf["165"]["inputs"]["value"] = int(height)
    wf["243"]["inputs"]["value"] = int(seconds)
    wf["141"]["inputs"]["seed"] = int(seed)
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


def build_director_timeline_one_image(
    *,
    image_name: str,
    prompt: str,
    seconds: int,
) -> tuple[str, str, str]:
    length = float(seconds)
    segment_prompt = f"{prompt}\n\n{DIRECTOR_FACELOCK_PROMPT}\n\nAvoid: {DIRECTOR_NEGATIVE_PROMPT}"
    segments = [
        {
            "id": f"tg_{uuid.uuid4().hex[:12]}",
            "start": 0.0,
            "length": length,
            "prompt": segment_prompt,
            "type": "image",
            "imageFile": image_name,
            "imageB64": f"/api/view?filename={quote(image_name)}&type=input&subfolder=",
        }
    ]
    prompts = [segment_prompt]
    lengths = [length]

    timeline = {
        "segments": segments,
        "audioSegments": [],
    }

    return (
        json.dumps(timeline, ensure_ascii=False),
        " | ".join(prompts),
        ",".join(str(x) for x in lengths),
    )


def apply_director_loras(wf: dict[str, Any], selected_loras: list[str]) -> None:
    node = wf.get("182")
    if not node:
        return

    inputs = node.setdefault("inputs", {})
    slots = [f"lora_{i}" for i in range(1, DIRECTOR_MAX_LORAS + 1)]
    for slot in slots:
        entry = inputs.setdefault(slot, {})
        entry["on"] = False

    valid_files = []
    for filename in selected_loras:
        opt = director_lora_by_file(filename)
        if opt and filename not in valid_files:
            valid_files.append(filename)

    for slot, filename in zip(slots, valid_files[:DIRECTOR_MAX_LORAS]):
        opt = director_lora_by_file(filename) or {}
        entry = inputs.setdefault(slot, {})
        entry["on"] = True
        entry["lora"] = filename
        entry["strength"] = float(opt.get("strength", DIRECTOR_LORA_STRENGTH_DEFAULT))


def patch_director_workflow(
    wf: dict[str, Any],
    *,
    prompt: str,
    image_name: str,
    width: int,
    height: int,
    seconds: int,
    seed: int,
    selected_loras: list[str] | None = None,
) -> dict[str, Any]:
    wf = json.loads(json.dumps(wf))

    frames = max(1, int(seconds) * DIRECTOR_FPS)

    timeline_data, local_prompts, segment_lengths = build_director_timeline_one_image(
        image_name=image_name,
        prompt=prompt,
        seconds=seconds,
    )

    director_prompt = f"{prompt}\n\n{DIRECTOR_FACELOCK_PROMPT}\n\nAvoid: {DIRECTOR_NEGATIVE_PROMPT}"

    # LTXDirector node
    wf["46"]["inputs"]["global_prompt"] = director_prompt
    wf["46"]["inputs"]["duration_seconds"] = int(seconds)
    wf["46"]["inputs"]["duration_frames"] = int(frames)
    wf["46"]["inputs"]["timeline_data"] = timeline_data
    wf["46"]["inputs"]["local_prompts"] = local_prompts
    wf["46"]["inputs"]["segment_lengths"] = segment_lengths
    wf["46"]["inputs"]["guide_strength"] = ",".join(["1.00"] * len(segment_lengths.split(",")))
    wf["46"]["inputs"]["frame_rate"] = int(DIRECTOR_FPS)
    wf["46"]["inputs"]["custom_width"] = int(width)
    wf["46"]["inputs"]["custom_height"] = int(height)

    # Seeds
    if "28" in wf and "noise_seed" in wf["28"]["inputs"]:
        wf["28"]["inputs"]["noise_seed"] = int(seed)
    if "166" in wf and "noise_seed" in wf["166"]["inputs"]:
        wf["166"]["inputs"]["noise_seed"] = int(make_seed())

    apply_director_loras(wf, selected_loras or [])

    # Final output node
    wf["94"]["inputs"]["filename_prefix"] = "tg_director"
    wf["94"]["inputs"]["format"] = "video/h264-mp4"
    wf["94"]["inputs"]["pix_fmt"] = "yuv420p"
    wf["94"]["inputs"]["save_output"] = True
    wf["94"]["inputs"]["trim_to_audio"] = False

    return wf


# ============================================================
# SIZE REFRESH
# ============================================================
def apply_size_to_media(st: dict[str, Any], media: dict[str, Any]) -> None:
    if media.get("orig_width") and media.get("orig_height"):
        fit_w, fit_h = fit_size_keep_aspect(media["orig_width"], media["orig_height"], st["max_side"])
        media["fit_width"] = fit_w
        media["fit_height"] = fit_h


def refresh_all_sizes(st: dict[str, Any]) -> None:
    apply_size_to_media(st, st["video_source"])
    for media in st["image_refs"]:
        apply_size_to_media(st, media)


# ============================================================
# RESULT HELPERS
# ============================================================
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


async def pick_result_from_history(prompt_id: str, mode: str) -> dict[str, Any]:
    history = await asyncio.to_thread(get_history, prompt_id)
    item = history.get(prompt_id)
    if not item or not item.get("outputs"):
        raise RuntimeError(f"No outputs yet for prompt_id={prompt_id}")

    outputs = item["outputs"]
    preferred_nodes = {
        "video": "314",
        "director": "94",
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



def build_director_faceswap_workflow(
    *,
    video_name: str,
    reference_image_name: str,
    fps: int,
) -> dict[str, Any]:
    return {
        "1": {
            "class_type": "VHS_LoadVideo",
            "inputs": {
                "video": video_name,
                "force_rate": int(fps),
                "custom_width": 0,
                "custom_height": 0,
                "frame_load_cap": 0,
                "skip_first_frames": 0,
                "select_every_nth": 1,
                "format": "None",
            },
        },
        "2": {
            "class_type": "LoadImage",
            "inputs": {
                "image": reference_image_name,
            },
        },
        "3": {
            "class_type": "RopeWrapper_LoadModels",
            "inputs": {
                "inswap_type": "Original",
            },
        },
        "4": {
            "class_type": "RopeWrapper_DetectNode",
            "inputs": {
                "models": ["3", 0],
                "input_image": ["1", 0],
                "SimilarityThreshold": float(DIRECTOR_ROPE_SIMILARITY),
                "detection_threshold": float(DIRECTOR_ROPE_DETECTION),
            },
        },
        "5": {
            "class_type": "RopeWrapper_OptionNode",
            "inputs": {
                "RestorerSwitch": False,
                "RestorerTypeTextSel": "CF",
                "RestorerDetTypeTextSel": "Blend",
                "RestorerSlider": 100,
                "OrientSwitch": False,
                "OrientSlider": 180,
                "StrengthSwitch": False,
                "StrengthSlider": 200,
                "BorderTopSlider": 10,
                "BorderSidesSlider": 10,
                "BorderBottomSlider": 10,
                "BorderBlurSlider": 10,
                "DiffSwitch": False,
                "DiffSlider": 4,
                "OccluderSwitch": False,
                "OccluderSlider": 0,
                "FaceParserSwitch": False,
                "FaceParserSlider": 0,
                "MouthParserSlider": 0,
                "CLIPSwitch": False,
                "CLIPTextEntry": " ",
                "CLIPSlider": 50,
                "BlendSlider": 5,
                "ColorSwitch": False,
                "ColorRedSlider": 0,
                "ColorGreenSlider": 0,
                "ColorBlueSlider": 0,
                "ColorGammaSlider": 0.0,
                "FaceAdjSwitch": False,
                "KPSXSlider": 0,
                "KPSYSlider": 0,
                "KPSScaleSlider": 0,
                "SwapperTypeTextSel": "128",
            },
        },
        "6": {
            "class_type": "RopeWrapper_SwapNode",
            "inputs": {
                "models": ["3", 0],
                "vm": ["3", 1],
                "input_image": ["1", 0],
                "source_face": ["2", 0],
                "detectResult": ["4", 1],
                "combineVideo": True,
                "frame_rate": int(fps),
                "filenamePrefix": "tg_director_rope",
                "saveOutput": True,
                "outputFrameIndex": 0,
                "ROPE_Options": ["5", 0],
                "source_target_matching": DIRECTOR_ROPE_MATCHING,
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


async def run_director_faceswap_postprocess(blob: bytes, meta: dict[str, Any], filename: str) -> tuple[bytes, str] | None:
    reference_image_name = meta.get("source_image_name")
    if not reference_image_name:
        return None

    input_name = f"tg_faceswap_{uuid.uuid4().hex}.mp4"
    input_path = COMFY_INPUT_DIR / input_name
    save_bytes(input_path, blob)

    try:
        wf = build_director_faceswap_workflow(
            video_name=input_name,
            reference_image_name=reference_image_name,
            fps=int(meta.get("fps") or DIRECTOR_FPS),
        )
        prompt_id = await asyncio.to_thread(queue_prompt, wf, str(uuid.uuid4()))
        result = await wait_for_result_from_prompt(
            prompt_id,
            preferred_node="6",
            timeout=DIRECTOR_FACE_SWAP_TIMEOUT,
        )
        swapped_blob = await asyncio.to_thread(
            fetch_file,
            result["filename"],
            result.get("subfolder", ""),
            result.get("type", "output"),
        )
        swapped_name = result.get("filename") or filename
        swapped_path = TMP_DIR / f"tg_rope_swapped_{uuid.uuid4().hex}.mp4"
        muxed_path = TMP_DIR / f"tg_rope_muxed_{uuid.uuid4().hex}.mp4"
        try:
            save_bytes(swapped_path, swapped_blob)
            await asyncio.to_thread(mux_video_with_audio, swapped_path, input_path, muxed_path)
            swapped_blob = muxed_path.read_bytes()
            swapped_name = muxed_path.name
        except Exception:
            log.exception("Director Rope audio mux failed; sending swapped video without remux")
        finally:
            try:
                swapped_path.unlink(missing_ok=True)
                muxed_path.unlink(missing_ok=True)
            except Exception:
                pass
        await asyncio.to_thread(
            delete_comfy_result_file,
            result["filename"],
            result.get("subfolder", ""),
        )
        return swapped_blob, swapped_name
    finally:
        try:
            input_path.unlink(missing_ok=True)
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
        DIRECTOR_FACE_SWAP
        and meta.get("mode") == "director"
        and filename.lower().endswith((".mp4", ".mov", ".webm"))
    ):
        try:
            processed = await run_director_faceswap_postprocess(blob, meta, filename)
            if processed:
                blob, filename = processed
                log.info("Director face swap postprocess applied for job #%s", meta.get("job_id"))
        except Exception:
            log.exception("Director face swap postprocess failed; sending original result")

    prompt_text = (meta.get("prompt") or "").strip()

    caption = "Готово."
    if prompt_text:
        caption = f"Готово.\n\n{prompt_text}"[:MAX_CAPTION]

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

    if meta["mode"] in ("video", "director"):
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
        job = await GEN_QUEUE.get()
        try:
            if job.mode == "video":
                await submit_video_job(app, job)
            elif job.mode == "director":
                await submit_director_job(app, job)
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


async def director_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_if_needed(update):
        return
    st = get_state(context)
    st["mode"] = "director"
    st["seconds"] = clamp_seconds(st["seconds"], st["mode"])
    await send_ui_message(
        update.message,
        context,
        "Режим: LTX Director / New.json — 1 фото + промт + звук",
        reply_markup=main_keyboard(),
    )


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
            f"Сейчас качество: {quality_label(st['max_side'])}. Используй /quality low | medium | high",
            reply_markup=main_keyboard(),
        )
        return

    name = (context.args[0] or "").strip().lower()
    if name not in QUALITY_PRESETS:
        await send_ui_message(update.message, context, "Используй: /quality low | medium | high", reply_markup=main_keyboard())
        return

    st["max_side"] = QUALITY_PRESETS[name]
    refresh_all_sizes(st)

    await send_ui_message(update.message, context, f"Качество: {name} ({st['max_side']} px).", reply_markup=main_keyboard())


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

    if st["mode"] in ("video", "director"):
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
        if name in QUALITY_PRESETS:
            st["max_side"] = QUALITY_PRESETS[name]
            refresh_all_sizes(st)
            await replace_ui_message_from_callback(query, context, f"Качество: {name} ({st['max_side']} px).", reply_markup=main_keyboard())
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
        if 0 <= idx < len(DIRECTOR_LORA_OPTIONS):
            filename = DIRECTOR_LORA_OPTIONS[idx]["file"]
            selected = list(st.get("director_loras") or [])
            if filename in selected:
                selected.remove(filename)
            elif len(selected) < DIRECTOR_MAX_LORAS:
                selected.append(filename)
            st["director_loras"] = selected
        await replace_ui_message_from_callback(query, context, lora_text(st), reply_markup=lora_keyboard(st))
        return

    if data == "lora:clear":
        st["director_loras"] = []
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

    if st["mode"] in ("video", "director"):
        if not st["video_source"].get("path"):
            await send_ui_message(target, context, f"Для {st['mode']} сначала пришли фото.", reply_markup=main_keyboard())
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
            seed=make_seed(),
            video_source=copy.deepcopy(st["video_source"]),
            image_refs=copy.deepcopy(st["image_refs"]),
            director_loras=list(st.get("director_loras") or []),
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
        seed=job.seed,
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
        "prompt": job.prompt,
    }


async def submit_director_job(app: Application, job: Job) -> None:
    src = job.video_source
    if not src or not src.get("path"):
        raise RuntimeError("Для LTX Director сначала пришли фото.")

    wf = await asyncio.to_thread(load_workflow, WORKFLOW_DIRECTOR)

    uploaded_name = await asyncio.to_thread(
        upload_image_to_comfy,
        src["path"],
        src["name"],
    )

    wf = await asyncio.to_thread(
        patch_director_workflow,
        wf,
        prompt=job.prompt,
        image_name=uploaded_name,
        width=src["fit_width"],
        height=src["fit_height"],
        seconds=job.seconds,
        seed=job.seed,
        selected_loras=job.director_loras,
    )

    prompt_id = await asyncio.to_thread(queue_prompt, wf, str(uuid.uuid4()))

    ACTIVE_PROMPTS[prompt_id] = {
        "job_id": job.job_id,
        "chat_id": job.chat_id,
        "mode": "director",
        "preferred_node": "94",
        "seconds": job.seconds,
        "fps": DIRECTOR_FPS,
        "width": src["fit_width"],
        "height": src["fit_height"],
        "prompt": job.prompt,
        "source_image_name": uploaded_name,
        "director_loras": job.director_loras,
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
    if not Path(WORKFLOW_DIRECTOR).exists():
        raise RuntimeError(f"Workflow file not found: {WORKFLOW_DIRECTOR}")

    app = Application.builder().token(BOT_TOKEN).post_init(post_init).build()

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("reset", reset_cmd))
    app.add_handler(CommandHandler("status", status_cmd))

    app.add_handler(CommandHandler("video", video_cmd))
    app.add_handler(CommandHandler("director", director_cmd))

    # Чтобы старая привычная команда /sound тоже включала новый Director.
    app.add_handler(CommandHandler("sound", director_cmd))

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
