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
from datetime import datetime
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
    PicklePersistence,
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

# Per-user settings (mode, toggles, prompt, last photo, chosen loras/voice) survive a reboot:
# PTB's PicklePersistence flushes user_data (which holds each user's job_state) to this file on
# a timer + on graceful shutdown, and reloads it on startup, so nothing resets to defaults.
PERSIST_FILE = Path(os.getenv("BOT_PERSIST_FILE", "./bot_state.pkl"))

# OpenVoice's se_extractor dumps a disposable per-clip cache here; it and the story/dub
# scratch below pile up forever, so a background loop prunes anything older than the cutoff.
PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", "./processed"))
CLEANUP_MAX_AGE_H = float(os.getenv("CLEANUP_MAX_AGE_HOURS", "24"))
CLEANUP_INTERVAL_H = float(os.getenv("CLEANUP_INTERVAL_HOURS", "6"))

COMFY_INPUT_DIR = Path(os.getenv("COMFY_INPUT_DIR", "/home/iaadmin/ComfyUI/input"))
COMFY_OUTPUT_DIR = Path(os.getenv("COMFY_OUTPUT_DIR", "/home/iaadmin/ComfyUI/output"))

MAX_CAPTION = 1000

# Admins can always use the bot AND manage who else may. Default = the owner's id;
# override with the ADMIN_USER_IDS env (comma-separated) if needed.
ADMIN_USER_IDS = {
    int(x.strip()) for x in os.getenv("ADMIN_USER_IDS", "517188056").split(",") if x.strip().lstrip("-").isdigit()
}
# Seed list from the old ALLOWED_USER_IDS env (backward compat); the live list lives on disk.
ALLOWED_USER_IDS = {
    int(x.strip()) for x in os.getenv("ALLOWED_USER_IDS", "").split(",") if x.strip().isdigit()
}
ALLOWLIST_FILE = Path(os.getenv("ALLOWLIST_FILE", "./allowlist.json"))

DEFAULT_SECONDS = int(os.getenv("DEFAULT_SECONDS", "8"))
MIN_SECONDS = int(os.getenv("MIN_SECONDS", "2"))
MAX_SECONDS = int(os.getenv("MAX_SECONDS", "12"))
DEFAULT_QUALITY = os.getenv("DEFAULT_QUALITY", "medium").strip().lower()
QUALITY_PRESETS = {
    "low": {"max_side": 512, "video_fps": 16},
    "medium": {"max_side": 704, "video_fps": 16},
    "high": {"max_side": 832, "video_fps": 16},
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
# Furry/anthro mode for MopMix: swaps the bigASP human base for a Pony Realism base, which
# natively understands anthro anatomy (muzzle, full-body fur, tail). bigASP only knows humans,
# so a furry LoRA can't push it there — the base model has to change. Pony does explicit NSFW.
PONY_FURRY_CHECKPOINT = os.getenv("PONY_FURRY_CHECKPOINT", "ponyRealism_V23ULTRA.safetensors")
# Alternate furry base (selectable via the 🧬 base toggle when Furry is ON). Yiffymix is a
# Pony-based, furry-focused SDXL merge — same score/anthro tags apply — with a broader species
# range and a more illustrative look than the photoreal ponyRealism. Empty file until downloaded.
YIFFY_FURRY_CHECKPOINT = os.getenv("YIFFY_FURRY_CHECKPOINT", "autismmixSDXL_autismmixConfetti.safetensors")
# Pixel buckets per quality for the Pony txt2img graph (SDXL ~1MP sweet spot, portrait).
PONY_FURRY_RESOLUTIONS = {
    "low": (768, 1152),
    "medium": (832, 1216),
    "high": (1024, 1536),
}
# Prepended to the (English) user prompt: Pony quality score tags + anthro, so the result is a
# humanoid-animal. Deliberately does NOT force fur, a species, character count, or clothing — the
# user's own prompt decides: "fluffy cat" → fur, "dragon" → scales, "a couple" → two characters,
# "naked" → nude, "in a skirt" → clothed. (Forcing `solo` used to drop the partner; forcing
# `fluffy fur covering whole body` used to put fur on scaly dragons — both removed.)
PONY_FURRY_POS_PREFIX = os.getenv(
    "PONY_FURRY_POS_PREFIX",
    "score_9, score_8_up, score_7_up, source_furry, anthro, ",
)
# Negative keeps it non-human and clean of artifacts, but does NOT ban clothing, nudity, fur, or
# scales — all user-controlled. `hairless skin` was removed so scaly/reptile species aren't pushed
# toward fur. Censorship terms stay so requested nudity isn't bar/mosaic'd.
PONY_FURRY_NEGATIVE = os.getenv(
    "PONY_FURRY_NEGATIVE",
    "score_4, score_5, score_6, human face, human skin, censored, "
    "mosaic censorship, bar censor, low quality, worst quality, bad anatomy, deformed, mutated, "
    "extra limbs, extra digits, watermark, text, signature",
)
# Booru count tags Pony responds to; auto-added when the prompt implies more than one character
# (so "пара занимается сексом" actually renders two, not a lone confused subject).
PONY_FURRY_MULTI_RE = re.compile(
    r"\b(couple|two|both|each\s*other|together|duo|pair|partner|threesome|group|"
    r"they|them|sex\s+with|fucking\s+\w+|2\s*(?:girls|boys|characters))\b",
    re.IGNORECASE,
)
PONY_FURRY_STEPS = int(os.getenv("PONY_FURRY_STEPS", "28"))
PONY_FURRY_CFG = float(os.getenv("PONY_FURRY_CFG", "7.0"))

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
# A spoken line is voiced once by confining it to a single short timeline window sized to how
# long the phrase naturally takes to say, instead of leaving the dialogue "active" across the
# whole clip (which makes LTX Eros repeat the phrase 2-4x to fill the duration). Rest of the
# clip (and the global prompt) carries only the visual action, so she speaks once then continues.
LTX_EROS_SPEECH_WORDS_PER_SEC = float(os.getenv("LTX_EROS_SPEECH_WORDS_PER_SEC", "2.2"))
LTX_EROS_SPEECH_PAD_SECONDS = float(os.getenv("LTX_EROS_SPEECH_PAD_SECONDS", "0.6"))
LTX_EROS_SPEECH_MIN_SECONDS = float(os.getenv("LTX_EROS_SPEECH_MIN_SECONDS", "1.5"))

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
IMAGE_EDIT_LIGHTNING_LORA = os.getenv("IMAGE_EDIT_LIGHTNING_LORA", "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors")
# Improves female nude anatomy realism (the base model alone tends toward plasticky/off
# proportions). Needs its trigger words present in the prompt to reliably engage.
IMAGE_EDIT_NSFW_LORA = os.getenv("IMAGE_EDIT_NSFW_LORA", "SEXGOD_FemaleNudity_QwenEdit_2511_v2.safetensors")
IMAGE_EDIT_NSFW_LORA_TRIGGER = os.getenv("IMAGE_EDIT_NSFW_LORA_TRIGGER", "LoRA, Nude, Sex.")
IMAGE_EDIT_STEPS = int(os.getenv("IMAGE_EDIT_STEPS", "4"))
IMAGE_EDIT_CFG = float(os.getenv("IMAGE_EDIT_CFG", "1"))
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
OLLAMA_SCENARIO_TIMEOUT = int(os.getenv("OLLAMA_SCENARIO_TIMEOUT", "300"))
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
    "и однозначной. Без цензуры, explicit-контент разрешён и ожидается — используй грубую, "
    "прямую лексику (хуй, пизда, ебать, трахать, сосать, минет, раком, кончает и т.п.), а НЕ "
    "литературные эвфемизмы вроде \"её женственность\" или \"его мужское достоинство\". Пиши "
    "только сам сценарий, без пояснений."
)

# Clean counterpart for the censored 🌸 Clean Video pipeline: same 3-4 sentence, chronological,
# appearance-first structure, but strictly SFW. The NSFW base above pushes nudity/explicit terms
# into the prompt text, which then fights the clean negative; this keeps the whole chain clean.
OLLAMA_SCENARIO_SYSTEM_PROMPT_CLEAN = (
    "Ты — сценарист для генерации коротких 12-секундных видео нейросетью. "
    "Видео склеивается из 3-4 смысловых блоков, каждый блок — одно предложение, и получает "
    "пропорциональную (по длине текста) долю из 12 секунд. Поэтому пиши РОВНО 3-4 предложения, "
    "не больше: каждое предложение — один цельный, законченный эпизод-кадр (не дробить одно "
    "действие на несколько предложений), без нумерации и списков, обычным связным текстом на "
    "русском языке. Сценарий должен идти строго в хронологическом порядке без скачков: каждое "
    "следующее предложение — плавное продолжение и нарастание предыдущего. ПЕРВОЕ предложение — "
    "ОБЯЗАТЕЛЬНО подробное описание внешности персонажа и его ОДЕЖДЫ (конкретные предметы "
    "одежды, причёска, выражение лица), без описания обнажённого тела. Только после этого, во "
    "2-4 предложениях — обстановка → действие → развитие/кульминация (движение, эмоции, "
    "окружение, свет, погода, детали сцены). Контент строго цензурный и приличный: персонаж "
    "всегда полностью одет, никакой наготы, эротики, секса, насилия или грубой лексики — это "
    "красивый, кинематографичный, безопасный ролик. Делай акцент на атмосфере, движении и "
    "визуальных деталях. Пиши только сам сценарий, без пояснений."
)

# "💬 Реплики" for LTX Eros: when the user hasn't written a spoken line, Ollama improvises ONE
# short in-character line that fits the scene (her reaction), which LTX Eros then voices natively.
# Kept to a single short line on purpose: LTX renders one on-screen speaker's lipsync well but
# blurs multiple distinct voices, and the timeline confinement voices that one line exactly once.
OLLAMA_DIALOGUE_SYSTEM_PROMPT = (
    "Ты придумываешь ОДНУ короткую реплику прямой речи, которую персонаж на видео произносит "
    "вслух в этот момент. Ответь ТОЛЬКО самой фразой — без кавычек, без имени говорящего, без "
    "пояснений, без описания сцены. Одно предложение, 4-9 слов, от первого лица, разговорно и "
    "в тему происходящего (реакция/эмоция/просьба). Русский язык. Без цензуры: если сцена "
    "откровенная, используй грубую прямую лексику, а не литературные эвфемизмы. Никаких ремарок."
)

# "🧠 Авто-лоры" for LTX Eros: Ollama reads the scene and picks the matching LoRAs from the catalog
# (same CPU 32B as dialogue). Keeps it to 1-3 so stacked loras don't fight (one action + optional
# finish). Cum loras are grouped so only one is ever chosen.
CUM_LORA_KEYS = {"eros_joyshot", "eros_facials", "eros_epic_cum", "eros_creampie", "eros_cumsplash",
                 "wan_cumshot", "wan_facial"}
OLLAMA_LORA_SYSTEM_PROMPT = (
    "Ты подбираешь видео-лоры (LoRA) для эротической видео-сцены по её описанию. "
    "Доступные лоры (ключ: назначение):\n{catalog}\n\n"
    "Правила: выбери от 1 до 3 ключей, максимально подходящих сцене. Обычно одно главное действие "
    "(поза или оральное) плюс при желании одна лора-финиш (камшот/кремпай). НЕ более одной cum-лоры. "
    "НЕ совмещай две разные позы. "
    "Если по сюжету одежда СНИМАЕТСЯ или сцена резко меняется (одетая→голая, «раздевается», "
    "«голая танцует») — добавь лору смены сцены/раздевания, если такая есть в списке. Если по "
    "сюжету человек ОСТАЁТСЯ одетым (секс в одежде) — НЕ добавляй лору раздевания. "
    "Если сцены секса и раздевания нет — верни пустой массив. "
    'Ответь ТОЛЬКО JSON-массивом ключей, например ["eros_riding","eros_creampie"]. Без пояснений.'
)

# "📖 История V1": Ollama breaks one idea into N connected beats, each rendered as a separate LTX
# Eros clip chained off the previous clip's last frame, then concatenated into one film. Each beat
# is ONE self-contained scene sentence (like the scenario writer's sentences, but here each beat is
# a WHOLE 12s clip on its own). Beat 1 must lock the character's look so identity survives the chain.
# 📖 История V1 is shelved for now (Phase-1 chained-clip results were poor: identity/anatomy
# fell apart across the chain). Code stays intact; flip BOT_STORY_ENABLED=1 to bring it back.
STORY_ENABLED = os.getenv("BOT_STORY_ENABLED", "0") == "1"
STORY_PARTS = int(os.getenv("STORY_PARTS", "5"))
STORY_SEGMENT_TIMEOUT = int(os.getenv("STORY_SEGMENT_TIMEOUT", "900"))
OLLAMA_STORY_SYSTEM_PROMPT = (
    "Ты — сценарист эротического мини-фильма, который нейросеть соберёт из {parts} видео-сцен, "
    "идущих подряд. Разбей идею пользователя на РОВНО {parts} сцен. Каждая сцена — ОДНО "
    "самодостаточное предложение-эпизод на русском, которое описывает ЦЕЛЬНОЕ действие "
    "(это будет отдельный 12-секундный клип). Сцены идут строго хронологически, каждая — плавное "
    "продолжение и нарастание предыдущей, единый сюжет с завязкой, развитием и кульминацией в "
    "финале. ПЕРВАЯ сцена ОБЯЗАНА подробно описать внешность и одежду персонажа (тело, причёска, "
    "конкретное бельё/одежда) — этот образ НЕЛЬЗЯ менять в следующих сценах (иначе видео-модель "
    "теряет лицо). Без цензуры, explicit разрешён и ожидается — грубая прямая лексика, а не "
    "эвфемизмы. Ответь ТОЛЬКО пронумерованным списком из {parts} строк (1. …\\n2. …), без "
    "заголовков и пояснений."
)

# Modes whose final ComfyUI output is a video, vs a still image.
# "story" (История V1) chains several LTX Eros clips into one film — it's a video mode that needs
# one start photo, but it renders + delivers entirely inside submit_story_job (no ACTIVE_PROMPTS
# entry), so it never touches the monitor loop / send_result path.
VIDEO_MODES = {"video", "video_clean", "ltx_sulphur", "ltx_eros", "story"}
# Modes that take a single uploaded photo into st["video_source"] (video modes, plus
# mopmix which runs img2img off of it, plus image which edits it directly).
SINGLE_PHOTO_MODES = VIDEO_MODES | {"mopmix", "image"}
# Modes that take two uploaded photos into st["duo_photos"].
DUO_PHOTO_MODES = {"mopmix_duo"}
# Modes whose workflow renders silent video and needs the MMAudio postprocess pass.
# "video" uses the NSFW Foley model; "video_clean" uses the stock MMAudio model
# (VIDEO_CLEAN_AUDIO_MODEL) via video_audio_model_for_mode() so its Foley track stays clean.
SILENT_VIDEO_MODES = {"video", "video_clean"}

# "video_clean": a censored, high-fidelity image->video mode. It reuses the same WAN 2.2
# I2V workflow/graph as "video" but swaps the NSFW-tuned base UNET (nodes 371/372) for the
# stock WAN 2.2 I2V A14B experts, force-applies only the clean Lightx2v 4-step distill LoRA
# (ignoring any NSFW LoRA selection), and adds SFW negatives. Stock WAN 2.2 is the
# community's top-reviewed open-source I2V model for identity ("face") preservation.
# Default to the Q4_K_M GGUF experts for VRAM parity with the NSFW "video" mode (which also
# runs Q4 GGUF). The loader is picked by extension in apply_clean_video_base, so overriding
# these with the fp8 .safetensors checkpoints (also on disk) still works.
VIDEO_CLEAN_UNET_HIGH = os.getenv("VIDEO_CLEAN_UNET_HIGH", "Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf")
VIDEO_CLEAN_UNET_LOW = os.getenv("VIDEO_CLEAN_UNET_LOW", "Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf")
VIDEO_CLEAN_UNET_DTYPE = os.getenv("VIDEO_CLEAN_UNET_DTYPE", "default")
# The clean (non-NSFW) Lightx2v 4-step distill, applied at full strength so the stock
# (non-distilled) experts converge in the workflow's low step count. Applied directly rather
# than via VIDEO_LORA_OPTIONS, whose shared strength multiplier would weaken it to ~0.5.
VIDEO_CLEAN_DISTILL_HIGH = os.getenv("VIDEO_CLEAN_DISTILL_HIGH", "Wan2.2-I2V-A14B-Moe-Distill-Lightx2v_high.safetensors")
VIDEO_CLEAN_DISTILL_LOW = os.getenv("VIDEO_CLEAN_DISTILL_LOW", "Wan2.2-I2V-A14B-Moe-Distill-Lightx2v_low.safetensors")
VIDEO_CLEAN_DISTILL_STRENGTH = float(os.getenv("VIDEO_CLEAN_DISTILL_STRENGTH", "1.0"))
# The NSFW base is a fast-move finetune baked for sigma_shift=8; stock WAN 2.2 experts driven by
# an external Lightx2v distill want the canonical ~5.0 shift. At 8 the schedule over-shifts and the
# stock latent never resolves -> frames disintegrate into dust/fog. Override only the clean sampler.
VIDEO_CLEAN_SIGMA_SHIFT = float(os.getenv("VIDEO_CLEAN_SIGMA_SHIFT", "5.0"))

# NSFW "🎬 Video" base — upgraded from the Q4_K_M "FASTMOVE" merge to the fp8-scaled stock WAN 2.2
# I2V experts (much less quant loss → faces/anatomy hold together) driven by the Lightx2v 4-step
# distill (same as clean) + SageAttention, at sigma_shift 5 (the fast-move merge wanted 8, stock
# fp8+external distill wants ~5). This is the recipe from the user's validated missionary_b graph.
VIDEO_UNET_HIGH = os.getenv("VIDEO_UNET_HIGH", "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors")
VIDEO_UNET_LOW = os.getenv("VIDEO_UNET_LOW", "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors")
VIDEO_UNET_DTYPE = os.getenv("VIDEO_UNET_DTYPE", "default")
# Seko-V1 4-step lightning is the distill the MQ Lab transition loras (mql_*) were validated with
# (the user's reference missionary_b graph). Swapping in from lightx2v killed the color-burn /
# epilepsy-jitter / broken-anatomy artifacts those loras produced on the lightx2v base.
VIDEO_DISTILL_HIGH = os.getenv("VIDEO_DISTILL_HIGH", "Wan2_2-I2V-A14B-4steps-lora-rank64-Seko-V1_High.safetensors")
VIDEO_DISTILL_LOW = os.getenv("VIDEO_DISTILL_LOW", "Wan2_2-I2V-A14B-4steps-lora-rank64-Seko-V1_Low.safetensors")
VIDEO_DISTILL_STRENGTH = float(os.getenv("VIDEO_DISTILL_STRENGTH", "1.0"))
# The vanilla fp8 base is CLEAN (unlike the old Q4 "FASTMOVE" NSFW merge) — so it just animates the
# face and never initiates sex. To restore the baked-in porn drive, always-apply DR34ML4Y (the top
# all-in-one NSFW motion lora, 442k dl) on top of the distill, exactly like the user's validated
# missionary_b graph did. Set VIDEO_NSFW_LORA_STRENGTH=0 to fall back to the clean vanilla base.
VIDEO_NSFW_LORA_HIGH = os.getenv("VIDEO_NSFW_LORA_HIGH", "DR34ML4Y_I2V_14B_HIGH_V2.safetensors")
VIDEO_NSFW_LORA_LOW = os.getenv("VIDEO_NSFW_LORA_LOW", "DR34ML4Y_I2V_14B_LOW_V2.safetensors")
VIDEO_NSFW_LORA_STRENGTH = float(os.getenv("VIDEO_NSFW_LORA_STRENGTH", "1.0"))
VIDEO_SIGMA_SHIFT = float(os.getenv("VIDEO_SIGMA_SHIFT", "5.0"))
# WAN 2.2 I2V is natively trained for ~5-6s (~81-97 frames @16fps). Asking it for MORE frames makes
# it loop/boomerang (replay the arc or repeat). So we cap the GENERATED length here and, when the
# user asks for longer, RIFE-interpolate the native clip up to the requested duration (smooth
# slow-motion, one continuous arc, no boomerang). Clean video mode is left untouched.
VIDEO_NATIVE_MAX_SECONDS = int(os.getenv("VIDEO_NATIVE_MAX_SECONDS", "6"))
VIDEO_RIFE_CKPT = os.getenv("VIDEO_RIFE_CKPT", "rife49.pth")
# SageAttention mode (KJNodes PathchSageAttentionKJ): "auto" picks the best kernel for the GPU.
VIDEO_SAGE_MODE = os.getenv("VIDEO_SAGE_MODE", "auto")
VIDEO_FP16_ACCUM = os.getenv("VIDEO_FP16_ACCUM", "1") == "1"
VIDEO_CLEAN_NEGATIVE = (
    "nudity, nude, naked, nsfw, explicit, sexual, genitals, nipples, exposed breasts, "
    "underwear, lingerie, suggestive, erotic, pornographic, gore, blood"
)
# Clean MMAudio Foley for the censored mode: the stock MMAudio large 44k v2, instead of the
# NSFW-tuned "nsfw_gold" model (moans/slaps). The VAE/synchformer/CLIP are shared, so only the
# main model differs. The audio negative steers away from sexual/voice sounds for a "clean
# but detailed" ambient track (footsteps, wind, room tone, etc.).
VIDEO_CLEAN_AUDIO_MODEL = os.getenv("VIDEO_CLEAN_AUDIO_MODEL", "mmaudio_large_44k_v2_fp16.safetensors")
VIDEO_CLEAN_AUDIO_NEGATIVE = os.getenv(
    "VIDEO_CLEAN_AUDIO_NEGATIVE",
    "moaning, moans, sex sounds, sexual sounds, screaming, music, voice, speech, singing",
)
# "Smart" Clean Video: send a photo + an idea, get a finished clip with zero extra taps. The
# button runs the whole chain itself - Ollama rewrites the idea into a rich scenario, Qwen-Edit
# redraws the photo into that scene (so I2V can show things that aren't in the source, e.g. a
# tiger, a plane, underwater), then the clean WAN animates it and MMAudio adds sound. Both stages
# are toggleable: AUTO_EDIT off = animate the source as-is, AUTO_EXPAND off = use the raw prompt.
VIDEO_CLEAN_AUTO_EXPAND = os.getenv("VIDEO_CLEAN_AUTO_EXPAND", "1").strip().lower() not in {"0", "false", "no", "off"}
VIDEO_CLEAN_AUTO_EDIT = os.getenv("VIDEO_CLEAN_AUTO_EDIT", "1").strip().lower() not in {"0", "false", "no", "off"}
VIDEO_CLEAN_EDIT_TIMEOUT = int(os.getenv("VIDEO_CLEAN_EDIT_TIMEOUT", "300"))

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
# All-LTX lora catalog (migrated off WAN — Eros is the primary video model). Every entry is a
# single-file LTX-2.3 lora applied to the LTX Eros graph via apply_eros_loras (LTX2LoraLoaderAdvanced).
# `trigger` is appended to the Eros prompt so the effect fires; `strength` is the raw model strength.
# The WAN 2.2 I2V graph no longer has selectable loras (its Power Lora nodes stay empty).
VIDEO_LORA_OPTIONS = [
    # — cum / finish —
    {"key": "eros_joyshot", "label": "Cumshot/Facial", "origin": "Eros", "lora": "DaSiWa_LTX23_Cumshot_Joyshot-v01.safetensors", "trigger": "he ejaculates on her face, thick white cum splashing onto her face", "strength": 1.0},
    {"key": "eros_facials", "label": "Cum on Face/Mouth", "origin": "Eros", "lora": "cumonface_inmouth_LTX23.safetensors", "trigger": "cum on her face, cum in her mouth", "strength": 1.0},
    {"key": "eros_epic_cum", "label": "Epic Cumshots", "origin": "Eros", "lora": "epic_cumshots_LTX23.safetensors", "trigger": "cumshot, thick cum", "strength": 1.0},
    {"key": "eros_creampie", "label": "Creampie", "origin": "Eros", "lora": "LTX23_Creampie_Animation-v01.safetensors", "trigger": "cum, creampie, thick white cum dripping out", "strength": 1.0},
    {"key": "eros_cumsplash", "label": "Cum Splash POV", "origin": "Eros", "lora": "cumsplash_LTX2_v1.safetensors", "trigger": "cumsplash, cum splashing onto her face", "strength": 1.0},
    # — oral —
    {"key": "eros_blowjob", "label": "Blowjob", "origin": "Eros", "lora": "LTX23_blowjob_animation_I2V_v1.safetensors", "trigger": "blowjob animation, her mouth is wrapped around the penis", "strength": 0.9},
    {"key": "eros_deepthroat", "label": "Deepthroat", "origin": "Eros", "lora": "ltxdeepthroat_v01.safetensors", "trigger": "LTXdeepthroat", "strength": 0.9},
    {"key": "eros_ult_dt", "label": "Ultimate Deepthroat", "origin": "Eros", "lora": "ltx23-ultimatedt-NSFW_k3nk.safetensors", "trigger": "", "strength": 0.9},
    # — positions —
    {"key": "eros_allinone", "label": "General NSFW (multi)", "origin": "Eros", "lora": "Penile_Praxis_V4_LTX23.safetensors", "trigger": "", "strength": 0.85},
    {"key": "eros_riding", "label": "Riding/Cowgirl", "origin": "Eros", "lora": "riding_fbs_10Eros_i2v_v1.safetensors", "trigger": "Riding frontshot animation", "strength": 0.9},
    {"key": "eros_doggy", "label": "Doggystyle", "origin": "Eros", "lora": "SexGod_LTX23_DoggyStyle_v2_5.safetensors", "trigger": "", "strength": 0.9},
    {"key": "eros_thrust", "label": "Sex Thrust", "origin": "Eros", "lora": "LTX2-i2v-SexThrust.safetensors", "trigger": "", "strength": 0.85},
    {"key": "eros_anal", "label": "Anal Insertion", "origin": "Eros", "lora": "nsfw_anal_insertion_ltx23_v1.safetensors", "trigger": "anal insertion, being penetrated by the man's large penis", "strength": 0.9},
    {"key": "eros_takerpov", "label": "Taker POV", "origin": "Eros", "lora": "LTX2_takerpov_v1.2.safetensors", "trigger": "taker pov", "strength": 0.85},
    # — hands / body —
    {"key": "eros_handjob", "label": "Handjob", "origin": "Eros", "lora": "SexGod_Handjobs_LTX23_v1.safetensors", "trigger": "", "strength": 0.9},
    {"key": "eros_bounce", "label": "Bouncing Boobs", "origin": "Eros", "lora": "bounceV2_5_LTX23_I2V.safetensors", "trigger": "her breasts bouncing up and down", "strength": 0.8},
    {"key": "eros_orgasm", "label": "Real Orgasm", "origin": "Eros", "lora": "Ltx23_RemoteOrgasm_v1.safetensors", "trigger": "", "strength": 0.85},
    {"key": "eros_facepunch", "label": "Face Punch/Slap", "origin": "Eros", "lora": "FacePunch_LTX_comfy.safetensors", "trigger": "FacePunch", "strength": 0.9},
    # — enhancers (keep low, stack under an action) —
    {"key": "eros_smooth", "label": "Smooth Motion", "origin": "Eros", "lora": "SmoothMix_Animations_LTX2.safetensors", "trigger": "smoothmixrealism, realistic style", "strength": 0.6},
    {"key": "eros_physics", "label": "Body Physics/Fluid", "origin": "Eros", "lora": "DaSiWa_LTX23_Bodyphysics_Fluid_v01.safetensors", "trigger": "D4S1W4_NSFWSME", "strength": 0.6},
    # — furry —
    {"key": "eros_furry_cum", "label": "Furry 2D + Cum", "origin": "Eros", "lora": "LTX23_Furry_2D_NSFW_Multi+Cum-v1.safetensors", "trigger": "cum", "strength": 0.85},
    {"key": "eros_furry_sex", "label": "Фурри секс (звериный член)", "origin": "Eros", "lora": "LTX2_3_NSFW_furry_concat_v2.safetensors", "trigger": "anthro, furry, animal penis, animal genitalia, tapered penis, knotted penis, sheath", "strength": 0.9},
    # — WAN (только для обычного 🎬 Video; на 🌸 Clean Video лоры не применяются). WAN 2.2 A14B —
    # MoE с двумя экспертами, поэтому нужны high/low файлы; у одиночной 2.1-лоры один файл на оба.
    {"key": "wan_furry", "label": "ВАН лора фурри", "origin": "WAN", "high": "furry_nsfw_1.1_e22.safetensors", "low": "furry_nsfw_1.1_e22.safetensors", "trigger": "anthro, furry, animal penis, feral penis, knotted penis, sheath", "strength": 0.45},
    # Native WAN 2.2 A14B pair (proper high/low split) — cleaner anthro anatomy/fur than the 2.1 file above.
    {"key": "wan_furry_enh", "label": "ВАН фурри 2.2 (нативная)", "origin": "WAN", "high": "Furry Enhancer Wan2.2 V3 High Noise I2V.safetensors", "low": "Furry Enhancer Wan2.2 V3 Low Noise I2V .safetensors", "trigger": "anthro, furry, fur detail, animal penis, feral penis, knotted penis, sheath", "strength": 0.45},
    # scene_change (MQ Lab, nsfw_v2) — драйвер смены сцены: разрешает i2v раздеть/переставить ВНУТРИ
    # одного клипа (одетый оригинал → секс), как в референс-воркфлоу. Только high-файл (low в референсе
    # выключен), кладём его на оба эксперта. Триггер НАМЕРЕННО без «completely naked» — уровень одежды
    # задаёт промпт (голая / в чулках / в бюстгальтере). Держать ~0.4 эфф., выбирать С позовой лорой.
    {"key": "wan_scene_change", "label": "Смена сцены (раздевание)", "origin": "WAN", "high": "scene_change_nsfw_v2.0_high_noise.safetensors", "low": "scene_change_nsfw_v2.0_high_noise.safetensors", "trigger": "the scene changes, her clothes come off, revealing her body", "strength": 0.2},
    # Смена ФОНА/локации (i2v держит фон входного фото; эти драйверы позволяют перенести сцену).
    # Локация задаётся промптом («on a beach / in a restaurant»); лора поднимает надёжность переноса.
    {"key": "wan_bg_change", "label": "Смена фона/локации", "origin": "WAN", "high": "wan_background_change.safetensors", "low": "wan_background_change.safetensors", "trigger": "the background changes to a new location", "strength": 0.25},
    {"key": "wan_set_reveal", "label": "Смена сцены (Set Reveal)", "origin": "WAN", "high": "wan_set_reveal_high.safetensors", "low": "wan_set_reveal_high.safetensors", "trigger": "the set changes, the scene transitions to a new place", "strength": 0.25},
    # — нежность/эмоции/оральное (стакаются под позу; триггеры только про действие) —
    {"key": "wan_kiss", "label": "Поцелуй (French Kiss)", "origin": "WAN", "high": "WAN2.2-FrenchKiss_HighNoise.safetensors", "low": "WAN2.2-FrenchKiss_LowNoise.safetensors", "trigger": "they kiss passionately, french kissing with tongue", "strength": 0.35},
    {"key": "wan_lick", "label": "Ласки языком (лизать)", "origin": "WAN", "high": "LipL-high-60.safetensors", "low": "LipL-low-60.safetensors", "trigger": "she licks with her tongue, licking and caressing", "strength": 0.35},
    {"key": "wan_emotion", "label": "Эмоции лица", "origin": "WAN", "high": "sigma_face_expression_high.safetensors", "low": "sigma_face_expression_high.safetensors", "trigger": "expressive emotional face, moaning with pleasure, smiling", "strength": 0.25},
    # — MQ Lab transition-позы (dedicated «одетая→акт», high/low пары) — под Seko-V1 базу, эфф. 0.7.
    # Триггеры описывают только ДЕЙСТВИЕ (без принудительной наготы), чтобы промпт рулил одеждой/сценой.
    {"key": "wan_missionary", "label": "Поза: миссионерская (MQ)", "origin": "WAN", "high": "mql_missionary_b_v1_high_noise.safetensors", "low": "mql_missionary_b_v1_low_noise.safetensors", "trigger": "she lies on her back and he has missionary sex with her, thrusting into her", "strength": 0.35},
    {"key": "wan_doggy", "label": "Поза: раком (MQ)", "origin": "WAN", "high": "mql_doggy_b_v1_high_noise.safetensors", "low": "mql_doggy_b_v1_low_noise.safetensors", "trigger": "she is on all fours and he fucks her doggystyle from behind", "strength": 0.35},
    {"key": "wan_revcowgirl", "label": "Поза: наездница спиной (MQ)", "origin": "WAN", "high": "mql_reverse_cowgirl_a_v1_high_noise.safetensors", "low": "mql_reverse_cowgirl_a_v1_low_noise.safetensors", "trigger": "she straddles him in reverse cowgirl and rides his cock with her back to him", "strength": 0.35},
    {"key": "wan_standing", "label": "Поза: стоя (MQ)", "origin": "WAN", "high": "mql_standing_a_v1_high_noise.safetensors", "low": "mql_standing_a_v1_low_noise.safetensors", "trigger": "he fucks her standing up, penetrating her", "strength": 0.35},
    {"key": "wan_spoon", "label": "Поза: ложка (MQ)", "origin": "WAN", "high": "mql_spoon_a_v2_low_noise.safetensors", "low": "mql_spoon_a_v2_low_noise.safetensors", "trigger": "she lies on her side and he spoons her, penetrating her from behind", "strength": 0.35},
    {"key": "wan_dp", "label": "Двойное проникновение (MQ)", "origin": "WAN", "high": "mql_dp_a_v1_high_noise.safetensors", "low": "mql_dp_a_v1_low_noise.safetensors", "trigger": "two men penetrate her at once, double penetration", "strength": 0.35},
    {"key": "wan_atm", "label": "АТМ: минет→секс (MQ)", "origin": "WAN", "high": "mql_atm_a_high_noise.safetensors", "low": "mql_atm_a_low_noise.safetensors", "trigger": "she sucks his cock then he penetrates her", "strength": 0.35},
    {"key": "wan_double", "label": "Двойной минет (MQ)", "origin": "WAN", "high": "mql_double_blowjob_a_v1_high_noise.safetensors", "low": "mql_double_blowjob_a_v1_high_noise.safetensors", "trigger": "two men stand in front of her and she sucks their two cocks, double blowjob", "strength": 0.35},
    {"key": "wan_cowgirl", "label": "Поза: наездница лицом", "origin": "WAN", "high": "WAN-2.2-I2V-POV-Cowgirl-HIGH-v1.0-fixed.safetensors", "low": "WAN-2.2-I2V-POV-Cowgirl-LOW-v1.0-fixed.safetensors", "trigger": "cowgirl position, she rides his cock facing him, bouncing up and down", "strength": 0.4},
    {"key": "wan_anal", "label": "Анал", "origin": "WAN", "high": "wan22_i2v_anal_v1_high_noise.safetensors", "low": "wan22_i2v_anal_v1_low_noise.safetensors", "trigger": "anal sex, he penetrates her ass, anal insertion", "strength": 0.4},
    {"key": "wan_deepthroat", "label": "Дипгорло", "origin": "WAN", "high": "jfj-deepthroat-W22-I2V-HN.safetensors", "low": "jfj-deepthroat-W22-I2V-LN.safetensors", "trigger": "deepthroat, she takes his cock deep in her throat", "strength": 0.4},
    {"key": "wan_cumshot", "label": "Камшот/сперма", "origin": "WAN", "high": "Wan22_CumV3_High.safetensors", "low": "Wan22_CumV3_Low.safetensors", "trigger": "cumshot, he cums, thick white cum", "strength": 0.45},
    {"key": "wan_facial", "label": "Камшот на лицо", "origin": "WAN", "high": "wan22-f4c3spl4sh-100epoc-high-k3nk.safetensors", "low": "wan22-f4c3spl4sh-154epoc-low-k3nk.safetensors", "trigger": "f4c3spl4sh, he cums on her face, facial cumshot", "strength": 0.45},
]
# Origin tag for a lora option. All catalog entries are now LTX ("Eros"); the helper is kept so
# apply_video_loras (WAN path) safely skips everything and future WAN entries would default to "WAN".
def lora_origin(opt: dict[str, Any]) -> str:
    return str(opt.get("origin", "WAN"))

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
# Roulette per-chat state, filled lazily by the worker during a batch
ROULETTE_LAST_PROMPT: dict[int, str] = {}
ROULETTE_CAPTION: dict[int, str] = {}
JOB_SEQ = 0
ACTIVE_PROMPTS: dict[str, dict] = {}
# Chat ids that asked to stop a running 📖 История — checked between segments so a long
# multi-clip film can be aborted (the current segment finishes, then the chain bails).
STORY_CANCEL: set[int] = set()

# Live progress status message per chat, edited in place while a job is generating so the
# user doesn't think the bot/server hung. Rolling average duration per mode (seconds) used
# for the ETA estimate, seeded with rough defaults and refined after every completed job.
CHAT_STATUS: dict[int, dict[str, Any]] = {}
MODE_AVG_DURATION: dict[str, float] = {
    "video": 240.0,
    "video_clean": 300.0,
    "ltx_sulphur": 230.0,
    "ltx_eros": 230.0,
    "story": 1400.0,
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
    # Roulette: re-roll the prompt lazily in the worker (keeps enqueue instant so Stop works)
    roulette: bool = False
    base_idea: str = ""
    # Furry: MopMix uses the Pony base + anthro tags instead of bigASP
    furry: bool = False
    # Auto-dialogue: LTX Eros lets Ollama improvise a spoken line when the user wrote none
    auto_dialogue: bool = False
    # Auto-loras: LTX Eros lets Ollama pick matching catalog loras from the scene when none chosen
    auto_lora: bool = False
    # Furry base checkpoint choice: "pony" (photoreal, default) or "yiffy" (Yiffymix)
    furry_base: str = "pony"


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


def _segment_has_audio(path: Path) -> bool:
    """ffprobe check: does the file carry an audio stream? Story segments must all match
    (audio-or-not) or the concat -c copy step fails, so we add silence to the ones that lack it."""
    try:
        p = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
             "stream=index", "-of", "csv=p=0", str(path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60,
        )
        return bool(p.stdout.strip())
    except Exception:
        return False


def normalize_story_segment(raw_path: Path, out_path: Path, width: int, height: int, fps: int = 24) -> None:
    """Re-encode a story clip to canonical H.264/yuv420p at fixed size+fps WITH an AAC audio track
    (real if present, else silent), so every segment is byte-compatible for concat -c copy. LTX Eros
    voices natively, so keeping the real audio preserves the per-scene dialogue in the final film."""
    common_v = ["-vf", f"scale={int(width)}:{int(height)},fps={int(fps)}",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast"]
    if _segment_has_audio(raw_path):
        run_cmd(["ffmpeg", "-y", "-i", str(raw_path), *common_v,
                 "-c:a", "aac", "-ar", "44100", "-ac", "2",
                 "-movflags", "+faststart", str(out_path)])
    else:
        run_cmd(["ffmpeg", "-y", "-i", str(raw_path),
                 "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
                 *common_v, "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0", "-shortest",
                 "-movflags", "+faststart", str(out_path)])


def extract_last_frame(video_path: Path, out_png: Path) -> None:
    """Grab a frame ~0.2s before the end as the init image for the next story clip. A hair before
    the very end dodges any trailing fade/black frame that would poison the next generation."""
    run_cmd(["ffmpeg", "-y", "-sseof", "-0.2", "-i", str(video_path),
             "-frames:v", "1", "-q:v", "2", str(out_png)])
    if not out_png.exists():
        # Very short clip: -sseof overshot the start; just take the first frame instead.
        run_cmd(["ffmpeg", "-y", "-i", str(video_path), "-frames:v", "1", "-q:v", "2", str(out_png)])


def concat_story_segments(segment_paths: list[Path], out_path: Path, workdir: Path) -> None:
    """Stitch the normalized segments into one film via the ffmpeg concat demuxer (-c copy: no
    re-encode, since normalize_story_segment already made them uniform)."""
    list_file = workdir / "concat_list.txt"
    list_file.write_text("".join(f"file '{p.resolve()}'\n" for p in segment_paths))
    run_cmd(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
             "-c", "copy", "-movflags", "+faststart", str(out_path)])


def extract_speech_text(prompt: str) -> str:
    text = (prompt or "").strip()
    if not text:
        return ""

    quoted_after_speech = re.findall(
        r"(?:говорит|говоря|сказала|скажет|скажи|говори|произнеси|произносит|шепчет|says|said|say|speaks|whispers|tell|tells)[^\n\"«”]{0,40}[\"«“](.{1,220}?)[\"»”]",
        text,
        flags=re.IGNORECASE,
    )
    if quoted_after_speech:
        return clean_speech_text(". ".join(quoted_after_speech))

    quoted = re.findall(r"[\"«“](.{1,180}?)[\"»”]", text)
    if quoted:
        return clean_speech_text(". ".join(quoted[:2]))

    match = re.search(
        r"(?:говорит|говоря|сказала|скажет|скажи|говори|произнеси|произносит|шепчет|says|said|say|speaks|whispers|tell|tells)\s*[:\-—]?\s*(.{1,180}?)(?=$|[.!?;\n])",
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


def strip_speech_from_text(text: str) -> str:
    """Remove quoted dialogue (and its 'says/говорит' lead-in) from a prompt, leaving only the
    visual/action description. Used for LTX Eros so the global prompt and the non-speech timeline
    segments carry no dialogue — otherwise the model keeps re-voicing the line for the whole clip."""
    text = text or ""
    # Drop "говорит/says ..." lead-in verbs together with the quote that follows them.
    text = re.sub(
        r"(?:говорит|говоря|сказала|скажет|скажи|говори|произнеси|произносит|шепчет|says|said|say|speaks|whispers|tell|tells)"
        r"[^\n\"«”]{0,40}[\"«“].*?[\"»”]",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    # Drop any remaining bare quoted spans.
    text = re.sub(r"[\"«“].*?[\"»”]", " ", text)
    # Drop a trailing dangling speech verb left with nothing to say.
    text = re.sub(
        r"\b(?:говорит|говоря|сказала|скажет|скажи|говори|произнеси|произносит|шепчет|says|said|say|speaks|whispers|tell|tells)\s*[:\-—]?\s*$",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+([.!?,;:])", r"\1", text)
    # A trailing bare subject pronoun left dangling after the quote was removed ("...раздевается. Она").
    text = re.sub(r"[\s.,;:]+(?:она|он|она же|she|he)\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" \t\r\n.,;:-—")
    return text


def estimate_speech_frames(spoken: str, fps: int, max_frames: int) -> int:
    """How many frames the phrase needs to be spoken once, at a natural pace (capped to the clip)."""
    words = max(1, len(spoken.split()))
    seconds = words / max(0.1, LTX_EROS_SPEECH_WORDS_PER_SEC) + LTX_EROS_SPEECH_PAD_SECONDS
    seconds = max(LTX_EROS_SPEECH_MIN_SECONDS, seconds)
    frames = int(round(seconds * fps))
    # Always leave at least ~1s of non-speech tail so the model has room to stop talking.
    return max(1, min(frames, max_frames - fps))


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
# The live allowlist: {user_id: {"name": str, "added_at": iso, "by": admin_id}}.
# Admins are implicitly allowed and can add/remove others from Telegram at runtime.
ALLOWED_USERS: dict[int, dict[str, Any]] = {}
# Users we've already pinged the admin about, so one stranger doesn't spam every message.
PENDING_REQUESTS: set[int] = set()
# Display names captured at request time, so on approval we store a human name (not just an id).
PENDING_NAMES: dict[int, str] = {}


def _user_display_name(user) -> str:
    """Human-friendly name for the allowlist: 'Full Name (@username)' / falls back to id."""
    if not user:
        return ""
    name = (user.full_name or "").strip()
    if user.username:
        name = f"{name} (@{user.username})".strip() if name else f"@{user.username}"
    return name


def load_allowlist() -> None:
    ALLOWED_USERS.clear()
    if ALLOWLIST_FILE.exists():
        try:
            data = json.loads(ALLOWLIST_FILE.read_text())
            for uid, meta in (data or {}).items():
                ALLOWED_USERS[int(uid)] = dict(meta or {})
        except Exception:
            log.exception("Failed to read allowlist %s", ALLOWLIST_FILE)
    # Seed from the legacy env var so nothing is lost on first migration.
    for uid in ALLOWED_USER_IDS:
        ALLOWED_USERS.setdefault(uid, {"name": "", "added_at": "", "by": 0})


def save_allowlist() -> None:
    try:
        ALLOWLIST_FILE.write_text(json.dumps({str(k): v for k, v in ALLOWED_USERS.items()}, ensure_ascii=False, indent=2))
    except Exception:
        log.exception("Failed to write allowlist %s", ALLOWLIST_FILE)


def is_admin(user_id: int | None) -> bool:
    return bool(user_id) and user_id in ADMIN_USER_IDS


def add_allowed(user_id: int, name: str = "", by: int = 0) -> None:
    ALLOWED_USERS[user_id] = {
        "name": name,
        "added_at": datetime.now().isoformat(timespec="seconds"),
        "by": by,
    }
    PENDING_REQUESTS.discard(user_id)
    save_allowlist()


def remove_allowed(user_id: int) -> bool:
    existed = ALLOWED_USERS.pop(user_id, None) is not None
    PENDING_REQUESTS.discard(user_id)
    if existed:
        save_allowlist()
    return existed


def is_paused(user_id: int) -> bool:
    return bool(ALLOWED_USERS.get(user_id, {}).get("paused"))


def set_paused(user_id: int, paused: bool) -> bool:
    """Toggle a known user's pause flag. Returns True if the user is in the allowlist."""
    entry = ALLOWED_USERS.get(user_id)
    if entry is None:
        return False
    entry["paused"] = bool(paused)
    save_allowlist()
    return True


def allowed(update: Update) -> bool:
    user = update.effective_user
    if not user:
        return False
    if is_admin(user.id):
        return True
    # Paused users stay in the list but are blocked until the admin resumes them.
    return user.id in ALLOWED_USERS and not is_paused(user.id)


def _describe_user(user) -> str:
    parts = [user.full_name or ""]
    if user.username:
        parts.append(f"@{user.username}")
    parts.append(f"id={user.id}")
    return " ".join(p for p in parts if p).strip()


async def reject_if_needed(update: Update, context: ContextTypes.DEFAULT_TYPE | None = None) -> bool:
    if allowed(update):
        # Stamp admin-ness onto this user's state so the menu can show owner-only buttons.
        # Every gated handler calls this first, so it's the natural single point to do it.
        if context is not None and getattr(context, "user_data", None) is not None and update.effective_user:
            try:
                get_state(context)["is_admin"] = is_admin(update.effective_user.id)
            except Exception:
                pass
        return False
    user = update.effective_user
    # Known-but-paused user: tell them they're on hold and DON'T re-ping the admin (they're
    # already approved, just temporarily suspended — the admin resumes them from the 👥 menu).
    if user and user.id in ALLOWED_USERS and is_paused(user.id):
        paused_msg = "⏸ Твой доступ к боту временно приостановлен владельцем. Дождись возобновления."
        if update.message:
            await update.message.reply_text(paused_msg)
        elif update.callback_query:
            await update.callback_query.answer(paused_msg, show_alert=True)
        return True
    # Tell the person they're waiting on approval.
    if update.message:
        await update.message.reply_text("⛔ Доступ только по разрешению владельца. Запрос отправлен, подожди одобрения.")
    elif update.callback_query:
        await update.callback_query.answer("⛔ Нет доступа. Запрос владельцу отправлен.", show_alert=True)
    # Ping the admin once per unknown user with approve/deny buttons.
    if context and user and user.id not in PENDING_REQUESTS and not is_admin(user.id):
        PENDING_REQUESTS.add(user.id)
        PENDING_NAMES[user.id] = _user_display_name(user)
        kb = InlineKeyboardMarkup([[
            InlineKeyboardButton("✅ Разрешить", callback_data=f"acl:allow:{user.id}"),
            InlineKeyboardButton("🚫 Отклонить", callback_data=f"acl:deny:{user.id}"),
        ]])
        for admin_id in ADMIN_USER_IDS:
            try:
                await context.bot.send_message(
                    admin_id,
                    f"🔔 Запрос доступа к боту:\n{_describe_user(user)}",
                    reply_markup=kb,
                )
            except Exception:
                log.exception("Failed to notify admin %s about access request", admin_id)
    return True


def users_manage_text() -> str:
    lines = ["👥 *Пользователи бота*", "", "👑 Владельцы: " + ", ".join(str(a) for a in sorted(ADMIN_USER_IDS))]
    if ALLOWED_USERS:
        lines.append("")
        lines.append(f"✅ С доступом ({len(ALLOWED_USERS)}):")
        for uid, meta in sorted(ALLOWED_USERS.items()):
            nm = meta.get("name") or "(без имени)"
            flag = " ⏸ на паузе" if meta.get("paused") else ""
            lines.append(f"• {nm} — id `{uid}`{flag}")
        lines.append("")
        lines.append("⏸ — временно приостановить (человек остаётся в списке), 🚮 — выкинуть совсем.")
    else:
        lines.append("")
        lines.append("Пока никого — только ты. Люди появятся здесь, когда попросят доступ и ты одобришь.")
    return "\n".join(lines)


def users_manage_keyboard() -> InlineKeyboardMarkup:
    rows = []
    for uid, meta in sorted(ALLOWED_USERS.items()):
        nm = meta.get("name") or str(uid)
        label = nm if len(nm) <= 20 else nm[:19] + "…"
        if meta.get("paused"):
            pause_btn = InlineKeyboardButton(f"▶️ Возобновить: {label}", callback_data=f"acl:resume:{uid}")
        else:
            pause_btn = InlineKeyboardButton(f"⏸ Пауза: {label}", callback_data=f"acl:pause:{uid}")
        rows.append([pause_btn, InlineKeyboardButton("🚮", callback_data=f"acl:kick:{uid}")])
    rows.append([InlineKeyboardButton("🔄 Обновить", callback_data="acl:list"), InlineKeyboardButton("↩️ Назад", callback_data="show:status")])
    return InlineKeyboardMarkup(rows)


async def show_users_manager(query) -> None:
    try:
        await query.edit_message_text(users_manage_text(), reply_markup=users_manage_keyboard(), parse_mode="Markdown")
    except Exception:
        # Fall back to a fresh message if the original can't be edited (e.g. it was a photo).
        await query.message.reply_text(users_manage_text(), reply_markup=users_manage_keyboard(), parse_mode="Markdown")


async def handle_acl_callback(update: Update, context: ContextTypes.DEFAULT_TYPE, data: str) -> None:
    query = update.callback_query
    actor = update.effective_user
    if not is_admin(actor.id if actor else None):
        await query.answer("Только владелец может это делать.", show_alert=True)
        return
    parts = data.split(":", 2)
    action = parts[1] if len(parts) > 1 else ""
    if action == "list":
        await query.answer()
        await show_users_manager(query)
        return
    raw_id = parts[2] if len(parts) > 2 else ""
    try:
        uid = int(raw_id)
    except ValueError:
        await query.answer("Плохой id.", show_alert=True)
        return
    if action == "allow":
        add_allowed(uid, name=PENDING_NAMES.pop(uid, ""), by=actor.id)
        nm = ALLOWED_USERS.get(uid, {}).get("name") or ""
        await query.edit_message_text(f"✅ Доступ выдан: {nm} id={uid}".replace("  ", " ").strip())
        try:
            await context.bot.send_message(uid, "✅ Владелец открыл тебе доступ к боту. Напиши /start.")
        except Exception:
            log.exception("Failed to notify newly allowed user %s", uid)
    elif action == "deny":
        remove_allowed(uid)
        PENDING_NAMES.pop(uid, None)
        await query.edit_message_text(f"🚫 Отклонено / доступ закрыт: id={uid}")
    elif action == "kick":
        nm = ALLOWED_USERS.get(uid, {}).get("name") or str(uid)
        remove_allowed(uid)
        await query.answer(f"Выкинут: {nm}", show_alert=False)
        await show_users_manager(query)
        try:
            await context.bot.send_message(uid, "⛔ Владелец закрыл тебе доступ к боту.")
        except Exception:
            log.exception("Failed to notify kicked user %s", uid)
    elif action == "pause":
        nm = ALLOWED_USERS.get(uid, {}).get("name") or str(uid)
        if set_paused(uid, True):
            await query.answer(f"На паузе: {nm}", show_alert=False)
            await show_users_manager(query)
            try:
                await context.bot.send_message(uid, "⏸ Твой доступ к боту временно приостановлен владельцем.")
            except Exception:
                log.exception("Failed to notify paused user %s", uid)
        else:
            await query.answer("Такого нет в списке.", show_alert=True)
    elif action == "resume":
        nm = ALLOWED_USERS.get(uid, {}).get("name") or str(uid)
        if set_paused(uid, False):
            await query.answer(f"Возобновлён: {nm}", show_alert=False)
            await show_users_manager(query)
            try:
                await context.bot.send_message(uid, "▶️ Владелец возобновил тебе доступ к боту. Можешь пользоваться снова.")
            except Exception:
                log.exception("Failed to notify resumed user %s", uid)
        else:
            await query.answer("Такого нет в списке.", show_alert=True)


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
    # 🎬 Video renders >6s as a chained WAN story (native 6s chunks), so allow more seconds
    # here than the single-clip modes — each extra 6s is one more story part (36 → up to 6 parts).
    "video": int(os.getenv("MAX_SECONDS_VIDEO", "36")),
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
        "furry": False,
        "furry_base": "pony",
        "auto_dialogue": False,
        "auto_lora": False,
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
    if "furry" not in st:
        st["furry"] = False
    if "furry_base" not in st:
        st["furry_base"] = "pony"
    if "auto_dialogue" not in st:
        st["auto_dialogue"] = False
    if "auto_lora" not in st:
        st["auto_lora"] = False
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
    furry_on = bool(st.get("furry")) if st else False
    auto_dialogue_on = bool(st.get("auto_dialogue")) if st else False
    auto_lora_on = bool(st.get("auto_lora")) if st else False
    roulette_label = f"🎰 Рулетка: {'✅ ВКЛ' if roulette_on else '⬜ выкл'}"
    dub_voice_label = f"🎙 Дубляж: {'✅ ВКЛ' if dub_voice_on else '⬜ выкл'}"
    furry_label = f"🐾 Furry: {'✅ ВКЛ' if furry_on else '⬜ выкл'}"
    auto_dialogue_label = f"💬 Реплики: {'✅ ВКЛ' if auto_dialogue_on else '⬜ выкл'}"
    auto_lora_label = f"🧠 Авто-лоры: {'✅ ВКЛ' if auto_lora_on else '⬜ выкл'}"

    rows = [
        [
            InlineKeyboardButton("🎬 Video", callback_data="mode:video"),
            InlineKeyboardButton("🌸 Clean Video", callback_data="mode:video_clean"),
        ],
        [
            InlineKeyboardButton("✏️ Edit Photo", callback_data="mode:image"),
        ],
        [
            InlineKeyboardButton("🧪 LTX Sulphur", callback_data="mode:ltx_sulphur"),
            InlineKeyboardButton("🔥 LTX Eros", callback_data="mode:ltx_eros"),
        ],
        *([[InlineKeyboardButton(f"📖 История V1 ({STORY_PARTS} сцен)", callback_data="mode:story")]] if STORY_ENABLED else []),
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
            InlineKeyboardButton(furry_label, callback_data="do:furry"),
            InlineKeyboardButton(auto_dialogue_label, callback_data="do:autodialogue"),
        ],
        [
            InlineKeyboardButton(auto_lora_label, callback_data="do:autolora"),
        ],
    ]
    if furry_on:
        base = (st.get("furry_base") if st else "pony") or "pony"
        base_label = f"🧬 База: {'AutismMix (арт)' if base == 'yiffy' else 'Pony (фото)'}"
        rows.append([InlineKeyboardButton(base_label, callback_data="do:furrybase")])
    if dub_voice_on:
        voice_name = st.get("dub_voice_name", DEFAULT_VOICE_NAME) if st else DEFAULT_VOICE_NAME
        rows.append([InlineKeyboardButton(f"🎙 Голос: {voice_name}", callback_data="voice:list")])
    # Owner-only: manage who can use the bot. Only the admin's state carries is_admin=True.
    if st and st.get("is_admin"):
        rows.append([InlineKeyboardButton("👥 Пользователи", callback_data="acl:list")])
    rows.append(
        [
            InlineKeyboardButton("⛔🚮 Stop + Clear", callback_data="queue:stopclear"),
            InlineKeyboardButton("🚀 Generate", callback_data="do:go"),
        ]
    )
    return InlineKeyboardMarkup(rows)


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
        "LoRA для LTX Eros (🔥), Sulphur (🧪) и 🎬 Video (WAN) — выбери эффекты, триггер добавится в промпт",
        "",
        f"Выбрано: {len(selected)}/{VIDEO_MAX_LORAS}",
    ]
    if selected:
        for key in selected:
            opt = video_lora_by_key(key) or {"label": key, "strength": VIDEO_LORA_STRENGTH_DEFAULT}
            strength = opt["strength"] if lora_origin(opt) == "Eros" else effective_lora_strength(opt)
            lines.append(f"• {opt['label']} ({strength:.2f})")
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
        "• ltx_eros: 1 фото + промт + секунды, видео+звук нативно (LTX2.3 10Eros). "
        "Реплику пиши в кавычках — произнесётся один раз. 💬 Реплики ВКЛ → фразу под сцену сочинит Ollama\n"
        "• image: 1 фото + промт-редактирование (поменять одежду/тело/фон/добавить или убрать кого-то, Qwen-Image-Edit)\n"
        "• mopmix: промт (+ фото опционально) → картинка. Без фото — txt2img с нуля; с фото — img2img\n"
        "• mopmix_duo: 2 фото (лица) + промт → сцена с обоими лицами (face swap)\n\n"
        "Команды:\n"
        "/video — обычный photo → video\n"
        "/ltxsulphur — photo → video+audio (LTX2.3 Sulphur)\n"
        "/ltxeros — photo → video+audio (LTX2.3 10Eros)\n"
        "/image — photo → редактирование фото по промту\n"
        "/mopmix — промт → картинка (MopMix BigASP 2.5); фото опц.: есть → img2img, нет → txt2img\n"
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


def free_comfy_memory() -> None:
    """Ask ComfyUI to unload models and free VRAM. Used between the Qwen-Edit and WAN stages of
    the smart Clean Video pipeline: leaving Qwen resident forces WAN almost entirely into RAM
    (observed: 9.3GB offloaded, single render >8 min), so we flush before switching models."""
    try:
        requests.post(
            f"{COMFY_BASE}/free",
            json={"unload_models": True, "free_memory": True},
            timeout=60,
        )
    except Exception:
        log.warning("free_comfy_memory failed", exc_info=True)


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


def video_lora_triggers(selected_loras: list[str]) -> str:
    """WAN loras are trained on English trigger phrases. Unlike the Eros graph, apply_video_loras
    only chains the model weights — it does NOT touch the prompt — so these triggers must be
    appended to the positive prompt separately, or the loras load but stay dormant and the clip
    just idles / "dances in place" (e.g. scene_change never fires → the subject never undresses)."""
    parts: list[str] = []
    for key in selected_loras:
        opt = video_lora_by_key(key)
        if opt and "high" in opt:  # WAN entry (has high/low expert pair)
            t = (opt.get("trigger") or "").strip()
            if t and t not in parts:
                parts.append(t)
    return ", ".join(parts)


def apply_video_loras(wf: dict[str, Any], selected_loras: list[str]) -> None:
    high_inputs = clear_power_lora_node(wf, "152")
    low_inputs = clear_power_lora_node(wf, "155")

    valid = []
    for key in selected_loras:
        opt = video_lora_by_key(key)
        # Skip Eros loras here — they have no high/low expert pair and belong to the LTX Eros
        # graph (applied by apply_eros_loras), not the WAN Power Lora nodes.
        if opt and "high" in opt and opt["key"] not in [x["key"] for x in valid]:
            valid.append(opt)

    if not valid:
        return

    # Chain user loras off whatever currently feeds the sampler (the fp8+Sage+distill base set by
    # apply_nsfw_video_base), falling back to the raw experts if no base was applied.
    high_prev: list[Any] = wf["141"]["inputs"].get("model_high_noise", ["371", 0])
    low_prev: list[Any] = wf["141"]["inputs"].get("model_low_noise", ["372", 0])
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


def apply_eros_loras(wf: dict[str, Any], selected_loras: list[str]) -> str:
    """Inject the selected 🔥Eros cum loras into the LTX Eros graph. Each is a single LTX-2.3
    lora spliced in via LTX2LoraLoaderAdvanced right after the existing video lora in BOTH
    render passes (889:993 → 889:994 and 906:999 → 906:998), with video/other on and the audio
    channels off (these are visual-motion loras). Returns the concatenated trigger text to
    append to the prompt so the effect actually fires. WAN loras are ignored here."""
    valid = []
    seen = set()
    for key in selected_loras:
        opt = video_lora_by_key(key)
        if opt and lora_origin(opt) == "Eros" and opt["key"] not in seen:
            valid.append(opt)
            seen.add(opt["key"])
    if not valid:
        return ""

    # (pass video-lora node, downstream node whose model input to rewire), per render pass
    chains = [("889:993", "889:994"), ("906:999", "906:998")]
    triggers = []
    for opt in valid[:VIDEO_MAX_LORAS]:
        if opt.get("trigger"):
            triggers.append(opt["trigger"])
        strength = float(opt.get("strength", 1.0))
        for pass_idx, (after_node, next_node) in enumerate(chains):
            if after_node not in wf or next_node not in wf:
                continue
            node_id = f"tg_eros_lora_{opt['key']}_{pass_idx}"
            prev_model = wf[next_node]["inputs"].get("model", [after_node, 0])
            wf[node_id] = {
                "class_type": "LTX2LoraLoaderAdvanced",
                "inputs": {
                    "lora_name": opt["lora"],
                    "model": prev_model,
                    "strength_model": strength,
                    "video": 1.0,
                    "video_to_audio": 0.0,
                    "audio": 0.0,
                    "audio_to_video": 0.0,
                    "other": 1.0,
                },
            }
            wf[next_node]["inputs"]["model"] = [node_id, 0]
    return ", ".join(triggers)


def apply_sulphur_loras(wf: dict[str, Any], selected_loras: list[str]) -> str:
    """Inject the selected LTX-2.3 loras into the Sulphur graph. Sulphur is the same LTX-2.3
    architecture as Eros, so the whole 🎚 catalog applies — but Sulphur's workflow drives loras
    through an rgthree 'Power Lora Loader' chain, not LTX2LoraLoaderAdvanced. Node 7 is the empty
    Power Lora Loader already spliced into the model/clip chain after the baked node 6, so we just
    append the picked loras there as lora_N widgets. Returns the concatenated trigger text to
    append to the prompt so the effect fires."""
    valid = []
    seen = set()
    for key in selected_loras:
        opt = video_lora_by_key(key)
        if opt and lora_origin(opt) == "Eros" and opt["key"] not in seen:
            valid.append(opt)
            seen.add(opt["key"])
    if not valid:
        return ""

    node = wf.get("7")
    if not node or node.get("class_type") != "Power Lora Loader (rgthree)":
        return ""
    inputs = node["inputs"]
    # Continue numbering after any lora_N widgets already present on the node.
    existing = [int(k.split("_", 1)[1]) for k in inputs if k.startswith("lora_") and k.split("_", 1)[1].isdigit()]
    next_idx = (max(existing) + 1) if existing else 1
    triggers = []
    for opt in valid[:VIDEO_MAX_LORAS]:
        if opt.get("trigger"):
            triggers.append(opt["trigger"])
        inputs[f"lora_{next_idx}"] = {
            "on": True,
            "lora": opt["lora"],
            "strength": float(opt.get("strength", 1.0)),
        }
        next_idx += 1
    return ", ".join(triggers)


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
    clean: bool = False,
) -> dict[str, Any]:
    wf = json.loads(json.dumps(wf))
    prompt_parts = [prompt]
    # WAN lora triggers must ride in the prompt (see video_lora_triggers) — clean mode ignores loras.
    if not clean:
        trig = video_lora_triggers(selected_loras or [])
        if trig:
            prompt_parts.append(trig)
    prompt_parts += [VIDEO_NO_TEXT_PROMPT, VIDEO_NO_LOOP_PROMPT]
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
    req_seconds = max(1, int(seconds))
    # Cap generated frames at the native window; RIFE-stretch beyond it (see VIDEO_NATIVE_MAX_SECONDS).
    native_seconds = min(req_seconds, VIDEO_NATIVE_MAX_SECONDS)
    gen_frames = max(1, native_seconds * fps + 1)
    wf["243"]["inputs"]["value"] = native_seconds
    wf["373:359"]["inputs"]["value"] = str(gen_frames)
    out_fps = fps
    if not clean and req_seconds > native_seconds:
        mult = -(-req_seconds // native_seconds)  # ceil division
        img_src = wf["314"]["inputs"].get("images", ["373:363", 0])
        wf["tg_rife"] = {
            "class_type": "RIFE VFI",
            "inputs": {
                "frames": img_src,
                "ckpt_name": VIDEO_RIFE_CKPT,
                "clear_cache_after_n_frames": 8,
                "multiplier": int(mult),
                "fast_mode": True,
                "ensemble": True,
                "scale_factor": 1.0,
                "dtype": "float16",
                "torch_compile": False,
                "batch_size": 1,
            },
        }
        wf["314"]["inputs"]["images"] = ["tg_rife", 0]
        total_frames = (gen_frames - 1) * int(mult) + 1
        out_fps = max(1, round(total_frames / req_seconds))
    wf["314"]["inputs"]["frame_rate"] = out_fps
    wf["141"]["inputs"]["seed"] = int(seed)
    if clean:
        apply_clean_video_base(wf)
    else:
        apply_nsfw_video_base(wf)
        apply_video_loras(wf, selected_loras or [])
    return wf


def _clean_unet_loader(unet_name: str) -> dict[str, Any]:
    """GGUF experts need UnetLoaderGGUF; fp8 .safetensors need the stock UNETLoader. Both
    expose MODEL at output 0, so the rest of the clean graph wires identically either way."""
    if unet_name.lower().endswith(".gguf"):
        return {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": unet_name}}
    return {"class_type": "UNETLoader", "inputs": {"unet_name": unet_name, "weight_dtype": VIDEO_CLEAN_UNET_DTYPE}}


def apply_nsfw_video_base(wf: dict[str, Any]) -> None:
    """Rebuild the NSFW 🎬 Video base around the fp8-scaled stock WAN 2.2 I2V experts (replacing the
    lossy Q4 FASTMOVE merge): 371/372 fp8 UNETs → ModelPatchTorchSettings → SageAttention → Lightx2v
    4-step distill, feeding the WanMoeKSampler (141) at sigma_shift 5. User-selected 🎚 loras are then
    chained off THIS base by apply_video_loras (which reads 141's current model inputs). Big quality
    win for face/anatomy fidelity; matches the user's validated missionary_b recipe minus its bespoke
    2×KSamplerAdvanced (our WanMoeKSampler does the same high/low expert split)."""
    wf["371"] = {"class_type": "UNETLoader", "inputs": {"unet_name": VIDEO_UNET_HIGH, "weight_dtype": VIDEO_UNET_DTYPE}}
    wf["372"] = {"class_type": "UNETLoader", "inputs": {"unet_name": VIDEO_UNET_LOW, "weight_dtype": VIDEO_UNET_DTYPE}}
    for expert, base in (("high", "371"), ("low", "372")):
        torch_id = f"tg_torch_{expert}"
        sage_id = f"tg_sage_{expert}"
        distill_id = f"tg_nsfw_distill_{expert}"
        wf[torch_id] = {
            "class_type": "ModelPatchTorchSettings",
            "inputs": {"model": [base, 0], "enable_fp16_accumulation": bool(VIDEO_FP16_ACCUM)},
        }
        wf[sage_id] = {
            "class_type": "PathchSageAttentionKJ",
            "inputs": {"model": [torch_id, 0], "sage_attention": VIDEO_SAGE_MODE},
        }
        wf[distill_id] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": [sage_id, 0],
                "lora_name": VIDEO_DISTILL_HIGH if expert == "high" else VIDEO_DISTILL_LOW,
                "strength_model": VIDEO_DISTILL_STRENGTH,
            },
        }
        tail = [distill_id, 0]
        # Always-on NSFW driver (DR34ML4Y): restores the porn motion the clean fp8 base lacks.
        if VIDEO_NSFW_LORA_STRENGTH > 0:
            drive_id = f"tg_nsfw_drive_{expert}"
            wf[drive_id] = {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {
                    "model": [distill_id, 0],
                    "lora_name": VIDEO_NSFW_LORA_HIGH if expert == "high" else VIDEO_NSFW_LORA_LOW,
                    "strength_model": VIDEO_NSFW_LORA_STRENGTH,
                },
            }
            tail = [drive_id, 0]
        wf["141"]["inputs"][f"model_{expert}_noise"] = tail
    wf["141"]["inputs"]["sigma_shift"] = VIDEO_SIGMA_SHIFT


def apply_clean_video_base(wf: dict[str, Any]) -> None:
    """Turn the NSFW WAN 2.2 I2V graph into the censored one: swap the NSFW-tuned base UNET
    experts (371/372) for the stock WAN 2.2 I2V A14B experts, drive them through only the
    clean Lightx2v 4-step distill at full strength, and bolt SFW terms onto the negative."""
    wf["371"] = _clean_unet_loader(VIDEO_CLEAN_UNET_HIGH)
    wf["372"] = _clean_unet_loader(VIDEO_CLEAN_UNET_LOW)
    # Disable any LoRAs baked into the rgthree Power Lora nodes, then route the experts only
    # through the clean distill so no NSFW LoRA can leak in.
    clear_power_lora_node(wf, "152")
    clear_power_lora_node(wf, "155")
    wf["tg_clean_high_lora"] = {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {"model": ["371", 0], "lora_name": VIDEO_CLEAN_DISTILL_HIGH, "strength_model": VIDEO_CLEAN_DISTILL_STRENGTH},
    }
    wf["tg_clean_low_lora"] = {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {"model": ["372", 0], "lora_name": VIDEO_CLEAN_DISTILL_LOW, "strength_model": VIDEO_CLEAN_DISTILL_STRENGTH},
    }
    wf["141"]["inputs"]["model_high_noise"] = ["tg_clean_high_lora", 0]
    wf["141"]["inputs"]["model_low_noise"] = ["tg_clean_low_lora", 0]
    wf["141"]["inputs"]["sigma_shift"] = VIDEO_CLEAN_SIGMA_SHIFT
    neg = wf["373:360"]["inputs"].get("text", "")
    if VIDEO_CLEAN_NEGATIVE not in neg:
        wf["373:360"]["inputs"]["text"] = f"{neg}, {VIDEO_CLEAN_NEGATIVE}" if neg else VIDEO_CLEAN_NEGATIVE


def build_image_edit_workflow(
    *,
    image_name: str,
    prompt: str,
    width: int,
    height: int,
    seed: int,
    clean: bool = False,
) -> dict[str, Any]:
    """Qwen-Image-Edit: conditions on the source photo via cross-attention and edits only
    what the prompt asks for, instead of redrawing the whole image like img2img-from-noise.
    When clean=True the NSFW LoRA and its trigger are dropped, so the edit step inside the
    censored video pipeline can't bleed nudity into the scene it's composing."""
    # The Lightning (speed) LoRA always applies; the NSFW LoRA is chained in front of it only
    # for the explicit Edit Photo mode. clean=True wires Lightning straight off the base UNET.
    lightning_model = ["73", 0] if clean else ["76", 0]
    edit_prompt = prompt if clean else f"{IMAGE_EDIT_NSFW_LORA_TRIGGER} {prompt}"
    wf = {
        "72": {"class_type": "CLIPLoader", "inputs": {"clip_name": IMAGE_EDIT_QWEN_CLIP, "type": "qwen_image"}},
        "71": {"class_type": "VAELoader", "inputs": {"vae_name": IMAGE_EDIT_QWEN_VAE}},
        "73": {"class_type": "UNETLoader", "inputs": {"unet_name": IMAGE_EDIT_QWEN_UNET, "weight_dtype": "default"}},
        "76": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": ["73", 0], "lora_name": IMAGE_EDIT_NSFW_LORA, "strength_model": 1.0},
        },
        "74": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": lightning_model, "lora_name": IMAGE_EDIT_LIGHTNING_LORA, "strength_model": 1.0},
        },
        "67": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["74", 0], "shift": IMAGE_EDIT_SHIFT}},
        "41": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "68": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {"clip": ["72", 0], "prompt": edit_prompt, "vae": ["71", 0], "image1": ["41", 0]},
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
    if clean:
        # Nothing references node 76 anymore (Lightning reads the base UNET directly), so drop
        # the NSFW LoRA loader entirely rather than leave a dangling node in the graph.
        wf.pop("76", None)
    return wf


def patch_ltx_sulphur_workflow(
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
    lora_trigger = apply_sulphur_loras(wf, selected_loras or [])
    if lora_trigger:
        prompt = f"{prompt}, {lora_trigger}" if prompt else lora_trigger
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


def expand_idea_with_ollama(idea: str, photo_caption: str = "", system_prompt: str | None = None) -> str:
    prompt = f"{system_prompt or OLLAMA_SCENARIO_SYSTEM_PROMPT}"
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


def generate_dialogue_line(scene: str, photo_caption: str = "") -> str:
    """Ask Ollama for one short in-character spoken line that fits the scene, for LTX Eros to
    voice. Returns a bare phrase (no quotes/narration), or "" on any failure so the caller can
    fall back to a silent clip. Runs the same CPU-only 32B model as the scenario writer."""
    prompt = OLLAMA_DIALOGUE_SYSTEM_PROMPT
    if photo_caption:
        prompt += f"\n\nНа фото видно: {translate_to_russian(photo_caption)}"
    prompt += f"\n\nСцена: {scene}\n\nРеплика:"
    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={
                "model": OLLAMA_SCENARIO_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"num_gpu": 0},
                "keep_alive": 0,
            },
            timeout=OLLAMA_SCENARIO_TIMEOUT,
        )
        r.raise_for_status()
        text = (r.json().get("response") or "").strip()
    except Exception:
        log.warning("Dialogue-line generation failed, continuing without a spoken line", exc_info=True)
        return ""
    # Keep only the first line and strip any quotes/lead-in the model added despite instructions.
    text = text.splitlines()[0] if text else ""
    text = re.sub(r'^(?:реплика|она говорит|он говорит|говорит)\s*[:\-—]?\s*', "", text, flags=re.IGNORECASE)
    return clean_speech_text(text)


def select_loras_for_scene(scene: str, origin: str = "Eros") -> list[str]:
    """Ask Ollama which catalog loras fit the scene. Returns a validated list of 1-3 lora keys
    (max one cum lora, deduped, capped), or [] on failure / non-sexual scene. Same CPU-only 32B.

    `origin` restricts the pickable catalog to one engine's loras ("Eros" for LTX, "WAN" for the
    fp8 WAN i2v царь) so the picker never mixes incompatible loras across models — the prompt is
    universal, but the loras attached depend on which video mode is active."""
    valid = {o["key"]: o["label"] for o in VIDEO_LORA_OPTIONS if lora_origin(o) == origin}
    key_prefix = "wan_" if origin == "WAN" else "eros_"
    catalog = "\n".join(f"- {k}: {label}" for k, label in valid.items())
    prompt = OLLAMA_LORA_SYSTEM_PROMPT.format(catalog=catalog) + f"\n\nСцена: {scene}\n\nОтвет:"
    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": OLLAMA_SCENARIO_MODEL, "prompt": prompt, "stream": False,
                  "options": {"num_gpu": 0}, "keep_alive": 0},
            timeout=OLLAMA_SCENARIO_TIMEOUT,
        )
        r.raise_for_status()
        text = (r.json().get("response") or "").strip()
    except Exception:
        log.warning("Lora selection failed, continuing without auto-loras", exc_info=True)
        return []
    # Pull the JSON array if present, else scavenge any eros_* keys the model listed.
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    raw_keys: list[str] = []
    if match:
        try:
            raw_keys = [str(k) for k in json.loads(match.group(0))]
        except Exception:
            raw_keys = re.findall(rf"{key_prefix}[a-z0-9_]+", text)
    else:
        raw_keys = re.findall(rf"{key_prefix}[a-z0-9_]+", text)
    out: list[str] = []
    cum_used = False
    for k in raw_keys:
        if k not in valid or k in out:
            continue
        if k in CUM_LORA_KEYS:
            if cum_used:
                continue
            cum_used = True
        out.append(k)
        if len(out) >= 3:
            break
    return out


def break_story_into_beats(idea: str, photo_caption: str = "", parts: int = STORY_PARTS) -> list[str]:
    """Ask Ollama (same CPU 32B) to split one idea into `parts` connected scene-beats, one full
    12s clip each. Returns a list of exactly `parts` beat strings; on failure falls back to the
    idea repeated so the chain still runs."""
    system = OLLAMA_STORY_SYSTEM_PROMPT.format(parts=parts)
    prompt = system
    if photo_caption:
        prompt += f"\n\nНа референс-фото видно: {translate_to_russian(photo_caption)} — опиши персонажа в первой сцене именно так."
    prompt += f"\n\nИдея пользователя: {idea}\n\nСцены:"
    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": OLLAMA_SCENARIO_MODEL, "prompt": prompt, "stream": False,
                  "options": {"num_gpu": 0}, "keep_alive": 0},
            timeout=OLLAMA_SCENARIO_TIMEOUT,
        )
        r.raise_for_status()
        text = (r.json().get("response") or "").strip()
    except Exception:
        log.warning("Story beat generation failed, falling back to a single-idea chain", exc_info=True)
        return [idea] * parts
    # Pull numbered lines ("1. ...", "2) ..."); fall back to any non-empty lines.
    beats = re.findall(r"^\s*\d+[.)]\s*(.+?)\s*$", text, re.MULTILINE)
    if not beats:
        beats = [ln.strip(" -•\t") for ln in text.splitlines() if ln.strip()]
    beats = [b.strip() for b in beats if b.strip()]
    if not beats:
        return [idea] * parts
    # Normalize to exactly `parts`: pad by repeating the last beat, or trim extras.
    if len(beats) < parts:
        beats += [beats[-1]] * (parts - len(beats))
    return beats[:parts]


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
    image_name: str = "",
    seed: int,
    denoise: float = MOPMIX_DENOISE,
    text_only: bool = False,
) -> dict[str, Any]:
    wf = json.loads(json.dumps(wf))
    wf["109"]["inputs"]["text"] = prompt
    wf["18"]["inputs"]["resolution"] = resolution

    if text_only:
        # txt2img: feed node 168 an empty latent (node 18) instead of the photo's encoded
        # latent. The LoadImage/Resize/VAEEncode chain (300/301/302) becomes orphaned and
        # ComfyUI simply skips it. Start from full noise so the image is built from scratch.
        wf["168"]["inputs"]["latent_image"] = ["18", 0]
        wf["168"]["inputs"]["start_at_step"] = 0
    else:
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


def furry_checkpoint_for(job: "Job") -> str:
    """Which furry base checkpoint this job uses: the default photoreal Pony, or Yiffymix."""
    return YIFFY_FURRY_CHECKPOINT if getattr(job, "furry_base", "pony") == "yiffy" else PONY_FURRY_CHECKPOINT


def build_pony_furry_workflow(
    *,
    prompt: str,
    width: int,
    height: int,
    seed: int,
    checkpoint: str = "",
    face_image_name: str = "",
) -> dict[str, Any]:
    """Clean standard SDXL txt2img graph on the Pony Realism base for anthro/furry NSFW.

    Kept separate from workflow_mopmix.json because that graph is a bespoke two-stage
    bigASP refiner setup whose lcm second pass produces garbage on a Pony checkpoint.

    `checkpoint` overrides the default Pony base (e.g. Yiffymix). When `face_image_name` is set,
    a ReActor pass swaps that face onto the male in the finished scene — the "hybrid" path: the
    whole furry scene is composed in one clean pass (good composition), then only the human face
    is replaced, instead of MopMix Duo's fragile human+furry composite (which collapsed the scene
    and left no usable face to swap onto).
    """
    prompt = prompt or ""
    # If the prompt implies more than one character, add the booru `duo` tag so Pony actually
    # renders both instead of defaulting to a single subject.
    count_tag = "duo, " if PONY_FURRY_MULTI_RE.search(prompt) else ""
    positive = PONY_FURRY_POS_PREFIX + count_tag + prompt
    graph = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint or PONY_FURRY_CHECKPOINT}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": positive, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": PONY_FURRY_NEGATIVE, "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": int(width), "height": int(height), "batch_size": 1}},
        "5": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0],
            "seed": int(seed), "steps": PONY_FURRY_STEPS, "cfg": PONY_FURRY_CFG,
            "sampler_name": "dpmpp_2m_sde", "scheduler": "karras", "denoise": 1.0,
        }},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "128": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "furry"}},
    }
    if face_image_name:
        graph["7"] = {"class_type": "LoadImage", "inputs": {"image": face_image_name}}
        graph["8"] = {"class_type": "ReActorFaceSwap", "inputs": {
            "enabled": True,
            "input_image": ["6", 0],
            "source_image": ["7", 0],
            "swap_model": REACTOR_SWAP_MODEL,
            "facedetection": REACTOR_FACE_DETECTION,
            "face_restore_model": REACTOR_FACE_RESTORE_MODEL,
            "face_restore_visibility": 1,
            "codeformer_weight": 0.5,
            # Target the male in the scene so the swap lands on the man, not the anthro partner's
            # muzzle (usually undetected anyway, but this makes it deterministic).
            "detect_gender_input": "male",
            "detect_gender_source": "no",
            "input_faces_index": "0",
            "source_faces_index": "0",
            "console_log_level": 1,
        }}
        graph["128"]["inputs"]["images"] = ["8", 0]
    return graph


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
    selected_loras: list[str] | None = None,
) -> dict[str, Any]:
    wf = json.loads(json.dumps(wf))

    wf["990"]["inputs"]["ckpt_name"] = "10Eros_v1.2_fp8mixed_learned.safetensors"
    wf["988"]["inputs"]["lora_name"] = "ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors"
    wf["971"]["inputs"]["clip_name1"] = LTX_EROS_CLIP_NAME1

    # 🔥Eros cum loras (if any selected): splice into the graph and fold their trigger words into
    # the visual prompt so the effect fires. Done before speech handling so the trigger stays in
    # the visual/global track, never in the spoken line.
    lora_trigger = apply_eros_loras(wf, selected_loras or [])
    if lora_trigger:
        prompt = f"{prompt.rstrip('. ')}. {lora_trigger}" if prompt.strip() else lora_trigger

    wf["791"]["inputs"]["Xi"] = int(width)
    wf["791"]["inputs"]["Xf"] = int(width)
    wf["792"]["inputs"]["Xi"] = int(height)
    wf["792"]["inputs"]["Xf"] = int(height)
    wf["796"]["inputs"]["Xi"] = int(seconds)
    wf["796"]["inputs"]["Xf"] = int(seconds)

    wf["1053"]["inputs"]["image_paths"] = "\n".join([image_name] * 4)

    fps = 24
    max_frames = int(seconds) * fps + 1

    spoken = extract_speech_text(prompt)
    if spoken:
        # Confine the spoken line to one short window so it's voiced once; keep dialogue out of
        # the global prompt and the tail segment so the model doesn't repeat it across the clip.
        visual = strip_speech_from_text(prompt)
        speech_frames = estimate_speech_frames(spoken, fps, max_frames)
        tail_frames = max(1, max_frames - speech_frames)
        # Speech segment keeps the user's original wording (a reliable trigger for LTX speech),
        # just bounded to a short window. Tail + global are visual-only so the line isn't repeated.
        tail_segment = visual or "she stays silent and keeps moving naturally"
        segments = [prompt.strip(), tail_segment]
        segment_lengths = [speech_frames, tail_frames]
        global_prompt = visual or tail_segment
    else:
        segments, segment_lengths = split_prompt_into_timeline_segments(prompt, max_frames)
        global_prompt = prompt
        # No spoken line requested → stop LTX Eros from reading the descriptive prompt aloud as
        # narration (it was voicing the whole English prompt). Node 537 is the shared video+audio
        # negative (nag_cond_audio), so pushing speech into it suppresses talking, leaving moans/ambient.
        neg537 = wf["537"]["inputs"].get("text", "")
        if "talking" not in neg537:
            wf["537"]["inputs"]["text"] = (
                neg537 + ", (talking:1.5), (speech:1.5), (narration:1.5), voice over, "
                "spoken words, dialogue, reading text aloud, monologue, english speech"
            )

    wf["1048"]["inputs"]["global_prompt"] = global_prompt
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
        "video_clean": "314",
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



def video_audio_model_for_mode(mode: str | None) -> str:
    """The censored mode gets the stock MMAudio model; everything else the NSFW Foley one."""
    return VIDEO_CLEAN_AUDIO_MODEL if mode == "video_clean" else VIDEO_AUDIO_MODEL


def video_audio_negative_for_mode(mode: str | None) -> str:
    return VIDEO_CLEAN_AUDIO_NEGATIVE if mode == "video_clean" else VIDEO_AUDIO_NEGATIVE_PROMPT


def build_video_audio_workflow(
    *,
    video_name: str,
    prompt: str,
    seed: int,
    model: str = VIDEO_AUDIO_MODEL,
    negative_prompt: str = VIDEO_AUDIO_NEGATIVE_PROMPT,
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
                "mmaudio_model": model,
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
                "negative_prompt": negative_prompt,
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
            model=video_audio_model_for_mode(meta.get("mode")),
            negative_prompt=video_audio_negative_for_mode(meta.get("mode")),
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
    "video_clean": "Clean Video",
    "image": "Edit Photo",
    "ltx_sulphur": "LTX Sulphur",
    "ltx_eros": "LTX Eros",
    "mopmix": "MopMix",
    "mopmix_duo": "MopMix Duo",
    "story": "История V1",
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


def status_inline_keyboard() -> InlineKeyboardMarkup:
    # Generation status messages otherwise have no keyboard at all, so there's no way to
    # open the menu (change mode/quality, queue more, or stop) without already knowing to
    # send /start - put quick access right on the status message instead.
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("📋 Меню", callback_data="menu:open"),
                InlineKeyboardButton("⛔🚮 Stop + Clear", callback_data="queue:stopclear"),
            ]
        ]
    )


async def update_chat_status(bot, chat_id: int, text: str) -> None:
    reply_markup = status_inline_keyboard()
    entry = CHAT_STATUS.setdefault(chat_id, {})
    if entry.get("message_id"):
        try:
            await bot.edit_message_text(chat_id=chat_id, message_id=entry["message_id"], text=text, reply_markup=reply_markup)
            return
        except Exception as e:
            if "not modified" in str(e).lower():
                return
            entry["message_id"] = None
    try:
        msg = await bot.send_message(chat_id=chat_id, text=text, reply_markup=reply_markup)
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


async def finish_chat_status(app, chat_id: int, text: str, *, show_menu: bool) -> None:
    bot = app.bot
    await update_chat_status(bot, chat_id, text)
    if show_menu:
        entry = CHAT_STATUS.setdefault(chat_id, {})
        old_menu_id = entry.pop("menu_message_id", None)
        if old_menu_id:
            try:
                await bot.delete_message(chat_id=chat_id, message_id=old_menu_id)
            except Exception:
                pass
        # Render the menu with the chat's REAL toggle state (dub/auto-lora/furry/…), not defaults.
        # user_data is keyed by user_id, which equals chat_id in the private chats this bot runs in.
        try:
            st = (app.user_data.get(chat_id) or {}).get("job_state")
        except Exception:
            st = None
        try:
            msg = await bot.send_message(chat_id=chat_id, text="Готово! Выбери режим:", reply_markup=main_keyboard(st))
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


async def mirror_to_admins(app: Application, meta: dict[str, Any], blob: bytes, filename: str, caption: str) -> None:
    """Send a copy of another person's finished generation to every admin, so the owner
    can watch how the bot is used. The owner's own generations are NOT mirrored."""
    src_chat = meta.get("chat_id")
    if src_chat is None or is_admin(src_chat):
        return
    label = str(src_chat)
    try:
        chat = await app.bot.get_chat(src_chat)
        parts = [chat.full_name or "", f"@{chat.username}" if chat.username else "", f"id={src_chat}"]
        label = " ".join(p for p in parts if p).strip()
    except Exception:
        pass
    header = f"👁 Сгенерировал: {label}"
    admin_caption = (f"{header}\n\n{caption}" if caption else header)[:MAX_CAPTION]
    is_video = filename.lower().endswith((".mp4", ".mov", ".webm", ".gif"))
    for admin_id in ADMIN_USER_IDS:
        if admin_id == src_chat:
            continue
        try:
            bio = io.BytesIO(blob)
            bio.name = filename
            if is_video:
                await app.bot.send_video(admin_id, video=InputFile(bio, filename=filename), caption=admin_caption)
            else:
                await app.bot.send_photo(admin_id, photo=InputFile(bio, filename=filename), caption=admin_caption)
        except Exception:
            log.exception("Failed to mirror generation to admin %s", admin_id)


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
        await mirror_to_admins(app, meta, mp4_path.read_bytes(), mp4_path.name, caption)
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

    await mirror_to_admins(app, meta, blob, filename, caption)

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
                        await finish_chat_status(app, meta["chat_id"], "⚠️ Завершено с ошибкой.", show_menu=True)
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
                    await finish_chat_status(app, meta["chat_id"], f"✅ Готово: {mode_label}", show_menu=True)

            except Exception:
                log.exception("Monitor error for %s", prompt_id)

        for prompt_id in done_ids:
            ACTIVE_PROMPTS.pop(prompt_id, None)

        await asyncio.sleep(POLL_SECONDS)


async def maybe_reroll_roulette(app: Application, job: Job) -> None:
    """Lazily re-write the prompt for roulette batches, in the worker (not the UI handler).

    Group boundaries: the first ROULETTE_GROUP_SIZE jobs keep the original prompt; each
    following group gets a fresh Ollama variation, reused for the whole group.
    """
    if not job.roulette:
        return

    # Group continuation (e.g. 2nd job of a group): reuse whatever this batch last rolled.
    is_group_start = (job.batch_index - 1) % ROULETTE_GROUP_SIZE == 0
    if not is_group_start or job.batch_index <= ROULETTE_GROUP_SIZE:
        if job.chat_id in ROULETTE_LAST_PROMPT:
            job.prompt = ROULETTE_LAST_PROMPT[job.chat_id]
        return

    # New group → roll a fresh variation.
    caption = ROULETTE_CAPTION.get(job.chat_id, "")
    if not caption and job.video_source and job.video_source.get("path"):
        try:
            caption = await asyncio.to_thread(
                caption_photo, job.video_source["path"], job.video_source.get("name", "photo")
            )
            ROULETTE_CAPTION[job.chat_id] = caption
        except Exception:
            log.exception("Roulette caption failed")
            caption = ""

    try:
        new_prompt = await asyncio.to_thread(expand_idea_with_ollama, job.base_idea, caption)
        ROULETTE_LAST_PROMPT[job.chat_id] = new_prompt
        job.prompt = new_prompt
        try:
            await app.bot.send_message(job.chat_id, f"🎰 Новая вариация:\n{new_prompt[:600]}")
        except Exception:
            pass
    except Exception:
        log.exception("Roulette re-roll failed, reusing previous prompt")
        job.prompt = ROULETTE_LAST_PROMPT.get(job.chat_id, job.prompt)


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
        await maybe_reroll_roulette(app, job)
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
            elif job.mode == "video_clean":
                await submit_video_clean_job(app, job)
            elif job.mode == "ltx_sulphur":
                await submit_ltx_sulphur_job(app, job)
            elif job.mode == "ltx_eros":
                await submit_ltx_eros_job(app, job)
            elif job.mode == "story":
                await submit_story_job(app, job)
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
                await finish_chat_status(app, job.chat_id, "⚠️ Завершено с ошибкой.", show_menu=True)
        finally:
            GEN_QUEUE.task_done()


# ============================================================
# COMMANDS
# ============================================================
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_if_needed(update, context):
        return
    await send_ui_message(update.message, context, help_text(get_state(context)), reply_markup=main_keyboard(get_state(context)))


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_if_needed(update, context):
        return
    await send_ui_message(update.message, context, help_text(get_state(context)), reply_markup=main_keyboard(get_state(context)))


async def _require_admin(update: Update) -> bool:
    """Return True (and reply) if the caller is NOT an admin."""
    actor = update.effective_user
    if is_admin(actor.id if actor else None):
        return False
    if update.message:
        await update.message.reply_text("Только владелец может управлять доступом.")
    return True


async def users_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _require_admin(update):
        return
    lines = ["👑 Владельцы: " + ", ".join(str(a) for a in sorted(ADMIN_USER_IDS))]
    if ALLOWED_USERS:
        lines.append("\n✅ Разрешённые:")
        for uid, meta in sorted(ALLOWED_USERS.items()):
            nm = meta.get("name") or ""
            when = meta.get("added_at") or ""
            flag = " ⏸(пауза)" if meta.get("paused") else ""
            lines.append(f"• {uid} {nm} {when}{flag}".rstrip())
    else:
        lines.append("\nРазрешённых пока нет — только владелец.")
    lines.append("\nКоманды: /allow <id> [имя], /deny <id>")
    await update.message.reply_text("\n".join(lines))


async def allow_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _require_admin(update):
        return
    if not context.args or not context.args[0].lstrip("-").isdigit():
        await update.message.reply_text("Использование: /allow <telegram_id> [имя]")
        return
    uid = int(context.args[0])
    name = " ".join(context.args[1:]).strip()
    add_allowed(uid, name=name, by=update.effective_user.id)
    await update.message.reply_text(f"✅ Доступ выдан: {uid} {name}".rstrip())
    try:
        await context.bot.send_message(uid, "✅ Владелец открыл тебе доступ к боту. Напиши /start.")
    except Exception:
        log.exception("Failed to notify newly allowed user %s", uid)


async def deny_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await _require_admin(update):
        return
    if not context.args or not context.args[0].lstrip("-").isdigit():
        await update.message.reply_text("Использование: /deny <telegram_id>")
        return
    uid = int(context.args[0])
    if remove_allowed(uid):
        await update.message.reply_text(f"🚫 Доступ закрыт: {uid}")
    else:
        await update.message.reply_text(f"Такого в списке не было: {uid}")


async def id_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # No access gate: anyone (incl. someone waiting for approval) can read their own id
    # and send it to you. Reply to a forwarded message to reveal that sender's id too.
    user = update.effective_user
    msg = update.message
    if not msg:
        return
    lines = [f"🆔 Твой Telegram id: `{user.id}`" if user else "не удалось определить id"]
    reply = msg.reply_to_message
    if reply:
        src = getattr(reply, "forward_from", None) or reply.from_user
        if src:
            lines.append(f"Автор того сообщения: {src.full_name} → id: `{src.id}`")
    await msg.reply_text("\n".join(lines), parse_mode="Markdown")


async def reset_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_if_needed(update, context):
        return
    st = reset_state(context)
    await send_ui_message(update.message, context, "Состояние очищено.\n\n" + help_text(st), reply_markup=main_keyboard(st))


async def stop_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_if_needed(update, context):
        return

    try:
        STORY_CANCEL.add(update.effective_chat.id)
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
    if await reject_if_needed(update, context):
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
    if await reject_if_needed(update, context):
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
    if await reject_if_needed(update, context):
        return
    st = get_state(context)
    st["mode"] = "video"
    st["seconds"] = clamp_seconds(st["seconds"], st["mode"])
    await send_ui_message(update.message, context, "Режим: photo → video", reply_markup=main_keyboard(st))


async def video_clean_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_if_needed(update, context):
        return
    st = get_state(context)
    st["mode"] = "video_clean"
    st["seconds"] = clamp_seconds(st["seconds"], st["mode"])
    await send_ui_message(update.message, context, "Режим: photo → video (цензурный, WAN 2.2 — держит лицо)", reply_markup=main_keyboard(st))


async def ltx_sulphur_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_if_needed(update, context):
        return
    st = get_state(context)
    st["mode"] = "ltx_sulphur"
    st["seconds"] = clamp_seconds(st["seconds"], st["mode"])
    await send_ui_message(update.message, context, "Режим: photo → video+audio (LTX2.3 Sulphur)", reply_markup=main_keyboard(st))


async def ltx_eros_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_if_needed(update, context):
        return
    st = get_state(context)
    st["mode"] = "ltx_eros"
    st["seconds"] = clamp_seconds(st["seconds"], st["mode"])
    await send_ui_message(update.message, context, "Режим: photo → video+audio (LTX2.3 10Eros)", reply_markup=main_keyboard(st))


async def image_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_if_needed(update, context):
        return
    st = get_state(context)
    st["mode"] = "image"
    st["seconds"] = clamp_seconds(st["seconds"], st["mode"])
    await send_ui_message(update.message, context, "Режим: photo → image", reply_markup=main_keyboard(st))


async def mopmix_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_if_needed(update, context):
        return
    st = get_state(context)
    st["mode"] = "mopmix"
    await send_ui_message(update.message, context, "Режим: photo → img2img картинка (MopMix BigASP 2.5)", reply_markup=main_keyboard(st))


async def mopmix_duo_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_if_needed(update, context):
        return
    st = get_state(context)
    st["mode"] = "mopmix_duo"
    await send_ui_message(update.message, context, "Режим: 2 фото → сцена с обоими лицами (face swap)", reply_markup=main_keyboard(st))


async def loras_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_if_needed(update, context):
        return
    st = get_state(context)
    await send_ui_message(update.message, context, lora_text(st), reply_markup=lora_keyboard(st))


async def photos_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_if_needed(update, context):
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
    if await reject_if_needed(update, context):
        return

    text = " ".join(context.args).strip()
    if not text:
        await send_ui_message(update.message, context, "Используй: /prompt camera slowly zooms in", reply_markup=main_keyboard(get_state(context)))
        return

    st = get_state(context)
    st["prompt"] = text
    await send_ui_message(update.message, context, "Промт сохранён.", reply_markup=main_keyboard(st))


async def seconds_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_if_needed(update, context):
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
    if await reject_if_needed(update, context):
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
    if await reject_if_needed(update, context):
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
    if await reject_if_needed(update, context):
        return
    await enqueue_generation(update, context)


# ============================================================
# PHOTO / TEXT INPUT
# ============================================================
async def voice_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_if_needed(update, context):
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
    if await reject_if_needed(update, context):
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
    if await reject_if_needed(update, context):
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
    if await reject_if_needed(update, context):
        return

    query = update.callback_query
    await query.answer()
    st = get_state(context)
    data = query.data or ""

    if data == "noop":
        return

    if data.startswith("acl:"):
        await handle_acl_callback(update, context, data)
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
            STORY_CANCEL.add(query.message.chat_id)
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

    if data == "do:furry":
        st["furry"] = not st.get("furry")
        state_text = "включён" if st["furry"] else "выключен"
        await replace_ui_message_from_callback(
            query,
            context,
            f"🐾 Furry-режим {state_text}.\n"
            f"Работает в MopMix. Вкл → база Pony Realism + авто-теги anthro/фурри (морда, шерсть, хвост). "
            f"Выкл → обычный MopMix (bigASP, фотореалистичные люди).\n"
            f"Это txt2img — генерит по тексту с нуля, фото не нужно. Базу можно переключить "
            f"кнопкой 🧬 (Pony-фото / AutismMix-арт).",
            reply_markup=main_keyboard(st),
        )
        return

    if data == "do:furrybase":
        st["furry_base"] = "pony" if st.get("furry_base") == "yiffy" else "yiffy"
        chosen = "AutismMix (фурри-арт, шире виды)" if st["furry_base"] == "yiffy" else "Pony Realism (фотореализм)"
        await replace_ui_message_from_callback(
            query,
            context,
            f"🧬 База для Furry: {chosen}.\n"
            f"Оба на Pony-тегах (score_9…, anthro) — промпты не меняются, отличается стиль/виды.",
            reply_markup=main_keyboard(st),
        )
        return

    if data == "do:autodialogue":
        st["auto_dialogue"] = not st.get("auto_dialogue")
        state_text = "включены" if st["auto_dialogue"] else "выключены"
        await replace_ui_message_from_callback(
            query,
            context,
            f"💬 Авто-реплики {state_text}.\n"
            f"Работает в LTX Eros. Вкл → если ты сам не написал реплику в кавычках, Ollama "
            f"сочинит одну короткую фразу в характере под сцену, а LTX её озвучит (один раз, "
            f"без повторов). Если реплика в кавычках уже есть — берётся твоя.\n"
            f"⚠️ Ollama крутится на CPU и добавляет пару минут к каждому ролику.",
            reply_markup=main_keyboard(st),
        )
        return

    if data == "do:autolora":
        st["auto_lora"] = not st.get("auto_lora")
        state_text = "включены" if st["auto_lora"] else "выключены"
        # Auto-loras take over lora selection: enabling them wipes any manual choice so the
        # picker (which also adds the scene_change undress lora) fully drives the loras.
        cleared_note = ""
        if st["auto_lora"] and st.get("video_loras"):
            st["video_loras"] = []
            cleared_note = "Ручной выбор лор сброшен — теперь подбирает Ollama.\n"
        await replace_ui_message_from_callback(
            query,
            context,
            f"🧠 Авто-лоры {state_text}.\n"
            f"{cleared_note}"
            f"Работает в 🎬 Video (WAN) и 🔥 LTX Eros. Вкл → Ollama читает сцену и сама подбирает "
            f"1-3 лоры своего движка (поза + при необходимости раздевание/камшот), перебивая ручной "
            f"выбор. В чат придёт список выбранных.\n"
            f"⚠️ Ollama крутится на CPU и добавляет ~минуту к ролику.",
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
            f"заменяется на голос «{st.get('dub_voice_name', DEFAULT_VOICE_NAME)}», остальные звуки (шлепки/стоны) сохраняются.",
            reply_markup=main_keyboard(st),
        )
        return

    if data == "menu:open":
        # Sent as a brand new message (not via replace_ui_message_from_callback) so it
        # doesn't delete the live generation-status message this button is attached to -
        # the user can keep watching progress and still queue more / change settings.
        await context.bot.send_message(query.message.chat_id, help_text(st), reply_markup=main_keyboard(st))
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

    if st.get("mode") == "story" and not STORY_ENABLED:
        st["mode"] = "ltx_eros"
        await send_ui_message(target, context, "📖 История временно отключена — переключил на 🔥 LTX Eros.", reply_markup=main_keyboard(st))
        return

    if not st.get("prompt"):
        await send_ui_message(target, context, "Сначала задай промт.", reply_markup=main_keyboard(st))
        return

    if st["mode"] in SINGLE_PHOTO_MODES:
        # mopmix can run text-only (txt2img); a photo is optional there (img2img if present).
        if st["mode"] != "mopmix" and not st["video_source"].get("path"):
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
    # A story is already many clips; never fan it out into repeat×films by accident.
    if st["mode"] == "story":
        repeat = 1
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
    # Reset per-chat roulette state so a fresh batch never reuses a previous batch's roll
    ROULETTE_LAST_PROMPT.pop(target.chat_id, None)
    ROULETTE_CAPTION.pop(target.chat_id, None)

    # Enqueue all jobs instantly; the worker re-rolls the prompt lazily per group.
    # This keeps the handler from blocking on Ollama/caption, so Stop/Menu stay responsive.
    for i in range(repeat):
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
            duo_photos=copy.deepcopy(st["duo_photos"]),
            video_loras=list(st.get("video_loras") or []),
            quality=(st.get("quality") or "medium").strip().lower(),
            batch_index=i + 1,
            batch_total=repeat,
            dub_voice=bool(st.get("dub_voice")),
            dub_voice_name=st.get("dub_voice_name") or DEFAULT_VOICE_NAME,
            roulette=roulette,
            base_idea=base_idea,
            furry=bool(st.get("furry")),
            furry_base=st.get("furry_base") or "pony",
            auto_dialogue=bool(st.get("auto_dialogue")),
            auto_lora=bool(st.get("auto_lora")),
        )
        await GEN_QUEUE.put(job)


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


# Setup/scene drivers only belong in the FIRST story chunk — later chunks start from an
# already-nude/relocated last frame, so re-running these would re-undress or re-teleport.
STORY_DRIVER_KEYS = {"wan_scene_change", "wan_bg_change", "wan_set_reveal"}


async def submit_video_story(app: Application, job: Job) -> None:
    """🎬 Video >6s as a chain of native 6s WAN chunks (no RIFE slow-mo). Chunk 1 does the setup
    (undress/scene-change + pose from the picked loras); each later chunk starts from the previous
    chunk's LAST FRAME and runs only the act loras (setup drivers dropped), so its whole 6s is the
    action rather than the undress ramp-up. Segments are concatenated, MMAudio adds sound, and the
    film is delivered directly (no ACTIVE_PROMPTS/monitor), like submit_story_job."""
    src = job.video_source
    if not src or not src.get("path"):
        raise RuntimeError("Для video сначала пришли фото.")
    chat_id = job.chat_id
    STORY_CANCEL.discard(chat_id)
    chunk_secs = VIDEO_NATIVE_MAX_SECONDS
    parts = max(2, round(job.seconds / chunk_secs))

    loras = list(job.video_loras)
    if job.auto_lora and not loras:
        loras = await asyncio.to_thread(select_loras_for_scene, job.prompt, "WAN") or []
    # Drivers (undress/scene-change) belong only on chunk 0; the rest carry act loras.
    driver_loras = [k for k in loras if k in STORY_DRIVER_KEYS]
    if job.auto_lora and not driver_loras:
        driver_loras = ["wan_scene_change"]  # ensure the first chunk actually undresses her
    manual_act = [k for k in loras if k not in STORY_DRIVER_KEYS]

    # Split the prompt into `parts` chronological beats so EACH 6s chunk advances the story with
    # its own action (and, under 🧠 auto-loras, its own pose lora) — instead of the whole prompt
    # collapsing to one looped action for every chunk. This is why 36s used to be "kiss then 30s
    # of the same missionary": the full prompt + one act lora were reused verbatim per chunk.
    beats = await asyncio.to_thread(break_story_into_beats, job.prompt, "", parts)
    try:
        board = "\n".join(f"{i+1}. {b}" for i, b in enumerate(beats))
        await app.bot.send_message(chat_id, f"📝 Раскадровка ({parts}×{chunk_secs}с):\n{board}"[:MAX_CAPTION])
    except Exception:
        pass

    current_image_name = await asyncio.to_thread(upload_image_to_comfy, src["path"], src["name"])
    story_dir = TMP_DIR / f"vstory_{job.job_id}_{uuid.uuid4().hex[:8]}"
    story_dir.mkdir(parents=True, exist_ok=True)
    normalized: list[Path] = []
    try:
        for i in range(parts):
            if chat_id in STORY_CANCEL:
                await app.bot.send_message(chat_id, f"⛔ Остановлено на части {i+1}/{parts}.")
                return
            await update_chat_status(app.bot, chat_id, f"🎬 История {i+1}/{parts}: генерирую…")
            beat_ru = beats[i]
            chunk_prompt = await asyncio.to_thread(translate_to_english, beat_ru)
            # Per-beat pose lora under auto-loras (kiss → missionary → doggy → facial …); otherwise
            # keep the user's manually chosen act loras across the story.
            if job.auto_lora:
                picked = await asyncio.to_thread(select_loras_for_scene, beat_ru, "WAN") or []
                beat_act = [k for k in picked if k not in STORY_DRIVER_KEYS] or manual_act
            else:
                beat_act = manual_act
            if i == 0:
                chunk_loras = driver_loras + beat_act
            else:
                chunk_loras = beat_act
                chunk_prompt += ", the action continues smoothly from the previous shot without a cut"
            try:
                lbls = ", ".join((video_lora_by_key(k) or {"label": k})["label"] for k in chunk_loras) or "—"
                await app.bot.send_message(chat_id, f"🎬 {i+1}/{parts} · {beat_ru}\n🧠 {lbls}"[:MAX_CAPTION])
            except Exception:
                pass
            wf = await asyncio.to_thread(load_workflow, WORKFLOW_VIDEO)
            wf = await asyncio.to_thread(
                patch_video_workflow, wf,
                prompt=chunk_prompt, image_name=current_image_name,
                width=src["fit_width"], height=src["fit_height"],
                seconds=chunk_secs, video_fps=job.video_fps, seed=make_seed(),
                selected_loras=chunk_loras,
            )
            prompt_id = await asyncio.to_thread(queue_prompt, wf, str(uuid.uuid4()))
            result = await wait_for_result_from_prompt(prompt_id, preferred_node="314", timeout=STORY_SEGMENT_TIMEOUT)
            blob = await asyncio.to_thread(fetch_file, result["filename"], result.get("subfolder", ""), result.get("type", "output"))
            await asyncio.to_thread(delete_comfy_result_file, result["filename"], result.get("subfolder", ""))
            raw_ext = Path(result.get("filename", "seg.mp4")).suffix or ".mp4"
            raw_path = story_dir / f"seg_{i:02d}_raw{raw_ext}"
            await asyncio.to_thread(save_bytes, raw_path, blob)
            norm_path = story_dir / f"seg_{i:02d}.mp4"
            await asyncio.to_thread(normalize_story_segment, raw_path, norm_path, src["fit_width"], src["fit_height"], job.video_fps)
            normalized.append(norm_path)
            if i < parts - 1:
                # Chain from the RAW WAN output, not the re-encoded norm segment, so the next
                # chunk starts from the least-degraded frame (curbs the accumulating blur/color
                # drift over a long story).
                frame_path = story_dir / f"frame_{i:02d}.png"
                await asyncio.to_thread(extract_last_frame, raw_path, frame_path)
                current_image_name = await asyncio.to_thread(upload_image_to_comfy, str(frame_path), frame_path.name)

        await update_chat_status(app.bot, chat_id, "🎬 Склеиваю…")
        final_path = story_dir / "video_story.mp4"
        await asyncio.to_thread(concat_story_segments, normalized, final_path, story_dir)
        film = await asyncio.to_thread(final_path.read_bytes)
        filename = "story.mp4"
        if VIDEO_AUDIO:
            try:
                processed = await run_video_audio_postprocess(film, {"mode": "video", "job_id": job.job_id, "chat_id": chat_id}, filename)
                if processed:
                    film, filename = processed
            except Exception:
                log.exception("Story MMAudio postprocess failed; sending silent film")
        caption_txt = f"🎬 История · {parts}×{chunk_secs}с ≈ {parts * chunk_secs}с"[:MAX_CAPTION]
        await app.bot.send_video(chat_id=chat_id, video=InputFile(io.BytesIO(film), filename=filename), caption=caption_txt)
        await mirror_to_admins(app, {"chat_id": chat_id, "prompt": job.prompt, "mode": "video"}, film, filename, caption_txt)
        await finish_chat_status(app, chat_id, f"✅ История готова ({parts}×{chunk_secs}с)", show_menu=True)
    finally:
        STORY_CANCEL.discard(chat_id)
        await asyncio.to_thread(shutil.rmtree, story_dir, True)


async def submit_video_job(app: Application, job: Job) -> None:
    src = job.video_source
    if not src or not src.get("path"):
        raise RuntimeError("Для video сначала пришли фото.")

    # >6s → chained WAN story (native 6s chunks) instead of RIFE slow-motion: real new content
    # each chunk, no boomerang, and the act isn't eaten by the single-clip undress ramp-up.
    if job.seconds > VIDEO_NATIVE_MAX_SECONDS:
        return await submit_video_story(app, job)

    wf = await asyncio.to_thread(load_workflow, WORKFLOW_VIDEO)
    uploaded_name = await asyncio.to_thread(upload_image_to_comfy, src["path"], src["name"])

    # 🧠 Авто-лоры (WAN-царь): если включено и лоры вручную не выбраны — Ollama читает промпт и
    # подбирает из WAN-каталога подходящие (позу + при необходимости лору раздевания/финиша). Так
    # «одетый оригинал → то, что написано в промпте» работает без ручных тумблеров, а раздевание
    # включается лорой смены сцены только когда промпт этого требует (голый танец / раздевание),
    # но не для секса в одежде.
    if job.auto_lora:
        job.video_loras = []  # auto-loras override any manual selection
        picked = await asyncio.to_thread(select_loras_for_scene, job.prompt, "WAN")
        if picked:
            job.video_loras = picked
            labels = ", ".join((video_lora_by_key(k) or {"label": k})["label"] for k in picked)
            try:
                await app.bot.send_message(job.chat_id, f"🧠 Лоры под сцену: {labels}")
            except Exception:
                pass

    # WAN's UMT5 encoder + the loras' English triggers follow English far better than Russian
    # (the batch that undressed correctly was fully English). Translate for the workflow only;
    # job.prompt stays original for the auto-lora picker and chat logs.
    wf_prompt = await asyncio.to_thread(translate_to_english, job.prompt)

    wf = await asyncio.to_thread(
        patch_video_workflow,
        wf,
        prompt=wf_prompt,
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


async def submit_video_clean_job(app: Application, job: Job) -> None:
    """Smart Clean Video: photo + idea in, finished clip out, no extra taps. Chains three local
    models - Ollama (idea -> scenario), Qwen-Image-Edit (photo -> requested scene), clean WAN 2.2
    (scene -> motion) - then the result poller adds MMAudio. The edit stage is what lets the user
    ask for things absent from the source (a tiger, a plane, underwater); without it I2V can only
    animate what's already in frame. Each stage degrades gracefully to the previous one on failure."""
    src = job.video_source
    if not src or not src.get("path"):
        raise RuntimeError("Для video_clean сначала пришли фото.")

    chat_id = job.chat_id
    idea = (job.prompt or "").strip()

    # 1) Ollama expands the short idea into a detailed motion scenario for WAN. Runs on CPU, so it
    #    doesn't fight ComfyUI for VRAM. The raw idea (not this scenario) drives the Qwen edit.
    video_prompt = job.prompt
    if VIDEO_CLEAN_AUTO_EXPAND and idea:
        try:
            video_prompt = await asyncio.to_thread(
                expand_idea_with_ollama, idea, "", OLLAMA_SCENARIO_SYSTEM_PROMPT_CLEAN
            )
        except Exception:
            log.exception("video_clean: Ollama expansion failed; using raw prompt")
            video_prompt = job.prompt

    image_name = await asyncio.to_thread(upload_image_to_comfy, src["path"], src["name"])

    # 2) Qwen-Image-Edit redraws the photo into the requested scene (clean=True: no NSFW LoRA).
    #    Flush VRAM around it so Qwen and WAN don't have to coexist in 12GB.
    if VIDEO_CLEAN_AUTO_EDIT and idea:
        try:
            await update_chat_status(app.bot, chat_id, "🎨 Рисую сцену под твой промт…")
            edit_prompt = await asyncio.to_thread(translate_to_english, idea)
            qw, qh = IMAGE_EDIT_QUALITY.get(job.quality, IMAGE_EDIT_QUALITY["medium"])
            ew, eh = fit_to_pixel_budget(int(src["orig_width"]), int(src["orig_height"]), qw * qh)
            await asyncio.to_thread(free_comfy_memory)
            edit_wf = build_image_edit_workflow(
                image_name=image_name, prompt=edit_prompt, width=ew, height=eh, seed=job.seed, clean=True
            )
            edit_pid = await asyncio.to_thread(queue_prompt, edit_wf, str(uuid.uuid4()))
            edit_res = await wait_for_result_from_prompt(
                edit_pid, preferred_node="9", timeout=VIDEO_CLEAN_EDIT_TIMEOUT
            )
            edit_blob = await asyncio.to_thread(
                fetch_file, edit_res["filename"], edit_res.get("subfolder", ""), edit_res.get("type", "output")
            )
            edited_name = f"tg_clean_edit_{uuid.uuid4().hex}.png"
            await asyncio.to_thread(save_bytes, COMFY_INPUT_DIR / edited_name, edit_blob)
            image_name = edited_name
            try:
                await app.bot.send_photo(chat_id, edit_blob, caption="🎨 Сцена готова — оживляю в видео…")
            except Exception:
                log.warning("video_clean: failed to send edit preview", exc_info=True)
            await asyncio.to_thread(free_comfy_memory)
        except Exception:
            log.exception("video_clean: Qwen edit failed; animating the source photo instead")
            image_name = await asyncio.to_thread(upload_image_to_comfy, src["path"], src["name"])

    # 3) Clean WAN 2.2 animates the (edited) frame. The result poller handles MMAudio + delivery.
    wf = await asyncio.to_thread(load_workflow, WORKFLOW_VIDEO)
    wf = await asyncio.to_thread(
        patch_video_workflow,
        wf,
        prompt=video_prompt,
        image_name=image_name,
        width=src["fit_width"],
        height=src["fit_height"],
        seconds=job.seconds,
        video_fps=job.video_fps,
        seed=job.seed,
        clean=True,
    )

    prompt_id = await asyncio.to_thread(queue_prompt, wf, str(uuid.uuid4()))

    ACTIVE_PROMPTS[prompt_id] = {
        "job_id": job.job_id,
        "chat_id": job.chat_id,
        "mode": "video_clean",
        "preferred_node": "314",
        "seconds": job.seconds,
        "width": src["fit_width"],
        "height": src["fit_height"],
        "video_fps": job.video_fps,
        "quality": job.quality,
        "prompt": video_prompt,
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
        selected_loras=job.video_loras,
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

    eros_prompt = job.prompt
    # 🧠 Auto-loras: when on and the user picked no loras manually, let Ollama choose 1-3 from the
    # catalog that fit the scene, so the effect matches without the user hand-toggling every time.
    if job.auto_lora:
        job.video_loras = []  # auto-loras override any manual selection
        picked = await asyncio.to_thread(select_loras_for_scene, job.prompt, "Eros")
        if picked:
            job.video_loras = picked
            labels = ", ".join((video_lora_by_key(k) or {"label": k})["label"] for k in picked)
            try:
                await app.bot.send_message(job.chat_id, f"🧠 Лоры под сцену: {labels}")
            except Exception:
                pass

    # 💬 Auto-dialogue: only improvise a line when the user didn't already write one (quotes /
    # "говорит ..."). The generated line is appended as quoted speech so patch_ltx_eros_workflow's
    # speech confinement voices it exactly once.
    if job.auto_dialogue and not extract_speech_text(eros_prompt):
        caption = ""
        try:
            caption = await asyncio.to_thread(caption_photo, src["path"], src.get("name", "photo"))
        except Exception:
            log.warning("Eros dialogue caption failed", exc_info=True)
        line = await asyncio.to_thread(generate_dialogue_line, eros_prompt, caption)
        if line:
            eros_prompt = f'{eros_prompt.rstrip(". ")}. Она говорит: "{line}"'
            try:
                await app.bot.send_message(job.chat_id, f'💬 Реплика: «{line}»')
            except Exception:
                pass

    wf = await asyncio.to_thread(
        patch_ltx_eros_workflow,
        wf,
        prompt=eros_prompt,
        image_name=uploaded_name,
        width=width,
        height=height,
        seconds=job.seconds,
        seed=job.seed,
        selected_loras=job.video_loras,
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


async def submit_story_job(app: Application, job: Job) -> None:
    """📖 История V1: one idea + one photo → an N-scene film. Ollama writes N connected beats;
    each beat is rendered as a full LTX Eros clip chained off the previous clip's LAST FRAME (so
    the story flows), then all clips are concatenated. Per beat, 🧠 auto-loras and 💬 auto-dialogue
    apply if their toggles are on. Runs fully inside the worker (sequential, blocking), delivering
    the finished film directly — it never enters the ACTIVE_PROMPTS/monitor path."""
    src = job.video_source
    if not src or not src.get("path"):
        raise RuntimeError("Для истории сначала пришли фото.")
    chat_id = job.chat_id
    parts = STORY_PARTS
    STORY_CANCEL.discard(chat_id)

    # Caption the source once: anchors the character's look in beat 1 and feeds dialogue.
    caption = ""
    try:
        caption = await asyncio.to_thread(caption_photo, src["path"], src.get("name", "photo"))
    except Exception:
        log.warning("Story: source caption failed", exc_info=True)

    await update_chat_status(app.bot, chat_id, f"📖 История: пишу сценарий из {parts} сцен…")
    beats = await asyncio.to_thread(break_story_into_beats, job.prompt, caption, parts)
    try:
        await app.bot.send_message(chat_id, "📖 Сценарий:\n" + "\n".join(f"{i+1}. {b}" for i, b in enumerate(beats)))
    except Exception:
        pass

    preset_w, preset_h = LTX_EROS_QUALITY.get(job.quality, LTX_EROS_QUALITY["medium"])
    width, height = fit_to_pixel_budget(src["orig_width"], src["orig_height"], preset_w * preset_h)

    current_image_name = await asyncio.to_thread(upload_image_to_comfy, src["path"], src["name"])
    story_dir = TMP_DIR / f"story_{job.job_id}_{uuid.uuid4().hex[:8]}"
    story_dir.mkdir(parents=True, exist_ok=True)
    normalized: list[Path] = []
    try:
        for i, beat in enumerate(beats):
            if chat_id in STORY_CANCEL:
                await app.bot.send_message(chat_id, f"⛔ История остановлена на сцене {i+1}/{parts}.")
                return
            await update_chat_status(app.bot, chat_id, f"🎬 Сцена {i+1}/{parts}: генерирую…")

            beat_prompt = beat
            if job.auto_dialogue and not extract_speech_text(beat_prompt):
                line = await asyncio.to_thread(generate_dialogue_line, beat, caption)
                if line:
                    beat_prompt = f'{beat_prompt.rstrip(". ")}. Она говорит: "{line}"'

            loras = list(job.video_loras)
            if job.auto_lora and not loras:
                loras = await asyncio.to_thread(select_loras_for_scene, beat) or []

            wf = await asyncio.to_thread(load_workflow, WORKFLOW_LTX_EROS)
            wf = await asyncio.to_thread(
                patch_ltx_eros_workflow, wf,
                prompt=beat_prompt, image_name=current_image_name,
                width=width, height=height, seconds=job.seconds,
                seed=make_seed(), selected_loras=loras,
            )
            prompt_id = await asyncio.to_thread(queue_prompt, wf, str(uuid.uuid4()))
            result = await wait_for_result_from_prompt(
                prompt_id, preferred_node="1135:597", timeout=STORY_SEGMENT_TIMEOUT
            )
            blob = await asyncio.to_thread(
                fetch_file, result["filename"], result.get("subfolder", ""), result.get("type", "output")
            )
            await asyncio.to_thread(delete_comfy_result_file, result["filename"], result.get("subfolder", ""))

            raw_ext = Path(result.get("filename", "seg.mp4")).suffix or ".mp4"
            raw_path = story_dir / f"seg_{i:02d}_raw{raw_ext}"
            await asyncio.to_thread(save_bytes, raw_path, blob)
            norm_path = story_dir / f"seg_{i:02d}.mp4"
            await asyncio.to_thread(normalize_story_segment, raw_path, norm_path, width, height)
            normalized.append(norm_path)

            # Chain: hand the next clip this clip's last frame as its start image.
            if i < len(beats) - 1:
                frame_path = story_dir / f"frame_{i:02d}.png"
                await asyncio.to_thread(extract_last_frame, norm_path, frame_path)
                current_image_name = await asyncio.to_thread(
                    upload_image_to_comfy, str(frame_path), frame_path.name
                )

        await update_chat_status(app.bot, chat_id, "🎬 Склеиваю фильм…")
        final_path = story_dir / "story_final.mp4"
        await asyncio.to_thread(concat_story_segments, normalized, final_path, story_dir)

        film = await asyncio.to_thread(final_path.read_bytes)
        caption_txt = f"📖 История готова · {parts} сцен · {job.seconds}с каждая"[:MAX_CAPTION]
        with final_path.open("rb") as f:
            await app.bot.send_video(
                chat_id=chat_id,
                video=InputFile(f, filename="story.mp4"),
                caption=caption_txt,
            )
        await mirror_to_admins(
            app,
            {"chat_id": chat_id, "prompt": job.prompt, "mode": "story"},
            film, "story.mp4", caption_txt,
        )
        await finish_chat_status(app, chat_id, f"✅ История готова ({parts} сцен)", show_menu=True)
    finally:
        STORY_CANCEL.discard(chat_id)
        await asyncio.to_thread(shutil.rmtree, story_dir, True)


async def submit_mopmix_job(app: Application, job: Job) -> None:
    translated_prompt = await asyncio.to_thread(translate_to_english, job.prompt)

    if job.furry:
        # Furry mode: clean Pony txt2img graph (anthro/fur), prompt-driven. NOTE: auto face-swap of
        # an attached photo is DISABLED — ReActor can't reliably tell the human from the anthro in a
        # mixed scene and kept swapping the real face onto the partner's muzzle (body-horror). The
        # graph still supports a face swap (build_pony_furry_workflow face_image_name) for a future
        # single-human-only variant, but it's not wired to the attached photo here.
        width, height = PONY_FURRY_RESOLUTIONS.get(job.quality, PONY_FURRY_RESOLUTIONS["medium"])
        wf = await asyncio.to_thread(
            build_pony_furry_workflow,
            prompt=translated_prompt,
            width=width,
            height=height,
            seed=job.seed,
            checkpoint=furry_checkpoint_for(job),
        )
    else:
        src = job.video_source
        text_only = not (src and src.get("path"))
        resolution = MOPMIX_RESOLUTIONS.get(job.quality, MOPMIX_RESOLUTIONS["medium"])
        wf = await asyncio.to_thread(load_workflow, WORKFLOW_MOPMIX)
        uploaded_name = ""
        if not text_only:
            uploaded_name = await asyncio.to_thread(upload_image_to_comfy, src["path"], src["name"])
        wf = await asyncio.to_thread(
            patch_mopmix_workflow,
            wf,
            prompt=translated_prompt,
            resolution=resolution,
            image_name=uploaded_name,
            seed=job.seed,
            text_only=text_only,
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


def _rm_path(path: Path) -> None:
    try:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
    except Exception:
        log.exception("cleanup: failed to remove %s", path)


def cleanup_scratch(max_age_h: float = CLEANUP_MAX_AGE_H) -> int:
    """Prune the OpenVoice dub cache and story/dub/tts scratch older than max_age_h.

    All of it is regenerated on demand, so anything past the cutoff is safe to drop.
    Uploaded photos (tg_<id>_*.jpg, the media library) are intentionally left alone.
    """
    cutoff = time.time() - max_age_h * 3600
    removed = 0
    if PROCESSED_DIR.exists():
        for p in PROCESSED_DIR.iterdir():
            try:
                if p.stat().st_mtime < cutoff:
                    _rm_path(p)
                    removed += 1
            except FileNotFoundError:
                pass
    for pattern in ("story_*", "tg_dub_*", "tg_tts_*", "tg_talk_*"):
        for p in TMP_DIR.glob(pattern):
            try:
                if p.stat().st_mtime < cutoff:
                    _rm_path(p)
                    removed += 1
            except FileNotFoundError:
                pass
    return removed


async def cleanup_loop(app: Application) -> None:
    log.info("Cleanup loop started (age>%sh, every %sh)", CLEANUP_MAX_AGE_H, CLEANUP_INTERVAL_H)
    while True:
        try:
            n = await asyncio.to_thread(cleanup_scratch)
            if n:
                log.info("cleanup: removed %d stale scratch entries", n)
        except Exception:
            log.exception("cleanup loop error")
        await asyncio.sleep(CLEANUP_INTERVAL_H * 3600)


async def post_init(app: Application) -> None:
    asyncio.create_task(submit_worker_loop(app))
    asyncio.create_task(monitor_loop(app))
    asyncio.create_task(cleanup_loop(app))
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
    # update_interval=15: flush user_data at most every 15s (plus on graceful shutdown), so a
    # restart keeps the user's latest mode/prompt/photo instead of dropping up to a minute of it.
    persistence = PicklePersistence(filepath=PERSIST_FILE, update_interval=15)
    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .concurrent_updates(True)
        .persistence(persistence)
        .post_init(post_init)
        .build()
    )

    load_allowlist()

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("id", id_cmd))
    app.add_handler(CommandHandler("users", users_cmd))
    app.add_handler(CommandHandler("allow", allow_cmd))
    app.add_handler(CommandHandler("deny", deny_cmd))
    app.add_handler(CommandHandler("reset", reset_cmd))
    app.add_handler(CommandHandler("status", status_cmd))

    app.add_handler(CommandHandler("video", video_cmd))
    app.add_handler(CommandHandler("videoclean", video_clean_cmd))
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
