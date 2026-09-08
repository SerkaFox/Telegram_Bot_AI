"""Safe Clean Video generation via the existing censored WAN 2.2 pipeline.

Reuses telegram_comfyui_bot.py building blocks (no new workflow): patch_video_workflow(clean=True)
to animate the source image, then the same MMAudio clean-Foley postprocess the bot uses, returning
one final normalized MP4. Imported lazily so mock/protocol tests never need ComfyUI or a GPU.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

import telegram_comfyui_bot as b

from .image import _source_dims, _wait_for_gpu_gate


def clean_video(
    prompt: str,
    source_path: str,
    *,
    quality: str = "medium",
    seconds: Optional[int] = None,
    seed: Optional[int] = None,
    out_dir: Path,
    timeout: int = 900,
    should_cancel: Optional[Callable[[], bool]] = None,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> list[Path]:
    """Animate `source_path` into a short clean clip described by `prompt` (SFW WAN 2.2 i2v), add
    clean MMAudio Foley, and write one final MP4 into out_dir. Duration is capped to the native
    window (no long batch). Returns [final_video_path]."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = Path(source_path)
    if not src.is_file():
        raise FileNotFoundError(f"source image not found: {source_path}")

    if on_progress:
        on_progress(5, "preparing")

    preset = b.QUALITY_PRESETS.get(quality, b.QUALITY_PRESETS["medium"])
    max_side, fps = int(preset["max_side"]), int(preset["video_fps"])
    sw, sh = _source_dims(src)
    fw, fh = b.fit_size_keep_aspect(sw, sh, max_side)
    # Short native duration only — never launch a long batch here.
    req_seconds = int(seconds) if seconds else b.VIDEO_NATIVE_MAX_SECONDS
    req_seconds = max(1, min(req_seconds, b.VIDEO_NATIVE_MAX_SECONDS))
    item_seed = int(seed) if seed is not None else b.make_seed()
    english = b.translate_to_english(prompt)

    uploaded = b.upload_image_to_comfy(str(src), src.name)
    wf = b.load_workflow(b.WORKFLOW_VIDEO)
    wf = b.patch_video_workflow(
        wf, prompt=english, image_name=uploaded, width=fw, height=fh,
        seconds=req_seconds, video_fps=fps, seed=item_seed, clean=True,
    )

    _wait_for_gpu_gate(should_cancel)
    if should_cancel and should_cancel():
        raise RuntimeError("cancelled before render")

    if on_progress:
        on_progress(15, "rendering")
    prompt_id = b.queue_prompt(wf, str(uuid.uuid4()))
    deadline = time.time() + timeout
    result = None
    while time.time() < deadline:
        item = b.get_history(prompt_id).get(prompt_id)
        if item and item.get("outputs"):
            result = b.pick_first_result_from_outputs(item["outputs"], preferred_node="314")
            break
        time.sleep(b.POLL_SECONDS)
    if result is None:
        raise TimeoutError(f"Clean video timed out (prompt_id={prompt_id})")

    silent_blob = b.fetch_file(result["filename"], subfolder=result.get("subfolder", ""),
                               file_type=result.get("type", "output"))
    try:
        b.delete_comfy_result_file(result["filename"], result.get("subfolder", ""))
    except Exception:
        pass

    # Clean MMAudio Foley (same model the bot uses for video_clean), returned as the final clip.
    if on_progress:
        on_progress(80, "audio")
    final_blob, final_name = silent_blob, result["filename"]
    try:
        meta = {"mode": "video_clean", "prompt": english}
        audio = asyncio.run(b.run_video_audio_postprocess(silent_blob, meta, result["filename"]))
        if audio:
            final_blob, final_name = audio
    except Exception:
        # Audio is best-effort; a silent clean clip is still a valid deliverable.
        pass

    if on_progress:
        on_progress(95, "encoding")
    dest = out_dir / f"clean_{final_name if final_name.endswith('.mp4') else final_name + '.mp4'}"
    dest.write_bytes(final_blob)
    return [dest]


def _build_talking_prompt(scene: str, dialogue: str) -> str:
    """Compose the LTX Sulphur prompt for a talking-head clip.

    The scene description drives the visuals; the dialogue is embedded verbatim as the exact spoken
    line so LTX-2.3's multilingual text encoder voices *those words* (RU / ES / EN alike) instead of
    improvising. FOX MIX owns the words — we never rewrite them here."""
    scene = (scene or "").strip()
    dialogue = (dialogue or "").strip()
    if not dialogue:
        return scene
    line = f'The character looks at the camera and clearly speaks these exact words aloud: "{dialogue}".'
    return f"{scene.rstrip('. ')}. {line}" if scene else line


def talking_video(
    prompt: str,
    source_path: str,
    *,
    dialogue_text: str = "",
    voice_mode: str = "native",
    voice_reference: Optional[str] = None,
    quality: str = "medium",
    seconds: Optional[int] = None,
    seed: Optional[int] = None,
    out_dir: Path,
    timeout: int = 900,
    should_cancel: Optional[Callable[[], bool]] = None,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> list[Path]:
    """SAFE talking-head / presenter clip from one source image via the existing LTX Sulphur graph.

    `prompt` describes the scene/gesture; `dialogue_text` is the exact spoken line (voiced natively
    by LTX-2.3). No loras are injected — this is strictly SFW, never LTX Eros / NSFW. With
    voice_mode="openvoice" the native track is re-timbred to a reference voice via the unchanged
    OpenVoice V2 pipeline; "native" leaves LTX's own voice. Returns [final_mp4]."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = Path(source_path)
    if not src.is_file():
        raise FileNotFoundError(f"source image not found: {source_path}")

    if on_progress:
        on_progress(5, "preparing")

    preset_w, preset_h = b.LTX_SULPHUR_QUALITY.get(quality, b.LTX_SULPHUR_QUALITY["medium"])
    sw, sh = _source_dims(src)
    width, height = b.fit_to_pixel_budget(sw, sh, preset_w * preset_h)
    max_seconds = b.MODE_MAX_SECONDS.get("ltx_sulphur", b.MAX_SECONDS)
    req_seconds = int(seconds) if seconds else b.DEFAULT_SECONDS
    req_seconds = max(1, min(req_seconds, max_seconds))
    item_seed = int(seed) if seed is not None else b.make_seed()

    full_prompt = _build_talking_prompt(prompt, dialogue_text)

    uploaded = b.upload_image_to_comfy(str(src), src.name)
    wf = b.load_workflow(b.WORKFLOW_LTX_SULPHUR)
    # selected_loras=[] → apply_sulphur_loras injects nothing: strictly SAFE, no Eros / NSFW lora.
    wf = b.patch_ltx_sulphur_workflow(
        wf, prompt=full_prompt, image_name=uploaded, width=width, height=height,
        seconds=req_seconds, seed=item_seed, selected_loras=[],
    )

    _wait_for_gpu_gate(should_cancel)
    if should_cancel and should_cancel():
        raise RuntimeError("cancelled before render")

    if on_progress:
        on_progress(15, "rendering")
    prompt_id = b.queue_prompt(wf, str(uuid.uuid4()))
    deadline = time.time() + timeout
    result = None
    while time.time() < deadline:
        item = b.get_history(prompt_id).get(prompt_id)
        if item and item.get("outputs"):
            result = b.pick_first_result_from_outputs(item["outputs"], preferred_node="61")
            break
        time.sleep(b.POLL_SECONDS)
    if result is None:
        raise TimeoutError(f"Talking video timed out (prompt_id={prompt_id})")

    blob = b.fetch_file(result["filename"], subfolder=result.get("subfolder", ""),
                        file_type=result.get("type", "output"))
    try:
        b.delete_comfy_result_file(result["filename"], result.get("subfolder", ""))
    except Exception:
        pass

    dest = out_dir / f"talking_{result['filename'] if result['filename'].endswith('.mp4') else result['filename'] + '.mp4'}"
    dest.write_bytes(blob)

    # voice_mode=openvoice: re-timbre LTX's native track to a reference voice via the unchanged
    # OpenVoice V2 pipeline. Best-effort — a failure keeps the valid native-voice clip.
    if str(voice_mode or "").strip().lower() == "openvoice":
        if on_progress:
            on_progress(88, "openvoice")
        voice_name = (voice_reference or "").strip() or b.DEFAULT_VOICE_NAME
        if b.voice_path(voice_name) is None:
            raise ValueError(f"voice_reference '{voice_name}' not found in {b.VOICES_DIR}")
        dubbed = out_dir / f"talking_openvoice_{dest.stem}.mp4"
        b.dub_voice_in_video(dest, dubbed, voice_name)
        try:
            dest.unlink(missing_ok=True)
        except Exception:
            pass
        dest = dubbed

    if on_progress:
        on_progress(95, "encoding")
    return [dest]
