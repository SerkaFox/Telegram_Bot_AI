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
