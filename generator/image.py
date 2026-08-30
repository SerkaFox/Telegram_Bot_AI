"""Safe image generation via the existing MopMix (SDXL bigASP) workflow.

This module is imported LAZILY (only when a real generation actually runs) so that protocol /
mock tests never need ComfyUI, python-telegram-bot, or a GPU. All heavy lifting reuses functions
already defined and battle-tested in telegram_comfyui_bot.py — we do not author a new workflow.
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Callable, Optional

# Reuse the running bot's ComfyUI client + workflow patchers. Importing the module does NOT start
# the bot (its polling lives under `if __name__ == "__main__"`); it only needs the deps the bot
# already has installed. Same pattern tools/complex_gen.py uses.
import telegram_comfyui_bot as b

_BOT_DIR = Path(b.__file__).resolve().parent
# Resolve the MopMix workflow to an absolute path so generation does not depend on cwd.
_MOPMIX_WORKFLOW = (_BOT_DIR / "workflow_mopmix.json")


def _png_dimensions(path: Path) -> tuple[Optional[int], Optional[int]]:
    """Read PNG width/height from the IHDR header (no PIL dependency). Returns (None, None) on any
    non-PNG / unreadable file so callers can still upload the artifact without dimensions."""
    try:
        with open(path, "rb") as f:
            head = f.read(26)
        if head[:8] != b"\x89PNG\r\n\x1a\n":
            return None, None
        return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")
    except Exception:
        return None, None


def _wait_for_gpu_gate(should_cancel: Optional[Callable[[], bool]], poll: float = 2.0) -> None:
    """Yield the single GPU to the Telegram bot: block while ComfyUI's own queue has anything
    running or pending. The bot is the only other producer, so a non-empty ComfyUI queue means an
    owner/manual job is in flight — FOX background work waits for it. Zero changes to the bot.

    ComfyUI serialises execution internally, so this is belt-and-suspenders for *priority*, not
    correctness. Bounded by nothing here on purpose (owner jobs always win); the caller's cancel
    check lets a pending FOX job bail out while it waits."""
    while True:
        if should_cancel and should_cancel():
            return
        try:
            q = b.get_queue_state()
            busy = bool(q.get("queue_running")) or bool(q.get("queue_pending"))
        except Exception:
            # If we cannot read the queue, do not hammer ComfyUI — treat as "clear" and let the
            # ComfyUI-side serialisation handle ordering.
            return
        if not busy:
            return
        time.sleep(poll)


def generate_mopmix_images(
    prompt: str,
    *,
    count: int = 1,
    quality: str = "medium",
    seed: Optional[int] = None,
    image_path: Optional[str] = None,
    out_dir: Path,
    timeout: int = 300,
    should_cancel: Optional[Callable[[], bool]] = None,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> list[Path]:
    """Render `count` safe images with the existing MopMix graph and write them into `out_dir`.

    txt2img when `image_path` is None, img2img when a photo is given (same as the bot). Honours a
    cooperative `should_cancel()` between batch items (finishes the current atomic render, does not
    start the next). Returns the list of written file paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    count = max(1, int(count))
    resolution = b.MOPMIX_RESOLUTIONS.get(quality, b.MOPMIX_RESOLUTIONS["medium"])
    text_only = not image_path

    english_prompt = b.translate_to_english(prompt)
    base_workflow = b.load_workflow(str(_MOPMIX_WORKFLOW))

    uploaded_name = ""
    if not text_only:
        src = Path(image_path)
        uploaded_name = b.upload_image_to_comfy(str(src), src.name)

    written: list[Path] = []
    for i in range(count):
        if should_cancel and should_cancel() and i > 0:
            # Cancel requested mid-batch: stop before starting the next atomic render.
            break

        item_seed = (int(seed) + i) if seed is not None else b.make_seed()
        wf = b.patch_mopmix_workflow(
            base_workflow,
            prompt=english_prompt,
            resolution=resolution,
            image_name=uploaded_name,
            seed=item_seed,
            text_only=text_only,
        )

        _wait_for_gpu_gate(should_cancel)
        if should_cancel and should_cancel() and i > 0:
            break

        prompt_id = b.queue_prompt(wf, str(uuid.uuid4()))

        # Synchronous poll for this prompt's result (the worker calls us in a thread, so blocking
        # here keeps the worker's async loop free for heartbeats).
        deadline = time.time() + timeout
        result = None
        while time.time() < deadline:
            history = b.get_history(prompt_id)
            item = history.get(prompt_id)
            if item and item.get("outputs"):
                result = b.pick_first_result_from_outputs(item["outputs"], preferred_node="128")
                break
            time.sleep(b.POLL_SECONDS)
        if result is None:
            raise TimeoutError(f"MopMix timed out (prompt_id={prompt_id})")

        blob = b.fetch_file(
            result["filename"],
            subfolder=result.get("subfolder", ""),
            file_type=result.get("type", "output"),
        )
        dest = out_dir / f"mopmix_{i:02d}_{result['filename']}"
        dest.write_bytes(blob)
        written.append(dest)

        # Free the ComfyUI copy so the shared output dir does not accumulate.
        try:
            b.delete_comfy_result_file(result["filename"], result.get("subfolder", ""))
        except Exception:
            pass

        if on_progress:
            pct = int(round((i + 1) / count * 100))
            on_progress(pct, f"rendered {i + 1}/{count}")

    if not written:
        raise RuntimeError("MopMix produced no images")
    return written


def _source_dims(path: Path) -> tuple[int, int]:
    """Source image dimensions via PIL (available in the bot venv); PNG-header fallback."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return int(im.width), int(im.height)
    except Exception:
        w, h = _png_dimensions(path)
        return (w or 1024, h or 1024)


def edit_qwen_images(
    instruction: str,
    source_path: str,
    *,
    count: int = 1,
    quality: str = "medium",
    seed: Optional[int] = None,
    out_dir: Path,
    timeout: int = 300,
    should_cancel: Optional[Callable[[], bool]] = None,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> list[Path]:
    """Edit `source_path` per `instruction` with the existing Qwen-Image-Edit graph (clean=True:
    no NSFW LoRA). Preserves the source aspect ratio, honours cooperative cancel between items."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    count = max(1, int(count))
    src = Path(source_path)
    if not src.is_file():
        raise FileNotFoundError(f"source image not found: {source_path}")

    if on_progress:
        on_progress(5, "loading")
    english = b.translate_to_english(instruction)
    sw, sh = _source_dims(src)
    qw, qh = b.IMAGE_EDIT_QUALITY.get(quality, b.IMAGE_EDIT_QUALITY["medium"])
    ew, eh = b.fit_to_pixel_budget(sw, sh, qw * qh)
    uploaded_name = b.upload_image_to_comfy(str(src), src.name)

    written: list[Path] = []
    for i in range(count):
        if should_cancel and should_cancel() and i > 0:
            break
        item_seed = (int(seed) + i) if seed is not None else b.make_seed()
        wf = b.build_image_edit_workflow(
            image_name=uploaded_name, prompt=english, width=ew, height=eh, seed=item_seed, clean=True,
        )
        _wait_for_gpu_gate(should_cancel)
        if should_cancel and should_cancel() and i > 0:
            break
        prompt_id = b.queue_prompt(wf, str(uuid.uuid4()))
        deadline = time.time() + timeout
        result = None
        while time.time() < deadline:
            item = b.get_history(prompt_id).get(prompt_id)
            if item and item.get("outputs"):
                result = b.pick_first_result_from_outputs(item["outputs"], preferred_node="9")
                break
            time.sleep(b.POLL_SECONDS)
        if result is None:
            raise TimeoutError(f"Qwen-Edit timed out (prompt_id={prompt_id})")
        blob = b.fetch_file(result["filename"], subfolder=result.get("subfolder", ""),
                            file_type=result.get("type", "output"))
        dest = out_dir / f"edit_{i:02d}_{result['filename']}"
        dest.write_bytes(blob)
        written.append(dest)
        try:
            b.delete_comfy_result_file(result["filename"], result.get("subfolder", ""))
        except Exception:
            pass
        if on_progress:
            on_progress(int(round((i + 1) / count * 100)), f"edited {i + 1}/{count}")
    if not written:
        raise RuntimeError("Qwen-Edit produced no images")
    return written
