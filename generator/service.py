"""Generation service layer with a hard SAFE / ADULT split.

Two separate service classes, on purpose:

  * SafeGeneratorService  — content_class="safe" jobs only. Phase-1 wires exactly one capability:
                            safe.image.mopmix (text2img / img2img via the existing MopMix graph).
  * AdultGeneratorService — every method refuses with UnsupportedInCurrentPhase. Adult generation is
                            intentionally NOT connected to FOX in this phase.

Routing to the right service is done by the caller (fox_worker) from the job's explicit
`content_class`, never by an implicit condition inside one shared method — so an adult job can never
silently fall through into the safe output pipeline.
"""
from __future__ import annotations

import struct
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional


class GenerationError(Exception):
    """A generation failure carrying a stable machine-readable error code."""

    def __init__(self, message: str, *, error_code: str = "generation_failed", retryable: bool = False):
        super().__init__(message)
        self.error_code = error_code
        self.retryable = retryable


class UnsupportedInCurrentPhase(GenerationError):
    def __init__(self, message: str = "capability not enabled in current phase"):
        super().__init__(message, error_code="unsupported_in_current_phase", retryable=False)


# ---- mime by extension (kept tiny; MopMix emits PNG) ----
_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
         ".mp4": "video/mp4", ".webm": "video/webm", ".wav": "audio/wav", ".mp3": "audio/mpeg"}


@dataclass
class ArtifactFile:
    path: Path
    kind: str                       # "image" | "video" | "audio"
    mime_type: str
    width: Optional[int] = None
    height: Optional[int] = None
    duration: Optional[float] = None


@dataclass
class GenerationResult:
    artifacts: list[ArtifactFile]
    metadata: dict = field(default_factory=dict)   # engine, mode, generation_seconds, ...


def _mime_for(path: Path) -> str:
    return _MIME.get(path.suffix.lower(), "application/octet-stream")


def _png_dims(path: Path) -> tuple[Optional[int], Optional[int]]:
    try:
        with open(path, "rb") as f:
            head = f.read(26)
        if head[:8] != b"\x89PNG\r\n\x1a\n":
            return None, None
        return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")
    except Exception:
        return None, None


def _write_test_png(path: Path, w: int = 64, h: int = 64, rgb: tuple[int, int, int] = (255, 140, 0)) -> None:
    """Pure-stdlib tiny solid-colour PNG for FOX_WORKER_MOCK_GENERATION (no PIL / no GPU)."""
    raw = bytearray()
    row = bytes(rgb) * w
    for _ in range(h):
        raw.append(0)          # filter type 0 (None)
        raw += row

    def _chunk(typ: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + typ + data
                + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF))

    png = (b"\x89PNG\r\n\x1a\n"
           + _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + _chunk(b"IDAT", zlib.compress(bytes(raw)))
           + _chunk(b"IEND", b""))
    path.write_bytes(png)


class SafeGeneratorService:
    """Safe (SFW) generation. Phase-1 capability: safe.image.mopmix."""

    def generate_image(
        self,
        *,
        prompt: str,
        out_dir: Path,
        count: int = 1,
        quality: str = "medium",
        seed: Optional[int] = None,
        image_path: Optional[str] = None,
        mock: bool = False,
        timeout: int = 300,
        should_cancel: Optional[Callable[[], bool]] = None,
        on_progress: Optional[Callable[[int, str], None]] = None,
    ) -> GenerationResult:
        if not prompt or not prompt.strip():
            raise GenerationError("empty prompt", error_code="invalid_prompt")

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        count = max(1, int(count))
        started = time.time()

        if mock:
            paths: list[Path] = []
            for i in range(count):
                # Same cooperative cancel rule as the real path: the first render is atomic, then
                # the batch bails before starting the next item.
                if should_cancel and should_cancel() and i > 0:
                    break
                p = out_dir / f"mock_{i:02d}.png"
                _write_test_png(p)
                paths.append(p)
                if on_progress:
                    on_progress(int(round((i + 1) / count * 100)), f"mock {i + 1}/{count}")
            engine, mode = "mock", "mopmix"
        else:
            # Lazy import so mock / protocol tests never pull ComfyUI / PTB / GPU deps.
            try:
                from .image import generate_mopmix_images
            except Exception as e:  # pragma: no cover - only when bot deps missing
                raise GenerationError(f"generator backend unavailable: {e}", error_code="backend_unavailable")
            try:
                paths = generate_mopmix_images(
                    prompt,
                    count=count,
                    quality=quality,
                    seed=seed,
                    image_path=image_path,
                    out_dir=out_dir,
                    timeout=timeout,
                    should_cancel=should_cancel,
                    on_progress=on_progress,
                )
            except TimeoutError as e:
                raise GenerationError(str(e), error_code="timeout", retryable=True)
            except Exception as e:
                raise GenerationError(str(e), error_code="generation_failed")
            engine, mode = "mopmix_bigasp25", "mopmix"

        artifacts: list[ArtifactFile] = []
        for p in paths:
            w, h = _png_dims(p)
            artifacts.append(ArtifactFile(path=p, kind="image", mime_type=_mime_for(p), width=w, height=h))

        return GenerationResult(
            artifacts=artifacts,
            metadata={"engine": engine, "mode": mode,
                      "generation_seconds": round(time.time() - started, 2)},
        )


class AdultGeneratorService:
    """Adult generation is deliberately not enabled for FOX in this phase — every call refuses
    cleanly so nothing routes an adult job into the safe pipeline."""

    def generate_image(self, **_kwargs) -> GenerationResult:
        raise UnsupportedInCurrentPhase("adult image generation is not enabled for FOX yet")

    def generate_video(self, **_kwargs) -> GenerationResult:
        raise UnsupportedInCurrentPhase("adult video generation is not enabled for FOX yet")
