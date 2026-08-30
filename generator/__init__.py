"""Non-Telegram generation service layer.

Exposes the server's generation capability through a plain Python API that does NOT depend on
Telegram handlers, so a second consumer (the FOX MEDIA WORKER) can drive generation the same way
the bot's queue does. Reuses the already-working ComfyUI workflows in telegram_comfyui_bot.py —
no new workflow is written here.

Safe and adult routing are kept in SEPARATE service classes on purpose (see service.py): an adult
job must never fall through into the safe output path by an implicit condition.
"""
from .service import (
    GenerationError,
    GenerationResult,
    ArtifactFile,
    SafeGeneratorService,
    AdultGeneratorService,
    UnsupportedInCurrentPhase,
)

__all__ = [
    "GenerationError",
    "GenerationResult",
    "ArtifactFile",
    "SafeGeneratorService",
    "AdultGeneratorService",
    "UnsupportedInCurrentPhase",
]
