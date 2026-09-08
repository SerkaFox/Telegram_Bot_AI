"""Capability registry advertised to FOX MIX.

Every capability the server *could* do is listed so FOX can discover/describe it, but only the
ENABLED subset actually executes. A capability that is described but not enabled is refused at
runtime with `unsupported_in_current_phase` — described != usable until its own tests pass.
"""
from __future__ import annotations

# Full catalogue the box is theoretically able to serve (discoverable by FOX MIX).
ALL_CAPABILITIES: tuple[str, ...] = (
    "safe.image.mopmix",
    "safe.image.edit",
    "safe.video.clean",
    "safe.video.talking",
    "safe.video.wan",
    "safe.video.ltx",
    "safe.video.v2v",
    "adult.image",
    "adult.video.wan",
    "adult.video.eros",
)

# Actually wired + tested. Phase G2 added safe.image.edit and safe.video.clean; this phase adds
# safe.video.talking (LTX Sulphur presenter / talking-head with native or OpenVoice speech).
# safe.video.wan/ltx/v2v stay advertised-but-disabled; all adult stays execution-disabled.
ENABLED_CAPABILITIES: frozenset[str] = frozenset({
    "safe.image.mopmix",
    "safe.image.edit",
    "safe.video.clean",
    "safe.video.talking",
})


def is_enabled(capability: str) -> bool:
    return capability in ENABLED_CAPABILITIES


def capability_for_job(job_type: str, mode: str, content_class: str) -> str | None:
    """Map a job's (type, mode, content_class) to a capability key, or None if we don't model it.

    Routing is explicit per (type, mode, content_class) so an adult job can never resolve to a safe
    capability, and image/mopmix vs image/edit never collide."""
    t = (job_type or "").strip().lower()
    m = (mode or "").strip().lower()
    c = (content_class or "").strip().lower()
    if c == "safe":
        if t == "image" and m in ("", "mopmix"):
            return "safe.image.mopmix"
        if t == "image" and m == "edit":
            return "safe.image.edit"
        if t == "video" and m == "clean":
            return "safe.video.clean"
        if t == "video" and m == "talking":
            return "safe.video.talking"
        if t == "video":
            return "safe.video.wan"        # advertised, not enabled
    elif c == "adult":
        if t == "image":
            return "adult.image"
        if t == "video":
            return "adult.video.wan"
    return None
