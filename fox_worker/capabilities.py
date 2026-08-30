"""Capability registry advertised to FOX MIX.

Every capability the server *could* do is listed so FOX can discover/describe it, but only the
Phase-1 subset is ENABLED. A capability that is described but not enabled is refused at runtime with
`unsupported_in_current_phase` — described != usable until its own tests pass.
"""
from __future__ import annotations

# Full catalogue the box is theoretically able to serve (discoverable by FOX MIX).
ALL_CAPABILITIES: tuple[str, ...] = (
    "safe.image.mopmix",
    "safe.image.edit",
    "safe.video.clean",
    "safe.video.wan",
    "safe.video.ltx",
    "safe.video.v2v",
    "adult.image",
    "adult.video.wan",
    "adult.video.eros",
)

# Actually wired + tested in this phase. ONLY safe MopMix images.
ENABLED_CAPABILITIES: frozenset[str] = frozenset({"safe.image.mopmix"})


def is_enabled(capability: str) -> bool:
    return capability in ENABLED_CAPABILITIES


def capability_for_job(job_type: str, content_class: str) -> str | None:
    """Map a job's (type, content_class) to a capability key, or None if we don't model it.

    Phase-1 only needs safe image → safe.image.mopmix. Other combinations resolve to their catalogue
    key (so the reason for refusal is 'unsupported_in_current_phase', not 'unknown')."""
    t = (job_type or "").strip().lower()
    c = (content_class or "").strip().lower()
    if t == "image" and c == "safe":
        return "safe.image.mopmix"
    if t == "image" and c == "adult":
        return "adult.image"
    if t == "video" and c == "safe":
        return "safe.video.wan"
    if t == "video" and c == "adult":
        return "adult.video.wan"
    return None
