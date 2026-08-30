"""Protocol data models for the FOX generator contract (V1)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# FOX quality codes → MopMix quality buckets.
_QUALITY = {"l": "low", "m": "medium", "h": "high",
            "low": "low", "medium": "medium", "high": "high"}


@dataclass
class Job:
    id: str
    type: str                       # "image" | "video" | "audio"
    mode: str                       # engine/mode, e.g. "mopmix"
    content_class: str              # "safe" | "adult"
    prompt: str
    negative_prompt: Optional[str] = None
    count: int = 1
    quality: str = "medium"         # normalised bucket name
    priority: int = 0
    options: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    @property
    def id_int(self) -> Optional[int]:
        try:
            return int(self.id)
        except (TypeError, ValueError):
            return None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "Job":
        jid = data.get("id") if data.get("id") is not None else data.get("job_id")
        q = str(data.get("quality") or "medium").strip().lower()
        return cls(
            id=str(jid),
            type=str(data.get("type") or "").strip().lower(),
            mode=str(data.get("mode") or "").strip().lower(),
            content_class=str(data.get("content_class") or "").strip().lower(),
            prompt=str(data.get("prompt") or ""),
            negative_prompt=data.get("negative_prompt"),
            count=int(data.get("count") or 1),
            quality=_QUALITY.get(q, "medium"),
            priority=int(data.get("priority") or 0),
            options=dict(data.get("options") or {}),
            raw=data,
        )
