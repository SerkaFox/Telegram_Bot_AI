"""Worker configuration, loaded from environment (.env). Never logs secret values."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_PROJECT_DIR = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader: KEY=VALUE lines into os.environ WITHOUT overriding already-set vars
    (so systemd Environment= and real env win). No external dependency, no logging of values."""
    path = path or (_PROJECT_DIR / ".env")
    if not path.exists():
        return
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    except Exception:
        # A malformed .env must never crash the worker; real env / defaults still apply.
        pass


def _to_bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Config:
    core_url: str
    worker_token: str
    worker_id: str
    worker_name: str
    poll_seconds: float
    heartbeat_seconds: float
    mock_generation: bool
    jobs_dir: Path
    retention_hours: float
    api_base: str
    comfy_base: str
    request_timeout: float

    @property
    def has_core(self) -> bool:
        return bool(self.core_url and self.worker_token)

    def base_url(self) -> str:
        return self.core_url.rstrip("/") + self.api_base


def _resolve_jobs_dir(raw: str) -> Path:
    """Prefer the configured (system) jobs dir; fall back to a project-local dir if it is not
    writable (e.g. no root to create /var/lib/...). Never raises."""
    candidate = Path(raw)
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        probe = candidate / ".write_test"
        probe.touch()
        probe.unlink()
        return candidate
    except Exception:
        fallback = _PROJECT_DIR / "fox_worker_jobs"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def load_config() -> Config:
    load_dotenv()
    return Config(
        core_url=os.getenv("FOX_CORE_URL", "").strip(),
        worker_token=os.getenv("FOX_WORKER_TOKEN", "").strip(),
        worker_id=os.getenv("FOX_WORKER_ID", "fox-media-worker-1").strip(),
        worker_name=os.getenv("FOX_WORKER_NAME", os.getenv("FOX_WORKER_ID", "fox-media-worker-1")).strip(),
        poll_seconds=float(os.getenv("FOX_POLL_SECONDS", "5")),
        heartbeat_seconds=float(os.getenv("FOX_HEARTBEAT_SECONDS", "30")),
        mock_generation=_to_bool(os.getenv("FOX_WORKER_MOCK_GENERATION", "false")),
        jobs_dir=_resolve_jobs_dir(os.getenv("FOX_JOBS_DIR", str(_PROJECT_DIR / "fox_worker_jobs"))),
        retention_hours=float(os.getenv("FOX_ARTIFACT_RETENTION_HOURS", "24")),
        api_base=os.getenv("FOX_API_BASE", "/internal/generator/v1"),
        comfy_base=os.getenv("COMFY_BASE", "http://127.0.0.1:8188"),
        request_timeout=float(os.getenv("FOX_REQUEST_TIMEOUT", "30")),
    )
