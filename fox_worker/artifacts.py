"""Per-job working directory + lifecycle + retention.

Lifecycle: claimed → rendering → generated → uploaded → cleanup. Files are kept until upload is
confirmed, then eligible for retention-based cleanup. Fully independent of the Telegram bot's dirs.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path


class JobWorkspace:
    def __init__(self, jobs_dir: Path, job_id: str):
        self.job_id = str(job_id)
        self.root = Path(jobs_dir) / self.job_id
        self.out_dir = self.root / "out"
        self.state = "claimed"

    def open(self) -> "JobWorkspace":
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.root / "state").write_text(self.state)
        return self

    def set_state(self, state: str) -> None:
        self.state = state
        try:
            (self.root / "state").write_text(state)
        except Exception:
            pass

    def cleanup(self) -> None:
        try:
            shutil.rmtree(self.root, ignore_errors=True)
        except Exception:
            pass


def prune_old_jobs(jobs_dir: Path, retention_hours: float) -> int:
    """Remove job workspaces older than the retention window. Safe to call periodically; never
    raises. Returns how many were removed."""
    jobs_dir = Path(jobs_dir)
    if not jobs_dir.exists() or retention_hours <= 0:
        return 0
    cutoff = time.time() - retention_hours * 3600
    removed = 0
    for child in jobs_dir.iterdir():
        try:
            if child.is_dir() and child.stat().st_mtime < cutoff:
                shutil.rmtree(child, ignore_errors=True)
                removed += 1
        except Exception:
            continue
    return removed
