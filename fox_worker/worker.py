"""FOX MEDIA WORKER — single async loop.

Responsibilities: heartbeat, claim one job, execute it (safe MopMix only in this phase), stream
progress, upload artifacts, complete/fail, honour cancel. Robust to FOX CORE downtime and network
errors: those are logged and retried with backoff and NEVER crash the process or touch ComfyUI /
the Telegram bot.

GPU sharing: the worker runs one job at a time (concurrency = 1) and the generator yields to any
in-flight ComfyUI work (see generator/image.py `_wait_for_gpu_gate`), so owner/manual Telegram jobs
keep priority without any change to the bot.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time

from generator import GenerationError

from . import capabilities as caps
from .artifacts import JobWorkspace, prune_old_jobs
from .client import FoxCoreClient, FoxCoreError
from .config import Config
from .models import Job

log = logging.getLogger("fox-worker")


class Metrics:
    """In-memory counters (no Prometheus yet) — logged so FOX-side dashboards can scrape stdout."""

    def __init__(self):
        self.jobs_claimed = 0
        self.jobs_completed = 0
        self.jobs_failed = 0
        self.gen_seconds_total = 0.0
        self.upload_seconds_total = 0.0

    def snapshot(self) -> dict:
        return {"jobs_claimed": self.jobs_claimed, "jobs_completed": self.jobs_completed,
                "jobs_failed": self.jobs_failed,
                "gen_seconds_total": round(self.gen_seconds_total, 1),
                "upload_seconds_total": round(self.upload_seconds_total, 1)}


class Worker:
    def __init__(self, config: Config, client: FoxCoreClient, safe_service, adult_service):
        self.cfg = config
        self.client = client
        self.safe = safe_service
        self.adult = adult_service
        self.metrics = Metrics()
        self._status = "idle"
        self._current_job_id: str | None = None
        self._stop = asyncio.Event()
        self._last_prune = 0.0

    # ---- lifecycle -------------------------------------------------------
    def request_stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        log.info("worker.started id=%s mock=%s capabilities=%s core=%s",
                 self.cfg.worker_id, self.cfg.mock_generation,
                 sorted(caps.ENABLED_CAPABILITIES), "set" if self.cfg.has_core else "MISSING")
        if not self.cfg.has_core:
            log.warning("FOX_CORE_URL / FOX_WORKER_TOKEN not set — worker idle until configured")

        hb_task = asyncio.create_task(self._heartbeat_loop())
        try:
            await self._poll_loop()
        finally:
            hb_task.cancel()
            try:
                await hb_task
            except asyncio.CancelledError:
                pass
            log.info("worker.stopped metrics=%s", self.metrics.snapshot())

    # ---- heartbeat -------------------------------------------------------
    async def _heartbeat_loop(self) -> None:
        while not self._stop.is_set():
            if self.cfg.has_core:
                try:
                    cur = int(self._current_job_id) if self._current_job_id else None
                    await asyncio.to_thread(
                        self.client.heartbeat,
                        status=self._status, current_job_id=cur,
                        capabilities=sorted(caps.ENABLED_CAPABILITIES), metadata=None)
                    log.debug("worker.heartbeat status=%s job=%s", self._status, self._current_job_id)
                except FoxCoreError as e:
                    log.warning("worker.heartbeat failed: %s", e)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.cfg.heartbeat_seconds)
            except asyncio.TimeoutError:
                pass

    # ---- main poll loop --------------------------------------------------
    async def _poll_loop(self) -> None:
        backoff = self.cfg.poll_seconds
        while not self._stop.is_set():
            self._maybe_prune()
            if not self.cfg.has_core:
                await self._sleep(self.cfg.poll_seconds)
                continue
            try:
                job_data = await asyncio.to_thread(self.client.claim)
                backoff = self.cfg.poll_seconds  # healthy call → reset backoff
            except FoxCoreError as e:
                if not e.retryable:
                    log.error("worker.error non-retryable claim failure: %s — pausing", e)
                    await self._sleep(min(backoff * 2, 60))
                else:
                    log.warning("worker.error claim failed (retry): %s", e)
                    backoff = min(backoff * 2, 60)
                    await self._sleep(backoff)
                continue

            if not job_data:
                self._status = "idle"
                await self._sleep(self.cfg.poll_seconds)
                continue

            job = Job.from_api(job_data)
            self.metrics.jobs_claimed += 1
            log.info("job.claimed id=%s type=%s class=%s count=%s quality=%s",
                     job.id, job.type, job.content_class, job.count, job.quality)
            await self._handle_job(job)

    # ---- job handling ----------------------------------------------------
    async def _handle_job(self, job: Job) -> None:
        self._status = "busy"
        self._current_job_id = job.id
        try:
            cap = caps.capability_for_job(job.type, job.mode, job.content_class)
            if cap is None or not caps.is_enabled(cap):
                reason = "unsupported_in_current_phase"
                log.info("job.failed id=%s reason=%s cap=%s", job.id, reason, cap)
                await asyncio.to_thread(self.client.failed, job.id, reason,
                                        f"capability {cap} not enabled")
                self.metrics.jobs_failed += 1
                return
            await self._execute_job(job, cap)
        except FoxCoreError as e:
            # Reporting to CORE failed — log and move on; do not crash.
            log.warning("job.error id=%s core reporting failed: %s", job.id, e)
        except Exception as e:  # pragma: no cover - defensive
            log.exception("job.failed id=%s unexpected: %s", job.id, e)
            try:
                await asyncio.to_thread(self.client.failed, job.id, "internal_error", str(e))
            except FoxCoreError:
                pass
            self.metrics.jobs_failed += 1
        finally:
            self._status = "idle"
            self._current_job_id = None

    async def _execute_job(self, job: Job, cap: str) -> None:
        """Shared executor for every enabled capability: started → progress → produce (behind the
        GPU gate) → upload → complete/failed. The GPU gate + cancel + finally-release live in the
        producer / this frame so one FOX job can never wedge the shared GPU or the bot."""
        ws = JobWorkspace(self.cfg.jobs_dir, job.id).open()
        cancel_event = threading.Event()
        control_task = asyncio.create_task(self._poll_control(job.id, cancel_event))
        log.info("job.started id=%s cap=%s", job.id, cap)
        await asyncio.to_thread(self.client.started, job.id)

        # Throttled progress reporter (called from the generation thread and this frame).
        last = {"pct": -10, "stage": ""}

        def on_progress(pct: int, stage: str) -> None:
            if pct - last["pct"] >= 5 or pct >= 100 or stage != last["stage"]:
                last["pct"], last["stage"] = pct, stage
                try:
                    self.client.progress(job.id, pct, stage)
                    log.info("job.progress id=%s pct=%s stage=%s", job.id, pct, stage)
                except FoxCoreError as e:
                    log.debug("progress post failed id=%s: %s", job.id, e)

        try:
            await asyncio.to_thread(self.client.progress, job.id, 1, "queued")
            ws.set_state("rendering")
            gen_start = time.time()
            result = await asyncio.to_thread(self._produce, job, cap, ws, cancel_event, on_progress)
            self.metrics.gen_seconds_total += time.time() - gen_start
            ws.set_state("generated")

            if cancel_event.is_set() and not result.artifacts:
                await asyncio.to_thread(self.client.failed, job.id, "cancelled", "cancelled by core")
                self.metrics.jobs_failed += 1
                log.info("job.failed id=%s reason=cancelled", job.id)
                return

            up_start = time.time()
            on_progress(90, "uploading")
            artifact_ids: list[str] = []
            for art in result.artifacts:
                aid = await asyncio.to_thread(
                    self.client.upload_artifact, job.id, art.path,
                    kind=art.kind, mime_type=art.mime_type,
                    width=art.width, height=art.height, duration=art.duration)
                artifact_ids.append(aid)
                log.info("artifact.uploaded id=%s artifact_id=%s kind=%s", job.id, aid, art.kind)
            self.metrics.upload_seconds_total += time.time() - up_start
            ws.set_state("uploaded")

            await asyncio.to_thread(self.client.complete, job.id)
            self.metrics.jobs_completed += 1
            log.info("job.completed id=%s artifacts=%s meta=%s", job.id, len(artifact_ids), result.metadata)
        except Exception as e:
            # Any generation/source error → report failed with its code; the gate is already released
            # (the producer thread has returned), so nothing stays locked.
            code = getattr(e, "error_code", "generation_failed")
            log.warning("job.failed id=%s code=%s: %s", job.id, code, e)
            try:
                await asyncio.to_thread(self.client.failed, job.id, code, str(e))
            except FoxCoreError:
                pass
            self.metrics.jobs_failed += 1
        finally:
            control_task.cancel()
            try:
                await control_task
            except asyncio.CancelledError:
                pass
            # Workspace kept for retention-based cleanup (prune runs in the main loop).

    def _fetch_source(self, job: Job, ws: JobWorkspace) -> str:
        """Fetch the source image a job points to. PROPOSED minimal contract: job.options.source_url
        (worker Bearer only when on the CORE host). Fails with `missing_source` when absent, so an
        image.edit / clean.video job without a source is reported cleanly (never a crash)."""
        opts = job.options or {}
        url = opts.get("source_url") or opts.get("source_image_url") or opts.get("source_download_url")
        if not url:
            raise GenerationError("no source_url in job.options", error_code="missing_source")
        dest = ws.root / "source"
        self.client.download(str(url), dest)
        return str(dest)

    def _produce(self, job: Job, cap: str, ws: JobWorkspace, cancel_event: threading.Event, on_progress):
        """Sync producer run in a worker thread → GenerationResult. Source fetch (edit/clean) happens
        here so a download failure is reported as a normal job failure, releasing the GPU gate."""
        mock = self.cfg.mock_generation
        if cap == "safe.image.mopmix":
            return self.safe.generate_image(
                prompt=job.prompt, out_dir=ws.out_dir, count=job.count, quality=job.quality,
                seed=job.options.get("seed"), image_path=None, mock=mock,
                should_cancel=cancel_event.is_set, on_progress=on_progress)
        if cap == "safe.image.edit":
            src = self._fetch_source(job, ws)
            on_progress(10, "waiting_gpu")
            return self.safe.edit_image(
                instruction=job.prompt, source_path=src, out_dir=ws.out_dir, count=job.count,
                quality=job.quality, seed=job.options.get("seed"), mock=mock,
                should_cancel=cancel_event.is_set, on_progress=on_progress)
        if cap == "safe.video.clean":
            src = self._fetch_source(job, ws)
            on_progress(10, "waiting_gpu")
            return self.safe.generate_clean_video(
                prompt=job.prompt, source_path=src, out_dir=ws.out_dir, quality=job.quality,
                seconds=job.options.get("duration") or job.options.get("seconds"),
                seed=job.options.get("seed"), mock=mock,
                should_cancel=cancel_event.is_set, on_progress=on_progress)
        raise GenerationError(f"no producer for {cap}", error_code="unsupported_in_current_phase")

    async def _poll_control(self, job_id: str, cancel_event: threading.Event) -> None:
        """Background: ask CORE whether this job was cancelled; flip the event so generation bails
        cooperatively (finishes the current atomic render, skips the rest of the batch)."""
        while not self._stop.is_set():
            try:
                ctl = await asyncio.to_thread(self.client.get_control, job_id)
                if ctl.get("cancel_requested"):
                    cancel_event.set()
                    log.info("job.cancel_requested id=%s", job_id)
                    return
            except FoxCoreError:
                pass
            await asyncio.sleep(3)

    # ---- helpers ---------------------------------------------------------
    def _maybe_prune(self) -> None:
        now = time.time()
        if now - self._last_prune > 3600:
            self._last_prune = now
            removed = prune_old_jobs(self.cfg.jobs_dir, self.cfg.retention_hours)
            if removed:
                log.info("cleanup pruned %s old job dirs", removed)

    async def _sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass
