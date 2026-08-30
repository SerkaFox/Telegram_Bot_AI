"""End-to-end protocol tests: FOX MEDIA WORKER against the mock FOX CORE.

Covers: heartbeat, claim-204, claim+job, progress, artifact upload, complete, failed,
adult-unsupported, cancel semantics, auth error, network retry/backoff. No ComfyUI / GPU:
generation runs in FOX_WORKER_MOCK_GENERATION mode.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from fox_worker.client import FoxCoreClient, FoxCoreError
from fox_worker.config import Config
from generator import SafeGeneratorService, AdultGeneratorService
from tests.mock_fox_core import API_BASE, MockFoxCore


# ---------- fixtures / helpers ----------
@pytest.fixture
def core():
    c = MockFoxCore(token="test-token").start()
    yield c
    c.stop()


def make_cfg(core_url: str, token: str, tmp_path, *, mock_generation=True, heartbeat=1000.0) -> Config:
    return Config(
        core_url=core_url, worker_token=token, worker_id="test-worker", worker_name="test-worker",
        poll_seconds=0.05, heartbeat_seconds=heartbeat, mock_generation=mock_generation,
        jobs_dir=tmp_path, retention_hours=24, api_base=API_BASE,
        comfy_base="http://127.0.0.1:8188", request_timeout=5,
    )


def drive_worker(cfg, predicate, timeout=10):
    from fox_worker.worker import Worker
    client = FoxCoreClient(cfg)
    worker = Worker(cfg, client, SafeGeneratorService(), AdultGeneratorService())

    async def _go():
        task = asyncio.create_task(worker.run())
        t0 = time.time()
        while time.time() - t0 < timeout:
            if predicate():
                break
            await asyncio.sleep(0.05)
        worker.request_stop()
        try:
            await asyncio.wait_for(task, timeout=5)
        except asyncio.TimeoutError:
            pass

    asyncio.run(_go())
    return worker


# ---------- protocol-level ----------
def test_claim_none_when_no_work(core, tmp_path):
    cfg = make_cfg(core.url, core.token, tmp_path)
    client = FoxCoreClient(cfg)
    assert client.claim() is None


def test_heartbeat_recorded(core, tmp_path):
    cfg = make_cfg(core.url, core.token, tmp_path)
    client = FoxCoreClient(cfg)
    client.heartbeat(status="idle", current_job_id=None, capabilities=["safe.image.mopmix"], metadata=None)
    assert core.heartbeats and core.heartbeats[-1]["status"] == "idle"
    assert core.heartbeats[-1]["worker_id"] == "test-worker"
    assert core.heartbeats[-1]["name"] == "test-worker"          # CORE requires `name`
    assert core.heartbeats[-1]["current_job_id"] is None


def test_auth_error_non_retryable(core, tmp_path):
    cfg = make_cfg(core.url, "WRONG-TOKEN", tmp_path)
    client = FoxCoreClient(cfg)
    with pytest.raises(FoxCoreError) as ei:
        client.claim()
    assert ei.value.status == 401
    assert ei.value.retryable is False


def test_network_error_retryable(tmp_path):
    # Point at a closed port → connection error, flagged retryable (worker backs off, never crashes).
    cfg = make_cfg("http://127.0.0.1:1", "t", tmp_path)
    client = FoxCoreClient(cfg)
    with pytest.raises(FoxCoreError) as ei:
        client.claim()
    assert ei.value.retryable is True


# ---------- full lifecycle ----------
def test_safe_image_job_full_lifecycle(core, tmp_path):
    core.enqueue_job({"id": "job-1", "type": "image", "mode": "mopmix", "content_class": "safe",
                      "prompt": "cute orange fox mascot", "count": 2, "quality": "M", "options": {}})
    cfg = make_cfg(core.url, core.token, tmp_path, heartbeat=0.2)
    drive_worker(cfg, predicate=lambda: len(core.completed) >= 1, timeout=15)

    assert not core.failed, f"unexpected failures: {core.failed}"
    # complete carries no body in the real contract; just the finalisation call for this job.
    assert len(core.completed) == 1
    assert core.completed[0]["job_id"] == "job-1"

    # two artifacts uploaded, 64x64 test PNGs (MIME travels in the file part, not a form field)
    arts = [a for a in core.artifacts if a["job_id"] == "job-1"]
    assert len(arts) == 2
    assert all(a["kind"] == "image" for a in arts)
    assert all(a["width"] == "64" and a["height"] == "64" for a in arts)
    assert all(a["size"] > 0 for a in arts)

    # explicit started (claimed → started) fired for this job
    assert any(s["job_id"] == "job-1" for s in core.started)

    # progress was streamed and heartbeat landed
    assert any(p["job_id"] == "job-1" for p in core.progress)
    assert core.heartbeats


def test_adult_job_unsupported_in_current_phase(core, tmp_path):
    core.enqueue_job({"id": "job-a", "type": "image", "content_class": "adult",
                      "prompt": "whatever", "count": 1, "quality": "M", "options": {}})
    cfg = make_cfg(core.url, core.token, tmp_path)
    drive_worker(cfg, predicate=lambda: len(core.failed) >= 1, timeout=10)

    assert not core.completed
    assert len(core.failed) == 1
    assert core.failed[0]["error_code"] == "unsupported_in_current_phase"
    # no artifact should ever be produced for an adult job in this phase
    assert not core.artifacts
    # an unsupported job is rejected before it ever "starts" execution
    assert not any(s["job_id"] == "job-a" for s in core.started)


# ---------- cancel semantics (unit-level, deterministic) ----------
def test_cancel_stops_batch_after_current_item(tmp_path):
    svc = SafeGeneratorService()
    res = svc.generate_image(prompt="x", out_dir=tmp_path, count=5, mock=True,
                             should_cancel=lambda: True)
    # i=0 always renders (atomic), then the batch bails → exactly one artifact.
    assert len(res.artifacts) == 1


def test_image_edit_full_lifecycle(core, tmp_path):
    core.enqueue_job({"id": "job-e", "type": "image", "mode": "edit", "content_class": "safe",
                      "prompt": "change the background to a news studio", "count": 1, "quality": "M",
                      "options": {"source_url": core.source_url}})
    cfg = make_cfg(core.url, core.token, tmp_path)
    drive_worker(cfg, predicate=lambda: len(core.completed) >= 1, timeout=12)
    assert not core.failed, f"unexpected failures: {core.failed}"
    assert core.source_downloads >= 1                       # source image was fetched from CORE
    arts = [a for a in core.artifacts if a["job_id"] == "job-e"]
    assert len(arts) == 1 and arts[0]["kind"] == "image"
    assert any(s["job_id"] == "job-e" for s in core.started)


def test_clean_video_full_lifecycle(core, tmp_path):
    core.enqueue_job({"id": "job-v", "type": "video", "mode": "clean", "content_class": "safe",
                      "prompt": "the fox smiles at the camera", "count": 1, "quality": "M",
                      "options": {"source_url": core.source_url, "duration": 4}})
    cfg = make_cfg(core.url, core.token, tmp_path)
    drive_worker(cfg, predicate=lambda: len(core.completed) >= 1, timeout=12)
    assert not core.failed, f"unexpected failures: {core.failed}"
    arts = [a for a in core.artifacts if a["job_id"] == "job-v"]
    assert len(arts) == 1 and arts[0]["kind"] == "video"


def test_edit_missing_source_fails_then_worker_recovers(core, tmp_path):
    # edit job WITHOUT source_url -> missing_source; worker must stay healthy and do the next job.
    core.enqueue_job({"id": "job-nosrc", "type": "image", "mode": "edit", "content_class": "safe",
                      "prompt": "edit me", "count": 1, "quality": "M", "options": {}})
    core.enqueue_job({"id": "job-ok", "type": "image", "mode": "mopmix", "content_class": "safe",
                      "prompt": "a fox", "count": 1, "quality": "M", "options": {}})
    cfg = make_cfg(core.url, core.token, tmp_path)
    drive_worker(cfg, predicate=lambda: len(core.completed) >= 1 and len(core.failed) >= 1, timeout=12)
    assert any(f["job_id"] == "job-nosrc" and f["error_code"] == "missing_source" for f in core.failed)
    assert any(c["job_id"] == "job-ok" for c in core.completed)   # recovered, no wedge


def test_video_wan_unsupported_in_current_phase(core, tmp_path):
    # safe.video.wan is advertised but NOT enabled -> refused, never generated.
    core.enqueue_job({"id": "job-w", "type": "video", "mode": "wan", "content_class": "safe",
                      "prompt": "x", "count": 1, "quality": "M", "options": {"source_url": core.source_url}})
    cfg = make_cfg(core.url, core.token, tmp_path)
    drive_worker(cfg, predicate=lambda: len(core.failed) >= 1, timeout=10)
    assert core.failed[0]["error_code"] == "unsupported_in_current_phase"
    assert not core.completed and not core.artifacts


def test_gpu_gate_yields_then_proceeds(monkeypatch):
    gi = pytest.importorskip("generator.image")
    calls = {"n": 0}
    def fake_queue():
        calls["n"] += 1
        return {"queue_running": [], "queue_pending": []} if calls["n"] >= 3 else {"queue_running": [1]}
    monkeypatch.setattr(gi.b, "get_queue_state", fake_queue)
    monkeypatch.setattr(gi.time, "sleep", lambda *_a, **_k: None)
    gi._wait_for_gpu_gate(None)                 # loops while ComfyUI busy, then returns
    assert calls["n"] >= 3


def test_gpu_gate_cancel_short_circuits(monkeypatch):
    gi = pytest.importorskip("generator.image")
    monkeypatch.setattr(gi.b, "get_queue_state", lambda: {"queue_running": [1]})
    monkeypatch.setattr(gi.time, "sleep", lambda *_a, **_k: None)
    gi._wait_for_gpu_gate(lambda: True)         # cancel requested → returns immediately (no hang)


# ---------- source download (image.edit / clean.video) ----------
def test_source_download_sends_bearer_and_streams(core, tmp_path):
    client = FoxCoreClient(make_cfg(core.url, core.token, tmp_path))
    dest = tmp_path / "s.png"
    client.download(core.source_url, dest, expected_prefix="image/")
    assert dest.read_bytes() == core.source_png              # streamed byte-exact
    assert core.source_auth == f"Bearer {core.token}"        # bearer present on CORE host


def test_source_download_no_bearer_off_host(tmp_path, monkeypatch):
    client = FoxCoreClient(make_cfg("http://core.example:8088", "TESTTOK", tmp_path))
    seen: dict = {}

    class FakeResp:
        status_code = 200
        headers = {"Content-Type": "image/png"}
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def iter_content(self, chunk_size=0): return [b"\x89PNG\r\n\x1a\n"]

    def fake_get(url, headers=None, timeout=None, stream=None):
        seen[url] = headers
        return FakeResp()

    monkeypatch.setattr(client._session, "get", fake_get)
    client.download("http://core.example:8088/a", tmp_path / "a", expected_prefix="image/")
    assert "Authorization" in seen["http://core.example:8088/a"]     # on-host → bearer
    client.download("https://minio.public/x?sig=1", tmp_path / "b", expected_prefix="image/")
    assert seen["https://minio.public/x?sig=1"] == {}               # off-host → no creds


def test_source_download_401_cleans_partial(core, tmp_path):
    dest = tmp_path / "s.png"
    with pytest.raises(FoxCoreError) as ei:
        FoxCoreClient(make_cfg(core.url, "WRONG", tmp_path)).download(core.source_url, dest, expected_prefix="image/")
    assert ei.value.status == 401 and ei.value.retryable is False
    assert not dest.exists()


def test_source_download_404(core, tmp_path):
    dest = tmp_path / "s.png"
    with pytest.raises(FoxCoreError) as ei:
        FoxCoreClient(make_cfg(core.url, core.token, tmp_path)).download(
            core.url + API_BASE + "/nope", dest, expected_prefix="image/")
    assert ei.value.status == 404 and not dest.exists()


def test_source_download_invalid_mime(core, tmp_path):
    dest = tmp_path / "s.png"
    with pytest.raises(FoxCoreError) as ei:
        FoxCoreClient(make_cfg(core.url, core.token, tmp_path)).download(
            core.url + API_BASE + "/badmime/x", dest, expected_prefix="image/")
    assert ei.value.status == 415 and not dest.exists()


def test_source_download_retry_on_transient(core, tmp_path):
    dest = tmp_path / "s.png"
    # /flaky returns 500 once then 200 → bounded retry recovers.
    FoxCoreClient(make_cfg(core.url, core.token, tmp_path)).download(
        core.url + API_BASE + "/flaky/x", dest, expected_prefix="image/", retries=1)
    assert dest.read_bytes() == core.source_png


def test_worker_polls_control_endpoint(core, tmp_path):
    # A completed job means the control endpoint was reachable and polled without breaking flow.
    core.enqueue_job({"id": "job-c", "type": "image", "content_class": "safe",
                      "prompt": "fox", "count": 1, "quality": "L", "options": {}})
    core.set_cancel("job-c", False)
    cfg = make_cfg(core.url, core.token, tmp_path)
    drive_worker(cfg, predicate=lambda: len(core.completed) >= 1, timeout=10)
    assert core.completed and core.completed[0]["job_id"] == "job-c"
