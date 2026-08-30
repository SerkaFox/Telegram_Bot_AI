"""HTTP client for the FOX generator protocol V1.

Synchronous (requests); the async worker calls these via asyncio.to_thread so the event loop stays
free for heartbeats. The worker ONLY makes OUTBOUND calls to FOX CORE — this box opens no inbound
port for FOX. The Bearer token is never logged.

Endpoints (base = FOX_CORE_URL + /internal/generator/v1) — matches the live CORE OpenAPI:
  POST  /claim                      {worker_id}                       -> 200 {job: null | ClaimedJob}
  POST  /jobs/{id}/started          (no body)                         -> {id, status}
  POST  /jobs/{id}/progress         {progress:0-100, stage?}          -> {id, status}
  POST  /jobs/{id}/failed           {error_code, error_message}       -> {id, status}
  POST  /jobs/{id}/artifacts        multipart: file, kind, width?, height?, duration?  -> {id, kind, size_bytes, sha256}
  POST  /jobs/{id}/complete         (no body)                         -> {id, status}
  POST  /heartbeat                  {worker_id, name, status, capabilities, current_job_id, metadata} -> {ok}
  GET   /jobs/{id}/control          -> {cancel_requested: bool}
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import requests


class FoxCoreError(Exception):
    def __init__(self, message: str, *, status: int | None = None, retryable: bool = True):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


class FoxCoreClient:
    def __init__(self, config):
        self.cfg = config
        self._session = requests.Session()

    # ---- internals -------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.cfg.worker_token}"}

    def _url(self, path: str) -> str:
        return self.cfg.base_url() + path

    def _post(self, path: str, *, json: Any = None, files: Any = None, data: Any = None,
              expect: tuple[int, ...] = (200, 204)) -> requests.Response:
        try:
            r = self._session.post(self._url(path), json=json, files=files, data=data,
                                   headers=self._headers(), timeout=self.cfg.request_timeout)
        except requests.RequestException as e:
            # Network / connection error — retryable; message intentionally omits any URL creds.
            raise FoxCoreError(f"network error on POST {path}: {e.__class__.__name__}", retryable=True)
        if r.status_code == 401 or r.status_code == 403:
            raise FoxCoreError(f"auth rejected on POST {path}", status=r.status_code, retryable=False)
        if r.status_code not in expect:
            raise FoxCoreError(f"unexpected {r.status_code} on POST {path}", status=r.status_code,
                               retryable=r.status_code >= 500)
        return r

    def _get(self, path: str, *, expect: tuple[int, ...] = (200,)) -> requests.Response:
        try:
            r = self._session.get(self._url(path), headers=self._headers(),
                                  timeout=self.cfg.request_timeout)
        except requests.RequestException as e:
            raise FoxCoreError(f"network error on GET {path}: {e.__class__.__name__}", retryable=True)
        if r.status_code in (401, 403):
            raise FoxCoreError(f"auth rejected on GET {path}", status=r.status_code, retryable=False)
        if r.status_code not in expect:
            raise FoxCoreError(f"unexpected {r.status_code} on GET {path}", status=r.status_code,
                               retryable=r.status_code >= 500)
        return r

    # ---- protocol --------------------------------------------------------
    def claim(self) -> Optional[dict]:
        """Ask for one job. CORE always answers 200 with {"job": null | ClaimedJob}; capabilities
        are advertised via heartbeat, not here. Returns the job dict, or None when idle."""
        r = self._post("/claim", json={"worker_id": self.cfg.worker_id}, expect=(200, 204))
        if r.status_code == 204 or not (r.content or b"").strip():
            return None
        return (r.json() or {}).get("job")

    def started(self, job_id: str) -> None:
        """Mark the job started (claimed → started) the moment execution begins. No request body."""
        self._post(f"/jobs/{job_id}/started", expect=(200, 204))

    def progress(self, job_id: str, progress: int, stage: str = "") -> None:
        self._post(f"/jobs/{job_id}/progress",
                   json={"progress": int(progress), "stage": stage or None}, expect=(200, 204))

    def failed(self, job_id: str, error_code: str, error_message: str = "") -> None:
        self._post(f"/jobs/{job_id}/failed",
                   json={"error_code": error_code, "error_message": (error_message or error_code)[:500]},
                   expect=(200, 204))

    def upload_artifact(self, job_id: str, path: Path, *, kind: str, mime_type: str,
                        width: int | None = None, height: int | None = None,
                        duration: float | None = None) -> str:
        # CORE takes file+kind (+optional dims); the MIME travels in the file part, not a form field.
        path = Path(path)
        data: dict[str, str] = {"kind": kind}
        if width is not None:
            data["width"] = str(width)
        if height is not None:
            data["height"] = str(height)
        if duration is not None:
            data["duration"] = str(duration)
        with open(path, "rb") as fh:
            files = {"file": (path.name, fh, mime_type)}
            r = self._post(f"/jobs/{job_id}/artifacts", data=data, files=files, expect=(200, 201))
        body = r.json() if (r.content or b"").strip() else {}
        return str(body.get("id", ""))

    def complete(self, job_id: str) -> None:
        """Finalise the job. Artifacts were already linked at upload time; no request body."""
        self._post(f"/jobs/{job_id}/complete", expect=(200, 204))

    def heartbeat(self, *, status: str, current_job_id: int | None, capabilities: list[str],
                  metadata: dict | None = None) -> None:
        self._post("/heartbeat",
                   json={"worker_id": self.cfg.worker_id, "name": self.cfg.worker_name,
                         "status": status, "capabilities": capabilities,
                         "current_job_id": current_job_id, "metadata": metadata},
                   expect=(200, 204))

    def download(self, url: str, dest: Path) -> Path:
        """Fetch a source artifact the job points to (proposed contract: job.options.source_url).
        Sends the worker Bearer token only when the URL is on the FOX CORE host; a presigned/public
        URL is fetched without our credentials. Streams to `dest`."""
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        headers = self._headers() if url.startswith(self.cfg.core_url.rstrip("/")) else {}
        try:
            with self._session.get(url, headers=headers, timeout=self.cfg.request_timeout,
                                   stream=True) as r:
                if r.status_code in (401, 403):
                    raise FoxCoreError(f"auth rejected downloading source ({r.status_code})",
                                       status=r.status_code, retryable=False)
                if r.status_code != 200:
                    raise FoxCoreError(f"source download failed ({r.status_code})",
                                       status=r.status_code, retryable=r.status_code >= 500)
                with open(dest, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=65536):
                        if chunk:
                            fh.write(chunk)
        except requests.RequestException as e:
            raise FoxCoreError(f"network error downloading source: {e.__class__.__name__}", retryable=True)
        return dest

    def get_control(self, job_id: str) -> dict:
        r = self._get(f"/jobs/{job_id}/control", expect=(200,))
        return r.json() if (r.content or b"").strip() else {}
