"""HTTP client for the FOX generator protocol V1.

Synchronous (requests); the async worker calls these via asyncio.to_thread so the event loop stays
free for heartbeats. The worker ONLY makes OUTBOUND calls to FOX CORE — this box opens no inbound
port for FOX. The Bearer token is never logged.

Endpoints (base = FOX_CORE_URL + /internal/generator/v1):
  POST  /jobs/claim                 -> 204 (no work) | 200 job json
  POST  /jobs/{id}/progress         {progress:0-100, stage}
  POST  /jobs/{id}/failed           {error_code, message, retryable}
  POST  /jobs/{id}/artifacts        multipart: file, kind, mime_type, width?, height?, duration?  -> {artifact_id}
  POST  /jobs/{id}/complete         {artifact_ids:[...], metadata:{...}}
  POST  /heartbeat                  {worker_id, status, job_id, capabilities, local_queue}
  GET   /jobs/{id}/control          -> {cancel: bool}
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
    def claim(self, capabilities: list[str]) -> Optional[dict]:
        """Ask for one job. Returns the job dict, or None when there is no work (HTTP 204)."""
        r = self._post("/jobs/claim",
                       json={"worker_id": self.cfg.worker_id, "capabilities": capabilities},
                       expect=(200, 204))
        if r.status_code == 204 or not (r.content or b"").strip():
            return None
        return r.json()

    def started(self, job_id: str) -> None:
        """Mark the job as started (claimed → started) the moment execution begins."""
        self._post(f"/jobs/{job_id}/started", json={"worker_id": self.cfg.worker_id},
                   expect=(200, 204))

    def progress(self, job_id: str, progress: int, stage: str = "") -> None:
        self._post(f"/jobs/{job_id}/progress",
                   json={"progress": int(progress), "stage": stage}, expect=(200, 204))

    def failed(self, job_id: str, error_code: str, message: str = "", retryable: bool = False) -> None:
        self._post(f"/jobs/{job_id}/failed",
                   json={"error_code": error_code, "message": message[:500], "retryable": bool(retryable)},
                   expect=(200, 204))

    def upload_artifact(self, job_id: str, path: Path, *, kind: str, mime_type: str,
                        width: int | None = None, height: int | None = None,
                        duration: float | None = None) -> str:
        path = Path(path)
        data = {"kind": kind, "mime_type": mime_type}
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
        return str(body.get("artifact_id", ""))

    def complete(self, job_id: str, artifact_ids: list[str], metadata: dict) -> None:
        self._post(f"/jobs/{job_id}/complete",
                   json={"artifact_ids": artifact_ids, "metadata": metadata}, expect=(200, 204))

    def heartbeat(self, *, status: str, job_id: str | None, capabilities: list[str],
                  local_queue: int = 0) -> None:
        self._post("/heartbeat",
                   json={"worker_id": self.cfg.worker_id, "status": status, "job_id": job_id,
                         "capabilities": capabilities, "local_queue": local_queue},
                   expect=(200, 204))

    def get_control(self, job_id: str) -> dict:
        r = self._get(f"/jobs/{job_id}/control", expect=(200,))
        return r.json() if (r.content or b"").strip() else {}
