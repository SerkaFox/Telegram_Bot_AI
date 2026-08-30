"""Mock FOX MIX CORE implementing generator protocol V1 over stdlib http.server.

Used by the test-suite (and runnable standalone: `python -m tests.mock_fox_core`) so the full worker
half can be developed and validated BEFORE the real FOX CORE exists. Records everything it receives
so tests can assert on it. No external web framework; multipart is parsed by hand (the stdlib `cgi`
module was removed in Python 3.13).
"""
from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

API_BASE = "/internal/generator/v1"


def parse_multipart(body: bytes, boundary: bytes) -> dict[str, Any]:
    """Return {field_name: value}. File fields become {'filename':..., 'content': bytes}."""
    out: dict[str, Any] = {}
    delim = b"--" + boundary
    for part in body.split(delim):
        part = part.strip(b"\r\n")
        if not part or part == b"--":
            continue
        if b"\r\n\r\n" not in part:
            continue
        raw_headers, _, content = part.partition(b"\r\n\r\n")
        headers = raw_headers.decode("utf-8", "replace")
        name = None
        filename = None
        for h in headers.split("\r\n"):
            if h.lower().startswith("content-disposition"):
                for token in h.split(";"):
                    token = token.strip()
                    if token.startswith("name="):
                        name = token[5:].strip('"')
                    elif token.startswith("filename="):
                        filename = token[9:].strip('"')
        if name is None:
            continue
        if filename is not None:
            out[name] = {"filename": filename, "content": content}
        else:
            out[name] = content.decode("utf-8", "replace")
    return out


class MockFoxCore:
    def __init__(self, token: str = "test-token"):
        self.token = token
        self._lock = threading.Lock()
        self.pending: list[dict] = []          # jobs to hand out on claim (FIFO)
        self.control: dict[str, dict] = {}      # job_id -> {"cancel": bool}
        # recorded interactions
        self.started: list[dict] = []
        self.progress: list[dict] = []
        self.failed: list[dict] = []
        self.completed: list[dict] = []
        self.artifacts: list[dict] = []
        self.heartbeats: list[dict] = []
        self._artifact_seq = 0
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # ---- test helpers ----
    def enqueue_job(self, job: dict) -> None:
        with self._lock:
            self.pending.append(job)

    def set_cancel(self, job_id: str, value: bool = True) -> None:
        with self._lock:
            self.control[str(job_id)] = {"cancel_requested": value}

    @property
    def url(self) -> str:
        assert self._server is not None
        host, port = self._server.server_address
        return f"http://127.0.0.1:{port}"

    def start(self) -> "MockFoxCore":
        core = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass
            def _auth_ok(self) -> bool:
                return self.headers.get("Authorization") == f"Bearer {core.token}"
            def _send(self, code: int, payload: dict | None = None):
                body = json.dumps(payload).encode() if payload is not None else b""
                self.send_response(code)
                if body:
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if body:
                    self.wfile.write(body)
            def _read_body(self) -> bytes:
                n = int(self.headers.get("Content-Length") or 0)
                return self.rfile.read(n) if n else b""
            def _read_json(self) -> dict:
                raw = self._read_body()
                return json.loads(raw) if raw else {}

            def do_GET(self):
                if not self._auth_ok():
                    return self._send(401, {"error": "unauthorized"})
                path = self.path
                if path.startswith(API_BASE) and path.endswith("/control"):
                    job_id = path[len(API_BASE):].strip("/").split("/")[1]
                    with core._lock:
                        return self._send(200, core.control.get(job_id, {"cancel_requested": False}))
                return self._send(404, {"error": "not found"})

            def do_POST(self):
                if not self._auth_ok():
                    return self._send(401, {"error": "unauthorized"})
                path = self.path
                rel = path[len(API_BASE):] if path.startswith(API_BASE) else path

                if rel == "/claim":
                    self._read_body()
                    with core._lock:
                        job = core.pending.pop(0) if core.pending else None
                    # Real CORE always answers 200 with {"job": null | ClaimedJob}.
                    return self._send(200, {"job": job})

                if rel == "/heartbeat":
                    core.heartbeats.append(self._read_json())
                    return self._send(200, {"ok": True})

                parts = rel.strip("/").split("/")  # ["jobs", "<id>", "<action>"]
                if len(parts) == 3 and parts[0] == "jobs":
                    job_id, action = parts[1], parts[2]
                    if action == "started":
                        d = self._read_json(); d["job_id"] = job_id
                        core.started.append(d); return self._send(200, {"ok": True})
                    if action == "progress":
                        d = self._read_json(); d["job_id"] = job_id
                        core.progress.append(d); return self._send(200, {"ok": True})
                    if action == "failed":
                        d = self._read_json(); d["job_id"] = job_id
                        core.failed.append(d); return self._send(200, {"ok": True})
                    if action == "complete":
                        d = self._read_json(); d["job_id"] = job_id
                        core.completed.append(d); return self._send(200, {"ok": True})
                    if action == "artifacts":
                        ctype = self.headers.get("Content-Type", "")
                        body = self._read_body()
                        fields = {}
                        if "boundary=" in ctype:
                            boundary = ctype.split("boundary=", 1)[1].strip().encode()
                            fields = parse_multipart(body, boundary)
                        with core._lock:
                            core._artifact_seq += 1
                            aid = core._artifact_seq
                        f = fields.get("file") or {}
                        content = f.get("content", b"") if isinstance(f, dict) else b""
                        core.artifacts.append({
                            "job_id": job_id, "artifact_id": aid,
                            "kind": fields.get("kind"),
                            "width": fields.get("width"), "height": fields.get("height"),
                            "duration": fields.get("duration"),
                            "filename": f.get("filename"), "size": len(content),
                        })
                        # Real CORE returns {id, kind, size_bytes, sha256}.
                        return self._send(200, {"id": aid, "kind": fields.get("kind"),
                                                "size_bytes": len(content),
                                                "sha256": hashlib.sha256(content).hexdigest()})
                return self._send(404, {"error": "not found"})

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        if self._thread:
            self._thread.join(timeout=2)


if __name__ == "__main__":
    core = MockFoxCore().start()
    print(f"Mock FOX CORE on {core.url}{API_BASE} (token=test-token). Ctrl-C to stop.")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        core.stop()
