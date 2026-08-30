"""Contract tests: the worker client must stay aligned with the live FOX CORE OpenAPI.

Uses the saved fixture docs/fox-generator-protocol-v1.json as the reference. If FOX changes the
protocol, refreshing the fixture makes these fail until the client is updated — catching drift
without hitting the live CORE (no network, no secrets)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURE = Path(__file__).resolve().parent.parent / "docs" / "fox-generator-protocol-v1.json"
API = "/internal/generator/v1"


@pytest.fixture(scope="module")
def spec():
    if not FIXTURE.exists():
        pytest.skip("contract fixture not present")
    return json.loads(FIXTURE.read_text())


def _required(spec, path, method="post", content="application/json"):
    op = spec["paths"][path][method]
    schema = op["requestBody"]["content"][content]["schema"]
    name = schema["$ref"].split("/")[-1] if "$ref" in schema else None
    model = spec["components"]["schemas"][name] if name else schema
    return set(model.get("required", [])), set(model.get("properties", {}))


def test_all_client_endpoints_exist(spec):
    for p in (f"{API}/claim", f"{API}/heartbeat",
              f"{API}/jobs/{{job_id}}/started", f"{API}/jobs/{{job_id}}/progress",
              f"{API}/jobs/{{job_id}}/failed", f"{API}/jobs/{{job_id}}/artifacts",
              f"{API}/jobs/{{job_id}}/complete", f"{API}/jobs/{{job_id}}/control"):
        assert p in spec["paths"], f"missing endpoint in contract: {p}"


def test_heartbeat_required_fields(spec):
    req, props = _required(spec, f"{API}/heartbeat")
    # The client sends these; CORE must still require exactly the ones we rely on being present.
    assert {"worker_id", "name", "status"} <= req
    assert {"capabilities", "current_job_id", "metadata"} <= props


def test_failed_required_fields(spec):
    req, _ = _required(spec, f"{API}/jobs/{{job_id}}/failed")
    assert {"error_code", "error_message"} <= req


def test_progress_required_fields(spec):
    req, props = _required(spec, f"{API}/jobs/{{job_id}}/progress")
    assert "progress" in req and "stage" in props


def test_claim_takes_worker_id(spec):
    req, _ = _required(spec, f"{API}/claim")
    assert "worker_id" in req


def test_artifacts_multipart_fields(spec):
    _, props = _required(spec, f"{API}/jobs/{{job_id}}/artifacts", content="multipart/form-data")
    assert {"file", "kind"} <= props
    # mime_type must NOT be a form field (MIME travels in the file part) — client relies on this.
    assert "mime_type" not in props


def test_fixture_has_no_secrets(spec):
    blob = json.dumps(spec)
    assert "Bearer " not in blob
    assert "fox_" not in blob  # our generated worker-token prefix must never be in the fixture
