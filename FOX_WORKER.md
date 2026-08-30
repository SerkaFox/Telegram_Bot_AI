# FOX MEDIA WORKER

Turns this generation box into an **outbound-only compute worker** for the external **FOX MIX**
system, without touching the existing Telegram bot. The bot, ComfyUI (`127.0.0.1:8188`) and Ollama
(`127.0.0.1:11434`) stay exactly as they were and are **never exposed to the internet**.

## Principle

```
FOX MIX CORE  (internet)
   ▲  │            worker makes ONLY outbound calls — no inbound port is opened here
   │  ▼
FOX MEDIA WORKER  (python -m fox_worker.main)
   │
   ├── generator/  (non-Telegram service layer) ──► ComfyUI 127.0.0.1:8188  (reuses MopMix workflow)
   └────────────────────────────────────────────►  Ollama  127.0.0.1:11434
tg-comfy-bot.service keeps running independently, and keeps GPU priority.
```

## Components

| Path | Role |
|------|------|
| `generator/service.py` | `SafeGeneratorService` / `AdultGeneratorService` — hard safe/adult split. Adult refuses with `unsupported_in_current_phase`. Includes `FOX_WORKER_MOCK_GENERATION` path (no GPU). |
| `generator/image.py` | Safe MopMix image generation, reusing `telegram_comfyui_bot.py` (no new workflow). Yields the GPU to the bot via a ComfyUI-queue gate. |
| `fox_worker/config.py` | Loads `.env` (never overrides real env), resolves jobs dir with fallback. |
| `fox_worker/models.py` | `Job` parsed from the claim payload. |
| `fox_worker/capabilities.py` | Capability registry; only `safe.image.mopmix` is ENABLED. |
| `fox_worker/client.py` | Protocol V1 HTTP client (Bearer auth, retry/backoff, never logs token). |
| `fox_worker/artifacts.py` | Per-job workspace lifecycle + retention prune. |
| `fox_worker/worker.py` | Single async loop: heartbeat + claim → started → progress → artifacts → complete/failed, with cancel. |
| `fox_worker/main.py` | Entrypoint. |
| `tests/mock_fox_core.py` | Stdlib mock FOX CORE implementing Protocol V1. |
| `tests/test_worker_protocol.py` | Full protocol test-suite (no GPU). |
| `deploy/fox-media-worker.service` | systemd unit (separate from the bot). |

## Protocol V1 (worker → FOX CORE, base `FOX_CORE_URL` + `/internal/generator/v1`)

| Call | Purpose |
|------|---------|
| `POST /jobs/claim` | `{worker_id, capabilities}` → `204` (idle) or job JSON `{id,type,content_class,prompt,negative_prompt,count,quality,options}` |
| `POST /jobs/{id}/started` | execution began |
| `POST /jobs/{id}/progress` | `{progress:0-100, stage}` |
| `POST /jobs/{id}/artifacts` | multipart `file,kind,mime_type,width?,height?,duration?` → `{artifact_id}` |
| `POST /jobs/{id}/complete` | `{artifact_ids[], metadata{engine,mode,generation_seconds}}` |
| `POST /jobs/{id}/failed` | `{error_code, message, retryable}` |
| `POST /heartbeat` | `{worker_id, status:idle\|busy\|error, job_id, capabilities, local_queue}` (~every 30s) |
| `GET  /jobs/{id}/control` | → `{cancel: bool}` (cooperative cancel) |

All calls send `Authorization: Bearer <FOX_WORKER_TOKEN>`.

## Capabilities

Advertised catalogue: `safe.image.mopmix`, `safe.image.edit`, `safe.video.clean`, `safe.video.wan`,
`safe.video.ltx`, `safe.video.v2v`, `adult.image`, `adult.video.wan`, `adult.video.eros`.

**Enabled now:** `safe.image.mopmix` only. Everything else (all video, all adult) is refused with
`unsupported_in_current_phase` until wired + tested in a later phase.

## GPU sharing

One physical GPU. The worker runs one job at a time and, before each render, waits while ComfyUI's
own queue is non-empty — so interactive Telegram/owner jobs always take priority. No change to the
bot was needed.

## Configure & run

1. Fill `.env` (copy from `.env.example`): set `FOX_CORE_URL` and the real `FOX_WORKER_TOKEN`.
2. Test offline: `python -m pytest tests/ -q`
3. Dry-run against the mock: `python -m tests.mock_fox_core` (one shell) + point `.env` at it.
4. Install the service (after approval): see header of `deploy/fox-media-worker.service`.

Set `FOX_WORKER_MOCK_GENERATION=true` to exercise the whole protocol with tiny fake artifacts (no GPU).
