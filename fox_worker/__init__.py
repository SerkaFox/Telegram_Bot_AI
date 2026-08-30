"""FOX MEDIA WORKER — outbound-only compute worker for FOX MIX CORE.

Runs alongside (never replacing) the Telegram bot. Makes only OUTBOUND HTTPS calls to FOX CORE;
opens no inbound port. Drives generation through the non-Telegram `generator` service layer, which
reuses the existing ComfyUI workflows. Localhost ComfyUI (8188) and Ollama (11434) are never exposed.
"""
