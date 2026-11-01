#!/usr/bin/env python3
"""Transcribe a rendered version (es) with timestamps to verify VO lines, order and cut-offs."""
import sys
from faster_whisper import WhisperModel
m = WhisperModel("medium", device="cpu", compute_type="int8")
segs, info = m.transcribe(sys.argv[1], language="es", vad_filter=True)
for s in segs: print(f"[{s.start:6.1f}-{s.end:6.1f}] {s.text.strip()}")
