#!/usr/bin/env python3
"""Chatterbox Multilingual (MIT) line synth. args: spec.json out_dir [device]. spec = {id: [voice, text, exaggeration?, cfg?]};
voice = "f"/"m" alias or path to a reference wav. Skips existing wavs."""
import sys, json, pathlib, os
import torch, torchaudio as ta
spec = json.load(open(sys.argv[1])); out = pathlib.Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
dev = sys.argv[3] if len(sys.argv) > 3 else "cpu"
todo = {k: v for k, v in spec.items() if not (out / f"{k}.wav").exists()}
if not todo: print("nothing to do"); sys.exit(0)
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
model = ChatterboxMultilingualTTS.from_pretrained(device=dev)
REF = {"f": os.environ.get("CB_F", ""), "m": os.environ.get("CB_M", "")}
for k, v in todo.items():
    ref = REF.get(v[0], v[0]) or None
    exg = v[2] if len(v) > 2 else 0.5; cfg = v[3] if len(v) > 3 else 0.5
    wav = model.generate(v[1], language_id="es", audio_prompt_path=ref, exaggeration=exg, cfg_weight=cfg)
    ta.save(str(out / f"{k}.wav"), wav, model.sr)
    print("VO", k, flush=True)
