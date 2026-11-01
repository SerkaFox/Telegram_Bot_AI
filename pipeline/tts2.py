#!/usr/bin/env python3
"""XTTS-v2 line synth v2. args: spec.json out_dir. spec = {id: [speaker, text, speed?]}; speaker "f"/"m" or an XTTS speaker name. Skips existing wavs."""
import os, sys, json, pathlib
os.environ["COQUI_TOS_AGREED"] = "1"
spec = json.load(open(sys.argv[1])); out = pathlib.Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
todo = {k: v for k, v in spec.items() if not (out / f"{k}.wav").exists()}
if not todo: print("nothing to do"); sys.exit(0)
from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False, gpu=False)
spks = tts.synthesizer.tts_model.speaker_manager.speaker_names
ALIAS = {"f": "Daisy Studious", "m": os.environ.get("MALE_VOICE", "Luis Moray")}
for k, v in todo.items():
    spk, text = ALIAS.get(v[0], v[0]), v[1]
    speed = v[2] if len(v) > 2 else 1.1
    if spk not in spks: print("NO SPEAKER", spk, flush=True); continue
    tts.tts_to_file(text=text, file_path=str(out / f"{k}.wav"), language="es", speaker=spk, speed=speed)
    print("VO", k, spk, flush=True)
