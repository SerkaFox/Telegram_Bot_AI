#!/usr/bin/env python3
"""Generic XTTS-v2 line synth. args: spec.json out_dir. spec = {id: ["f"|"m", text]}. Skips existing wavs."""
import os, sys, json, pathlib
os.environ["COQUI_TOS_AGREED"] = "1"
spec = json.load(open(sys.argv[1])); out = pathlib.Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
todo = {k: v for k, v in spec.items() if not (out / f"{k}.wav").exists()}
if not todo: print("nothing to do"); sys.exit(0)
from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False, gpu=False)
spks = tts.synthesizer.tts_model.speaker_manager.speaker_names
def pick(c):
    for x in c:
        if x in spks: return x
    return spks[0]
F = pick(["Daisy Studious", "Sofia Hellen"]); M = pick(["Damien Black", "Viktor Eka", "Luis Moray"])
for k, (g, text) in todo.items():
    tts.tts_to_file(text=text, file_path=str(out / f"{k}.wav"), language="es", speaker=F if g == "f" else M)
    print("VO", k, flush=True)
