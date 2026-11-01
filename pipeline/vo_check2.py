#!/usr/bin/env python3
"""ASR gate for tts2 specs. args: spec.json dir [--delete-bad]. Prints ratio + transcript, duration; deletes failing wavs if asked."""
import sys, json, re, difflib, pathlib, subprocess
from faster_whisper import WhisperModel
spec = json.load(open(sys.argv[1])); d = pathlib.Path(sys.argv[2]); delete = "--delete-bad" in sys.argv
m = WhisperModel("medium", device="cpu", compute_type="int8")
NUM = {"100%": "cien por cien", "100": "cien"}
BRAND = re.compile(r"me[aá]?[tdc]?[sz]e|meatze|meatse")
def norm(t):
    t = t.lower().replace("meátse", "meatze").replace("baracaldo", "barakaldo")
    for a, b in NUM.items(): t = re.sub(rf"(?<![\w%]){re.escape(a)}(?![\w%])", b, t)
    t = re.sub(r"[^a-záéíóúñü0-9 ]", " ", t)
    return " ".join(w for w in t.split() if w not in ("meatze", "meatse", "meaze", "meatzé"))
bad = []
for k, v in spec.items():
    p = d / f"{k}.wav"
    if not p.exists(): continue
    text = v[1]
    dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nk=1:nw=1", str(p)]))
    hyp = " ".join(s.text for s in m.transcribe(str(p), language="es")[0])
    r = difflib.SequenceMatcher(None, norm(text).split(), norm(hyp).split()).ratio()
    brand_ok = ("meátse" not in text.lower()) or bool(BRAND.search(hyp.lower()))
    rate = len(text.split()) / dur
    flag = r < 0.85 or not brand_ok or len(hyp.split()) > len(text.split()) + 3 or rate < 1.6
    if flag:
        bad.append(k)
        if delete: p.unlink()
    print(f"{'BAD ' if flag else 'ok  '}{r:.2f} {dur:4.1f}s {rate:.1f}w/s {k} | {hyp.strip()}", flush=True)
print("BAD:", len(bad), bad)
