#!/usr/bin/env python3
"""ASR every VO line, compare with its script text, print + list mismatches (word similarity)."""
import sys, json, re, difflib, pathlib
sys.path.insert(0, "/home/iaadmin/tg_comfy_bot/outputs")
import versions as v
from faster_whisper import WhisperModel
m = WhisperModel("medium", device="cpu", compute_type="int8")
NUM = {"100%": "cien por cien", "100": "cien", "300": "trescientas", "3": "tres", "4": "cuatro", "16": "dieciséis", "17": "diecisiete", "3d": "tres de"}
BRAND = re.compile(r"me[aá]?[tdc]?[sz]e|meatze|meatse")
def norm(t):
    t = t.lower().replace("meátse", "meatze").replace("baracaldo", "barakaldo").replace(".eus", " punto eus")
    for a, b in NUM.items(): t = re.sub(rf"(?<![\w%]){re.escape(a)}(?![\w%])", b, t)
    t = re.sub(r"[^a-záéíóúñü0-9 ]", " ", t)
    return " ".join(w for w in t.split() if w not in ("meatze", "meatse", "meaze", "meatzé"))
bad = []
for k, (g, text) in v.all_vos().items():
    p = v.VO / f"{k}.wav"
    if not p.exists(): continue
    hyp = " ".join(s.text for s in m.transcribe(str(p), language="es")[0])
    r = difflib.SequenceMatcher(None, norm(text).split(), norm(hyp).split()).ratio()
    brand_ok = ("meátse" not in text.lower()) or bool(BRAND.search(hyp.lower()))
    flag = r < 0.85 or not brand_ok or len(hyp.split()) > len(text.split()) + 3
    if flag: bad.append(k)
    print(f"{'BAD ' if flag else 'ok  '}{r:.2f} {k} | {text} || {hyp.strip()}", flush=True)
json.dump(bad, open("/home/iaadmin/tg_comfy_bot/outputs/vo_bad.json", "w"))
print("BAD:", len(bad))
