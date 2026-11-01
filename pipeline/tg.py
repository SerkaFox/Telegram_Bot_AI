#!/usr/bin/env python3
"""Telegram helper: tg.py msg "text" | tg.py video file "caption" | tg.py audio file "caption" | tg.py album "caption" img1 img2 ..."""
import sys, subprocess, json
tok = subprocess.check_output(["grep", "-oP", r"TELEGRAM_BOT_TOKEN=\K\S+", "/etc/systemd/system/tg-comfy-bot.service"]).decode().strip()
API = f"https://api.telegram.org/bot{tok}/"; CHAT = "517188056"
def call(method, *args):
    r = subprocess.run(["curl", "-s", API + method, "-F", f"chat_id={CHAT}", *args], capture_output=True, text=True)
    ok = json.loads(r.stdout or "{}").get("ok"); print(method, "ok" if ok else ("FAIL " + r.stdout[:200]))
kind = sys.argv[1]
if kind == "msg": call("sendMessage", "-F", f"text={sys.argv[2]}")
elif kind == "video": call("sendVideo", "-F", f"video=@{sys.argv[2]}", "-F", f"caption={sys.argv[3] if len(sys.argv) > 3 else ''}", "-F", "supports_streaming=true")
elif kind == "audio": call("sendAudio", "-F", f"audio=@{sys.argv[2]}", "-F", f"caption={sys.argv[3] if len(sys.argv) > 3 else ''}")
elif kind == "album":
    files = sys.argv[3:]
    media = [{"type": "photo", "media": f"attach://f{i}", **({"caption": sys.argv[2]} if i == 0 else {})} for i, _ in enumerate(files)]
    call("sendMediaGroup", "-F", "media=" + json.dumps(media, ensure_ascii=False), *sum([["-F", f"f{i}=@{f}"] for i, f in enumerate(files)], []))
