#!/usr/bin/env python3
"""MEATZE shot library: varied, professional shots shared by the 10 edit versions.
Phase 1 stills (Qwen-Image-Edit two/three-ref + ReActor), phase 2 talking-line TTS,
phase 3 motion (clean Wan 2.2 I2V, strict professional prompts, x2 upscale),
phase 4 talking close-ups (InfiniteTalk). Resumable: anything already on disk is skipped."""
import json, subprocess, sys, time, pathlib, urllib.request, urllib.parse

ROOT = pathlib.Path("/home/iaadmin/tg_comfy_bot")
OUT = ROOT / "outputs"; STILLS = OUT / "stills"; CLIPS = OUT / "lib"; VOICES = OUT / "voices"
for d in (STILLS, CLIPS, VOICES): d.mkdir(parents=True, exist_ok=True)
CIN = pathlib.Path("/home/iaadmin/ComfyUI/input")
PY = "/home/iaadmin/ComfyUI/venv/bin/python"
LOG = OUT / "library_log.txt"
sys.path.insert(0, str(OUT))
import make_fullbody as fb   # noqa: E402
import batch_meatze as bm    # noqa: E402  (chest-up still recipe + post/wait/fetch)

WOMAN_WR = "a navy business suit with a white blouse, hair in a low ponytail"
MAN_WR = "a light blue shirt and dark trousers"
WHO = {"f": "A professional woman in a navy business suit with a white blouse, hair in a low ponytail, in a training centre.",
       "m": "A professional man in a light blue shirt and dark trousers, in a training centre."}
STYLE = ("Serious professional demeanor, mouth closed, only a slight closed-mouth smile without showing teeth, upright composed posture, minimal "
         "dignified movement, no swaying. Corporate informational video, realistic, natural light.")

def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"; print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")

# ---------- phase 1: stills ----------
STILL_SPECS = [  # id, presenter, room, gender, pose
    ("w_reception", "meatze_woman.jpg", "meatze_room_b1.jpg", "female", "standing beside the reception desk, upright, hands relaxed, about to greet"),
    ("w_hall",      "meatze_woman.jpg", "meatze_room_b8.jpg", "female", "walking toward the camera through the hall, mid-stride, upright"),
    ("w_class",     "meatze_woman.jpg", "meatze_room_b2.jpg", "female", "standing in the aisle of the computer classroom, one hand slightly extended toward the computers"),
    ("w_meeting",   "meatze_woman.jpg", "meatze_room_b3.jpg", "female", "standing next to the meeting table with laptops, one hand resting near the table"),
    ("m_lobby",     "meatze_man.jpg",   "meatze_room_p2.jpg", "male",   "standing in the lobby next to the MEATZE floor mat, upright, about to greet"),
    ("m_class",     "meatze_man.jpg",   "meatze_room_p3.jpg", "male",   "standing in the computer classroom beside the desks, one hand gesturing toward the screens"),
    ("m_walk",      "meatze_man.jpg",   "meatze_room_p4.jpg", "male",   "walking slowly toward the camera along the aisle between the desks, mid-stride, upright"),
    ("m_class2",    "meatze_man.jpg",   "meatze_room_p7.jpg", "male",   "standing at the front of the classroom, upright, presenting the room"),
]

def both_wf(seed=777):
    pos = ("A photorealistic wide photograph of the woman from the first image and the man from the second image "
           "standing side by side at the reception desk of the room in the third image, full figures visible, "
           "upright professional posture, slight polite smiles, facing the camera. The woman wears " + WOMAN_WR +
           "; the man wears " + MAN_WR + ". Keep both exact faces. Keep the room EXACTLY as in the third image. "
           "Realistic scale, natural lighting, soft contact shadows, photographic.")
    neg = ("extra people, third person, duplicated face, swapped genders, cartoon, deformed hands, cropped heads, "
           "distorted furniture, text, watermark, nudity")
    g = fb.wf("meatze_woman.jpg", "meatze_room_b1.jpg", "female", WOMAN_WR, "", seed, 1152, 640)
    g["43"] = {"class_type": "LoadImage", "inputs": {"image": "meatze_man.jpg"}}
    g["44"] = {"class_type": "LoadImage", "inputs": {"image": "meatze_room_b1.jpg"}}
    for nid, txt in (("68", pos), ("69", neg)):
        g[nid] = {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["72", 0], "prompt": txt, "vae": ["71", 0],
                  "image1": ["41", 0], "image2": ["43", 0], "image3": ["44", 0]}}
    g["80"] = fb.reactor(["8", 0], "41", "female")
    g["81"] = fb.reactor(["80", 0], "43", "male")
    g["128"]["inputs"]["images"] = ["81", 0]
    return g

def run_graph_image(graph, dst):
    pid = fb.post(graph); res = fb.wait(pid)
    if not res or res.get("status", {}).get("status_str") != "success":
        raise RuntimeError(f"graph failed {res.get('status') if res else 'timeout'}")
    img = None
    for o in res.get("outputs", {}).values():
        for im in o.get("images", []): img = im
    bm.fetch(img, dst)

def phase_stills():
    for sid, pres, room, g, pose in STILL_SPECS:
        dst = STILLS / f"lib_{sid}.jpg"
        if dst.exists(): continue
        log(f"still {sid}")
        try:
            wr = WOMAN_WR if g == "female" else MAN_WR
            run_graph_image(fb.wf(pres, room, g, wr, pose, seed=abs(hash(sid)) % 99999, w=1152, h=640), dst)
            (CIN / f"meatze_lib_{sid}.jpg").write_bytes(dst.read_bytes())
        except Exception as e: log(f"  FAIL {sid}: {e}")
    dst = STILLS / "lib_both.jpg"
    if not dst.exists():
        log("still both (3-ref)")
        try:
            run_graph_image(both_wf(), dst); (CIN / "meatze_lib_both.jpg").write_bytes(dst.read_bytes())
        except Exception as e: log(f"  FAIL both: {e}")
    dst = STILLS / "lib_m_chest_lobby.jpg"
    if not dst.exists():
        log("still man chest (lobby)")
        try:
            run_graph_image(bm.still_wf("meatze_man.jpg", "meatze_room_p2.jpg", "male", MAN_WR, seed=5151), dst)
            (CIN / "meatze_lib_m_chest_lobby.jpg").write_bytes(dst.read_bytes())
        except Exception as e: log(f"  FAIL chest: {e}")

# ---------- phase 2: talking-line TTS ----------
TALK_LINES = {
    "t_w_hello": ("f", "Hola, bienvenidos a Meátse. Hoy te enseño nuestros centros y cómo formarte para encontrar empleo."),
    "t_w_ai":    ("f", "Y una gran novedad: ya ha comenzado nuestra formación en inteligencia artificial."),
    "t_m_subv":  ("m", "Para personas desempleadas, nuestros cursos son cien por cien subvencionados e incluyen prácticas en empresa."),
    "t_m_close": ("m", "Infórmate sin compromiso en Barakaldo o en Bilbao. Te esperamos en Meátse."),
}

def phase_tts():
    need = {k: v for k, v in TALK_LINES.items() if not (CIN / f"meatze_{k}.wav").exists()}
    if not need: return
    log(f"tts talking lines {list(need)}")
    spec = OUT / "talk_lines.json"; spec.write_text(json.dumps(need, ensure_ascii=False))
    subprocess.run(["/home/iaadmin/tts_venv/bin/python", str(OUT / "tts_lines.py"), str(spec), str(VOICES)], check=True)
    for k in need:
        raw = VOICES / f"{k}.wav"
        dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nk=1:nw=1", str(raw)]))
        total = max(6.0, dur + 1.0)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-af", f"adelay=400|400,apad=whole_dur={total:.2f}",
                        "-ar", "24000", "-ac", "1", str(CIN / f"meatze_{k}.wav")], check=True)

# ---------- phase 3: motion ----------
MOTION = [  # out, still(in comfy input), gender, action+camera, w, h
    ("mo_w_welcome",  "meatze_lib_w_reception.jpg", "f", "She gives one brief polite welcoming wave and a small nod, then lowers her hand. Slow smooth camera dolly-in.", 832, 480),
    ("mo_both",       "meatze_lib_both.jpg",        "both", "", 832, 480),
    ("mo_w_hall",     "meatze_lib_w_hall.jpg",      "f", "She walks calmly toward the camera through the hall; the camera smoothly tracks backward ahead of her.", 832, 480),
    ("mo_w_class",    "meatze_lib_w_class2.jpg",    "f", "She keeps facing the camera and calmly gestures with one open hand toward the computers beside her, then lowers her hand. She never turns her back. Slow lateral camera move.", 832, 480),
    ("mo_w_bigclass", "meatze_fb_shot3w.jpg",       "f", "She calmly presents the large computer classroom with one open hand. Slow smooth camera pan across the room.", 832, 480),
    ("mo_w_meeting",  "meatze_lib_w_meeting.jpg",   "f", "She calmly gestures toward the laptops on the table as if explaining. Slow camera push-in.", 832, 480),
    ("mo_m_lobby",    "meatze_lib_m_lobby.jpg",     "m", "He gives one brief polite welcoming wave and a small nod. Slow smooth camera dolly-in.", 832, 480),
    ("mo_m_class",    "meatze_lib_m_class.jpg",     "m", "He calmly presents the screens with one open hand, then looks at the camera. Slow lateral camera move.", 832, 480),
    ("mo_m_walk",     "meatze_lib_m_walk.jpg",       "m", "He walks calmly between the desks of the classroom; the camera tracks smoothly with him.", 832, 480),
    ("mo_m_class2",   "meatze_lib_m_class2.jpg",    "m", "He calmly extends one open palm sideways toward the rows of computers to present the classroom, then lowers his hand and looks at the camera. No pointing finger. Slow camera push-in.", 832, 480),
]

def motion_prompt(g, action):
    if g == "both":
        return ("A professional woman in a navy business suit and a professional man in a light blue shirt stand still side by side "
                "at a training-centre reception with their arms relaxed at their sides the whole time. They only blink naturally "
                "and give one small calm nod. No hand gestures at all. The camera slowly dollies in. " + STYLE)
    return f"{WHO[g]} {action} {STYLE}"

def phase_motion():
    for out, still, g, action, w, h in MOTION:
        dst = CLIPS / f"{out}.mp4"
        if dst.exists(): continue
        if not (CIN / still).exists(): log(f"  skip {out}: still missing"); continue
        log(f"motion {out}")
        r = subprocess.run([PY, str(OUT / "make_motion.py"), out, still, "5", motion_prompt(g, action), str(w), str(h)],
                           capture_output=True, text=True)
        src = OUT / "clips" / f"{out}.mp4"
        if r.returncode == 0 and src.exists(): src.rename(dst); log(f"  ok {out}")
        else: log(f"  FAIL {out}: {r.stdout[-300:]} {r.stderr[-300:]}")

# ---------- phase 4: talking ----------
TALK = [("tk_w_hello", "meatze_lib_w_chest_b4.jpg", "t_w_hello"),
        ("tk_w_ai",    "meatze_lib_w_chest_b2.jpg", "t_w_ai"),
        ("tk_m_subv",  "meatze_chest_shot10m.jpg", "t_m_subv"),
        ("tk_m_close", "meatze_lib_m_chest_lobby.jpg", "t_m_close")]

def phase_talk():
    for out, still, line in TALK:
        dst = CLIPS / f"{out}.mp4"
        if dst.exists(): continue
        if not (CIN / still).exists(): log(f"  skip {out}: still missing"); continue
        log(f"talk {out}")
        r = subprocess.run([PY, str(OUT / "make_talk.py"), out, still, f"meatze_{line}.wav"], capture_output=True, text=True)
        src = OUT / "clips" / f"{out}.mp4"
        if r.returncode == 0 and src.exists(): src.rename(dst); log(f"  ok {out}")
        else: log(f"  FAIL {out}: {r.stdout[-300:]} {r.stderr[-300:]}")

if __name__ == "__main__":
    phases = sys.argv[1:] or ["stills", "tts", "motion", "talk"]
    log(f"LIBRARY START {phases}")
    for p in phases: globals()[f"phase_{p}"]()
    log("LIBRARY DONE")
