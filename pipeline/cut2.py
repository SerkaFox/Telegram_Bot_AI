#!/usr/bin/env python3
"""MEATZE v2 editor — real-time cuts at 50 fps, no time-stretching.
A timeline is a list of items rendered to silent 1920x1080@50 segments, joined by hard cuts or short
xfades; voice lines live on their own track (they may run across cuts), talk clips bring their own audio,
music is side-chain ducked under the voice and the master is loudness-normalised for TV (-16 LUFS).
Cards / lower thirds / photo moves reuse the layout helpers from versions.py."""
import hashlib, json, pathlib, subprocess
import versions as v

OUT = v.OUT; V2 = OUT / "v2"; WORK = V2 / "work"; REND = V2 / "renders"
for d in (WORK, REND): d.mkdir(parents=True, exist_ok=True)
W, H, FPS = 1920, 1080, 50
v.FPS = FPS

def dur(p): return v.dur(p)

# ---------------------------------------------------------------- timeline items
def clip(src, d=None, inn=0.0, lt=None, speed=None, tin=None, vo=None, vo_at=0.0, fx=None, pip=None):
    """motion clip; 48 fps sources play at 50 (x1.04) unless speed is given"""
    return {"k": "clip", "src": str(src), "d": d, "inn": inn, "lt": lt, "speed": speed, "tin": tin, "vo": vo, "vo_at": vo_at, "fx": fx,
            "pip": str(pip) if pip else None}

def talk(src, d=None, inn=0.0, lt=None, tin=None, fx=None, pip=None):
    """pip = path of a clip shown in a round 'video call' bubble in the top-right corner"""
    return {"k": "talk", "src": str(src), "d": d, "inn": inn, "lt": lt, "tin": tin, "fx": fx, "pip": str(pip) if pip else None}

def photo(src, move="in", d=3.0, lt=None, tin=None, vo=None, vo_at=0.0):
    return {"k": "photo", "src": str(src), "move": move, "d": d, "lt": lt, "tin": tin, "vo": vo, "vo_at": vo_at}

def card(els, d=3.0, bg=None, icon=None, sweep=False, tin=None, vo=None, vo_at=0.0):
    return {"k": "card", "els": els, "d": d, "bg": str(bg) if bg else None, "icon": icon, "sweep": sweep, "tin": tin, "vo": vo, "vo_at": vo_at}

def real(ss, d, lt=None, tin=None, vo=None, vo_at=0.0):
    return {"k": "real", "ss": ss, "d": d, "lt": lt, "tin": tin, "vo": vo, "vo_at": vo_at}

def tvkey(src, insert, d=None, inn=0.0, ins_inn=0.0, lt=None, tin=None, fx=None):
    """talk/motion clip with a chroma-green TV screen; `insert` video is keyed into the screen"""
    return {"k": "tvkey", "src": str(src), "insert": str(insert), "d": d, "inn": inn, "ins_inn": ins_inn, "lt": lt, "tin": tin, "fx": fx}

def split(left, right, d, labels=("ANTES", "DESPUÉS"), tin=None, vo=None, vo_at=0.0):
    """two motion clips side by side (each 960 wide) with labels"""
    return {"k": "split", "left": str(left), "right": str(right), "d": d, "labels": list(labels), "tin": tin, "vo": vo, "vo_at": vo_at}

def alive(r, d, lt=None, tin=None, vo=None, vo_at=0.0, clips="/home/iaadmin/tg_comfy_bot/outputs/v2/clips/"):
    """real classroom comes alive: monitors switch on in the empty real room, then students fade in (dissolve) and work"""
    a = min(1.6, d * 0.4)
    son = f"{clips}son_{r}.mp4"
    if not pathlib.Path(son).exists():
        return [clip(f"{clips}cls_{r}.mp4", d=d, lt=lt, tin=tin, vo=vo, vo_at=vo_at)]
    return [clip(son, d=a, tin=tin, vo=vo, vo_at=vo_at),
            clip(f"{clips}cls_{r}.mp4", d=d - a + 0.6, lt=lt, tin=("dissolve", 0.6))]

def green_box(src, inn):
    """bbox + key colour of the green screen in the first frame (after cover-scaling to WxH)"""
    import numpy as np
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-ss", f"{inn:.3f}", "-i", src, "-frames:v", "1", "-vf", v.cover(W, H),
                                   "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    a = np.frombuffer(raw, np.uint8).reshape(H, W, 3).astype(int)
    m = (a[..., 1] > 120) & (a[..., 1] > a[..., 0] + 60) & (a[..., 1] > a[..., 2] + 60)
    ys, xs = np.where(m)
    y0, y1 = np.percentile(ys, [0.5, 99.5]).astype(int); x0, x1 = np.percentile(xs, [0.5, 99.5]).astype(int)
    col = np.median(a[m], axis=0).astype(int)
    return x0, y0, x1 - x0, y1 - y0, "0x%02x%02x%02x" % tuple(col)

# ---------------------------------------------------------------- segment render
def enc(inputs, fc, out, D):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", fc, "-map", "[v]", "-an",
                    "-t", f"{D:.3f}", "-r", str(FPS), "-c:v", "libx264", "-preset", "medium", "-crf", "16",
                    "-pix_fmt", "yuv420p", str(out)], check=True)

FX = {
    # gentle punch-in over the clip (adds camera energy to static talk shots)
    "push": "scale=iw*1.08:-1,zoompan=z='1+0.06*on/{N}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={W}x{H}:fps={FPS}",
    # digital screen look for the AI presenter
    "screen": "eq=contrast=1.05:saturation=1.05,noise=alls=6:allf=t,drawgrid=w=iw:h=3:t=1:c=black@0.10",
    "warm": "eq=saturation=0.92:gamma=1.02,colortemperature=temperature=5600",
    "desat": "eq=saturation=0.88",
    "recre": "drawtext=fontfile=" + v.MED + ":text=Recreación:fontsize=24:fontcolor=white@0.75:x=w-text_w-40:y=h-60:shadowcolor=black@0.6:shadowx=1:shadowy=1",
}

def seg(it, out, bug=None):
    k = it["k"]
    over = ",".join(v.chrome(W, H, bug) + v.lower_third(it.get("lt"), W, H)) if k != "card" else ""
    if k == "card" and it["bg"] and it["bg"].endswith(".mp4"):
        # moving background: blurred, darkened clip under the card text
        D = it["d"]; els = ",".join(v.render_els(it["els"], W, H))
        k = max(1 / 1.04, D / max(0.1, dur(it["bg"]) - 0.05))   # stretch the blurred bg to the card length: never loops / jumps
        fc = (f"[0:v]setpts=(PTS-STARTPTS)*{k:.4f},fps={FPS},{v.cover(W, H)},gblur=sigma=10,eq=brightness=-0.06:saturation=0.85,"
              f"drawbox=color={v.NAVY}@0.5:t=fill,vignette=PI/4,{els},fade=in:st=0:d=0.3[v]")
        enc(["-i", it["bg"]], fc, out, D)
        return D
    if k == "card" and not it["bg"] and not it["icon"]:
        # title background with a slow push-in and a soft light sweep (never a frozen frame)
        D = it["d"]; N = int(D * FPS); els = ",".join(v.render_els(it["els"], W, H))
        fc = (f"[0:v]scale={W * 2}:{H * 2},zoompan=z='1+0.07*on/{N}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={W}x{H}:fps={FPS},setsar=1[bg];"
              f"[1:v]format=rgba,geq=r=255:g=255:b=255:a='255*0.16*exp(-pow((X-260)/110,2))'[sw];"
              f"[bg][sw]overlay=x='-520+(W+1040)*t/{max(D, 2.0):.2f}':y=0:shortest=1,{els},fade=in:st=0:d=0.3[v]")
        enc(["-loop", "1", "-framerate", str(FPS), "-t", f"{D + 0.1:.3f}", "-i", str(v.GFX / "title_bg.png"),
             "-f", "lavfi", "-t", f"{D + 0.1:.3f}", "-i", f"color=c=white:s=520x{H}:r={FPS}"], fc, out, D)
        return D
    if k == "card":
        s = {"k": "card", "els": it["els"], "bg": it["bg"], "d": it["d"], "vo": None, "icon": it["icon"], "sweep": it["sweep"]}
        tmp = out.with_suffix(".a.mp4"); D = v.seg_render(s, W, H, None, tmp)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-an", "-c:v", "copy", str(out)], check=True); tmp.unlink()
        return D
    if k == "photo":
        s = {"k": "photo", "src": it["src"], "move": it["move"], "d": it["d"], "vo": None, "lt": it.get("lt")}
        tmp = out.with_suffix(".a.mp4"); D = v.seg_render(s, W, H, bug, tmp)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-an", "-c:v", "copy", str(out)], check=True); tmp.unlink()
        return D
    if k == "real":
        D = it["d"]
        enc(["-ss", str(it["ss"]), "-t", f"{D + 0.2:.3f}", "-i", str(v.V1)], f"[0:v]fps={FPS},{v.cover(W, H)},{over}[v]", out, D)
        return D
    if k == "split":
        D = it["d"]; hw = W // 2
        lab = lambda t, x: (f"drawtext=fontfile={v.BLACK}:text={t}:fontsize=40:fontcolor=white:x={x}:y=60:"
                            f"box=1:boxcolor={v.ACCENT}:boxborderw=14")
        fc = (f"[0:v]setpts=(PTS-STARTPTS)/1.04,fps={FPS},scale={hw}:{H}:force_original_aspect_ratio=increase,crop={hw}:{H},{lab(it['labels'][0], 60)}[l];"
              f"[1:v]setpts=(PTS-STARTPTS)/1.04,fps={FPS},scale={hw}:{H}:force_original_aspect_ratio=increase,crop={hw}:{H},{lab(it['labels'][1], 60)}[r];"
              f"[l][r]hstack=2,drawbox=x={hw - 3}:y=0:w=6:h={H}:color=white:t=fill,{','.join(v.chrome(W, H, bug))}[v]")
        enc(["-i", it["left"], "-i", it["right"]], fc, out, D)
        return D
    if k == "tvkey":
        src = it["src"]; D = min(it["d"] or dur(src) - it["inn"], dur(src) - it["inn"])
        x, y, bw, bh, col = green_box(src, it["inn"] + 0.2)
        x, y, bw, bh = x - 6, y - 6, bw + 12, bh + 12
        fc = (f"[1:v]setpts=PTS-STARTPTS,fps={FPS},scale={bw}:{bh}:force_original_aspect_ratio=increase,crop={bw}:{bh},"
              f"eq=brightness=0.03:saturation=1.05[ins];"
              f"[0:v]setpts=PTS-STARTPTS,fps={FPS},{v.cover(W, H)},split[b][bb];"
              f"[bb]crop={bw}:{bh}:{x}:{y},format=yuva420p,chromakey={col}:0.13:0.05,despill=type=green[kc];"
              f"[ins][kc]overlay=format=auto[tv];[b][tv]overlay={x}:{y}{',' + over if over else ''}[v]")
        enc(["-ss", f"{it['inn']:.3f}", "-i", src, "-stream_loop", "-1", "-ss", f"{it['ins_inn']:.3f}", "-i", it["insert"]], fc, out, D)
        return D
    src = pathlib.Path(it["src"]); L = dur(src)
    sfps = eval(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
                                         "-of", "csv=p=0", str(src)]).decode().strip())
    speed = it.get("speed") or (FPS / sfps if k == "clip" and 44 <= sfps <= 49 else 1.0)
    if k == "talk" and not it["d"]: L = speech_end(src)
    avail = (L - it["inn"]) / speed
    D = min(it["d"] or avail, avail)
    N = int(D * FPS)
    chain = [f"setpts=(PTS-STARTPTS)/{speed:.5f}", f"fps={FPS}", v.cover(W, H)]
    if it.get("fx"):
        for f in it["fx"].split("+"): chain.append(FX[f].format(N=N, W=W, H=H, FPS=FPS))
    if it.get("pip") and pathlib.Path(it["pip"].replace(".mp4", "_mask.mp4")).exists():
        # "pop-out" bubble: the cut-out subject (matte video) may break out of the circle above its centre (head, waving hand)
        R = 380; X, Y = W - R - 90, 210; Sq = 760; CX, CY = 832, 520; PW = int(R * 1.8); Sp = int(Sq * 1.8)
        m = it["pip"].replace(".mp4", "_mask.mp4"); pr = "setpts=(PTS-STARTPTS)/1.04," + f"fps={FPS},pad=iw+600:ih+600:300:300:black"
        cx, cy = CX + 300, CY + 300
        fc = (f"[0:v]{','.join(chain)}[base];"
              f"[1:v]{pr},split[s1][s2];[2:v]{pr},format=gray[mk];"
              f"[s1]crop={Sq}:{Sq}:{cx - Sq // 2}:{cy - Sq // 2},scale={R}:{R},format=rgba,"
              f"geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='if(lte(hypot(X-{R / 2},Y-{R / 2}),{R / 2 - 2}),255,0)',fade=in:st=0.15:d=0.35:alpha=1[pc];"
              f"[s2][mk]alphamerge,crop={Sp}:{Sp}:{cx - Sp // 2}:{cy - Sp // 2},scale={PW}:{PW},"
              f"geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='alpha(X,Y)*max(lte(hypot(X-{PW / 2},Y-{PW / 2}),{R / 2 - 2}),clip(({PW / 2 + R * 0.42}-Y)/60,0,1)*clip(({PW / 2 + R * 0.12}-X)/60,0,1)+clip(({PW / 2 - R * 0.05}-Y)/40,0,1))',"
              f"fade=in:st=0.15:d=0.35:alpha=1[pop];"
              f"color=c=white:s={R + 12}x{R + 12}:r={FPS},format=rgba,geq=r=255:g=255:b=255:a='if(lte(hypot(X-{(R + 12) / 2},Y-{(R + 12) / 2}),{(R + 12) / 2}),235,0)',"
              f"fade=in:st=0.15:d=0.35:alpha=1[ring];"
              f"[base][ring]overlay={X - 6}:{Y - 6}:shortest=1[b2];[b2][pc]overlay={X}:{Y}:shortest=1[b3];"
              f"[b3][pop]overlay={X + R // 2 - PW // 2}:{Y + R // 2 - PW // 2}:shortest=1{(',' + over) if over else ''}[v]")
        enc(["-ss", f"{it['inn']:.3f}", "-i", str(src), "-i", it["pip"], "-i", m], fc, out, D)
        return D
    if it.get("pip"):
        R = 380; X, Y = W - R - 70, 70
        fc = (f"[0:v]{','.join(chain)}[base];"
              f"[1:v]setpts=(PTS-STARTPTS)/1.04,fps={FPS},crop=ih:ih:max(0\\,(iw-ih)/2-iw*0.08):0,scale={R}:{R},format=rgba,"
              f"geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='if(lte(hypot(X-{R / 2},Y-{R / 2}),{R / 2 - 2}),255,0)',fade=in:st=0.15:d=0.35:alpha=1[pc];"
              f"color=c=white:s={R + 12}x{R + 12}:r={FPS},format=rgba,geq=r=255:g=255:b=255:a='if(lte(hypot(X-{(R + 12) / 2},Y-{(R + 12) / 2}),{(R + 12) / 2}),235,0)',"
              f"fade=in:st=0.15:d=0.35:alpha=1[ring];"
              f"[base][ring]overlay={X - 6}:{Y - 6}:shortest=1[b2];[b2][pc]overlay={X}:{Y}:shortest=1{(',' + over) if over else ''}[v]")
        enc(["-ss", f"{it['inn']:.3f}", "-i", str(src), "-stream_loop", "-1", "-i", it["pip"]], fc, out, D)
        return D
    if over: chain.append(over)
    enc(["-ss", f"{it['inn']:.3f}", "-i", str(src)], f"[0:v]{','.join(chain)}[v]", out, D)
    return D

def speech_end(src, thr="-35dB"):
    """time of the last spoken sound: cut only a silence that runs to the very end of the clip (never a pause between sentences)"""
    r = subprocess.run(["ffmpeg", "-i", str(src), "-vn", "-af", f"silencedetect=noise={thr}:d=0.3", "-f", "null", "-"],
                       capture_output=True, text=True).stderr
    L = dur(src)
    st = [float(l.split("silence_start: ")[1]) for l in r.splitlines() if "silence_start" in l]
    en = [float(l.split("silence_end: ")[1].split()[0]) for l in r.splitlines() if "silence_end" in l]
    if st and st[-1] > 0.5 and (len(en) < len(st) or en[-1] >= L - 0.08):
        return min(L, st[-1] + 0.3)
    return L

def talk_audio(it, D, out):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{it['inn']:.3f}", "-i", it["src"], "-vn", "-t", f"{D:.3f}",
                    "-ar", "48000", "-ac", "2", str(out)], check=True)

# ---------------------------------------------------------------- assembly
def build(name, items, music, mvol=0.22, bug=None, credit=True):
    wdir = WORK / name; wdir.mkdir(exist_ok=True)
    segs, starts, t = [], [], 0.0
    voice = []                                   # (wav, start, gain)
    for i, it in enumerate(items):
        key = hashlib.md5((json.dumps(it, sort_keys=True) + str(bug) +
                           "".join(str(pathlib.Path(it[f]).stat().st_mtime) for f in ("src", "insert", "left", "right", "pip") if it.get(f))).encode()).hexdigest()[:10]
        out = wdir / f"s{i:02d}_{key}.mp4"
        if not out.exists(): seg(it, out, bug)
        D = dur(out); tin = it.get("tin")
        if tin and i: t -= tin[1]
        starts.append(t); segs.append((out, D, tin)); t += D
        if it["k"] == "talk":
            wa = wdir / f"s{i:02d}_{key}.wav"
            if not wa.exists(): talk_audio(it, D, wa)
            voice.append((wa, starts[-1], 1.0))
        if it.get("vo"):
            voice.append((pathlib.Path(it["vo"]), starts[-1] + it.get("vo_at", 0.0), 1.0))
    T = t
    ev = sorted((st, st + dur(str(w)), pathlib.Path(w).name) for w, st, g in voice)
    for (a0, a1, an), (b0, b1, bn) in zip(ev, ev[1:]):
        if b0 < a1 - 0.25: print(f"  VOICE OVERLAP {an} [{a0:.1f}-{a1:.1f}] x {bn} [{b0:.1f}-{b1:.1f}]", flush=True)
    for i, (p, D, tin) in enumerate(segs): print(f"  {starts[i]:6.2f} {D:5.2f} {items[i]['k']:5s} {pathlib.Path(items[i].get('src') or items[i].get('bg') or '-').name}", flush=True)
    # video: chain of hard cuts / xfades
    inputs, fc, cur = [], [], "n0"
    for i, (p, _, _) in enumerate(segs):
        inputs += ["-i", str(p)]; fc.append(f"[{i}:v]fps={FPS},settb=1/{FPS},setsar=1[n{i}]")
    for i in range(1, len(segs)):
        tin = segs[i][2]
        if tin:
            fc.append(f"[{cur}][n{i}]xfade=transition={tin[0]}:duration={tin[1]}:offset={starts[i]:.3f},settb=1/{FPS}[x{i}]")
        else:
            fc.append(f"[{cur}][n{i}]concat=n=2:v=1:a=0,settb=1/{FPS}[x{i}]")
        cur = f"x{i}"
    fc.append(f"[{cur}]fade=t=out:st={T - 0.7:.3f}:d=0.7,format=yuv420p[v]")
    # audio: voice bus + ducked music
    n = len(segs); ain = []
    for j, (wav, st, g) in enumerate(voice):
        inputs += ["-i", str(wav)]
        ms = int(round(st * 1000))
        fc.append(f"[{n + j}:a]aformat=sample_rates=48000:channel_layouts=stereo,volume={g},adelay={ms}|{ms}[vo{j}]"); ain.append(f"[vo{j}]")
    inputs += ["-stream_loop", "-1", "-i", str(v.MUS / f"{music}.mp3")]
    mi = n + len(voice)
    fc.append(f"{''.join(ain)}amix=inputs={len(ain)}:duration=longest:normalize=0,apad,atrim=0:{T:.3f},asplit[vb][vk]")
    fc.append(f"[{mi}:a]volume={mvol},atrim=0:{T:.3f},afade=t=in:d=0.6,afade=t=out:st={T - 2.0:.3f}:d=2.0,"
              f"aformat=sample_rates=48000:channel_layouts=stereo[m]")
    fc.append("[m][vk]sidechaincompress=threshold=0.025:ratio=10:attack=10:release=350[md]")
    fc.append("[vb][md]amix=inputs=2:duration=first:normalize=0,loudnorm=I=-16.5:TP=-1.5:LRA=11[a]")
    final = REND / f"MEATZE_{name}.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", ";".join(fc), "-map", "[v]", "-map", "[a]",
                    "-t", f"{T:.3f}", "-r", str(FPS), "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-maxrate", "5500k", "-bufsize", "11M",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", str(final)], check=True)
    print(f"{name}: {final} {dur(final):.1f}s", flush=True)
    return final
