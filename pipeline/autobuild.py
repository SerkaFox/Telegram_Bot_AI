"""Builds every version as soon as all of its clips / voice files exist; rebuilds when an input is newer than the render.
Loops until all 10 are built. Log: v2/logs/autobuild.log"""
import sys, time, pathlib, importlib, json, traceback
sys.path.insert(0, "/home/iaadmin/tg_comfy_bot/outputs"); sys.path.insert(0, "/home/iaadmin/tg_comfy_bot/outputs/v2")
import cut2 as c
V2 = pathlib.Path("/home/iaadmin/tg_comfy_bot/outputs/v2")

def versions():
    out = []
    for mod, fn, name, music, bug in [("v1_cut", None, "V01_Entra_en_la_pantalla", "Bright_Wish", None),
                                      ("v2_cut", None, "V02_Reportaje_IA", "Life_of_Riley", "REPORTAJE")]:
        m = importlib.reload(importlib.import_module(mod))
        out.append((name, lambda m=m: m.items, music, bug, 0.2 if mod == "v1_cut" else 0.16))
    cu = importlib.reload(importlib.import_module("cuts"))
    for vid in ["V03", "V04", "V05", "V06", "V07", "V08", "V09", "V10"]:
        out.append((f"{vid}_{cu.NAMES[vid]}", lambda f=getattr(cu, vid): f(), None, None, 0.18))
    return cu, out

def deps(obj):
    if isinstance(obj, dict): return [p for v in obj.values() for p in deps(v)]
    if isinstance(obj, (list, tuple)): return [p for v in obj for p in deps(v)]
    if isinstance(obj, str) and obj.endswith((".mp4", ".wav")) and obj.startswith("/"): return [obj]
    return []

done = set(n for n in sys.argv[1:])
while True:
    try:
        cu, vers = versions()
        try: cu.robot_voice()
        except Exception: pass
        for name, get, music, bug, mvol in vers:
            if name in done: continue
            try: res = get()
            except FileNotFoundError: continue          # a VO file needed for durations is not ready yet
            items, music, bug = (res if isinstance(res, tuple) else (res, music, bug))
            missing = [p for p in deps(items) if not pathlib.Path(p).exists()]
            if missing: continue
            out = c.REND / f"MEATZE_{name}.mp4"
            print(time.strftime("%H:%M:%S"), "BUILD", name, flush=True)
            c.build(name, items, music, mvol=mvol, bug=bug)
            print(time.strftime("%H:%M:%S"), "BUILT", name, flush=True)
            done.add(name)
    except Exception:
        traceback.print_exc()
    if len(done) >= 10: print("ALL BUILT", flush=True); break
    time.sleep(90)
