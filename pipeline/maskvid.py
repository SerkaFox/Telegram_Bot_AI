"""maskvid.py in.mp4 out_name [model] — per-frame foreground matte (BiRefNet) -> grayscale mask video <out_name>.mp4 (same fps/size)"""
import sys, subprocess, urllib.parse, urllib.request
sys.path.insert(0, "/home/iaadmin/tg_comfy_bot/outputs")
import make_motion2 as mm
src, out = sys.argv[1], sys.argv[2]; model = sys.argv[3] if len(sys.argv) > 3 else "BiRefNet-general"
fps = eval(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", src]).decode())
N = int(subprocess.check_output(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", src]).decode())
name = mm.upload(src); parts = []
for start in range(0, N, 48):
    g = {"1": {"class_type": "VHS_LoadVideo", "inputs": {"video": name, "force_rate": 0, "custom_width": 0, "custom_height": 0,
               "frame_load_cap": 48, "skip_first_frames": start, "select_every_nth": 1, "format": "AnimateDiff"}},
         "2": {"class_type": "BiRefNetRMBG", "inputs": {"image": ["1", 0], "model": model, "mask_blur": 1, "mask_offset": 0,
               "invert_output": False, "refine_foreground": False, "background": "Alpha"}},
         "3": {"class_type": "VHS_VideoCombine", "inputs": {"images": ["2", 2], "frame_rate": fps, "loop_count": 0, "filename_prefix": "meatze_mask",
               "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 12, "save_metadata": False, "trim_to_audio": False, "pingpong": False, "save_output": True}}}
    res = mm.wait(mm.post(g))
    if not res or res.get("status", {}).get("status_str") != "success": print("FAILED", str((res or {}).get("status"))[:400]); sys.exit(1)
    v = [x for o in res["outputs"].values() for x in o.get("gifs", []) + o.get("videos", [])][-1]
    p = mm.CLIPS / f"_{out}_{start:04d}.mp4"
    p.write_bytes(urllib.request.urlopen(mm.COMFY + "/view?" + urllib.parse.urlencode({"filename": v["filename"], "subfolder": v.get("subfolder", ""), "type": v.get("type", "output")})).read())
    parts.append(p)
lst = mm.CLIPS / f"_{out}.txt"; lst.write_text("".join(f"file '{p}'\n" for p in parts))
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(mm.CLIPS / f"{out}.mp4")], check=True)
for p in parts: p.unlink()
lst.unlink(); print("SAVED", out, flush=True)
