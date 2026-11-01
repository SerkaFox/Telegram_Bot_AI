"""facelock.py in.mp4 out_name [--woman ref.png] [--man ref.png] — keep presenter identities stable in a motion clip
(Wan drifts faces when the camera moves). ReActor per frame, gender-matched: female face <- woman ref, male face <- man ref."""
import sys, argparse, pathlib, subprocess
sys.path.insert(0, "/home/iaadmin/tg_comfy_bot/outputs")
import make_motion2 as mm

V = "/home/iaadmin/tg_comfy_bot/outputs/v2/"
ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("out")
ap.add_argument("--woman", default=V + "codex1/woman_ai.png"); ap.add_argument("--man", default=V + "codex1/man_a.png")
a = ap.parse_args()
fps = eval(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", a.src]).decode())

def swap(chain, src, gender):
    return {"class_type": "ReActorFaceSwap", "inputs": {
        "enabled": True, "input_image": chain, "source_image": [src, 0], "swap_model": "inswapper_128.onnx",
        "facedetection": "retinaface_resnet50", "face_restore_model": "GFPGANv1.3.pth", "face_restore_visibility": 0.55,
        "codeformer_weight": 0.5, "detect_gender_input": gender, "detect_gender_source": gender,
        "input_faces_index": "0", "source_faces_index": "0", "console_log_level": 0}}

N = int(subprocess.check_output(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", a.src]).decode())
import urllib.parse, urllib.request, tempfile
vname, wname, mname = mm.upload(a.src), mm.upload(a.woman), mm.upload(a.man)
parts = []; CH = 40
for start in range(0, N, CH):     # chunks: a whole 1664x960 clip in one ReActor batch runs out of VRAM
    g = {
        "1": {"class_type": "VHS_LoadVideo", "inputs": {"video": vname, "force_rate": 0, "custom_width": 0, "custom_height": 0,
              "frame_load_cap": CH, "skip_first_frames": start, "select_every_nth": 1, "format": "AnimateDiff"}},
        "2": {"class_type": "LoadImage", "inputs": {"image": wname}},
        "3": {"class_type": "LoadImage", "inputs": {"image": mname}},
    }
    g["4"] = swap(["1", 0], "2", "female")
    g["5"] = swap(["4", 0], "3", "male")
    g["6"] = {"class_type": "VHS_VideoCombine", "inputs": {"images": ["5", 0], "frame_rate": fps, "loop_count": 0, "filename_prefix": "meatze_facelock",
              "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 15, "save_metadata": False, "trim_to_audio": False,
              "pingpong": False, "save_output": True}}
    pid = mm.post(g); res = mm.wait(pid)
    if not res or res.get("status", {}).get("status_str") != "success":
        print("FAILED chunk", start, str((res or {}).get("status"))[:300]); sys.exit(1)
    vid = [v for o in res["outputs"].values() for v in o.get("gifs", []) + o.get("videos", [])][-1]
    url = mm.COMFY + "/view?" + urllib.parse.urlencode({"filename": vid["filename"], "subfolder": vid.get("subfolder", ""), "type": vid.get("type", "output")})
    part = mm.CLIPS / f"_{a.out}_{start:04d}.mp4"; part.write_bytes(urllib.request.urlopen(url).read()); parts.append(part)
    print("chunk", start, flush=True)
lst = mm.CLIPS / f"_{a.out}.txt"; lst.write_text("".join(f"file '{p}'\n" for p in parts))
dst = mm.CLIPS / f"{a.out}.mp4"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(dst)], check=True)
for p in parts: p.unlink()
lst.unlink(); print("SAVED", dst, flush=True)
