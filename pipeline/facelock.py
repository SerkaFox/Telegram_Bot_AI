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

g = {
    "1": {"class_type": "VHS_LoadVideo", "inputs": {"video": mm.upload(a.src), "force_rate": 0, "custom_width": 0, "custom_height": 0,
          "frame_load_cap": 0, "skip_first_frames": 0, "select_every_nth": 1, "format": "AnimateDiff"}},
    "2": {"class_type": "LoadImage", "inputs": {"image": mm.upload(a.woman)}},
    "3": {"class_type": "LoadImage", "inputs": {"image": mm.upload(a.man)}},
}
g["4"] = swap(["1", 0], "2", "female")
g["5"] = swap(["4", 0], "3", "male")
g["6"] = {"class_type": "VHS_VideoCombine", "inputs": {"images": ["5", 0], "frame_rate": fps, "loop_count": 0, "filename_prefix": "meatze_facelock",
          "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 17, "save_metadata": False, "trim_to_audio": False,
          "pingpong": False, "save_output": True}}
pid = mm.post(g); print("submitted", a.out, pid, flush=True)
res = mm.wait(pid)
if not res or res.get("status", {}).get("status_str") != "success":
    print("FAILED", (res or {}).get("status")); sys.exit(1)
import urllib.parse, urllib.request
vid = [v for o in res["outputs"].values() for v in o.get("gifs", []) + o.get("videos", [])][-1]
url = mm.COMFY + "/view?" + urllib.parse.urlencode({"filename": vid["filename"], "subfolder": vid.get("subfolder", ""), "type": vid.get("type", "output")})
dst = mm.CLIPS / f"{a.out}.mp4"; dst.write_bytes(urllib.request.urlopen(url).read()); print("SAVED", dst, flush=True)
