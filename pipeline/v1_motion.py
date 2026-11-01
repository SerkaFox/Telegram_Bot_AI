import sys; sys.path.insert(0, "/home/iaadmin/tg_comfy_bot/outputs")
import make_motion2 as mm, pathlib
C1 = "/home/iaadmin/tg_comfy_bot/outputs/v2/codex1/"; C2 = "/home/iaadmin/tg_comfy_bot/outputs/v2/codex2/"
RT = "Real-time natural speed, normal everyday pace. "
SHOTS = [
 ("v1_walk", C1 + "man_c.png", 5, RT + "The man walks forward between the rows of computers toward the camera with relaxed natural steps, glances at a computer, then looks at the camera with a friendly closed-mouth smile. The camera tracks backward smoothly in front of him."),
 ("v1_sit", C1 + "man_desk.png", 5, RT + "The man turns his head from the camera to the monitor and types a few keys on the keyboard; the dark monitor screen lights up with a soft blue glow. Calm, natural movements. Static camera with a slight push-in."),
 ("v1_power", C2 + "hands_power.png", 3, RT + "The finger presses the power button and the monitor screen glows brighter blue. Slight slow camera push-in. Shallow depth of field."),
 ("v1_screen", C1 + "monitor_close.png", 4, RT + "The camera pushes in steadily toward the monitor screen. The woman on the screen blinks, gently tilts her head and gives a light closed-mouth smile; the blue neural network lines glow and pulse softly."),
 ("v1_r_office", C2 + "robot_office.png", 4, RT + "The white robot picks up a sheet of paper and places it into a blue folder; the woman next to it looks at the laptop and nods. Gentle camera dolly to the right."),
 ("v1_r_shop", C2 + "robot_shop.png", 4, RT + "The robot holds the tablet and turns it slightly toward the shop owner, who leans in, looks at it and nods, pleased. Slow camera push-in."),
 ("v1_r_service", C2 + "robot_service.png", 4, RT + "The robot with the headset hands the paper to the young woman, who takes it and smiles politely. Gentle camera move to the left."),
 ("v1_r_home", C2 + "robot_home.png", 4, RT + "The robot places a plate into the dishwasher; in the background the father and child set the table. Slow camera dolly."),
 ("v1_r_teach", C2 + "robot_teach.png", 4, RT + "The robot at the front gestures toward the large screen while the adult students look at it and type on their computers. Slow camera push-in."),
]
only = set(sys.argv[1:])
for name, still, sec, prompt in SHOTS:
    if only and name not in only: continue
    if (mm.CLIPS / f"{name}.mp4").exists(): print("skip", name); continue
    if not pathlib.Path(still).exists(): print("MISSING", still); continue
    mm.run(name, still, sec, prompt, cfg_raw=3.0)
print("V1 MOTION DONE", flush=True)
