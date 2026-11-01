"""Sequential GPU job queue for versions 3-9 (motion + talk). Resumable; talk jobs wait for their VO."""
import sys, time, pathlib; sys.path.insert(0, "/home/iaadmin/tg_comfy_bot/outputs")
import make_motion2 as mm, make_talk2 as mt
V = "/home/iaadmin/tg_comfy_bot/outputs/v2/"; C1, C2, C3, C4, C5, C6 = (V + f"codex{i}/" for i in range(1, 7))
KIT = "/home/iaadmin/tg_comfy_bot/MEATZE_CLAUDE_CODE_KIT/assets/"
RT = "Real-time natural speed, normal everyday pace. "
def tr(v, k): return V + f"vo_{v}/trim/{k}.wav"

def M(name, still, sec, prompt): return ("m", name, still, sec, RT + prompt)
def T(name, still, wav, prompt=None, seed=21): return ("t", name, still, wav, prompt or mt.POS, seed)

ANCHOR = ("A professional presenter seated at a table talks calmly and warmly to the camera, lively natural expressions, eyebrows move with "
          "the words, gentle nods, natural blinking, light closed-mouth smile between phrases, correct lip sync, realistic, natural real-time speed.")
T1, T2 = V + "vo_v1/trim/", V + "vo_v2/trim/"
VOX = ("A person is being interviewed on a city street and answers naturally, talking to the interviewer standing just to the left of the camera. "
       "Natural lively facial expressions, eyebrows and head move with the words, natural blinking, small nods, relaxed. "
       "Clear articulate lip movement, correct lip sync. Realistic documentary look, natural real-time speed, slight handheld camera.")
REP = ("A friendly TV reporter on a city street talks warmly to the camera holding a microphone, natural lively expressions, "
       "gentle head movement, natural blinking, light closed-mouth smile between phrases, correct lip sync, realistic, natural real-time speed.")
JOBS = [
 # ---- real classrooms come alive: monitors switch on (empty real photo), then populated class (codex edit of the same room)
 *[M(f"son_{r}", f"{KIT}{'B_Bagatza' if r[0] == 'b' else 'P_second_office'}/{r}.jpeg", 3,
     "The computer monitors in the empty classroom switch on one after another, each screen lighting up with a soft blue glow. "
     "Static camera with a very slight slow push-in. The room stays exactly the same, no people, nothing else changes.")
   for r in ("b2", "b6", "p3", "p4", "p6", "p7")],
 M("cls_b2", C6 + "cls_b2.png", 4, "The teacher draws arrows on the whiteboard with the marker and turns to the class explaining; the students type and look at their screens, one nods. Slow camera dolly."),
 M("cls_b2_board", C6 + "cls_b2_board.png", 3, "The hand draws an arrow connecting two boxes on the whiteboard with the blue marker, the line appears smoothly as it is drawn. Static close-up."),
 M("cls_b6", C6 + "cls_b6.png", 4, "The students type and click, looking at their screens; the teacher walks slowly between the rows and stops to look at a screen. Slow camera pan."),
 M("cls_b3", C6 + "cls_b3.png", 4, "The woman points at the laptop screen and explains; the others lean in, nod and one types. Natural collaborative discussion. Slow camera push-in."),
 M("cls_p3", C6 + "cls_p3.png", 4, "The teacher points at the student's monitor and explains; she nods, types and smiles slightly; other students work at their computers. Slow camera dolly."),
 M("cls_p4", C6 + "cls_p4.png", 4, "The students look at the projector screen; the student lowers her raised hand and the others take notes and type. Slow camera push-in."),
 M("cls_p6", C6 + "cls_p6.png", 4, "In pairs, one student explains and points at the screen, the other nods and types. Slow camera move along the desks."),
 M("cls_p7", C6 + "cls_p7.png", 4, "The teacher gestures toward the diagram on the whiteboard and talks; the students look up from their screens and type. Slow camera push-in."),
 ("m", "flf_b2", f"{KIT}B_Bagatza/b2.jpeg", 5, RT + "Students walk into the empty classroom and sit down at the computers, the monitors switch on, the teacher walks to the whiteboard. Static camera.", C6 + "cls_b2.png"),
 T("v1_tk_w12", C1 + "woman_ai.png", T1 + "v1_w12.wav", seed=7),
 # ---- V1 / V2 (priority)
 T("v1_tk_m1", V + "crops/v1_walk_end.png", T1 + "v1_m1.wav", seed=7),
 T("v1_tk_w4", C1 + "woman_ai.png", T1 + "v1_w4.wav", seed=7),
 T("v1_tk_m4", C1 + "man_a.png", T1 + "v1_m4.wav", seed=7),
 T("v2_tk_w1", C3 + "rep_woman_street.png", T2 + "v2_w1.wav", REP, seed=11),
 T("v2_tk_girl", C3 + "vp_girl.png", T2 + "v2_girl.wav", VOX, seed=11),
 T("v2_tk_man46", C3 + "vp_man46.png", T2 + "v2_man46.wav", VOX, seed=11),
 T("v2_tk_woman35", C3 + "vp_woman35.png", T2 + "v2_woman35.wav", VOX, seed=11),
 T("v2_tk_w2", C3 + "rep_woman_street.png", T2 + "v2_w2.wav", REP, seed=12),
 T("v2_tk_man52", C3 + "vp_man52.png", T2 + "v2_man52.wav", VOX, seed=11),
 T("v2_tk_woman41", C3 + "vp_woman41.png", T2 + "v2_woman41.wav", VOX, seed=11),
 T("v2_tk_man28", C3 + "vp_man28.png", T2 + "v2_man28.wav", VOX, seed=11),
 T("v2_tk_woman58", C3 + "vp_woman58.png", T2 + "v2_woman58.wav", VOX, seed=11),
 T("v2_tk_m1", C3 + "rep_man_lobby.png", T2 + "v2_m1.wav", seed=11),
 T("v2_tk_m2", C1 + "man_b.png", T2 + "v2_m2.wav", seed=11),
 T("v2_tk_m3", C3 + "rep_man_lobby.png", T2 + "v2_m3.wav", seed=13),
 # ---- V3 robots on TV
 M("v3_tv_wide", C5 + "w_tv_green.png", 4, "The woman turns slightly toward the TV and gestures gently at it with an open hand, then looks back at the camera with a light closed-mouth smile; she does not speak. The TV screen stays plain uniform green. Static camera."),
 T("v3_tk_w1", C5 + "w_tv_green.png", tr("v3", "v3_w1"), "A presenter in a classroom talks warmly to the camera and gestures gently toward the TV next to her. The TV screen stays plain uniform green. Natural lively expressions, correct lip sync, natural real-time speed."),
 M("v3_r_web", C5 + "robot_web.png", 4, "The young developer points at the monitor and talks; the white robot turns its head to look and types on the keyboard. Gentle camera dolly."),
 M("v3_r_mkt", C5 + "robot_marketing.png", 4, "The woman points at a tile on the large monitor and smiles; the robot nods and moves the tiles with its hand on the desk. Slow camera push-in."),
 T("v3_tk_m1", C5 + "m_desk_mess.png", tr("v3", "v3_m1"), "A man at a messy desk talks to the camera with an amused, self-deprecating smile, shrugging slightly, lively expressions, correct lip sync, natural real-time speed."),
 T("v3_tk_w2", C1 + "woman_ai.png", tr("v3", "v3_w2"), seed=33),
 T("v3_tk_m2", C5 + "m_board.png", tr("v3", "v3_m2")),
 # ---- V4 myths
 T("v4_tk_m1", C5 + "m_board.png", tr("v4", "v4_m1"), seed=34),
 T("v4_tk_w1", C5 + "w_table.png", tr("v4", "v4_w1"), ANCHOR),
 T("v4_tk_m2", C1 + "man_a.png", tr("v4", "v4_m2"), seed=35),
 T("v4_tk_w2", C5 + "w_table.png", tr("v4", "v4_w2"), ANCHOR, seed=36),
 # ---- V5 a day with Laura
 M("v5_kitchen", C4 + "day_kitchen.png", 4, "The woman sips her coffee, looks at the laptop screen and sighs thoughtfully, then scrolls with one finger on the touchpad. Soft morning light, slow camera push-in."),
 M("v5_street", C4 + "day_street.png", 4, "The woman walks confidently along the street toward the camera with natural steps, the tote bag swinging slightly. The camera tracks backward with her."),
 M("v5_door", C4 + "day_door.png", 4, "The woman arrives at the reception desk and says hello; the receptionist in the navy suit greets her with a polite nod and light closed-mouth smile and hands her a leaflet."),
 M("v5_class", C4 + "day_class.png", 4, "The woman types on the keyboard and looks at the screen with focus, then smiles slightly; the other students work at their computers. Slow camera dolly."),
 M("v5_help", C4 + "day_help.png", 4, "The teacher points at the monitor and explains; the woman nods, understands and smiles, then types. Calm natural movements."),
 M("v5_team", C4 + "day_team.png", 4, "The three classmates look at the monitor together, one points at the screen, they exchange looks and smile naturally. Handheld camera."),
 M("v5_office", C4 + "day_office.png", 4, "The woman explains something to her colleague, pointing at the laptop; the colleague nods. Professional office atmosphere, slow camera move."),
 M("v5_job", C4 + "day_job.png", 4, "The two women shake hands firmly and smile; a brief friendly nod. Slow camera push-in."),
 T("v5_tk_w1", C5 + "w_table.png", tr("v5", "v5_w1"), ANCHOR, seed=37),
 T("v5_tk_m1", C5 + "m_lobby_p2.png", tr("v5", "v5_m1")),
 # ---- V6 before / after
 M("v6_before", C4 + "before_sofa.png", 4, "The tired man scrolls on his phone, sighs, rubs his forehead and lowers the phone, discouraged. Dim warm light, static camera."),
 M("v6_after", C4 + "after_office.png", 4, "The man works confidently at his desk, types, looks at the monitor and nods with a satisfied light smile. Bright daylight, slow camera dolly."),
 T("v6_tk_m1", C1 + "man_b.png", tr("v6", "v6_m1"), seed=38),
 # ---- V7 the future calls
 T("v7_tk_w1", C5 + "w_phone.png", tr("v7", "v7_w1"), "A woman at a reception desk answers the telephone, holding the receiver at her ear, curious and surprised, talks into the phone. Correct lip sync, natural real-time speed."),
 M("v7_robot", C5 + "robot_phone.png", 4, "The white robot holds the telephone receiver to its head and tilts its head as if talking, its blue eyes pulse softly, small friendly gestures with the free hand."),
 T("v7_tk_m1", C5 + "m_board.png", tr("v7", "v7_m1"), seed=39),
 T("v7_tk_w2", C5 + "w_phone.png", tr("v7", "v7_w2"), "A woman at a reception desk holding a telephone receiver at her ear turns to the man next to her and tells him something, light amused smile, correct lip sync, natural real-time speed.", seed=40),
 T("v7_tk_m2", C1 + "man_a.png", tr("v7", "v7_m2"), seed=41),
 T("v7_tk_w3", C5 + "w_table.png", tr("v7", "v7_w3"), ANCHOR, seed=42),
 M("v7_both", C5 + "both_lobby.png", 5, "The two presenters stand side by side, arms relaxed at their sides; they glance at each other, then look at the camera and give one small nod with light closed-mouth smiles. The camera slowly pushes in."),
 # ---- V8 two centres
 T("v8_tk_w1", C5 + "w_call.png", tr("v8", "v8_w1"), "A woman in a classroom holds a tablet and talks to it in a video call, friendly lively expressions, correct lip sync, natural real-time speed."),
 T("v8_tk_w2", C5 + "w_call.png", tr("v8", "v8_w2"), "A woman in a classroom holds a tablet and talks to it in a video call, friendly lively expressions, correct lip sync, natural real-time speed.", seed=43),
 T("v8_tk_m1", C5 + "m_call.png", tr("v8", "v8_m1"), "A man in a classroom holds a tablet and talks to it in a video call, warm friendly lively expressions, correct lip sync, natural real-time speed."),
 M("v8_m_walk", C5 + "m_lobby_p2.png", 4, "The man walks toward the camera through the corridor with relaxed natural steps and a warm smile. The camera tracks backward."),
 T("v8_tk_w3", C1 + "woman_ai.png", tr("v8", "v8_w3"), seed=44),
 # ---- V9 upcoming courses
 T("v9_tk_w1", C5 + "w_table.png", tr("v9", "v9_w1"), ANCHOR, seed=45),
 T("v9_tk_w2", C5 + "w_table.png", tr("v9", "v9_w2"), ANCHOR, seed=46),
 T("v9_tk_m_ai", C5 + "m_board.png", tr("v9", "v9_ai"), seed=47),
]

def done(name): return (mm.CLIPS / f"{name}.mp4").exists()
if __name__ == "__main__":
    only = set(sys.argv[1:])
    pending = [j for j in JOBS if (not only or j[1] in only) and not done(j[1])]
    while pending:
        progressed = False
        for j in list(pending):
            src = j[2]
            if not pathlib.Path(src).exists(): continue
            if j[0] == "t" and not pathlib.Path(j[3]).exists(): continue
            if j[0] == "m" and len(j) > 5 and not pathlib.Path(j[5]).exists(): continue
            if j[0] == "m": mm.run(j[1], src, j[3], j[4], cfg_raw=3.0, last=j[5] if len(j) > 5 else None)
            else: mt.run(j[1], src, j[3], prompt=j[4], seed=j[5])
            pending.remove(j); progressed = True
        if not progressed:
            print("waiting for inputs:", [j[1] for j in pending], flush=True); time.sleep(120)
    print("JOBS DONE", flush=True)
