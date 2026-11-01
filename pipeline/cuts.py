"""Edit timelines for versions 3-10 (V1/V2 live in v1_cut.py / v2_cut.py).
python v2/cuts.py V03 V05 ...  -> renders into v2/renders/"""
import sys, subprocess; sys.path.insert(0, "/home/iaadmin/tg_comfy_bot/outputs")
import cut2 as c, versions as v
from versions import E, BLACK, MED, SEMI, SOFT, ACCENT
V = "/home/iaadmin/tg_comfy_bot/outputs/v2/"; CL = V + "clips/"; C4, C5 = V + "codex4/", V + "codex5/"
FADE = ("fade", 0.3)
def vo(ver, k): return V + f"vo_{ver}/trim/{k}.wav"
def vd(ver, k, pad=0.35): return v.dur(vo(ver, k)) + pad
def intro(theme): return [c.real(1.0, 1.4), c.card(v.intro_card(theme), d=2.4, sweep=True, tin=FADE)]
def outro(): return [c.card(v.contact_card(v.FOOT), d=5.5, tin=FADE)]
RED = "0xd62828"
def myth(n, text):
    return [E(f"MITO {n}", BLACK, 40, "white", 0.3, d=0.1, box=RED), E(f"«{text}»", BLACK, 80, "white", 0.42, d=0.35)]
SUBV = v.title_card("100% subvencionado", "Para personas desempleadas", kicker="Barakaldo · Bilbao")

def V03():
    a = "v3"
    return intro("LA IA EN TU DÍA A DÍA") + [
        c.talk(CL + "v3_tk_w1.mp4", lt=("Inteligencia artificial", "En el día a día"), tin=FADE),
        c.tvkey(CL + "v3_tv_wide.mp4", CL + "v1_r_teach.mp4", d=2.8),
        c.clip(CL + "v1_r_office.mp4", d=vd(a, "v3_f_office"), vo=vo(a, "v3_f_office"), lt=("En la oficina", None), tin=("zoomin", 0.4)),
        c.clip(CL + "v1_r_shop.mp4", d=vd(a, "v3_f_shop") / 2, vo=vo(a, "v3_f_shop"), lt=("En el comercio", None)),
        c.clip(CL + "v3_r_mkt.mp4", d=vd(a, "v3_f_shop") / 2),
        c.clip(CL + "v1_r_service.mp4", d=vd(a, "v3_f_service"), vo=vo(a, "v3_f_service"), lt=("Atención al cliente", None)),
        c.clip(CL + "v1_r_home.mp4", d=vd(a, "v3_f_home", 0.6), vo=vo(a, "v3_f_home"), lt=("En casa", None)),
        c.talk(CL + "v3_tk_m1.mp4", tin=("fade", 0.2)),
        c.clip(CL + "v3_desk_wide.mp4", d=vd(a, "v3_w2", 0.4), vo=vo(a, "v3_w2"), vo_at=0.15),
        c.talk(CL + "v3_tk_m2.mp4"),
        c.card(v.ai_card(), d=vd(a, "v3_f_ai", 0.6), bg="ai", icon="ia", vo=vo(a, "v3_f_ai"), vo_at=0.2, tin=FADE),
        *c.alive("p6", vd(a, "v3_m_subv") / 2, lt=("100% subvencionado", "Barakaldo"), vo=vo(a, "v3_m_subv"), vo_at=0.1, tin=FADE),
        *c.alive("p3", vd(a, "v3_m_subv") / 2 + 0.3, lt=("Para personas desempleadas", "Bilbao")),
    ] + outro(), "Easy_Lemon", None

def V04():
    a = "v4"
    return intro("MITOS Y REALIDADES") + [
        c.talk(CL + "v4_tk_m1.mp4", tin=FADE),
        c.card(myth(1, "La IA es solo para informáticos"), d=vd(a, "v4_f_mito1", 0.5), vo=vo(a, "v4_f_mito1"), vo_at=0.2, tin=FADE),
        c.talk(CL + "v4_tk_w1.mp4", lt=("REALIDAD", "Se usa en casi cualquier trabajo"), tin=("wipeleft", 0.3)),
        c.card(myth(2, "Ya es tarde para aprender"), d=vd(a, "v4_m_mito2", 0.5), vo=vo(a, "v4_m_mito2"), vo_at=0.2, tin=FADE),
        c.talk(CL + "v2_tk_woman58.mp4", fx="recre", tin=("wipeleft", 0.3)),
        c.talk(CL + "v4_tk_m2.mp4", lt=("REALIDAD", "Se aprende practicando")),
        c.card(myth(3, "La IA lo hace todo sola"), d=vd(a, "v4_f_mito3", 0.5), vo=vo(a, "v4_f_mito3"), vo_at=0.2, tin=FADE),
        c.talk(CL + "v2_tk_man28.mp4", fx="recre", tin=("wipeleft", 0.3)),
        c.talk(CL + "v4_tk_w2.mp4", lt=("REALIDAD", "Necesita personas con criterio")),
        c.card(v.ai_card(), d=vd(a, "v4_m_ai", 0.6), bg="ai", icon="ia", vo=vo(a, "v4_m_ai"), vo_at=0.2, tin=FADE),
        c.card(SUBV, d=vd(a, "v4_f_subv", 0.5), bg=CL + "cls_b6.mp4", vo=vo(a, "v4_f_subv"), vo_at=0.2, tin=FADE),
    ] + outro(), "Wallpaper", "MITOS Y REALIDADES"

def V05():
    a = "v5"
    L = lambda f, k, **kw: c.clip(CL + f + ".mp4", d=vd(a, k, 0.5), vo=vo(a, k), vo_at=0.15, fx="recre", **kw)
    return intro("UN DÍA CON LAURA") + [
        c.clip(CL + "v5_kitchen.mp4", vo=vo(a, "v5_n1"), vo_at=0.15, fx="recre", tin=FADE),
        c.clip(CL + "v5_kitchen_b.mp4", d=max(0.8, vd(a, "v5_n1", 0.5) - 3.8), fx="recre"),
        L("v5_street", "v5_n2"),
        c.clip(CL + "v5_door.mp4", vo=vo(a, "v5_n3"), vo_at=0.15, fx="recre", lt=("Barakaldo", None)),
        c.clip(CL + "v5_door_b.mp4", d=max(0.8, vd(a, "v5_n3", 0.5) - 3.8), fx="recre"),
        L("v5_class", "v5_n4"),
        L("v5_help", "v5_n5"),
        L("v5_team", "v5_n6"),
        c.clip(CL + "v5_office.mp4", d=vd(a, "v5_n7", 0.3) / 2, vo=vo(a, "v5_n7"), vo_at=0.15, fx="recre", lt=("Prácticas en empresa", None)),
        c.clip(CL + "v5_job.mp4", d=vd(a, "v5_n7", 0.6) / 2, fx="recre"),
        c.talk(CL + "v5_tk_w1.mp4", tin=FADE),
        c.talk(CL + "v5_tk_m1.mp4", lt=("Barakaldo · Bilbao", "Te esperamos")),
    ] + outro(), "Dreamer", None

def V06():
    a = "v6"
    return intro("ANTES Y DESPUÉS") + [
        c.clip(CL + "v6_before.mp4", d=vd(a, "v6_m_before", 0.6), vo=vo(a, "v6_m_before"), vo_at=0.2, fx="desat", tin=FADE),
        c.clip(CL + "v6_after.mp4", d=vd(a, "v6_m_after", 0.6), vo=vo(a, "v6_m_after"), vo_at=0.1, tin=("wipeleft", 0.4)),
        c.split(CL + "v6_before.mp4", CL + "v6_after.mp4", 2.6, tin=FADE),
        c.talk(CL + "v6_tk_m1.mp4", tin=FADE),
        *c.alive("p6", vd(a, "v6_f_areas") / 2, lt=("Informática · Desarrollo web", None), vo=vo(a, "v6_f_areas"), vo_at=0.1, tin=FADE),
        c.card(v.ai_card(), d=vd(a, "v6_f_areas") / 2 + 0.4, bg="ai", icon="ia", tin=FADE),
        c.card(SUBV, d=vd(a, "v6_f_subv", 0.5), bg=CL + "cls_p4.mp4", vo=vo(a, "v6_f_subv"), vo_at=0.2, tin=FADE),
        c.card(v.contact_card(v.FOOT), d=5.5, vo=vo(a, "v6_m2"), vo_at=0.4, tin=FADE),
    ], "Ascending_the_Vale", None

def V07():
    a = "v7"
    return intro("EL FUTURO LLAMA") + [
        c.talk(CL + "v7_tk_w1.mp4", tin=FADE),
        c.clip(CL + "v7_robot.mp4", d=v.dur(V + "vo_v7/robot.wav") + 0.4, vo=V + "vo_v7/robot.wav", vo_at=0.15, tin=("zoomin", 0.3)),
        c.talk(CL + "v7_tk_m1.mp4", tin=("fade", 0.2)),
        c.talk(CL + "v7_tk_w2.mp4"),
        c.talk(CL + "v7_tk_m2.mp4", lt=("IA · Informática · Desarrollo web", "Administración · Comercio")),
        c.talk(CL + "v7_tk_w3.mp4", lt=("100% subvencionado", "Para personas desempleadas")),
        c.clip(CL + "v7_both.mp4", d=vd(a, "v7_m3", 0.8), vo=vo(a, "v7_m3"), vo_at=0.3, lt=("Barakaldo · Bilbao", None), tin=FADE,
               pip=CL + "v7_robot_wave.mp4"),
    ] + outro(), "Life_of_Riley", None

def V08():
    a = "v8"
    two = [E("DOS CENTROS", BLACK, 36, "white", 0.2, d=0.1, box=ACCENT),
           E("BARAKALDO", BLACK, 76, "white", 0.32, x=0.1, d=0.3, mw=0.38), E("Gabriel Aresti, 3", SEMI, 42, SOFT, 0.43, x=0.1, d=0.5, mw=0.38),
           E("94 478 03 88", SEMI, 42, "white", 0.5, x=0.1, d=0.6, mw=0.38),
           E("BILBAO", BLACK, 76, "white", 0.32, x=0.56, d=0.8, mw=0.38), E("Nicolás Alcorta, 4", SEMI, 42, SOFT, 0.43, x=0.56, d=1.0, mw=0.38),
           E("94 421 27 02", SEMI, 42, "white", 0.5, x=0.56, d=1.1, mw=0.38),
           E("Una misma forma de enseñar", MED, 46, "white", 0.7, d=1.4)]
    return intro("DOS CENTROS") + [
        c.talk(CL + "v8_tk_w1.mp4", lt=("Barakaldo", "Gabriel Aresti, 3"), tin=FADE),
        *c.alive("b2", vd(a, "v8_f_b") / 3, vo=vo(a, "v8_f_b"), vo_at=0.1, tin=FADE),
        *c.alive("b3", vd(a, "v8_f_b") / 3),
        *c.alive("b6", vd(a, "v8_f_b") / 3 + 0.2),
        c.talk(CL + "v8_tk_w2.mp4", tin=FADE),
        c.talk(CL + "v8_tk_m1.mp4", lt=("Bilbao", "Nicolás Alcorta, 4"), tin=("slideleft", 0.25)),
        c.clip(CL + "v8_m_walk.mp4", d=vd(a, "v8_m_p", 0.5), vo=vo(a, "v8_m_p"), vo_at=0.15, tin=FADE),
        c.card(two, d=vd(a, "v8_f_map", 2.2), vo=vo(a, "v8_f_map"), vo_at=1.2, tin=FADE),
        c.card(v.ai_card(), d=vd(a, "v8_m_ai") / 2, bg="ai", icon="ia", vo=vo(a, "v8_m_ai"), vo_at=0.2, tin=FADE),
        c.card(SUBV, d=vd(a, "v8_m_ai") / 2 + 0.4, bg=CL + "cls_p4.mp4", tin=FADE),
        c.talk(CL + "v8_tk_w3.mp4", fx="screen", tin=FADE),
    ] + outro(), "Inspired", "DOS CENTROS"

def V09():
    a = "v9"
    cc = [c.card(v.course_card(*v.UPCOMING[i]), d=vd(a, f"v9_c{i + 1}", 0.6), bg=bg, vo=vo(a, f"v9_c{i + 1}"), vo_at=0.25, tin=FADE)
          for i, bg in enumerate([CL + f"cls_{r}.mp4" for r in ("p4", "b2", "p3", "b6")])]
    return intro("PRÓXIMOS CURSOS") + [c.talk(CL + "v9_tk_w1.mp4", tin=FADE)] + cc + [
        c.talk(CL + "v9_tk_m_ai.mp4", tin=FADE),
        c.card(v.ai_card(), d=2.4, bg="ai", icon="ia", tin=FADE),
        c.talk(CL + "v9_tk_w2.mp4", lt=("100% subvencionado", "Para personas desempleadas"), tin=FADE),
    ] + outro(), "Easy_Lemon", "AGENDA"

def V10():
    a = "v10"; f1, m1, f2 = vd(a, "v10_f1"), vd(a, "v10_m1"), vd(a, "v10_f2")
    kin = lambda t: [E(t, BLACK, 120, "white", 0.42, d=0.05)]
    k = m1 / 5
    return [c.real(1.0, 1.0), c.card(v.intro_card("MEATZE"), d=1.6, sweep=True, tin=("fade", 0.2)),
            c.clip(CL + "v1_walk.mp4", inn=1.5, d=f1, vo=vo(a, "v10_f1"), vo_at=0.1, tin=("fade", 0.2)),
            c.clip(CL + "v1_r_teach.mp4", d=k, vo=vo(a, "v10_m1"), vo_at=0.05, lt=("Inteligencia artificial", None)),
            c.clip(CL + "v5_class.mp4", d=k, lt=("Informática", None)),
            c.clip(CL + "v3_r_web.mp4", d=k, lt=("Desarrollo web", None)),
            c.clip(CL + "v1_r_office.mp4", d=k, lt=("Administración", None)),
            c.clip(CL + "v1_r_shop.mp4", d=k + 0.2, lt=("Comercio", None)),
            c.card(kin("100% subvencionado"), d=1.4, tin=("fade", 0.2), vo=vo(a, "v10_f2"), vo_at=0.1),
            c.clip(CL + "v7_both.mp4", d=max(1.5, f2 - 1.2), lt=("Barakaldo · Bilbao", None)),
            c.card(v.contact_card(v.FOOT), d=4.5, tin=FADE)], "Bright_Wish", None

NAMES = {"V03": "La_IA_en_tu_dia", "V04": "Mitos_y_realidades", "V05": "Un_dia_con_Laura", "V06": "Antes_y_despues",
         "V07": "El_futuro_llama", "V08": "Dos_centros", "V09": "Proximos_cursos", "V10": "Spot_20s"}

def robot_voice():
    src, dst = V + "vo_v7/trim/v7_robot.wav", V + "vo_v7/robot.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-af",
                    "asetrate=48000*0.94,aresample=48000,atempo=1.06,flanger=delay=2:depth=2:speed=0.6,aecho=0.8:0.6:12:0.3,highpass=f=180",
                    dst], check=True)

if __name__ == "__main__":
    robot_voice()
    for vid in sys.argv[1:]:
        items, music, bug = globals()[vid]()
        c.build(f"{vid}_{NAMES[vid]}", items, music, mvol=0.18, bug=bug)
