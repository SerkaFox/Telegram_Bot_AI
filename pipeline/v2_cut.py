import sys; sys.path.insert(0, "/home/iaadmin/tg_comfy_bot/outputs")
import cut2 as c, versions as v
V = "/home/iaadmin/tg_comfy_bot/outputs/v2/"; CL = V + "clips/"; T = V + "vo_v2/trim/"
FADE = ("fade", 0.3); Q1 = ("¿Qué le pedirías a la IA?", None); Q2 = ("¿Quita trabajo o lo transforma?", None)
items = [
    c.real(1.0, 1.4),
    c.card(v.intro_card("REPORTAJE"), d=2.4, sweep=True, tin=FADE),
    c.talk(CL + "v2_tk_w1.mp4", lt=("Reportaje", "¿Qué piensas de la inteligencia artificial?"), tin=FADE),
    c.talk(CL + "v2_tk_boy.mp4", lt=Q1, fx="recre"),
    c.talk(CL + "v2_tk_girl.mp4", lt=Q1, fx="recre"),
    c.talk(CL + "v2_tk_man46.mp4", lt=Q1, fx="recre"),
    c.talk(CL + "v2_tk_woman35.mp4", lt=Q1, fx="recre"),
    c.talk(CL + "v2_tk_w2.mp4", tin=("fade", 0.2)),
    c.talk(CL + "v2_tk_man52.mp4", lt=Q2, fx="recre"),
    c.talk(CL + "v2_tk_woman41.mp4", lt=Q2, fx="recre"),
    c.talk(CL + "v2_tk_man28.mp4", lt=Q2, fx="recre"),
    c.talk(CL + "v2_tk_woman58.mp4", lt=Q2, fx="recre"),
    c.talk(CL + "v2_tk_m1.mp4", lt=("MEATZE", "Centro de Barakaldo"), tin=("fadewhite", 0.2)),
    c.talk(CL + "v2_tk_m2.mp4"),
    c.card(v.ai_card(), d=2.6, bg="ai", icon="ia", tin=FADE),
    *c.alive("b6", 3.4, lt=("Informática · Desarrollo web", "Barakaldo"), vo=T + "v2_w3.wav", vo_at=0.1, tin=FADE),
    *c.alive("p4", 3.2, lt=("Administración · Comercio", "Bilbao")),
    c.card(v.title_card("100% subvencionado", "Para personas desempleadas"), d=3.6, bg=CL + "cls_p6.mp4", tin=FADE),
    c.talk(CL + "v2_tk_m3.mp4", lt=("Barakaldo · Bilbao", "Te esperamos"), tin=FADE),
    c.card(v.contact_card(v.FOOT), d=5.5, tin=FADE),
]
if __name__ == "__main__":
    c.build("V02_Reportaje_IA", items, "Life_of_Riley", mvol=0.16, bug="REPORTAJE")
