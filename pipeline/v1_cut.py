import sys; sys.path.insert(0, "/home/iaadmin/tg_comfy_bot/outputs")
import cut2 as c, versions as v
V = "/home/iaadmin/tg_comfy_bot/outputs/v2/"; CL = V + "clips/"; T = V + "vo_v1/trim/"
FADE = ("fade", 0.3)

items = [
    c.real(1.0, 1.4),
    c.card(v.intro_card("NOVEDAD · IA"), d=2.4, sweep=True, tin=FADE),
    c.clip(CL + "v1_walk.mp4", d=3.4, lt=("MEATZE", "Formación para el empleo"), tin=FADE),
    c.talk(CL + "v1_tk_m1.mp4"),
    c.clip(CL + "v1_sit.mp4", d=1.3),
    c.clip(CL + "v1_power.mp4", d=1.8),
    c.clip(CL + "v1_sit.mp4", inn=3.5, d=1.4),
    c.clip(CL + "v1_screen.mp4", d=2.4),
    c.talk(CL + "v1_tk_w12.mp4", fx="screen", tin=("zoomin", 0.5)),
    c.clip(CL + "v1_r_office.mp4", d=2.1, vo=T + "v1_w3.wav", vo_at=0.05, tin=("fade", 0.25)),
    c.clip(CL + "v1_r_service.mp4", d=2.0),
    c.clip(CL + "v1_r_shop.mp4", d=2.0),
    c.clip(CL + "v1_r_home.mp4", d=2.3),
    c.talk(CL + "v1_tk_m2.mp4", tin=("fade", 0.25)),
    c.talk(CL + "v1_tk_w4.mp4", fx="screen"),
    c.card(v.ai_card(), d=2.6, bg="ai", icon="ia", tin=FADE),
    *c.alive("b2", 2.7, lt=("Informática · Desarrollo web", "Barakaldo"), vo=T + "v1_m3.wav", vo_at=0.1, tin=FADE),
    c.clip(CL + "cls_b2_board.mp4", d=1.5),
    *c.alive("p3", 2.6, lt=("Administración · Comercio", "Bilbao")),
    c.card(v.title_card("100% subvencionado", "Para personas desempleadas", kicker="Con prácticas en empresa"),
           d=5.4, bg=CL + "v1_r_teach.mp4", vo=T + "v1_w5.wav", vo_at=0.15, tin=FADE),
    c.talk(CL + "v1_tk_m4.mp4", lt=("Barakaldo · Bilbao", "Te esperamos"), tin=FADE),
    c.card(v.title_card("Aprende hoy.", "Trabaja con las herramientas de mañana."), d=2.6, tin=FADE),
    c.card(v.contact_card(v.FOOT), d=5.5, tin=FADE),
]

if __name__ == "__main__":
    c.build("V01_Entra_en_la_pantalla", items, "Bright_Wish", mvol=0.2)
