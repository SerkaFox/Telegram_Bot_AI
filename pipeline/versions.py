#!/usr/bin/env python3
"""MEATZE — 10 edit versions built from the shared shot library + real photos/video.
  python versions.py tts            # synth every VO line (XTTS, CPU) — cached by text hash
  python versions.py render V01 ... # render versions (missing assets are skipped and logged)
  python versions.py send V01 ...   # push rendered versions to Telegram
Facts: meatze.eus (areas, 100% subvencionado + prácticas, contacts) and Lanbide listing (upcoming
courses with dates). The AI course comes from the client (no public listing yet — no dates claimed)."""
import hashlib, json, pathlib, subprocess, sys, tempfile, time

ROOT = pathlib.Path("/home/iaadmin/tg_comfy_bot")
OUT = ROOT / "outputs"; LIB = OUT / "lib"; CLIPS = OUT / "clips"; VO = OUT / "voices" / "vo_lines"
REND = OUT / "versions"; WORK = OUT / "vwork"
for d in (VO, REND, WORK): d.mkdir(parents=True, exist_ok=True)
KIT = ROOT / "MEATZE_CLAUDE_CODE_KIT/assets"
V1 = ROOT / "meatze_client_new/Новая папка/WhatsApp Video 2026-10-28 at 18.00.02.mp4"
NEWP = ROOT / "meatze_client_new/Новая папка"
FD = "/usr/share/fonts/truetype/lato/"
BLACK, HEAVY, BOLD, SEMI, MED = (FD + f for f in ("Lato-Black.ttf", "Lato-Heavy.ttf", "Lato-Bold.ttf", "Lato-Semibold.ttf", "Lato-Medium.ttf"))
NAVY, ACCENT, SOFT = "0x0a1630", "0x2f6bff", "0xbcd0ff"
FPS, XF = 25, 0.45
LOG = OUT / "versions_log.txt"
MUS = OUT / "music"; GFX = OUT / "gfx"
MUSIC = {"V01": "Inspired", "V02": "Wallpaper", "V03": "Dreamer", "V04": "Wallpaper", "V05": "Easy_Lemon",
         "V06": "Inspired", "V07": "Bright_Wish", "V08": "Inspired", "V09": "Easy_Lemon", "V10": "Ascending_the_Vale"}
CREDIT = "Música: Kevin MacLeod (incompetech.com) · CC BY 4.0"

def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"; print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")

def B(n): return KIT / "B_Bagatza" / f"b{n}.jpeg"
def P(n): return KIT / "P_second_office" / f"p{n}.jpeg"
NEW_IT = NEWP / "WhatsApp Image 2026-10-28 at 18.00.52.jpeg"
NEW_2 = NEWP / "WhatsApp Image 2026-10-28 at 18.00.52 (1).jpeg"
REAL = {"logo": (1.0, 2.4), "reception": (3.5, 2.0), "students": (15.6, 4.6)}

# ---------------------------------------------------------------- cards (layout as elements)
def E(t, f=HEAVY, s=56, c="white", y=0.4, x="c", d=0.0, box=None, mw=None):
    return {"t": t, "f": f, "s": s, "c": c, "y": y, "x": x, "d": d, "box": box, "mw": mw}

def title_card(main, sub=None, badge=None, kicker=None):
    els = []
    y = 0.34
    if badge: els.append(E(badge, BLACK, 34, "white", y - 0.09, d=0.1, box=ACCENT))
    els.append(E(main, BLACK, 104, "white", y, d=0.2))
    if sub: els.append(E(sub, MED, 46, SOFT, y + 0.2, d=0.55))
    if kicker: els.append(E(kicker, SEMI, 36, "white", y + 0.32, d=0.85))
    return els

def list_card(heading, items, badge=None):
    els = []
    if badge: els.append(E(badge, BLACK, 32, "white", 0.13, x=0.1, d=0.1, box=ACCENT))
    els.append(E(heading, BLACK, 66, "white", 0.2, x=0.1, d=0.2))
    for i, it in enumerate(items):
        els.append(E("—  " + it, SEMI, 42, "white", 0.36 + i * 0.1, x=0.1, d=0.5 + i * 0.25))
    return els

def course_card(name, where_when, detail, badge="PRÓXIMO INICIO"):
    return [E(badge, BLACK, 32, "white", 0.2, x=0.1, d=0.1, box=ACCENT),
            E(name, BLACK, 70, "white", 0.29, x=0.1, d=0.25),
            E(where_when, BOLD, 46, SOFT, 0.5, x=0.1, d=0.6),
            E(detail, MED, 40, "white", 0.58, x=0.1, d=0.85)]

def step_card(n, title, sub):
    return [E(f"PASO {n}", BLACK, 36, "white", 0.24, d=0.1, box=ACCENT),
            E(title, BLACK, 92, "white", 0.36, d=0.3), E(sub, MED, 44, SOFT, 0.58, d=0.6)]

def area_card(area, items, extra=None):
    els = [E("ÁREA DE FORMACIÓN", BLACK, 30, "white", 0.2, x=0.1, d=0.1, box=ACCENT),
           E(area, BLACK, 92, "white", 0.29, x=0.1, d=0.25),
           E(items, SEMI, 42, SOFT, 0.52, x=0.1, d=0.6, mw=0.6)]
    if extra: els.append(E(extra, MED, 36, "white", 0.7, x=0.1, d=0.95, mw=0.6))
    return els

def ai_card(line2="Ya ha comenzado"):
    return [E("NUEVO", BLACK, 40, "white", 0.24, x=0.1, d=0.1, box=ACCENT),
            E("Inteligencia", BLACK, 112, "white", 0.33, x=0.1, d=0.3),
            E("Artificial", BLACK, 112, "white", 0.45, x=0.1, d=0.45),
            E("Formación en MEATZE · " + line2, MED, 46, SOFT, 0.64, x=0.1, d=0.8)]

THEME = {"V01": "PRESENTACIÓN", "V02": "REPORTAJE", "V03": "TOUR GUIADO", "V04": "NOVEDAD · IA", "V05": "PRÓXIMOS CURSOS",
         "V06": "DOS CENTROS", "V07": "MEATZE", "V08": "¿BUSCAS EMPLEO?", "V09": "ÁREAS DE FORMACIÓN", "V10": "MEATZE"}

def intro_card(theme):
    return [E(theme, BLACK, 34, "white", 0.27, d=0.15, box=ACCENT),
            E("MEATZE", BLACK, 190, "white", 0.36, d=0.35),
            E("Centro de Enseñanzas Profesionales", MED, 46, SOFT, 0.62, d=0.9),
            E("Barakaldo  ·  Bilbao", SEMI, 38, "white", 0.71, d=1.2)]

def contact_card(foot=None):
    els = [E("MEATZE", BLACK, 96, "white", 0.14, d=0.1),
           E("Centro de Enseñanzas Profesionales", MED, 40, SOFT, 0.3, d=0.3),
           E("BARAKALDO", BLACK, 40, "white", 0.42, d=0.5, box=ACCENT),
           E("Gabriel Aresti, 3  ·  94 478 03 88", SEMI, 42, "white", 0.5, d=0.6),
           E("BILBAO", BLACK, 40, "white", 0.61, d=0.8, box=ACCENT),
           E("Nicolás Alcorta, 4  ·  94 421 27 02", SEMI, 42, "white", 0.69, d=0.9),
           E("meatze.eus   ·   secretaria@meatze.es", BOLD, 40, SOFT, 0.82, d=1.1)]
    if foot: els.append(E(foot, MED, 24, "white", 0.905, d=1.3))
    els.append(E(CREDIT, MED, 22, SOFT, 0.94, d=1.4))
    return els

# ---------------------------------------------------------------- versions
UPCOMING = [
    ("Gestión ágil de procesos administrativos y de RRHH", "Bilbao · 3 nov 2026 – 11 feb 2027", "300 h · lunes a viernes, 15:00–20:00"),
    ("Comunicación e interacción social en entornos TIC", "Barakaldo · 16 nov 2026 – 8 ene 2027", "135 h · lunes a viernes, 15:00–20:00"),
    ("Desarrollo web adaptativo con WordPress", "Bilbao · 17 nov 2026 – 17 feb 2027", "270 h"),
    ("Programación de servicios y procesos", "Barakaldo · 15 dic 2026 – 26 ene 2027", "80 h · lunes a viernes, 9:00–14:00"),
]
AREAS = ["Informática", "Videojuegos y animación 3D", "Administración", "Marketing", "Comercio"]
FOOT = "Fechas y horarios sujetos a confirmación en la matrícula · Fuente: meatze.eus / Lanbide"

def clip(src, vo=None, lt=None): return {"k": "clip", "src": src, "vo": vo, "lt": lt}
def talk(src, lt=None): return {"k": "talk", "src": src, "lt": lt}
def photo(src, move="in", d=4.0, vo=None, lt=None): return {"k": "photo", "src": src, "move": move, "d": d, "vo": vo, "lt": lt}
def real(win, vo=None, lt=None): return {"k": "real", "win": win, "vo": vo, "lt": lt}
def card(els, bg=None, d=3.5, vo=None, tr=None, icon=None): return {"k": "card", "els": els, "bg": bg, "d": d, "vo": vo, "tr": tr, "icon": icon}

def F(t): return ("f", t)
def M(t): return ("m", t)

VERSIONS = {
 "V01": {"name": "Presentación institucional", "aspect": "h", "trans": "fade", "segs": [
    card(title_card("MEATZE", "Centro de Enseñanzas Profesionales", kicker="Barakaldo · Bilbao"), bg=B(1), d=3.5),
    clip("lib/mo_both.mp4", F("Bienvenidos a Meatze, centro de enseñanzas profesionales en Barakaldo y en Bilbao."), ("MEATZE", "Barakaldo · Bilbao")),
    talk("lib/tk_w_hello.mp4", ("Barakaldo", "Gabriel Aresti, 3")),
    clip("lib/mo_w_hall.mp4", F("Nuestro centro de Barakaldo está pensado para aprender haciendo."), ("Centro de Barakaldo", None)),
    clip("lib/mo_w_class.mp4", F("Aulas de informática totalmente equipadas, para una formación práctica desde el primer día."), ("Aulas de informática", None)),
    photo(B(6), "right", 4.5, F("Personas que se forman para dar un nuevo paso en su vida profesional.")),
    talk("lib/tk_w_ai.mp4", ("NUEVO", "Inteligencia Artificial")),
    card(ai_card(), bg="ai", icon="ia", d=4.0, vo=F("La inteligencia artificial ya forma parte de nuestra oferta formativa.")),
    photo(P(1), "in", 5.0, M("Y en Bilbao, en la calle Nicolás Alcorta, te espera nuestro segundo centro."), ("Centro de Bilbao", "Nicolás Alcorta, 4")),
    clip("lib/mo_m_class.mp4", M("Con la misma metodología práctica y aulas preparadas para cada especialidad."), None),
    card(list_card("Áreas de formación", AREAS), bg=P(4), d=5.0, vo=M("Informática, videojuegos, administración, marketing y comercio.")),
    talk("lib/tk_m_subv.mp4", ("100% subvencionado", "Para personas desempleadas")),
    card(list_card("Próximos inicios", [f"{c[0]} — {c[1].split(' · ')[0]}" for c in UPCOMING], badge="NOV – DIC 2026"), bg=P(3), d=6.5,
         vo=M("Y los próximos cursos empiezan ya en noviembre y diciembre.")),
    talk("lib/tk_m_close.mp4"),
    card(contact_card(FOOT), d=6.0),
 ]},
 "V02": {"name": "Reportaje", "aspect": "h", "trans": "smoothleft", "bug": "REPORTAJE", "segs": [
    real("logo"),
    photo(P(1), "in", 5.0, M("Estamos en Bilbao, a las puertas de Meatze, un centro dedicado a la formación para el empleo."), ("MEATZE Bilbao", "Nicolás Alcorta, 4")),
    clip("lib/mo_m_lobby.mp4", M("Aquí, igual que en su sede de Barakaldo, la formación se vive en el aula."), None),
    photo(P(4), "right", 4.5, M("Aulas de informática, equipos preparados y grupos de trabajo.")),
    photo(P(6), "in", 4.5, M("Cursos dirigidos a personas desempleadas y también a trabajadores en activo.")),
    card(title_card("Barakaldo", "Gabriel Aresti, 3"), bg=B(1), d=2.5, tr="fadeblack"),
    clip("lib/mo_w_welcome.mp4", F("Desde Barakaldo continuamos el recorrido."), ("MEATZE Barakaldo", "Gabriel Aresti, 3")),
    photo(B(6), "right", 4.5, F("Estas son las aulas donde se forman nuestros alumnos.")),
    clip("lib/mo_w_bigclass.mp4", F("Y la gran novedad de este curso: la formación en inteligencia artificial ya ha comenzado."), ("NUEVO", "Inteligencia Artificial")),
    talk("clips/s02_woman_b2.mp4", ("Formación práctica", None)),
    photo(B(3), "left", 5.0, F("Para personas desempleadas, los cursos son cien por cien subvencionados e incluyen prácticas en empresa.")),
    talk("lib/tk_m_close.mp4"),
    card(contact_card(), d=5.5),
 ]},
 "V03": {"name": "Tour guiado", "aspect": "h", "trans": "fade", "segs": [
    card(title_card("Acompáñanos", "Un recorrido por MEATZE", badge="TOUR"), bg=B(8), d=3.0),
    clip("lib/mo_w_welcome.mp4", F("¡Hola! Acompáñame a conocer nuestro centro por dentro.")),
    clip("lib/mo_w_hall.mp4", F("Empezamos en Barakaldo. Este es nuestro vestíbulo."), ("Barakaldo", "Vestíbulo")),
    clip("lib/mo_w_class.mp4", F("Aquí están las aulas de informática, donde se aprende practicando."), ("Aula de informática", None)),
    clip("lib/mo_w_meeting.mp4", F("Esta sala está preparada para trabajar en equipo."), ("Sala de trabajo", None)),
    clip("lib/mo_w_bigclass.mp4", F("Y ahora, una gran novedad: formación en inteligencia artificial."), ("NUEVO", "Inteligencia Artificial")),
    card(title_card("Bilbao", "Nicolás Alcorta, 4"), bg=P(1), d=2.5, tr="fadeblack"),
    photo(P(1), "in", 5.0, M("Ahora vamos a Bilbao, a nuestro segundo centro"), ("Bilbao", "Nicolás Alcorta, 4")),
    clip("lib/mo_m_lobby.mp4", M("Bienvenidos. Aquí empieza tu formación.")),
    clip("lib/mo_m_walk.mp4", M("Aulas amplias y luminosas, equipadas para cada especialidad."), ("Aulas de informática", None)),
    clip("lib/mo_m_class2.mp4", M("Cursos para personas desempleadas y también para trabajadores.")),
    talk("lib/tk_m_close.mp4"),
    card(contact_card(), d=5.5),
 ]},
 "V04": {"name": "Novedad: Inteligencia Artificial", "aspect": "h", "trans": "fade", "segs": [
    card(ai_card(), bg="ai", icon="ia", d=4.0, vo=F("Novedad en Meatze: formación en inteligencia artificial.")),
    talk("lib/tk_w_ai.mp4", ("NUEVO", "Inteligencia Artificial")),
    clip("lib/mo_w_class.mp4", F("Aprende a usar las herramientas de inteligencia artificial que ya utilizan las empresas.")),
    photo(B(6), "right", 4.5, F("Una formación práctica, pensada para mejorar tu empleabilidad.")),
    clip("lib/mo_m_class.mp4", M("Pregunta por ella en nuestros centros de Barakaldo y de Bilbao."), ("Barakaldo · Bilbao", None)),
    card(list_card("Además en MEATZE", AREAS), bg=P(4), d=5.0, vo=M("Y además, informática, videojuegos, administración, marketing y comercio.")),
    talk("lib/tk_m_close.mp4"),
    card(contact_card(), d=5.5),
 ]},
 "V05": {"name": "Próximos cursos", "aspect": "h", "trans": "fade", "segs": [
    card(title_card("Próximos cursos", "Noviembre – Diciembre 2026", badge="MEATZE"), bg=B(2), d=3.0),
    clip("lib/mo_both.mp4", F("¿Quieres formarte este otoño? Estos son nuestros próximos cursos.")),
    card(course_card(*UPCOMING[0]), bg=P(2), d=5.5, vo=M("En Bilbao, gestión ágil de procesos administrativos y de recursos humanos, a partir del tres de noviembre.")),
    clip("lib/mo_m_class.mp4", M("Trescientas horas, de lunes a viernes por la tarde.")),
    card(course_card(*UPCOMING[1]), bg=B(2), d=5.5, vo=F("En Barakaldo, comunicación e interacción social en entornos digitales, desde el dieciséis de noviembre.")),
    card(course_card(*UPCOMING[2]), bg=P(3), d=5.0, vo=M("Desarrollo web adaptativo con WordPress, en Bilbao, desde el diecisiete de noviembre.")),
    card(course_card(*UPCOMING[3]), bg=B(6), d=5.0, vo=F("Y en diciembre, programación de servicios y procesos, en Barakaldo.")),
    talk("lib/tk_w_ai.mp4", ("NUEVO", "Inteligencia Artificial")),
    card(list_card("Para personas desempleadas", ["Sistemas microinformáticos", "Montaje y reparación de sistemas microinformáticos",
                                                  "Actividades de gestión administrativa", "Atención al cliente", "Recepción telefónica de oficina"],
                   badge="100% SUBVENCIONADO + PRÁCTICAS"), bg=P(6), d=6.5,
         vo=M("Y para personas desempleadas, cursos cien por cien subvencionados con prácticas en empresa.")),
    talk("lib/tk_m_close.mp4"),
    card(contact_card(FOOT), d=6.0),
 ]},
 "V06": {"name": "Dos centros", "aspect": "h", "trans": "fade", "segs": [
    card(title_card("Dos centros", "Una misma forma de aprender", kicker="Barakaldo · Bilbao"), bg=B(1), d=3.0),
    clip("lib/mo_both.mp4", F("Tenemos dos centros para estar más cerca de ti.")),
    card(title_card("Barakaldo", "Gabriel Aresti, 3 · 94 478 03 88"), bg=B(1), d=3.5, vo=F("En Barakaldo, en la calle Gabriel Aresti, número tres."), tr="fadeblack"),
    talk("lib/tk_w_hello.mp4"),
    clip("lib/mo_w_bigclass.mp4", F("Aulas amplias y bien equipadas"), ("Barakaldo", "Aula de informática")),
    photo(B(5), "in", 4.0, F("Y un equipo que te atiende desde el primer momento.")),
    card(title_card("Bilbao", "Nicolás Alcorta, 4 · 94 421 27 02"), bg=P(1), d=3.5, vo=M("Y en Bilbao, en la calle Nicolás Alcorta, número cuatro."), tr="fadeblack"),
    clip("lib/mo_m_lobby.mp4", M("Un espacio pensado para aprender."), ("Bilbao", "Recepción")),
    clip("lib/mo_m_walk.mp4", M("Con aulas preparadas para cada especialidad."), ("Bilbao", "Aula de informática")),
    photo(NEW_IT, "in", 4.0, M("Y equipos renovados para tu formación.")),
    talk("lib/tk_m_close.mp4"),
    card(contact_card(), d=6.0, vo=M("Llámanos o ven a vernos al centro que te quede más cerca.")),
 ]},
 "V07": {"name": "Teaser 30 s", "aspect": "h", "trans": "slideleft", "segs": [
    real("logo"),
    clip("lib/mo_w_welcome.mp4", F("¿Buscas formación para encontrar empleo?")),
    card(ai_card("Novedad"), bg="ai", icon="ia", d=3.0, vo=F("Ahora, también en inteligencia artificial.")),
    clip("lib/mo_m_walk.mp4", M("Cursos cien por cien subvencionados para personas desempleadas."), ("100% subvencionado", "Para personas desempleadas")),
    photo(B(6), "right", 4.5, M("En Barakaldo y en Bilbao.")),
    card(title_card("MEATZE", "meatze.eus · 94 478 03 88", kicker="Barakaldo · Bilbao"), bg=None, d=3.5, vo=M("Infórmate ya.")),
 ]},
 "V08": {"name": "¿Buscas empleo?", "aspect": "h", "trans": "fade", "segs": [
    card(title_card("¿Buscas empleo?", "Fórmate con MEATZE"), bg=B(2), d=3.5, vo=F("¿Estás buscando empleo o quieres mejorar tu perfil profesional?")),
    talk("lib/tk_w_hello.mp4"),
    card(step_card(1, "Elige tu área", " · ".join(AREAS)), bg=B(6), d=4.5, vo=F("Primero, elige tu área: informática, videojuegos, administración, marketing o comercio.")),
    clip("lib/mo_w_class.mp4", F("Te formas en aulas equipadas, con un enfoque práctico.")),
    card(step_card(2, "Fórmate gratis", "100% subvencionado para personas desempleadas"), bg=P(4), d=4.0, vo=M("Segundo: si estás en desempleo, tu formación es cien por cien subvencionada.")),
    clip("lib/mo_m_class.mp4", M("Sin coste para ti, en Barakaldo o en Bilbao."), ("100% subvencionado", "Para personas desempleadas")),
    card(step_card(3, "Prácticas en empresa", "Al finalizar el curso"), bg=P(6), d=4.0, vo=M("Y tercero: al terminar, haces prácticas en empresa.")),
    clip("lib/mo_m_class2.mp4", M("Así das el salto al mundo laboral con experiencia real.")),
    talk("lib/tk_w_ai.mp4", ("NUEVO", "Inteligencia Artificial")),
    talk("lib/tk_m_close.mp4"),
    card(contact_card(), d=5.5),
 ]},
 "V09": {"name": "Áreas de formación", "aspect": "h", "trans": "fade", "segs": [
    card(title_card("Áreas de formación", "MEATZE · Barakaldo · Bilbao"), bg=B(6), d=3.0),
    clip("lib/mo_both.mp4", F("Puedes formarte en cinco grandes áreas.")),
    card(area_card("Informática", "Ofimática · Redes · Sistemas · Mantenimiento · Desarrollo web"), bg=B(2), d=5.0, icon="informatica",
         vo=F("Informática: ofimática, redes, sistemas, mantenimiento y desarrollo web.")),
    clip("lib/mo_w_class.mp4"),
    card(area_card("Videojuegos", "Programación · Modelado · Animación 2D y 3D · Concept art",
                   "Ciclo de Grado Superior en Animaciones 3D, Juegos y Entornos Interactivos"), bg=B(6), d=6.5,
         vo=F("Videojuegos: programación, modelado, animación y concept art, con el ciclo de grado superior en animaciones tres de, juegos y entornos interactivos.")),
    clip("lib/mo_w_bigclass.mp4", F("Y como novedad, formación en inteligencia artificial."), ("NUEVO", "Inteligencia Artificial")),
    card(area_card("Administración", "Contabilidad · Recursos humanos · Creación de empresas · Secretariado · Gestión fiscal"), bg=P(6), d=5.5, icon="administracion",
         vo=M("Administración: contabilidad, recursos humanos, creación de empresas, secretariado y gestión fiscal.")),
    clip("lib/mo_m_class.mp4"),
    card(area_card("Marketing", "Redes sociales · Publicación web · Plan de marketing · Branding · Venta online"), bg=P(3), d=5.0, icon="marketing",
         vo=M("Marketing: redes sociales, publicación web, branding y venta online.")),
    card(area_card("Comercio", "Gestión de ventas · Escaparatismo · Personal shopper · Atención al cliente · Idiomas"), bg=P(7), d=5.0, icon="comercio",
         vo=M("Y comercio: ventas, escaparatismo, personal shopper, atención al cliente e idiomas.")),
    talk("lib/tk_m_subv.mp4", ("100% subvencionado", "Para personas desempleadas")),
    talk("lib/tk_m_close.mp4"),
    card(contact_card(), d=5.5),
 ]},
 "V10": {"name": "Spot TV 20 s", "aspect": "h", "trans": "fade", "segs": [
    real("logo"),
    clip("lib/mo_both.mp4", F("Formación para el empleo, en Barakaldo y en Bilbao."), ("MEATZE", "Barakaldo · Bilbao")),
    card(ai_card("Novedad"), bg="ai", icon="ia", d=3.5, vo=F("Novedad: formación en inteligencia artificial.")),
    clip("lib/mo_m_walk.mp4", M("Cursos cien por cien subvencionados para personas desempleadas, con prácticas en empresa."), ("100% subvencionado", "Con prácticas en empresa")),
    card(title_card("MEATZE", "meatze.eus", kicker="Barakaldo 94 478 03 88 · Bilbao 94 421 27 02"), bg=None, d=4.0, vo=M("Infórmate en meatze punto eus.")),
 ]},
}

# ---------------------------------------------------------------- TTS
def tts_text(t): return t.replace("Meatze", "Meátse").replace("MEATZE", "Meátse").replace("meatze", "meátse")
def vo_key(vo): return hashlib.md5((vo[0] + vo[1]).encode()).hexdigest()[:12]
def vo_path(vo): return VO / f"{vo_key(vo)}.wav"

def all_vos():
    seen = {}
    for v in VERSIONS.values():
        for s in v["segs"]:
            if s.get("vo"): seen[vo_key(s["vo"])] = [s["vo"][0], tts_text(s["vo"][1])]
    return seen

def do_tts():
    spec = WORK / "vo_spec.json"; spec.write_text(json.dumps(all_vos(), ensure_ascii=False))
    subprocess.run(["/home/iaadmin/tts_venv/bin/python", str(OUT / "tts_lines.py"), str(spec), str(VO)], check=True)

# ---------------------------------------------------------------- render helpers
def dur(p):
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nk=1:nw=1", str(p)]))

def tf(text):
    f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, dir=WORK); f.write(text); f.close(); return f.name

def wrap(text, size, maxw):
    maxc = max(8, int(maxw / (size * 0.52)))
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > maxc and cur: lines.append(cur); cur = w
        else: cur = (cur + " " + w).strip()
    if cur: lines.append(cur)
    return lines

def alpha(d, fade=0.45): return f"'if(lt(t,{d}),0,if(lt(t,{d + fade}),(t-{d})/{fade},1))'"

def drawtext(text, font, size, color, x, y, d, box=None, slide=True):
    yy = f"'{y}+18*min(1,max(0,1-(t-{d})/0.45))'" if slide else str(y)
    s = (f"drawtext=fontfile={font}:textfile={tf(text)}:expansion=none:fontsize={size}:fontcolor={color}:"
         f"x={x}:y={yy}:alpha={alpha(d)}:shadowcolor=black@0.55:shadowx=2:shadowy=2")
    if box: s += f":box=1:boxcolor={box}:boxborderw=14"
    return s

def render_els(els, W, H):
    out, scale = [], min(W, H) / 1080
    for e in els:
        size = int(e["s"] * scale * (0.86 if W < H else 1.0))
        x0 = 0.1 if e["x"] == "c" else e["x"]
        maxw = W * (e.get("mw") or ((1 - 2 * x0) if e["x"] != "c" else 0.84))
        y = int(e["y"] * H)
        for i, line in enumerate(wrap(e["t"], size, maxw)):
            x = "(w-text_w)/2" if e["x"] == "c" else str(int(e["x"] * W))
            out.append(drawtext(line, e["f"], size, e["c"], x, y + int(i * size * 1.22), e["d"] + i * 0.08, e["box"]))
    return out

def lower_third(lt, W, H):
    if not lt: return []
    title, sub = lt
    vertical = W < H
    ts, ss = (50, 34) if not vertical else (46, 32)
    bw = min(int(W * 0.92), int(max(len(title) * ts * 0.56, len(sub or "") * ss * 0.52) + 130))
    y = H - (250 if not vertical else 520)
    bh = 140 if sub else 96
    return [f"drawbox=x=0:y={y}:w={bw}:h={bh}:color={NAVY}@0.8:t=fill:enable='gte(t,0.25)'",
            f"drawbox=x=0:y={y}:w=12:h={bh}:color={ACCENT}:t=fill:enable='gte(t,0.25)'",
            drawtext(title, HEAVY, ts, "white", "56", y + 20, 0.35, slide=False)] + \
           ([drawtext(sub, MED, ss, SOFT, "56", y + 86, 0.55, slide=False)] if sub else [])

def chrome(W, H, bug):
    s = [f"drawtext=fontfile={BLACK}:text=MEATZE:fontsize={int(30 * min(W, H) / 1080)}:fontcolor=white@0.85:"
         f"x=w-text_w-48:y=40:shadowcolor=black@0.5:shadowx=2:shadowy=2"]
    if bug:
        s.append(f"drawtext=fontfile={BLACK}:textfile={tf(bug)}:expansion=none:fontsize={int(28 * min(W, H) / 1080)}:"
                 f"fontcolor=white:x=48:y=40:box=1:boxcolor=0xd62828@0.95:boxborderw=12")
    return s

def cover(W, H): return f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1"

def audio_chain(vo, D, idx):
    """returns (input args, filter producing [a])"""
    if vo and vo_path(vo).exists():
        return (["-i", str(vo_path(vo))],
                f"[{idx}:a]adelay=250:all=1,apad,atrim=0:{D:.3f},aformat=sample_rates=48000:channel_layouts=stereo[a]")
    return (["-f", "lavfi", "-t", f"{D:.3f}", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"],
            f"[{idx}:a]atrim=0:{D:.3f}[a]")

def vo_need(vo):
    return dur(vo_path(vo)) + 0.95 if vo and vo_path(vo).exists() else 0

def encode(inputs, fc, out, D):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", fc, "-map", "[v]", "-map", "[a]",
                    "-t", f"{D:.3f}", "-r", str(FPS), "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "48000", "-b:a", "192k", str(out)], check=True)

def seg_render(s, W, H, bug, out):
    k = s["k"]
    over = ",".join(chrome(W, H, bug) + lower_third(s.get("lt"), W, H))
    if k == "card":
        D = max(s["d"], vo_need(s.get("vo")))
        bg = s.get("bg")
        if bg == "ai" and (GFX / "ai_bg.png").exists():
            inp = ["-loop", "1", "-framerate", str(FPS), "-t", f"{D:.3f}", "-i", str(GFX / "ai_bg.png")]
            base = f"[0:v]{cover(W, H)},zoompan=z='1+0.05*on/{int(D * FPS)}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={W}x{H}:fps={FPS},drawbox=color={NAVY}@0.25:t=fill"
        elif bg and bg != "ai" and pathlib.Path(bg).exists():
            inp = ["-loop", "1", "-t", f"{D:.3f}", "-i", str(bg)]
            base = f"[0:v]{cover(W, H)},gblur=sigma=30,eq=brightness=-0.18:saturation=0.6,drawbox=color={NAVY}@0.74:t=fill,vignette=PI/4"
        elif (GFX / "title_bg.png").exists():
            inp = ["-loop", "1", "-t", f"{D:.3f}", "-i", str(GFX / "title_bg.png")]
            base = f"[0:v]{cover(W, H)}"
        else:
            inp = ["-f", "lavfi", "-t", f"{D:.3f}", "-i", f"color=c={NAVY}:s={W}x{H}:r={FPS}"]
            base = f"[0:v]setsar=1"
        els = ",".join(render_els(s["els"], W, H))
        icon = GFX / f"icon_{s['icon']}.png" if s.get("icon") else None
        if icon and icon.exists():
            inp += ["-loop", "1", "-t", f"{D:.3f}", "-i", str(icon)]
            a_in, a_fc = audio_chain(s.get("vo"), D, 2)
            isz = int(H * 0.36)
            fc = (f"[1:v]format=rgba,scale={isz}:{isz},fade=in:st=0.5:d=0.6:alpha=1[ic];"
                  f"{base},fps={FPS}[bgv];[bgv][ic]overlay=W-w-{int(W * 0.08)}:(H-h)/2:shortest=1,{els},fade=in:st=0:d=0.3[v];{a_fc}")
        elif s.get("sweep"):
            inp += ["-f", "lavfi", "-t", f"{D:.3f}", "-i", f"color=c=white:s=520x{H}:r={FPS}"]
            a_in, a_fc = audio_chain(s.get("vo"), D, 2)
            fc = (f"[1:v]format=rgba,geq=r=255:g=255:b=255:a='255*0.22*exp(-pow((X-260)/95,2))'[sw];{base},fps={FPS},{els}[tx];"
                  f"[tx][sw]overlay=x='-400+(W+800)*(t-1.0)/1.3':y=0:enable='between(t,1.0,2.3)':shortest=1,"
                  f"fade=in:st=0:d=0.35[v];{a_fc}")
        else:
            a_in, a_fc = audio_chain(s.get("vo"), D, 1)
            fc = f"{base},fps={FPS},{els},fade=in:st=0:d=0.3[v];{a_fc}"
        encode(inp + a_in, fc, out, D); return D
    if k == "photo":
        D = max(s["d"], vo_need(s.get("vo"))); N = int(D * FPS)
        z = {"in": ("1+0.12*on/{N}", "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"),
             "out": ("1.12-0.12*on/{N}", "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"),
             "left": ("1.14", "(iw-iw/zoom)*on/{N}", "ih/2-(ih/zoom/2)"),
             "right": ("1.14", "(iw-iw/zoom)*(1-on/{N})", "ih/2-(ih/zoom/2)")}[s["move"]]
        zz, xx, yy = (e.format(N=N) for e in z)
        a_in, a_fc = audio_chain(s.get("vo"), D, 1)
        fc = (f"[0:v]scale={2 * W}:{2 * H}:force_original_aspect_ratio=increase,crop={2 * W}:{2 * H},"
              f"zoompan=z='{zz}':x='{xx}':y='{yy}':d={N}:s={W}x{H}:fps={FPS},setsar=1,{over}[v];{a_fc}")
        encode(["-i", str(s["src"])] + a_in, fc, out, D); return D
    if k == "real":
        ss, avail = REAL[s["win"]]
        D = max(avail, vo_need(s.get("vo"))); kfac = D / avail
        a_in, a_fc = audio_chain(s.get("vo"), D, 1)
        fc = f"[0:v]setpts=PTS*{kfac:.4f},fps={FPS},{cover(W, H)},{over}[v];{a_fc}"
        encode(["-ss", str(ss), "-t", str(avail), "-i", str(V1)] + a_in, fc, out, D); return D
    src = OUT / s["src"]
    L = dur(src)
    if k == "talk":
        D = L
        fc = f"[0:v]fps={FPS},{cover(W, H)},{over}[v];[0:a]aformat=sample_rates=48000:channel_layouts=stereo,atrim=0:{D:.3f}[a]"
        encode(["-i", str(src)], fc, out, D); return D
    # motion clip: slow down up to 1.25x, then hold the last frame if the line is longer
    need = max(L, vo_need(s.get("vo")))
    kfac = min(need / L, 1.25); hold = max(0.0, need - L * kfac)
    a_in, a_fc = audio_chain(s.get("vo"), need, 1)
    fc = (f"[0:v]setpts=PTS*{kfac:.4f},fps={FPS},{cover(W, H)},tpad=stop_mode=clone:stop_duration={hold:.3f},{over}[v];{a_fc}")
    encode(["-i", str(src)] + a_in, fc, out, need); return need

def available(s):
    if s["k"] in ("clip", "talk"): return (OUT / s["src"]).exists()
    if s["k"] == "photo": return pathlib.Path(s["src"]).exists()
    if s.get("vo") and not vo_path(s["vo"]).exists(): return False
    return True

def render(vid, prewarm_only=False):
    v = VERSIONS[vid]
    W, H = (1920, 1080) if v["aspect"] == "h" else (1080, 1920)
    wdir = WORK / vid; wdir.mkdir(exist_ok=True)
    body = list(v["segs"])
    if body and ((body[0]["k"] == "real" and body[0]["win"] == "logo") or (body[0]["k"] == "card" and not body[0].get("vo"))):
        body = body[1:]
    intro = [real("logo"), {"k": "card", "els": intro_card(THEME[vid]), "bg": None, "d": 4.0, "vo": None, "tr": None, "sweep": True}]
    segs, trans = [], []
    for i, s in enumerate(intro + body):
        if not available(s): log(f"{vid}: skip seg {i} ({s['k']} {s.get('src', '')})"); continue
        key = hashlib.md5((json.dumps(s, sort_keys=True, default=str) + f"{W}x{H}{v.get('bug')}" +
                           (str((OUT / s["src"]).stat().st_mtime) if s["k"] in ("clip", "talk") else "") +
                           (str(vo_path(s["vo"]).stat().st_mtime) if s.get("vo") and vo_path(s["vo"]).exists() else "")).encode()).hexdigest()[:10]
        out = wdir / f"seg_{i:02d}_{key}.mp4"
        if prewarm_only and s["k"] in ("clip", "talk"): continue
        try:
            if not out.exists(): seg_render(s, W, H, v.get("bug"), out)
            segs.append((out, dur(out))); trans.append(s.get("tr") or v["trans"])
        except subprocess.CalledProcessError as e:
            log(f"{vid}: seg {i} FAILED {e}")
    if prewarm_only: log(f"{vid}: prewarmed {len(segs)} segs"); return None
    if len(segs) < 2: log(f"{vid}: not enough segments"); return None
    inputs, fch = [], []
    for p, _ in segs: inputs += ["-i", str(p)]
    vcur, acur, off = "0:v", "0:a", segs[0][1] - XF
    for i in range(1, len(segs)):
        fch.append(f"[{vcur}][{i}:v]xfade=transition={trans[i]}:duration={XF}:offset={off:.3f}[v{i}]")
        fch.append(f"[{acur}][{i}:a]acrossfade=d={XF}[a{i}]")
        vcur, acur = f"v{i}", f"a{i}"; off += segs[i][1] - XF
    rough = wdir / "rough.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", ";".join(fch), "-map", f"[{vcur}]", "-map", f"[{acur}]",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "48000", str(rough)], check=True)
    T = dur(rough)
    final = REND / f"MEATZE_{vid}_{v['name'].replace(' ', '_').replace(':', '').replace('¿', '').replace('?', '')}.mp4"
    mvol = v.get("music", 0.24)
    mfile = MUS / f"{MUSIC.get(vid, 'Inspired')}.mp3"
    fc = (f"[1:a]volume={mvol},atrim=0:{T:.3f},afade=t=in:d=1.2,afade=t=out:st={max(0, T - 2.5):.3f}:d=2.5,"
          f"aformat=sample_rates=48000:channel_layouts=stereo[m];[0:a]asplit[x][y];"
          f"[m][x]sidechaincompress=threshold=0.03:ratio=9:attack=15:release=400[md];"
          f"[y][md]amix=inputs=2:duration=first:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=11[a];"
          f"[0:v]fade=t=out:st={max(0, T - 0.8):.3f}:d=0.8[v]")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(rough), "-stream_loop", "-1", "-i", str(mfile), "-filter_complex", fc,
                    "-map", "[v]", "-map", "[a]", "-t", f"{T:.3f}", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(final)], check=True)
    log(f"{vid}: DONE {final.name} {dur(final):.1f}s ({len(segs)}/{len(v['segs'])} segs)")
    return final

def send(vid):
    files = sorted(REND.glob(f"MEATZE_{vid}_*.mp4"))
    if not files: return
    f = files[-1]; v = VERSIONS[vid]
    tok = subprocess.check_output(["grep", "-oP", r"TELEGRAM_BOT_TOKEN=\K\S+", "/etc/systemd/system/tg-comfy-bot.service"]).decode().strip()
    cap = f"🎬 MEATZE {vid} — «{v['name']}» · {dur(f):.0f} с · {'9:16' if v['aspect'] == 'v' else '16:9'}"
    subprocess.run(["curl", "-s", f"https://api.telegram.org/bot{tok}/sendVideo", "-F", "chat_id=517188056",
                    "-F", f"video=@{f}", "-F", f"caption={cap}", "-F", "supports_streaming=true", "-o", "/dev/null"], timeout=600)
    log(f"{vid}: sent to Telegram")

if __name__ == "__main__":
    cmd, ids = sys.argv[1], sys.argv[2:] or list(VERSIONS)
    if cmd == "tts": do_tts()
    elif cmd == "render":
        for i in ids: render(i)
    elif cmd == "prewarm":
        for i in ids: render(i, prewarm_only=True)
    elif cmd == "send":
        for i in ids: send(i)
