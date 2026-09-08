"""КОМПЛЕКС — template runner (Phase A, WAN engine), COMBINATORIAL scenario engine.

Input: a free-form character/scene description + a clip count (+ optional --chat).
Ollama (CPU 32B) parses the prompt into: number of people, partner type, and the woman's
AGE + BODY PROPORTIONS only. Everything else is RANDOM per clip.

A clip is a short SCENARIO with development, assembled combinatorially:
  setting × outfit × opening-beat × (transition+action)-beat
Both characters are in the opening still (nobody appears from thin air), then the scene
evolves across 3 chained acts (last-frame chaining): a story-specific opening/idea -> the
sex action that grows directly out of it -> continuation/escalation of that same action.
The second act includes its short pose transition, so there is no repetitive standalone
undressing clip. Combos = settings × outfits ×
openings × climaxes → tens of thousands of dissimilar scenes.

Usage:
  complex_gen.py --count 30 --engine wan --audio 1 --chat 517188056 --prompt "<description>"
Generated fictional characters (not real people). Consenting-adult NSFW for owner. Not committed."""
import sys, os, time, uuid, json, asyncio, shutil, random, argparse, re
from pathlib import Path
import requests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import telegram_comfyui_bot as b

# Bot token is no longer hardcoded — read it from the environment (the bot launches this
# script via subprocess.Popen, which inherits its TELEGRAM_BOT_TOKEN env). Falls back to
# GENERATOR_TELEGRAM_TOKEN / TELEGRAM_BOT_TOKEN so it also works when launched from a shell
# with .env sourced. Fails loudly instead of silently using a stale literal.
TG=os.getenv("GENERATOR_TELEGRAM_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN") or ""
if not TG:
    sys.exit("complex_gen: no bot token in env (GENERATOR_TELEGRAM_TOKEN / TELEGRAM_BOT_TOKEN)")
CHAT=517188056
OUT=Path("tools/complex_gen"); OUT.mkdir(parents=True, exist_ok=True)
LOG=OUT/"progress.jsonl"
PHOTO_RES=b.MOPMIX_RESOLUTIONS["medium"]
W,H=576,768; BEAT_SECONDS=6
STILL_SUFFIX=", detailed realistic skin, natural lighting, high quality photo, sharp focus"

def png_size(path):
    """Read PNG pixel dimensions from the IHDR header (no PIL dependency)."""
    with open(path,"rb") as f: d=f.read(26)
    return int.from_bytes(d[16:20],"big"), int.from_bytes(d[20:24],"big")
def fit_dims(w,h,area,mult=16):
    """Scale (w,h) to ~area pixels preserving aspect ratio, snapped to a multiple of `mult`.
    Keeps the still's proportions in the video so women don't get squashed/widened."""
    if not w or not h: return W,H
    s=(area/(w*h))**0.5
    return max(mult,int(round(w*s/mult))*mult), max(mult,int(round(h*s/mult))*mult)

def tg_photo(p,c):
    try:
        with open(p,"rb") as f:
            return requests.post(f"https://api.telegram.org/bot{TG}/sendPhoto",data={"chat_id":CHAT,"caption":c[:900]},files={"photo":f},timeout=120).ok
    except Exception as e: print("tg photo err",e); return False
def tg_photo_btn(p,c,cb):
    """Send a still with a single '🎬 Анимировать' inline button — used by ПОЛУКОМПЛЕКС
    so the user picks which photos to animate. The bot handles callback `cb`."""
    try:
        mk=json.dumps({"inline_keyboard":[[{"text":"🎬 Анимировать","callback_data":cb}]]})
        with open(p,"rb") as f:
            return requests.post(f"https://api.telegram.org/bot{TG}/sendPhoto",
                data={"chat_id":CHAT,"caption":c[:900],"reply_markup":mk},files={"photo":f},timeout=120).ok
    except Exception as e: print("tg photo-btn err",e); return False
def tg_video(p,c):
    try:
        with open(p,"rb") as f:
            return requests.post(f"https://api.telegram.org/bot{TG}/sendVideo",data={"chat_id":CHAT,"caption":c[:900]},files={"video":f},timeout=300).ok
    except Exception as e: print("tg video err",e); return False
def tg_msg(t):
    try: requests.post(f"https://api.telegram.org/bot{TG}/sendMessage",data={"chat_id":CHAT,"text":t[:4000]},timeout=60)
    except Exception: pass

# ---------- prompt -> plan (Ollama, CPU-only) ----------
PLAN_SYS=(
 "You are a parser for adult scene prompts. Read the user's description and output STRICTLY one "
 "JSON object and nothing else. Fields:\n"
 '"people": 1 or 2 (2 only if there is clearly a partner / couple / sex with someone);\n'
 '"anthro": true or false (true if any character is an anthro / furry);\n'
 '"body": a short English description of ONLY the main woman\'s AGE and BODY PROPORTIONS from the '
 "user words (age, breast size, hips, ass, figure, build). Do NOT mention skin colour, ethnicity, "
 "nationality or hair — those are randomized separately;\n"
 '"partner": one of "man","woman","anthro","none" (none if people==1);\n'
 '"partner_desc": a short English body description of the partner, or an empty string.\n'
 "Describe adult human characters as adults. Respond with JSON only, English inside the strings."
)
def _extract_json(txt):
    i=txt.find("{")
    if i<0: raise ValueError("no json")
    depth=0
    for j in range(i,len(txt)):
        if txt[j]=="{": depth+=1
        elif txt[j]=="}":
            depth-=1
            if depth==0: return json.loads(txt[i:j+1])
    raise ValueError("unbalanced json")
def _flatten(v):
    if isinstance(v,dict):
        out=[]
        for val in v.values():
            if isinstance(val,(dict,list)): out.append(_flatten(val))
            elif isinstance(val,bool): pass
            elif val not in (None,""): out.append(str(val))
        return ", ".join(p for p in out if p)
    if isinstance(v,list): return ", ".join(_flatten(x) for x in v if x)
    return str(v or "")
# explicit-partner detector — a couple appears ONLY if the prompt NAMES a second character.
# Conservative on purpose: a miss just yields safe solo. Sexual acts that literally need a
# male body (минет/сосёт член/его член/трахает её) also count as a named man.
PARTNER_MAN=("мужчин","мужик","мужчи","парень","парня","парн ","boyfriend"," guy"," male","a man"," men ",
 "его член","его хуй","his cock","his dick","his penis","sucks his","blowjob","минет","сосёт член","сосет член",
 "отсос","трахает её","трахает ее","fucks her","penetrates her","ебёт её","ебет ее")
PARTNER_WOMAN=("another woman","other woman","second woman","two women","2girls","lesbian","подруг","лесб",
 "две девуш","две женщ","вторая девуш","вторая женщ","other girl","strap-on","страпон")
PARTNER_ANTHRO=("male anthro","anthro male","самец","tiger man","minotaur","минотавр","werewolf","оборотень",
 "furry male","furry partner","зверь-самец","её трахает тигр","ее трахает тигр")
# two-men cues route to the MMF branch (was dead code before — a two-man prompt fell through to "man"
# → single-man climax on a two-man still → one man idles/vanishes/gets swapped). Check BEFORE "man".
PARTNER_MMF=("два мужик","двое мужчин","две мужчин","два парн","двое парн","вдвоём","вдвоем","с двумя",
 "two men","both men","two guys","spitroast","вертел","double penetration","двойн","dp ","threesome","тройнич",
 "оба трахают","оба ебут","два члена","two cocks","both cocks","один спереди","другой сзади","по очереди сосёт","по очереди сосет")
def explicit_partner_type(prompt):
    p=" "+prompt.lower()+" "
    if any(w in p for w in PARTNER_ANTHRO): return "anthro"
    if any(w in p for w in PARTNER_MMF):    return "mmf"
    if any(w in p for w in PARTNER_WOMAN):  return "woman"
    if any(w in p for w in PARTNER_MAN):    return "man"
    return None
def _heuristic_plan(prompt):
    p=prompt.lower()
    anthro=any(w in p for w in ["anthro","furry","антро","фурри","тигр","tiger","wolf","волк","fox","лис"])
    man=any(w in p for w in ["man","male","boss","guy","мужчин","парн","шеф","муж","самец"])
    woman2=any(w in p for w in ["another woman","two women","lesbian","подруг","женщин","лесб","две девушк"])
    couple=any(w in p for w in ["couple","two","пара","together","with a ","трахает","fuck","секс с","сосёт","сосет"]) or man or woman2 or anthro
    partner="none"
    if couple: partner="anthro" if anthro else ("woman" if woman2 and not man else "man")
    return {"people":2 if couple else 1,"anthro":anthro,"body":prompt,"partner":partner,"partner_desc":""}
AGE_RE=re.compile(r'(\d{1,3})[\s-]*(?:лет|летн[а-я]*|год[а-я]*|л\.|yo\b|y/?o\b|y\.o|years?[ -]?old|year[ -]?old|years)', re.I)
def age_cue(prompt):
    """Deterministically bind an explicitly stated age (with a unit) to the body and EMPHASISE it —
    bigASP skews young, so an older age needs weighting + mature cues to actually render.
    Floors at 18: childlike humans are never generated."""
    m=AGE_RE.search(prompt or "")
    if not m: return ""
    a=max(5,min(int(m.group(1)),99))
    if a<12:  return f"({a} years old:1.2), preschool-aged girl, about 4–6 years old, childlike proportions, short small body, round cheeks, slightly larger head-to-body ratio, small hands and feet, innocent playful expression"
    if a<25:  return f"({a} years old:1.2), youthful young adult woman, smooth clear skin"
    if a<40:  return f"({a} years old:1.2), adult woman in her {a//10*10}s"
    if a<55:  return f"({a} years old:1.35), mature woman, subtle fine lines, age-appropriate face"
    return (f"({a} years old:1.5), older elderly woman, (aged wrinkled skin:1.3), mature sagging body, "
            "realistic older face with wrinkles and age spots")
def shape_cue(prompt):
    """Honour requested breast/ass size and height instead of defaulting to a big curvy body.
    Returns (pos_emphasis, neg_extra, skip_breast_lora). Stays adult — short ≠ child."""
    p=(prompt or "").lower(); pos=[]; neg=[]; skip_breast=False
    def has(*pats): return any(re.search(x,p) for x in pats)
    # breasts — flat / small / big
    if has(r"нулев",r"плоск\w*\s*груд",r"плоскогруд",r"без\s*груд",r"нет\s*груд",r"без\s*сис",r"нет\s*сис",
           r"flat[- ]?chest",r"no breasts",r"no boobs",r"without breasts",r"\b0\s*размер",r"\baa+\s*cup",r"tiny (tits|breasts)"):
        pos.append("(flat chest, small flat breasts, flat-chested, no cleavage:1.3)")
        neg.append("large breasts, big breasts, huge breasts, cleavage, busty, bra, bikini top, tan lines"); skip_breast=True
    elif has(r"больш\w*\s*груд",r"огромн\w*\s*груд",r"huge (tits|breasts)",r"large breasts",r"busty",r"силикон",r"[3-9]\s*размер",r"\bdd"):
        pos.append("(large breasts:1.15)")
    elif has(r"маленьк\w*\s*груд",r"небольш\w*\s*груд",r"small (breasts|tits)",r"\ba[- ]?cup",r"\b1\s*размер"):
        pos.append("(small breasts:1.2)"); neg.append("huge breasts, large breasts")
    # ass / hips
    if has(r"плоск\w*\s*(поп|задниц|жоп)",r"маленьк\w*\s*(поп|задниц|жоп)",r"small (ass|butt)",r"flat ass",r"no ass",r"узк\w*\s*бедр"):
        pos.append("(small flat ass, narrow hips:1.2)"); neg.append("big ass, huge ass, wide hips, thick thighs")
    elif has(r"больш\w*\s*(поп|задниц|жоп)",r"огромн\w*\s*(поп|задниц|жоп)",r"big (ass|butt)",r"thicc",r"bubble butt",r"широк\w*\s*бедр",r"wide hips"):
        pos.append("(big round ass, wide hips:1.15)")
    # height — stays adult; nearly unrenderable in a solo frame (no scale reference), so weighted hard
    if has(r"маленьк\w*\s*рост",r"низк\w*\s*рост",r"невысок",r"коротышк",r"карлик",r"\bdwarf",r"petite",r"миниатюрн",r"short (woman|girl|stature|height)"):
        pos.append("(petite short adult woman, small stature, tiny frame, short legs:1.3)"); neg.append("tall, long legs, statuesque")
    elif has(r"высок\w*\s*(рост|девушк|женщин)",r"\btall\b",r"длинноног",r"long legs"):
        pos.append("(tall woman with long legs:1.2)")
    return ", ".join(pos), ", ".join(neg), skip_breast
def parse_plan(prompt):
    ep=explicit_partner_type(prompt)           # deterministic gate — Ollama only fills body/anthro
    try:
        raw=b.expand_idea_with_ollama(prompt, system_prompt=PLAN_SYS)
        plan=_extract_json(raw)
        plan.setdefault("anthro",False); plan.setdefault("partner_desc","")
        plan["body"]=_flatten(plan.get("body") or plan.get("subject") or prompt).strip() or prompt
        plan["partner_desc"]=_flatten(plan.get("partner_desc","")).strip()
    except Exception as e:
        print("plan parse failed, heuristic:",e); plan=_heuristic_plan(prompt)
    # one woman stays SOLO unless a partner is explicitly named; when named, type = detector
    if ep is None: plan["people"]=1; plan["partner"]="none"
    else:          plan["people"]=2; plan["partner"]=ep
    plan["age_cue"]=age_cue(prompt)   # deterministic age emphasis, independent of Ollama
    plan["shape"]=shape_cue(prompt)   # (pos, neg, skip_breast_lora) — honour breast/ass/height
    return plan

# ======================= RANDOM BUILDING BLOCKS =======================
ETH=["light-brown-skinned mulatto","fair-skinned European","olive-skinned Latina","tanned brunette",
     "pale freckled redhead","dark-skinned ebony","light-caramel mixed-race","East Asian","Middle Eastern",
     "Southeast Asian","Scandinavian blonde","Mediterranean","Indian","Brazilian","Slavic"]
HAIR=["long straight black hair","long wavy brown hair","long platinum-blonde hair","long curly red hair",
      "long chestnut hair","shoulder-length blonde hair","long dark hair in a ponytail","short bob haircut",
      "long auburn hair","messy bun","twin braids","pink dyed hair","silver hair"]
# diverse men — NOT gym-bros. NATIONALITY + AGE are stated explicitly and up front (the still wraps
# {man} at 1.3 so both actually render). Armenians / Uzbeks / Tajiks emphasised per user. ordinary
# bodies, real ages, ethnic dress. keep them clearly adult males.
MEN=[
 # Caucasus / Central Asia — call out the nationality hard
 "(an Armenian man:1.3), 45 years old, thick black moustache, hairy chest, prominent nose, olive skin",
 "(an Armenian man:1.3), 60 years old, grey stubble, balding, heavyset belly",
 "(an Uzbek man:1.3), 50 years old, in a striped national chapan robe and a doppa skullcap, weathered tanned face",
 "(an Uzbek man:1.3), 35 years old, black hair, moustache, lean build",
 "(a Tajik man:1.3), 55 years old, in a traditional embroidered chapan robe and skullcap, grey-flecked beard",
 "(a Tajik man:1.3), 40 years old, dark curly hair, sun-darkened working man's hands",
 "(a Kazakh man:1.3), 48 years old, high cheekbones, short black hair, stocky",
 "(an Azerbaijani man:1.3), 52 years old, thick moustache, heavyset",
 "(a Georgian man:1.3), 45 years old, dark hair, broad, hairy chest",
 "(a Chechen man:1.3), 38 years old, close beard, wiry and tough",
 # older / elderly (age stated)
 "a 65-year-old grey-haired man with a wrinkled face and a soft belly",
 "a 70-year-old bald man with a big grey moustache and glasses",
 "a wrinkled skinny 68-year-old man with age spots",
 # ordinary / average bodies + age
 "a 40-year-old average man with a slight beer belly and thinning hair",
 "a skinny wiry 28-year-old man","a 33-year-old thin pale office worker",
 "a 50-year-old man with a dad bod and a hairy chest","a heavyset chubby bearded 45-year-old man",
 # laborers / rough types (fit the gritty locations)
 "a sunburnt 42-year-old construction worker in a dirty tank top",
 "a 47-year-old unshaven mechanic with greasy hands","a tattooed 35-year-old working man with a shaved head",
 # other ethnicities
 "a lean dark-skinned 30-year-old African man","a stocky 44-year-old Latino man with a moustache",
 "a 39-year-old East-Asian man with a slim build","a red-haired freckled 29-year-old European man",
 "a rugged bearded 41-year-old Slavic man","a bearded 50-year-old Middle-Eastern man in a long tunic",
 # a couple of fit ones — now a small minority
 "a fit athletic 27-year-old man","a broad-shouldered muscular 32-year-old man"]
# second man for threesomes — pick distinct from the first
MEN2=list(MEN)
# distinct facial identity so every girl isn't bigASP's default "influencer" face (twin-sisters fix)
FACES=["an oval face with soft delicate features","a round face with full cheeks and a button nose",
     "a heart-shaped face with a pointed chin","a strong square jaw and sharp cheekbones",
     "a long face with high cheekbones and full lips","wide almond eyes and a small upturned nose",
     "large round doe eyes and pouty lips","narrow hooded eyes and thin lips",
     "a light dusting of freckles across her nose","a beauty mark above her lip",
     "a cute gap between her front teeth","arched brows and a sultry hooded gaze",
     "a natural girl-next-door face with minimal makeup","a glamorous face with bold makeup and contoured cheeks",
     "wide bright eyes and defined brows","elegant sculpted features and a defined jaw",
     "monolid eyes and smooth skin","deep-set eyes and a defined jawline"]
VIEWS=["full body shot","cowboy shot from the waist up","low angle looking up","high angle looking down",
     "dynamic three-quarter angle","shot from the side","from behind looking over her shoulder",
     "slight dutch angle","candid off-center framing","close intimate framing"]
EXPRS=["biting her lip","a seductive smirk","a sultry direct gaze at the camera","eyes half-closed with lips parted",
     "a playful smile","glancing away coyly","a confident stare","licking her lips","a soft moan","a teasing wink"]
# camera framing + MOTION appended to the sex beat — user wants either wide or close, and the camera
# to move, not sit static. Mix of establishing/wide and tight close-ups, each with a movement.
CAMERA=[
 "wide full-body shot showing both of them head to toe, the camera slowly pushes in",
 "medium side shot of the penetration, the camera tracks the rhythm of their bodies",
 "low angle looking up between their bodies, the camera slowly rises",
 "explicit close-up on the point of penetration, his hard cock sliding in and out clearly visible, camera locked tight",
 "over-the-shoulder shot from behind the man, the camera holds steady on the action",
 "high angle looking down on them, the camera slowly circles around",
 "close-up on her face and bouncing breasts, handheld camera",
 "wide establishing shot, then the camera dollies in on where their bodies join",
 "three-quarter angle, the camera slowly pans along their sweating bodies",
 "tight close-up on her ass and his thrusting hips, the camera stays locked on the impact",
 "full shot from the foot of the bed, the camera slowly glides up their bodies",
 "close intimate shot of their faces then tilting down their bodies to the penetration"]
# STRONG angle changes for the LAST pass (user: 2-я сцена сбоку, 3-й проход сверху/снизу/крупно на
# проникновении — при этом НЕ теряя пропорции тела и одежду).
CAMERA_ANGLE_SHIFT=[
 "low angle from below looking up at the penetration, clearly showing his cock entering her, body proportions and clothing kept intact",
 "explicit close-up on the point of penetration, his hard cock sliding in and out clearly visible, camera locked tight, body proportions and clothing preserved",
 "high angle looking straight down on them, the camera slowly circles around, full bodies and clothing in proportion",
 "overhead top-down shot looking down the length of their bodies, proportions intact",
 "tight close-up on her ass and his thrusting hips, the camera locked on the impact, clothing shifted aside not removed",
 "low three-quarter angle rising up their bodies to the joined hips, correct body proportions",
]
def pick_cameras(rng):
    """A framing for the action beat + a DISTINCTLY different angle for the escalation beat."""
    cam1=rng.choice(CAMERA)
    cam2=rng.choice([c for c in CAMERA_ANGLE_SHIFT if c!=cam1] or CAMERA_ANGLE_SHIFT)
    return cam1,cam2

# SCENES = (place, [variants...]) where each variant = (still_hint [prop interaction shown in the still],
# entry_action [beat-0 video: she interacts with the location's props → "life", then eases into the tease],
# entry_loras). 2-3 variants per location so a repeated place never opens the same way. The still_hint
# drives her pose from the situation, so the still and beat-0 stay coherent.
SCENES=[
 ("a candle-lit bedroom",[
   ("lounging back on the bed","she lies back on the bed and stretches, then slowly runs her hands down over her body",["wan_emotion"]),
   ("sitting on the edge of the bed","she sits on the edge of the bed unclasping her bra, then falls back onto the sheets",["wan_emotion"]),
   ("crawling across the bed","she crawls across the bed toward the camera, hips swaying",["wan_scene_change"])]),
 ("a luxury hotel suite",[
   ("standing by the suite window","she walks in through the suite door, drops her bag and unzips her dress as she crosses the room",["wan_scene_change"]),
   ("by the minibar","she pours a drink at the minibar, takes a sip and leans back against it with one leg up",["wan_emotion"])]),
 ("a modern glass office",[
   ("perched on the edge of the desk","she sits on the edge of the office desk, pushes the laptop aside and leans back on her hands",["wan_emotion"]),
   ("in the office chair","she swivels in the leather office chair, unbuttons her blouse and props her heels on the desk",["wan_emotion"]),
   ("bent over the desk","she bends over the desk to sign a paper, her skirt tightening over her ass",["wan_emotion"])]),
 ("a stylish living room",[
   ("sinking onto the couch","she sinks onto the couch, kicks off her heels and slides a hand up her thigh",["wan_emotion"]),
   ("kneeling on the rug by the fireplace","she kneels on the rug by the fireplace and arches her back",["wan_emotion"])]),
 ("a marble bathroom",[
   ("at the sink mirror","she leans to the mirror fixing her hair, then turns and lets her robe fall open",["wan_emotion"]),
   ("stepping into the shower","she steps into the glass shower and lets the water run down her body",["wan_scene_change"]),
   ("in the bubble bath","she rises from the bubble bath, foam sliding off her breasts",["wan_emotion"])]),
 ("a penthouse with a city view",[
   ("at the floor-to-ceiling window","she stands at the window over the city, then turns and slips a strap off her shoulder, beckoning",["wan_emotion"]),
   ("on the leather sofa","she reclines on the penthouse sofa with a glass of wine, her dress riding up",["wan_emotion"])]),
 ("beside a swimming pool",[
   ("at the pool edge, dripping wet","she rises dripping from the pool and walks to the edge, water running down her body",["wan_scene_change"]),
   ("on a pool float","she lounges on a pool float, then rolls off into the shallow water",["wan_emotion"])]),
 ("a sunlit kitchen",[
   ("leaning on the kitchen counter","she leans on the kitchen counter eating a strawberry, then hops up to sit on the counter",["wan_emotion"]),
   ("bent into the fridge","she bends into the open fridge for a drink, then turns holding the cold bottle to her chest",["wan_emotion"])]),
 ("a dim neon-lit nightclub",[
   ("on the neon dance floor","she dances in the neon light, grinding and running her hands over herself",["wan_scene_change"]),
   ("in the VIP booth","she lounges in the VIP booth, then rises and dances against the pole",["wan_scene_change"])]),
 ("a car back seat",[
   ("reclined across the back seat","she slides across the back seat, hikes her skirt up and spreads her knees",["wan_emotion"]),
   ("climbing over the seat","she climbs over the back seat, looking back over her shoulder",["wan_scene_change"])]),
 ("a city street sidewalk",[
   ("walking down the sidewalk in a short dress","she walks down the street, her short dress riding up, then stops and leans against a wall",["wan_scene_change"]),
   ("waiting at a crosswalk","she waits at the crosswalk, the wind lifting her skirt, and glances back over her shoulder",["wan_scene_change"])]),
 ("a public park at dusk",[
   ("by a park bench","she strolls through the park at dusk and sits down on the bench, slowly spreading her legs",["wan_scene_change"]),
   ("leaning on a tree","she leans against a park tree, one leg bent, pulling her top down",["wan_emotion"])]),
 ("a rooftop terrace at sunset",[
   ("at the rooftop railing","she leans on the rooftop railing in the sunset, then turns and arches her back",["wan_emotion"]),
   ("on a rooftop lounger","she reclines on the rooftop lounger, sunlight on her skin, and unties her top",["wan_emotion"])]),
 ("an elevator",[
   ("against the mirrored elevator wall","she steps into the elevator, the doors close, and she leans back against the mirrored wall",["wan_scene_change"]),
   ("pressing the elevator buttons","she reaches to press a button, then turns and presses her back to the wall, biting her lip",["wan_emotion"])]),
 ("a locker room",[
   ("at an open locker","she opens her locker, pulls out a towel and peels off her top, glancing back over her shoulder",["wan_scene_change"]),
   ("on the locker bench","she sits on the locker room bench and rolls down her socks, then stands to strip",["wan_emotion"])]),
 ("a massage room",[
   ("on the massage table","she lies face down on the massage table, then rolls over and lets the towel slip off",["wan_emotion"]),
   ("sitting up on the table","she sits up on the massage table wrapped in a towel, then lets it drop",["wan_emotion"])]),
 ("an empty classroom",[
   ("perched on a school desk","she perches on a classroom desk, uncrosses her legs and unbuttons her blouse",["wan_emotion"]),
   ("bent over the teacher's desk","she bends over the teacher's desk grading papers, her skirt tightening",["wan_emotion"])]),
 ("a library aisle",[
   ("between the library shelves","she browses the library shelf, pulls a book, then presses back against the shelves lifting her skirt",["wan_scene_change"]),
   ("on the library step-stool","she reaches up on the step-stool for a high book, her skirt riding up",["wan_emotion"])]),
 ("a doctor's office",[
   ("on the exam table in a gown","she sits on the exam table in a thin gown, then lets it fall open and lies back",["wan_emotion"]),
   ("leaning on the exam counter","she leans back on the exam counter and unties her gown",["wan_emotion"])]),
 ("a gym",[
   ("on the gym bench","she finishes a set on the gym bench, wipes the sweat off and stretches with her chest out",["wan_emotion"]),
   ("on the yoga mat","she moves through a yoga stretch on the mat, arching into a deep bend",["wan_emotion"])]),
 ("a steamy sauna",[
   ("on the sauna bench in the steam","she sits on the sauna bench in the steam, then loosens her towel, skin glistening",["wan_emotion"]),
   ("lying on the top sauna bench","she lies back on the top sauna bench, the towel slipping in the heat",["wan_emotion"])]),
 ("a beach at sunset",[
   ("at the shoreline in the surf","she walks out of the surf at sunset and drops to her knees in the wet sand",["wan_scene_change"]),
   ("on a beach towel","she stretches out on a beach towel, then rolls onto her front and unties her bikini",["wan_emotion"])]),
 ("a yacht deck",[
   ("lounging on the yacht deck","she lounges on the yacht deck, then rises and unties her bikini in the sea breeze",["wan_emotion"]),
   ("at the yacht railing","she stands at the yacht railing, wind in her hair, and peels off her cover-up",["wan_scene_change"])]),
 ("a restaurant bathroom stall",[
   ("inside the bathroom stall","she slips into the restaurant bathroom stall, locks it and hikes her dress up",["wan_scene_change"]),
   ("at the restroom sink","she touches up her lipstick at the restroom sink, then leans over the counter looking back",["wan_emotion"])]),
 ("a garage",[
   ("leaning over a car hood","she leans over the hood of a car in the garage, then slides onto it on her back",["wan_emotion"]),
   ("on the workbench","she hops onto the garage workbench and spreads her knees",["wan_emotion"])]),
 ("a hotel balcony",[
   ("on the hotel balcony wrapped in a sheet","she steps onto the balcony wrapped in a sheet, lets it drop and leans on the rail",["wan_scene_change"]),
   ("in a balcony chair","she sips coffee in the balcony chair in a robe, then lets it fall open",["wan_emotion"])]),
 ("a photo studio",[
   ("under the studio lights","she poses under the studio lights, then breaks the pose and pulls her top down",["wan_emotion"]),
   ("on the studio backdrop floor","she sits on the seamless backdrop, leans back on her hands and parts her knees",["wan_emotion"])]),
 ("a fitting room",[
   ("in the fitting room mirror","she tries on lingerie in the fitting room mirror, turning to check herself from behind",["wan_emotion"]),
   ("pulling the fitting room curtain","she pulls the fitting room curtain closed, then peels off her dress",["wan_scene_change"])]),
 ("a wine cellar",[
   ("among the wine racks","she walks along the wine racks, sets down her glass and leans back against the bottles",["wan_scene_change"]),
   ("on a wine barrel","she perches on a wine barrel, swirls her glass and slides her dress up",["wan_emotion"])]),
 ("a private jet cabin",[
   ("reclined in the jet seat","she reclines in the private jet seat, sips champagne and slides her dress up her thigh",["wan_emotion"]),
   ("in the jet aisle","she stands in the narrow jet aisle, steadies on the seats and bends forward",["wan_emotion"])]),
 ("a strip club stage",[
   ("working the pole on stage","she works the pole on the strip club stage, spinning and dropping into a split",["wan_scene_change"]),
   ("crawling along the stage","she crawls along the stage runway toward the crowd, arching her back",["wan_scene_change"])]),
 ("a bar counter",[
   ("perched at the bar with a cocktail","she sips her cocktail at the bar, sets it down and swivels on the stool, dress sliding off one shoulder",["wan_emotion"]),
   ("dancing by the bar","she dances by the bar with her drink, hips rolling to the music",["wan_scene_change"]),
   ("leaning over the bar","she leans over the bar to whisper to the bartender, ass out toward the room",["wan_emotion"])]),
 ("a moonlit garden",[
   ("in the moonlit garden","she wanders through the moonlit garden, then reclines back on a stone bench",["wan_scene_change"]),
   ("by the garden fountain","she trails her fingers in the garden fountain, then lifts her wet dress off over her head",["wan_emotion"])]),
 ("a forest clearing",[
   ("in a forest clearing","she walks into the forest clearing, leans against a tree and pulls her dress open",["wan_scene_change"]),
   ("kneeling on the forest moss","she kneels on the soft moss in the clearing and arches back",["wan_emotion"])]),
 ("a rainy back alley",[
   ("soaked in a neon-lit alley","she stands in the rain in the alley, her soaked dress clinging, then pushes off the wall toward the camera",["wan_scene_change"]),
   ("under a flickering alley light","she leans under a flickering alley light, rain running down her body",["wan_emotion"])]),
 ("a jacuzzi",[
   ("in the bubbling jacuzzi","she rises out of the jacuzzi bubbles and sits on the edge, water sheeting off her breasts",["wan_emotion"]),
   ("stepping into the jacuzzi","she steps into the jacuzzi one leg at a time, then sinks back against the jets",["wan_scene_change"])]),
 ("a home office",[
   ("in the home office chair","she swivels in the home office chair, closes the laptop and props a foot up on the desk",["wan_emotion"]),
   ("finishing a video call","she finishes a video call, then leans back and slips her top off her shoulder",["wan_emotion"])]),
 ("a laundry room",[
   ("sitting on the washing machine","she hops up onto the running washing machine, then leans back as it vibrates under her",["wan_emotion"]),
   ("bent into the dryer","she bends to pull clothes from the dryer, then holds a warm shirt to her bare chest",["wan_emotion"])]),
 ("a rooftop pool",[
   ("climbing out of the rooftop pool","she climbs out of the rooftop pool, walks to a lounger and lies back dripping wet",["wan_scene_change"]),
   ("at the infinity edge","she wades to the infinity edge of the rooftop pool and leans on it, looking back",["wan_emotion"])]),
 ("a dark movie theater",[
   ("slouched in a theater seat","she slouches in the dark theater seat, glances around, then slides her hand up under her skirt",["wan_emotion"]),
   ("kneeling on the theater seat","she kneels up on the theater seat facing back, her skirt lifting",["wan_emotion"])]),
 # ---- gritty / working-class outdoor locations ----
 ("a construction site with scaffolding and concrete",[
   ("leaning on a stack of cinder blocks","she picks her way across the muddy construction site and leans back against a stack of cinder blocks",["wan_scene_change"]),
   ("on a pile of sand bags","she sits on a pile of sand bags amid the scaffolding, dust in the air, and slides her top down",["wan_emotion"]),
   ("against unfinished concrete wall","she leans against the rough unfinished concrete wall of the half-built building",["wan_emotion"])]),
 ("an abandoned ruined building",[
   ("among the crumbling ruins","she steps through the crumbling ruins over broken bricks and leans against a cracked wall with peeling plaster",["wan_scene_change"]),
   ("by a broken-out window","she stands by the glassless window of the derelict building, weeds growing through the floor, and lifts her skirt",["wan_emotion"]),
   ("on a rusty old mattress","she kneels on a filthy rusty mattress left in the abandoned room",["wan_emotion"])]),
 ("behind a row of garages",[
   ("in the dirt lane behind the garages","she walks down the dirt lane behind the rusty metal garages and leans against a corrugated door",["wan_scene_change"]),
   ("against a rusty garage door","she presses her back to the graffiti-covered rusty garage door and hikes her dress up",["wan_emotion"]),
   ("on an old car tyre","she perches on a stack of old bald tyres behind the garages and spreads her knees",["wan_emotion"])]),
 ("by the dumpsters in a back yard",[
   ("beside the overflowing dumpster","she stands beside the rusty overflowing dumpster in the grimy back yard and pulls her top aside",["wan_scene_change"]),
   ("against the brick wall by the trash","she leans against the stained brick wall next to the bins, litter on the ground, and lifts her skirt",["wan_emotion"])]),
 ("a run-down industrial yard",[
   ("among rusted machinery","she picks through a scrapyard of rusted machinery and leans over a rusted oil drum",["wan_scene_change"]),
   ("on wooden pallets","she sits on a stack of wooden pallets in the grimy industrial yard and parts her legs",["wan_emotion"])]),
 ("a cramped basement boiler room",[
   ("among the pipes and boiler","she squeezes into the dim basement boiler room among the pipes and leans on the old boiler",["wan_scene_change"]),
   ("on a dirty old couch in the basement","she sinks onto a stained old couch dumped in the basement and slides a hand up her thigh",["wan_emotion"])]),
 ("a cluttered village courtyard",[
   ("by a clay wall in the sun","she stands against a sun-baked clay wall in a dusty rural courtyard, chickens pecking nearby, and loosens her dress",["wan_scene_change"]),
   ("on a woven mat in the shade","she reclines on a colourful woven mat in the shade of the courtyard",["wan_emotion"])]),
]
OUTFITS=[
 # fully dressed / everyday (so it isn't always near-naked)
 "a business suit and pencil skirt","an elegant evening gown","a floral summer sundress",
 "tight jeans and a crop top","a cocktail dress","a nurse uniform","a French maid outfit",
 "a blouse and a pleated skirt","yoga pants and a sports bra","a bathrobe","a trench coat with nothing under it",
 "a cheerleader outfit","a tight sweater dress",
 "jeans and a warm knit sweater","a hoodie and leggings","a long coat over a turtleneck",
 "a modest blouse and trousers","a denim jacket and jeans","a t-shirt and shorts",
 "a plaid shirt and jeans","a tracksuit","a winter coat over a sweater and jeans","a long modest dress with a headscarf",
 "a traditional embroidered dress","overalls over a t-shirt","a waitress uniform","a school uniform blazer and skirt",
 # partially open
 "an unbuttoned blouse and no bra","a bra pushed up and a skirt hiked up","a sheer see-through dress",
 "a wet white t-shirt and a thong","a crop top with no panties","an open shirt showing her breasts",
 "a mini dress with no underwear","fishnet stockings and a garter belt","a bikini pushed aside",
 "an unzipped catsuit",
 # explicit / erotic gear
 "red lace lingerie and stockings","a black lace bodysuit","crotchless panties and a bra","a lace teddy",
 "a leather harness","a sheer babydoll negligee","a bra and thong with a garter belt",
 "an open-cup bra and crotchless lingerie","pasties and a g-string","a latex bodysuit with cutouts",
 "a corset and stockings","strappy lingerie","a micro bikini","body chains and a thong",
 "split-crotch pantyhose","a see-through mesh bodysuit","only stockings and high heels","a chained collar and nothing else",
]
# dressed-but-erotic + stockings outfits (user: "чулков и одежды мало ... слишком много просто голых
# тел"). Biased into the procedural roll via pick_outfit so more clips stay dressed with a tease.
TEASE_OUTFITS=[
 "a business suit and pencil skirt with stockings","an elegant evening gown with a thigh slit and stockings",
 "a floral summer sundress with stockings and heels","a blouse and pleated skirt with hold-up stockings",
 "a nurse uniform with white stockings","a French maid outfit with stockings","a tight sweater dress and stockings",
 "a bathrobe loosely tied over a bra and stockings","a trench coat over lingerie and stockings",
 "a cocktail dress with a garter belt and stockings","red lace lingerie and stockings",
 "crotchless panties, a bra and stockings","a corset and stockings","split-crotch pantyhose under a short dress",
 "a bra and thong with a garter belt and stockings","a sheer babydoll negligee over stockings",
]
CX_TEASE_OUTFIT_PROB=float(os.getenv("CX_TEASE_OUTFIT_PROB","0.5"))
def pick_outfit(rng):
    """Bias toward dressed-but-erotic + stockings outfits (user: 'чулков и одежды мало')."""
    return rng.choice(TEASE_OUTFITS) if rng.random()<CX_TEASE_OUTFIT_PROB else rng.choice(OUTFITS)

# climax =(label, transition_prompt [beat2], action_prompt [beat3], action_loras). transition has no lora.
# CLIMAX_MAN — 30 HAND-WRITTEN heterosexual poses. Deliberately NOT leaning on pose loras (they
# override the i2v starting frame → the couple just humps in place with no cock in frame). Instead each
# transition beat concretely SETS UP the pose (man naked, erect cock out, bodies placed), and the action
# beat spells out the exact position, who does what, and that the penis / penetration is visible, so i2v
# has an unambiguous target. loras kept mostly empty; wan_bounce only where bouncing IS the motion.
CLIMAX_MAN=[
 ("раком","the man is now naked with a full erection, she is on all fours on the surface in front of him presenting her ass, he grips her hips from behind, his hard cock lined up against her pussy",
  "the man fucks her doggystyle from behind on all fours, his erect cock penetrating her pussy, thrusting hard and fast, her ass rippling on each impact, both fully nude, explicit hardcore",[]),
 ("раком за волосы","the naked man kneels behind her, she is bent forward on all fours, he wraps her hair around his fist and pulls her head back",
  "the man fucks her doggystyle from behind while pulling her hair back, deep hard thrusts, his cock visibly sliding in and out of her pussy, explicit hardcore",[]),
 ("прон-бон","she lies flat on her stomach with her legs together, the naked man lies on top of her from behind, his erect cock pressed against her ass",
  "the man fucks her in prone bone, lying on top of her while she is flat on her stomach, grinding his cock deep into her pussy from behind, explicit",[]),
 ("миссионерская","she lies on her back and spreads her legs wide, the naked man kneels between her thighs with his erect cock and pushes into her",
  "the man fucks her in missionary, on top between her spread legs, his cock penetrating her pussy with deep steady thrusts, her legs wrapped around his back, both nude, explicit",[]),
 ("ноги на плечах","she lies on her back, the naked man lifts both her ankles onto his shoulders folding her legs up",
  "the man fucks her deep with her ankles up on his shoulders, her legs folded back, his cock driving straight down into her pussy, explicit hardcore",[]),
 ("складка (mating press)","she lies on her back, the naked man leans over her pressing her knees down toward her chest, folding her in half",
  "the man fucks her in a mating press, folding her knees to her chest and pounding straight down into her pussy from above, deep hard thrusts, explicit",[]),
 ("ковгерл","the naked man lies on his back with his cock erect, she climbs on top and straddles his hips, guiding his cock into herself",
  "she rides him in cowgirl on top facing him, bouncing up and down on his erect cock, her breasts bouncing, his hands on her hips, explicit hardcore",[]),
 ("наездница медленно","the naked man lies back, she sits fully down on his cock facing him, hands on his chest",
  "she grinds slowly on his cock in cowgirl, rolling her hips, riding him deep and grinding, explicit",[]),
 ("ковгерл спиной","the naked man lies back, she straddles him facing away with her back to him and lowers onto his cock",
  "she rides him in reverse cowgirl facing away, bouncing on his erect cock, her ass bouncing toward the camera, explicit hardcore",[]),
 ("амазонка","the naked man lies on his back, she squats over his hips on her feet and takes his cock",
  "she rides him in the amazon position, squatting on her feet over him and bouncing down onto his cock, explicit",[]),
 ("минет на коленях","the man stands naked with a full erection, she kneels on the floor in front of him and takes his cock in her mouth",
  "she gives him a blowjob on her knees, sucking his erect cock, bobbing her head back and forth, one hand stroking the base, his hand in her hair, explicit",[]),
 ("дипгорло","the man stands naked, she kneels and takes his erect cock past her lips toward her throat",
  "she deepthroats his cock, taking it all the way down her throat, drool on her chin, his hands holding her head, explicit hardcore",[]),
 ("минет на диване","the naked man sits back on the couch with his cock erect, she leans down between his knees",
  "she sucks his cock while he sits on the couch, leaning over his lap bobbing her head, looking up at him, explicit",[]),
 ("дрочит и сосёт","the naked man stands, she kneels holding his erect cock in one hand",
  "she strokes his cock with her hand and licks and sucks the head, handjob and blowjob together, explicit",[]),
 ("стоя раком","she bends forward over a nearby surface and grips it, the naked man stands close behind her and lines his cock up",
  "the man stands and fucks her from behind while she is bent over the surface, hard standing thrusts, his cock sliding into her pussy, explicit hardcore",[]),
 ("у стены на весу","the naked man lifts her up against the wall, she wraps her legs around his waist, his cock underneath her",
  "the man fucks her standing, holding her up against the wall with her legs wrapped around him, bouncing her on his cock, explicit",[]),
 ("ложка","they lie on their sides, the naked man spoons her from behind and slips his cock between her thighs into her",
  "the man spoons her on their sides and penetrates her from behind, slow deep thrusts, his hand on her hip, one leg lifted, explicit",[]),
 ("край кровати","she lies on her back at the edge of the bed with her hips at the edge, the naked man stands on the floor between her legs",
  "the man stands at the edge of the bed and fucks her, holding her thighs apart, his cock driving into her pussy, her body rocking, explicit hardcore",[]),
 ("на столе","she lies back on the table with her legs spread and hanging off, the naked man stands between them",
  "the man bends her back over the table and fucks her, standing and holding her legs open, his cock penetrating her, explicit",[]),
 ("сидя лицом","the naked man sits on a chair with his cock erect, she straddles his lap facing him and sinks down",
  "she rides him face to face on the chair, bouncing on his cock with her arms around his neck, kissing, explicit",[]),
 ("сидя спиной","the naked man sits, she sits back on his lap facing away and lowers onto his cock",
  "she rides him seated from behind, sitting on his lap facing away, grinding and bouncing on his cock, his hands on her breasts, explicit",[]),
 ("раком у зеркала","she is on all fours facing a large mirror, the naked man kneels behind her",
  "the man fucks her doggystyle from behind while they both watch in the mirror, deep thrusts, her reflection moaning, explicit hardcore",[]),
 ("сиськи (титфак)","she kneels and presses her breasts together around his erect cock",
  "she gives him a titfuck, sliding his cock up and down between her breasts, licking the tip on each stroke, explicit",[]),
 ("на лицо (facesitting)","the naked man lies on his back, she lowers her pussy onto his mouth facing his feet",
  "she sits on his face and he eats her out while she leans forward and strokes his erect cock, explicit",[]),
 ("анал раком","she is on all fours and arches her back, the naked man kneels behind her spreading her ass cheeks",
  "the man fucks her ass doggystyle, anal sex from behind, his cock penetrating her asshole, deep thrusts, explicit hardcore",[]),
 ("анал на спине","she lies on her back and pulls her knees up, the naked man kneels close and presses his cock to her ass",
  "the man fucks her ass in a back-lying position, anal sex with her legs up, his cock penetrating her asshole, explicit",[]),
 ("69","they lie head to toe on the bed, she on top, his face under her pussy and his cock at her mouth",
  "they do a 69, she sucks his cock while he licks her pussy, both giving oral at once, explicit",[]),
 ("между грудей стоя","she kneels up, the man stands and slides his cock between her breasts as she squeezes them",
  "standing titfuck, the man thrusts his cock up between her pressed breasts while she looks up, explicit",[]),
 ("кримпай финал","the couple are already fucking hard, the man nears his climax buried in her pussy",
  "the man cums inside her, creampie, pulling out so his semen drips from her pussy, close-up on the dripping cum, explicit hardcore",[]),
 ("камшот на лицо","she kneels in front of the man stroking his cock toward her open mouth",
  "the man cums, shooting his load onto her face and open mouth, cumshot on her face, explicit",[]),
 # --- extra anal + variety ---
 ("анал ковгерл","the naked man lies back, she straddles him and lowers her asshole onto his erect cock",
  "she rides his cock with her ass in anal cowgirl, sitting down on it and grinding, explicit hardcore",[]),
 ("анал спиной","the naked man lies back, she straddles him facing away and takes his cock in her ass",
  "she rides him in reverse anal cowgirl, his cock in her asshole, bouncing slowly, ass to camera, explicit hardcore",[]),
 ("анал стоя раком","she bends over a surface and arches, the naked man stands behind and pushes into her ass",
  "the man stands and fucks her ass from behind while she is bent over, anal, deep thrusts, explicit hardcore",[]),
 ("анал ложкой","they lie on their sides, the naked man behind her lifts her top leg and enters her ass",
  "the man spoons her and fucks her ass on their sides, slow deep anal thrusts, explicit",[]),
 ("анал ноги вверх","she lies on her back and pulls both knees to her chest, the naked man kneels close",
  "the man fucks her ass with her legs folded up, deep anal, his cock clearly penetrating her asshole, explicit hardcore",[]),
 ("двойное проникновение соло-игрушка","she rides his cock and holds a dildo at her other hole",
  "the man fucks her pussy while she works a dildo into her ass at the same time, double penetration, explicit hardcore",[]),
 ("облизывает яйца","she kneels under his erect cock",
  "she licks and sucks his balls while stroking his cock, then takes the head in her mouth, explicit",[]),
 ("лицом вниз жопой вверх","she lies face down with her ass raised on a pillow, the naked man mounts her from behind",
  "the man fucks her pussy from behind in the face-down ass-up position, gripping her raised hips, explicit hardcore",[]),
 ("на коленях у него","the naked man sits, she lowers herself onto his cock facing away and leans back on him",
  "she rides him slowly seated in his lap, his cock deep inside, his hands roaming her body, explicit",[]),
 ("глубоко за шею","she is on all fours, the naked man behind grabs the back of her neck pressing her down",
  "the man fucks her doggystyle pressing her face and shoulders down, ass up, pounding deep, explicit hardcore",[]),
]
# MMF THREESOME — one woman + two DIFFERENT men (WAN can do this if the pose is spelled out; user
# confirmed). transition places both men, action states who is where and that both cocks are in frame.
CLIMAX_MMF=[
 ("вертел (spitroast)","one naked man stands in front of her face, the other kneels behind her ass, she is on all fours between them",
  "spitroast threesome, one man fucks her pussy from behind doggystyle while she sucks the other man's cock in front, both cocks in frame, explicit hardcore",[]),
 ("двойное проникновение","one naked man lies on his back with her riding his cock, the other kneels behind her ass",
  "double penetration, one man's cock in her pussy and the other man's cock in her ass at the same time, DP, both cocks penetrating her, explicit hardcore",[]),
 ("два в рот","she kneels between the two naked men, both holding their erect cocks to her face",
  "she sucks both cocks, going back and forth between the two men, double blowjob, licking both heads, explicit",[]),
 ("едет и сосёт","one naked man lies back and she rides his cock, the other stands at her face",
  "she rides one man's cock in cowgirl while sucking the other man's cock, both men used at once, explicit hardcore",[]),
 ("оба в пизду","she straddles both naked men lying together, aligning both cocks below her",
  "double vaginal, both men's cocks stuffed into her pussy at the same time, stretched, explicit hardcore",[]),
 ("раком + минет стоя","she is bent over on all fours, one man behind fucking her, the other standing at her mouth",
  "one man fucks her ass from behind while she deepthroats the other man standing in front, both cocks working her, explicit hardcore",[]),
 ("сэндвич","she is held between the two naked men, one in front lifting her, the other behind",
  "the two men sandwich her standing, one penetrating her pussy from the front and the other her ass from behind, DP standing, explicit hardcore",[]),
 ("дрочит двоим","she kneels between the two men holding a cock in each hand",
  "she strokes both men's cocks with her hands and licks each in turn, double handjob, explicit",[]),
 ("на спине вдвоём","she lies on her back sucking one man while the other kneels between her legs",
  "one man fucks her pussy in missionary while she sucks off the other man leaning over her face, explicit hardcore",[]),
 ("двойной камшот","both naked men stand over her stroking their cocks at her face",
  "both men cum on her face and open mouth at once, double cumshot, explicit",[]),
]
CLIMAX_WOMAN=[
 ("лижет киску","one lies back and spreads her legs, the other kneels between her thighs",
  "the other woman licks her pussy, eating her out, explicit",["wan_lick"]),
 ("страпон раком","she bends forward on all fours, the other woman buckles on a strap-on and steps behind her",
  "the woman with the strap-on fucks her doggystyle from behind, the dildo sliding in and out, explicit hardcore",[]),
 ("страпон миссионер","she lies back and spreads her legs, the other woman with a strap-on moves between them",
  "the woman fucks her with the strap-on in missionary, thrusting deep, explicit",[]),
 ("трибадизм","they intertwine their legs together, pressing close",
  "the two women scissor and grind their pussies together, tribadism, explicit",["wan_lick"]),
 ("69","they lie head to toe on top of each other",
  "the two women lick each other in the 69 position, both eating pussy at once, explicit",["wan_lick"]),
 ("пальцы+поцелуй","one lies back, the other leans over her kissing her",
  "she fingers her pussy while kissing her passionately, explicit",["wan_finger_lick","wan_kiss"]),
 ("двойное дилдо","the two women sit facing each other sharing a double-ended dildo",
  "the two women fuck themselves on a double-ended dildo between them, grinding toward each other, explicit hardcore",[]),
 ("сидит на лице","one woman lies back, the other lowers her pussy onto her face",
  "one woman sits on the other's face and rides her tongue, facesitting, while reaching back to rub her pussy, explicit",[]),
]
TIGER=("(male anthro tiger:1.5), (tiger head:1.45), (orange fur with black stripes:1.4), (feline face:1.35), "
       "(furry:1.2), huge muscular male body")
CLIMAX_ANTHRO=[
 ("тигр раком","she bends forward, the tiger steps up close behind her",
  "the tiger fucks her doggystyle from behind, his feline penis penetrating her, explicit",["wan_furry_enh","wan_doggy"]),
 ("тигр ковгерл","the tiger lies back and she straddles him",
  "she rides the tiger's feline penis in cowgirl, bouncing, explicit",["wan_furry_enh","wan_cowgirl"]),
 ("тигр минет","she kneels down in front of the tiger",
  "she sucks the tiger's feline penis, bobbing her head, explicit",["wan_furry_enh","wan_chasing_bj"]),
 ("тигр миссионер","she lies back and spreads her legs, the tiger moves over her",
  "the tiger fucks her in missionary, his feline penis penetrating her, explicit",["wan_furry_enh","wan_missionary"]),
]
# SOLO — kept small on purpose (solo % is now low). No bounce/finger-lick lora; prose-driven, and the
# dildo scenes are ACTIVE and cock-oriented (she treats the dildo like a real cock) per user note.
CLIMAX_SOLO=[
 ("мастурбация","she lies back and spreads her legs wide open",
  "she rubs and fingers her pussy hard and caresses her breasts, masturbating, back arching, moaning",[]),
 ("дилдо верхом","she straddles a big realistic dildo suction-cupped to the floor",
  "she rides the thick realistic cock-shaped dildo like a real cock, sinking down on it and grinding, moaning",[]),
 ("сосёт дилдо","she kneels holding a big realistic dildo to her lips",
  "she sucks and deepthroats the realistic cock-shaped dildo like a blowjob, drool on her chin, then licks it, explicit",[]),
 ("дилдо в пизде","she lies back and pushes a big dildo into herself",
  "she fucks herself hard with the thick dildo, sliding it deep in and out of her pussy, rubbing her clit, moaning",[]),
 ("дилдо в жопе","she gets on all fours and reaches back with a dildo",
  "she pushes a dildo into her ass and works it in and out, anal toy play, explicit",[]),
]

ANTI_CLONE=("2girls, 3girls, multiple girls, twins, sisters, clone, duplicate person, identical women, "
            "same face twice, man with breasts, feminine male, futanari, two women instead of a couple, "
            "girl where the man should be")
NEG_WOMAN="3girls, 4girls, extra woman, penis, three characters"
NEG_SOLO="1boy, man, penis, 2girls, another person, duo, couple"
NEG_MMF=("3girls, 2girls, extra woman, twins, identical men, cloned man, same face, man with breasts, feminine male, futanari, "
         "two men kissing, men kissing each other, gay, homosexual, male-male contact, men touching each other, men holding hands")
# still templates per partner type (both present, in {setting}, wearing {outfit})
STILL_MAN=("1girl, 1man, (heterosexual couple:1.2), a woman and a clothed man together "
           "in {setting}, the woman is {subj} wearing {outfit}, "
           "((the man is {man}:1.3)), (clearly an adult male, no breasts:1.25)")
STILL_MMF=("1girl, 2boys, (mmf threesome, one woman between two men:1.2), a woman flanked by two "
           "DIFFERENT clothed men in {setting}, the woman is {subj} wearing {outfit}, "
           "((one man is {man}:1.2)), ((the other man is {man2}:1.2)), "
           "(two distinct adult males, clearly male, no breasts:1.2), "
           "(the two men do not touch or kiss each other, both face the woman:1.2)")
STILL_WOMAN="2girls, two women together in {setting}, one is {subj}, wearing {outfit}, the other a nude woman"
STILL_SOLO="solo, 1girl, a woman alone in {setting}, {subj}, wearing {outfit}"
STILL_ANTHRO=("duo, interspecies, human on anthro, "+TIGER+", standing next to a nude human woman in {setting}, "
              "(human woman:1.3), (human skin:1.3), (human face:1.3), {subj}, wearing {outfit}")
def config_for(plan):
    if plan["people"]==1: return CLIMAX_SOLO,"solo",NEG_SOLO,STILL_SOLO
    if plan["partner"]=="anthro": return CLIMAX_ANTHRO,"pony","",STILL_ANTHRO
    if plan["partner"]=="woman":  return CLIMAX_WOMAN,"mopmix",NEG_WOMAN,STILL_WOMAN
    if plan["partner"]=="mmf":    return CLIMAX_MMF,"mopmix",NEG_MMF,STILL_MMF
    return CLIMAX_MAN,"mopmix",ANTI_CLONE,STILL_MAN

# ======================= COUPLE / MMF BEAT AUTHORING =======================
# Fixes (user field-report): (1) the man was only in the STILL, never in the beats → identity drifted
# and a fresh man appeared by beat-3; now his description is carried into EVERY beat. (2) beat-0 was
# solo ("she alone with props") → in a couple the man was dropped/idle; now beat-0 is the couple
# together (walk + grope + kiss neck/cheek, she laughs) like the user's best hand-written clips.
# (3) two men used to kiss each other / one idled → NO_MM forbids M/M contact and both are placed busy.
# (4) wan_scene_change kept re-dressing her → dropped from entry, undress is now prose off the SAME outfit.
NO_MM=" The two men never kiss or touch each other, no contact between the men, they both focus only on the woman."
DOG=", he is walking his dog on a leash beside them, but the dog is not just trotting. The dog is fully erect, aggressively sniffing her wet cunt from behind, his thick red knot pressing hard against her ass, his heavy balls slapping her thighs. The bitch bends over the wooden fence, her skirt hiked up to her waist, and the dog's massive cock buries deep inside her tight, soaking pussy, stretching her walls to the limit. He grips her hips with his paws and starts violently pumping into her with a brutal, animalistic rhythm, his knot locking inside her as she screams and moans, arching her back. The dog pounds her relentlessly until his hot, thick semen floods her aching hole and spills down her legs, while the owner just smiles and watches."
# CONTINUITY + LIFE: Once the dog is in beat-0 it must NOT vanish (last-frame chain carries it, but the
# beat prompt must keep it or WAN drops/morphs it). A live pet reacts to its owner's commotion — wagging,
# jumping up at its OWNER, licking his hand/face, barking. Every reaction is OWNER-directed, and the dog
# is fully engaged in the act, dominating the female. The dog "knows its place" as the wild animal in the
# scene, actively and violently fucking the woman while the owner proudly watches his pet work.
DOG_REACT=[
 "the horny dog mounts the woman from behind, thrusting its thick knot deep into her tight pussy, tail wagging wildly with each savage pump",
 "the eager stud dog pins her down, sliding his heavy, pulsing cock in and out of her dripping hole, yanking her hips back against him",
 "the perverted dog licks her soaked cunt clean, then shoves his huge tongue inside her asshole before driving his dick into her wet mouth",
 "the dog aggressively fucks her doggy style while she screams, his massive balls slapping loudly against her thighs, cumming deep inside her",
 "the animal pushes her onto all fours, ramming his thick shaft into her tight virgin hole, knotting her and filling her belly with hot semen",
 "the dog growls with raw lust, tearing off her panties and burying his long red cock deep inside her pulsating cunt, making her beg for more",
 "the brutal beast slams her against the wall, lifting her leg and pounding her so hard she can't breathe, spilling his hot load all over her face"
]

DOG_GUARD = ("The dog is a rampant, dominant stud. The dog actively mounts the woman from behind, "
             "penetrating her deeply with its massive, thick, veined member. The dog fucks her hard, "
             "rhythmically, stretching her tight pussy until she is full of hot, heavy canine seed. "
             "The dog takes complete possession of her body, and the woman moans in pleasure as the dog "
             "fills her up:1.5")

def dog_clause(rng):
    """One owner-directed, emotionally raw dog reaction for the ultimate humiliation scene (acts 2 & 3)."""
    return ", "+rng.choice(DOG_REACT)+DOG_GUARD
OUTDOOR_WALK={"a city street sidewalk","a public park at dusk","a beach at sunset","a yacht deck",
 "a rooftop terrace at sunset","behind a row of garages","by the dumpsters in a back yard",
 "a run-down industrial yard","a cluttered village courtyard","a construction site with scaffolding and concrete"}
def _short_man(m):
    """Lighter identity reminder for in-beat reuse: drop the (…:1.3) prompt weighting, keep the description."""
    return re.sub(r"[()]","",re.sub(r":1\.\d+","",m)).strip()
def _bare(o):
    """Drop a leading article so 'strips off her {outfit}' doesn't read 'her a dress'."""
    return re.sub(r"^(a|an|the)\s+","",o.strip(),flags=re.I)
# ---- beat-0 opener pool (fixes "beat-0 всегда эмоции / всегда walk+grope") -----------------------
# Each entry: (template, [loras]).  Placeholders: {m1}/{m2}=man desc, {loc}=location, {o}=bare outfit,
# {d}=dog-on-leash suffix (only present in WALK templates, so the dog only appears where it makes
# sense).  Loras vary per opener (emotion / kiss / lick / none) so the first chunk is not always the
# emotion lora.  All openers keep BOTH people present and dressed → identity + outfit are established.
MAN_OPENERS=[
 ("she walks together with the man {loc}, he is {m1}, arm around her, groping her ass and breasts and kissing her neck while she laughs, both still dressed, she wears {o}{d}",["wan_emotion"]),
 ("they arrive {loc} and he pushes her back against the wall, kissing her deeply, hands roaming over her {o}, the man is {m1}",["wan_kiss"]),
 ("she sits on his lap and grinds slowly against him while they kiss, the man is {m1}, both still dressed, she wears {o}",["wan_kiss"]),
 ("he stands close behind her kissing her neck, his hands sliding over her {o} and squeezing her breasts, the man is {m1}",["wan_lick"]),
 ("she unbuttons his shirt and kisses down his chest, the man is {m1}, she is still in her {o}",["wan_kiss"]),
 ("they make out on the couch {loc}, his hands slipping under her {o}, the man is {m1}",["wan_kiss"]),
 ("he gropes her ass and breasts from behind while she leans forward laughing, both dressed, the man is {m1}, she wears {o}",["wan_emotion"]),
 ("she pulls him in by his collar and kisses him hard, pressing her body against his, the man is {m1}, she wears {o}",["wan_kiss"]),
 ("he lifts her onto a surface and steps between her knees, kissing her, the man is {m1}, she wears {o}",["wan_kiss"]),
 ("she kneels in front of him and slowly unzips his trousers looking up at him, freeing his stiffening cock, the man is {m1}",[]),
 ("he slides his hand up her thigh under her {o} while kissing her, the man is {m1}",["wan_lick"]),
 ("she straddles him on a chair facing him and rocks her hips against him, the man is {m1}, she wears {o}",["wan_emotion"]),
 ("he spanks her ass and pulls her against him, both grinning, still dressed, the man is {m1}, she wears {o}",["wan_emotion"]),
 ("she leans back against his chest, his hands cupping her breasts while he kisses her neck, the man is {m1}, she wears {o}",["wan_lick"]),
 ("he bends her gently forward over a surface, kissing her spine, hands gripping her hips, the man is {m1}, she wears {o}",["wan_lick"]),
 ("she palms his hardening cock through his trousers while kissing him, the man is {m1}, she wears {o}",[]),
 ("he carries her in and drops her onto the bed, climbing over her and kissing her, the man is {m1}, she wears {o}",["wan_emotion"]),
 ("she bites her lip and guides his hand between her legs while he kisses her throat, the man is {m1}, she wears {o}",["wan_lick"]),
 ("he kisses along her collarbone while peeling her {o} strap off her shoulder, the man is {m1}",["wan_lick"]),
 ("she pushes him into a chair and climbs onto his lap, kissing him, the man is {m1}, she wears {o}",["wan_emotion"]),
 ("he pins her wrists above her head and kisses her throat, pressing against her, the man is {m1}, she wears {o}",["wan_kiss"]),
 ("she rubs her ass back against his crotch, grinding on him while he gropes her, the man is {m1}, she wears {o}",["wan_emotion"]),
 ("they walk in together and he peels her {o} down kissing her bare shoulder, the man is {m1}{d}",["wan_lick"]),
 ("he grips her jaw and kisses her deep, his other hand squeezing her ass, the man is {m1}, she wears {o}",["wan_kiss"]),
 ("she runs her hands over his chest as he squeezes her breasts through her {o}, the man is {m1}",["wan_emotion"]),
 ("he kneels and kisses up her inner thigh as she sits on the edge, the man is {m1}, she wears {o}",["wan_lick"]),
 ("walking arm in arm {loc} he keeps groping under her {o} and she giggles, the man is {m1}{d}",["wan_emotion"]),
 ("she strokes his freed cock with one hand while kissing him, the man is {m1}, she wears {o}",[]),
]
MMF_OPENERS=[
 ("she walks between two men {loc}, on her right {m1} and on her left {m2}, both groping her ass and breasts and kissing her neck while she laughs, everyone dressed, she wears {o}{d}",["wan_emotion"]),
 ("she sits between the two men on a couch, a hand of each roaming her body, {m1} on one side and {m2} on the other, she wears {o}",["wan_emotion"]),
 ("one man kisses her mouth while the other kisses her neck from behind, {m1} and {m2}, she wears {o}",["wan_kiss"]),
 ("both men peel her {o} off her shoulders kissing her skin, {m1} and {m2}",["wan_lick"]),
 ("she strokes both men through their trousers, {m1} on one side and {m2} on the other, she wears {o}",[]),
 ("one man grinds against her ass from behind while the other kisses her in front, {m1} and {m2}, she wears {o}",["wan_emotion"]),
 ("she kneels between the two standing men and unzips both of them, freeing their cocks, {m1} and {m2}",[]),
 ("one squeezes her breasts from behind while she kisses the other, {m1} and {m2}, she wears {o}",["wan_lick"]),
 ("both men grope her ass and thighs as she arches between them, {m1} and {m2}, she wears {o}",["wan_emotion"]),
 ("she turns her head to kiss one then the other, a hand on each, {m1} and {m2}, she wears {o}",["wan_kiss"]),
 ("one man lifts her {o} while the other kisses down her neck, {m1} and {m2}",["wan_lick"]),
 ("she rubs her ass back on one man while palming the other's cock, {m1} and {m2}, she wears {o}",[]),
 ("the two men sandwich her with hands everywhere and she laughs, {m1} and {m2}, she wears {o}",["wan_emotion"]),
 ("one kisses her deeply while the other kneels kissing her thigh, {m1} and {m2}, she wears {o}",["wan_lick"]),
 ("she sits on one man's lap grinding while the other kisses her, {m1} and {m2}, she wears {o}",["wan_emotion"]),
 ("she licks and kisses one man's neck while the other gropes her from behind, {m1} and {m2}, she wears {o}",["wan_lick"]),
 ("they bend her forward between them, {m1} at her front and {m2} at her back, she wears {o}",["wan_emotion"]),
 ("she takes a cock in each hand as the two men stand on either side of her, {m1} and {m2}",[]),
]
# LOCATION-SPECIFIC couple setups (user: "надо историю завязывать ... на каждую локацию по парочке").
# ~2 narrative openers per SCENES location: she is doing something that fits the place, the man
# watches / approaches, ending in first contact (so beat-0 still establishes couple + outfit + identity).
# {m1}=man desc, {o}=bare outfit, {d}=dog-on-leash suffix (only in walk setups). NO {loc} placeholder —
# the location is written into the prose, so a setup can never land in the wrong place ("кухня на пляже").
CX_LOC_OPENER_PROB=float(os.getenv("CX_LOC_OPENER_PROB","0.7"))
COUPLE_SETUP={
 "a candle-lit bedroom":[
  "she sits at the vanity brushing her hair in her {o}, unaware, as the man watches from the doorway, then crosses the candle-lit bedroom and slides his hands over her shoulders, the man is {m1}",
  "she lies reading on the bed in her {o} when the man walks in, sets down his glass and climbs over her kissing her, the man is {m1}"],
 "a luxury hotel suite":[
  "she stands at the suite window admiring the view in her {o}, the man steps out of the bathroom, watches a moment, then comes up behind her and wraps his arms around her, the man is {m1}",
  "she unpacks on the hotel bed in her {o} when the man pulls her back onto the sheets kissing her neck, the man is {m1}"],
 "a modern glass office":[
  "she leans over the desk sorting papers in her {o}, the man watches through the glass wall, then walks in, lowers the blinds and presses against her from behind, the man is {m1}",
  "she sits in the office chair in her {o} when the man leans over, spins the chair to face him and kisses her, the man is {m1}"],
 "a stylish living room":[
  "she curls on the sofa scrolling her phone in her {o}, the man watches from the hallway, then sits beside her and slides a hand up her thigh, the man is {m1}",
  "she sways to music alone in the living room in her {o}, the man catches her from behind with his hands on her hips, the man is {m1}"],
 "a marble bathroom":[
  "she does her makeup at the marble sink in her {o}, the man leans in the doorway watching, then steps in and kisses her bare shoulder, the man is {m1}",
  "she steps out of the shower reaching for a towel, the man is waiting, takes the towel away and presses her to the wall, the man is {m1}"],
 "a penthouse with a city view":[
  "she stands at the floor-to-ceiling glass with a drink in her {o}, the man crosses the penthouse behind her and slides his arms around her waist, the man is {m1}",
  "she reclines on the designer sofa in her {o}, the man sits close, trailing a hand along her leg, the man is {m1}"],
 "beside a swimming pool":[
  "she suns herself on a lounger by the pool in her {o}, the man walks over dripping, leans down and kisses her, the man is {m1}",
  "she dangles her legs in the pool in her {o}, the man wades up between her knees and pulls her close, the man is {m1}"],
 "a sunlit kitchen":[
  "she cooks at the stove in the sunlit kitchen in her {o}, humming, as the man leans in the doorway watching, then crosses over and wraps his arms around her kissing her neck, the man is {m1}",
  "she reaches into a high cupboard in the kitchen in her {o}, the man steps up behind her, hands settling on her hips, the man is {m1}"],
 "a dim neon-lit nightclub":[
  "she dances in the neon-lit club crowd in her {o}, the man watches from the bar, then moves in behind her grinding to the beat, the man is {m1}",
  "she sips a cocktail in a club booth in her {o} when the man slides in beside her, a hand on her thigh, the man is {m1}"],
 "a car back seat":[
  "she waits in the back seat of the parked car in her {o}, the man climbs in, shuts the door and pulls her onto his lap, the man is {m1}",
  "she stretches across the back seat in her {o}, the man leans over her kissing her as the windows fog, the man is {m1}"],
 "a city street sidewalk":[
  "she walks the city sidewalk in her {o} when the man falls into step beside her, slips an arm around her waist and squeezes her ass, the man is {m1}{d}",
  "she waits at the corner in her {o}, the man walks up smiling, pulls her into a doorway and kisses her, the man is {m1}{d}"],
 "a public park at dusk":[
  "she strolls the dusk park in her {o} when the man on a bench smiles, gets up and falls into step, slipping a hand onto her hip, the man is {m1}{d}",
  "she sits on the park bench in her {o}, the man sits close, an arm around her, kissing her cheek as she laughs, the man is {m1}{d}"],
 "a rooftop terrace at sunset":[
  "she leans on the rooftop railing watching the sunset in her {o}, the man approaches from behind and wraps his arms around her, the man is {m1}{d}",
  "she lounges on the terrace sofa in her {o}, the man sits beside her, trailing fingers up her thigh, the man is {m1}"],
 "an elevator":[
  "she rides the elevator alone in her {o} when the man steps in, the doors close, and he presses her to the mirrored wall, the man is {m1}",
  "she checks her phone in the elevator in her {o}, the man leans in, hits the stop button and kisses her, the man is {m1}"],
 "a locker room":[
  "she changes by the lockers in her {o}, the man watches from the row of benches, then walks over and gropes her from behind, the man is {m1}",
  "she ties her shoe on the locker-room bench in her {o}, the man steps up between her knees, the man is {m1}"],
 "a massage room":[
  "she lies on the massage table in her {o}, the man's hands slide from her back down over her ass, leaning in to her neck, the man is {m1}",
  "she waits on the massage table in her {o}, the man oils his hands, then works them under her towel, the man is {m1}"],
 "an empty classroom":[
  "she erases the board in the empty classroom in her {o}, the man watches from a desk, then comes up behind her, the man is {m1}",
  "she grades papers at the desk in her {o}, the man leans over her, closing the folder and kissing her, the man is {m1}"],
 "a library aisle":[
  "she reaches for a book high on the shelf in her {o}, the man watches from the next aisle, then steps close behind her, the man is {m1}",
  "she reads against the stacks in her {o}, the man crowds her quietly and kisses her, the man is {m1}"],
 "a doctor's office":[
  "she waits on the exam table in her {o}, the man in a white coat steps in, sets down the chart and runs a hand up her thigh, the man is {m1}",
  "she sits in the doctor's office in her {o}, the man leans in close under the pretext of an exam, the man is {m1}"],
 "a gym":[
  "she stretches on the gym mat in her {o}, the man racks his weights, walks over and settles his hands on her hips, the man is {m1}",
  "she does squats by the mirror in her {o}, the man spots her from behind, hands sliding to her waist, the man is {m1}"],
 "a steamy sauna":[
  "she sits wrapped in a towel in the steamy sauna in her {o}, the man ladles water, then slides down the bench against her, the man is {m1}",
  "she leans back in the sauna heat in her {o}, the man's hand finds her thigh through the steam, the man is {m1}"],
 "a beach at sunset":[
  "she walks the shoreline at sunset in her {o} when the man scoops a hand over her ass and kisses her shoulder, she laughs, the man is {m1}{d}",
  "she lies on a beach towel at sunset in her {o}, the man kneels beside her, leaning down to kiss her, the man is {m1}{d}"],
 "a yacht deck":[
  "she stands at the yacht rail in the wind in her {o}, the man comes up behind her, hands circling her waist, the man is {m1}{d}",
  "she sunbathes on the yacht deck in her {o}, the man crouches over her with a grin, the man is {m1}"],
 "a restaurant bathroom stall":[
  "she touches up her lipstick in the restaurant restroom in her {o}, the man slips in, pulls her into a stall and locks it, the man is {m1}",
  "she waits in the stall in her {o}, the man crowds in behind her, the man is {m1}"],
 "a garage":[
  "she leans on the car hood in the garage in her {o}, the man wipes his hands, walks over and pins her to the car, the man is {m1}{d}",
  "she looks for something on the garage shelves in her {o}, the man steps up behind her, the man is {m1}"],
 "a hotel balcony":[
  "she leans on the hotel balcony rail in her {o}, the man steps out behind her, arms around her waist, kissing her neck, the man is {m1}",
  "she sips wine on the balcony in her {o}, the man draws her back against him, the man is {m1}"],
 "a photo studio":[
  "she poses under the studio lights in her {o}, the man lowers the camera, walks over and guides her hips closer, the man is {m1}",
  "she waits on the studio backdrop in her {o}, the man steps in, adjusting her pose with roaming hands, the man is {m1}"],
 "a fitting room":[
  "she tries on clothes in the fitting-room mirror in her {o}, the man slips through the curtain behind her, hands on her waist, the man is {m1}",
  "she zips a dress in the fitting room in her {o}, the man steps in, unzips it again kissing her shoulder, the man is {m1}"],
 "a wine cellar":[
  "she browses the racks in the wine cellar in her {o}, the man follows her down, corners her against the bottles, the man is {m1}",
  "she pours a taste in the cellar in her {o}, the man sets down his glass and pulls her in, the man is {m1}"],
 "a private jet cabin":[
  "she reclines in the jet cabin seat in her {o}, the man leans over from the aisle, a hand on her thigh, the man is {m1}",
  "she stands to stow her bag in the jet in her {o}, the man draws her into his lap, the man is {m1}"],
 "a strip club stage":[
  "she works the pole on the club stage in her {o}, the man watches from the front row, then climbs up and pulls her down to him, the man is {m1}",
  "she struts the stage in her {o}, the man steps up, hands sliding to her hips, the man is {m1}"],
 "a bar counter":[
  "she sits alone at the bar in her {o}, the man slides onto the next stool, leans in close and rests a hand on her knee, the man is {m1}",
  "she leans on the bar counter in her {o}, the man steps up behind her, an arm around her waist, the man is {m1}"],
 "a moonlit garden":[
  "she wanders the moonlit garden in her {o}, the man steps from the shadows, drawing her close among the flowers, the man is {m1}{d}",
  "she sits on the garden bench in the moonlight in her {o}, the man joins her, a hand on her thigh, the man is {m1}{d}"],
 "a forest clearing":[
  "she walks into the sunlit forest clearing in her {o}, the man catches up, pressing her back to a tree, the man is {m1}{d}",
  "she rests on a fallen log in the clearing in her {o}, the man kneels before her parting her knees, the man is {m1}"],
 "a rainy back alley":[
  "she shelters from the rain in the alley in her {o}, the man steps close, backing her to the wet wall, the man is {m1}",
  "she hurries down the rainy alley in her {o} when the man pulls her into a doorway kissing her hard, the man is {m1}"],
 "a jacuzzi":[
  "she soaks in the bubbling jacuzzi in her {o}, the man slides in behind her, hands gliding over her under the water, the man is {m1}",
  "she rests her arms on the jacuzzi edge in her {o}, the man wades up between her legs, the man is {m1}"],
 "a home office":[
  "she works late at the home-office desk in her {o}, the man leans in the doorway watching, then comes over and turns her chair to him, the man is {m1}",
  "she files papers by the shelves in the home office in her {o}, the man steps up behind her, hands on her hips, the man is {m1}"],
 "a laundry room":[
  "she loads the machine in the laundry room in her {o}, the man watches from the door, then presses her over the warm dryer, the man is {m1}",
  "she folds clothes on the counter in her {o}, the man comes up behind her, hands roaming, the man is {m1}"],
 "a rooftop pool":[
  "she floats at the edge of the rooftop pool in her {o}, the man lowers in beside her, drawing her close, the man is {m1}{d}",
  "she suns on the rooftop-pool deck in her {o}, the man kneels over her lounger, the man is {m1}"],
 "a dark movie theater":[
  "she watches the screen in the dark theater in her {o}, the man slides an arm around her, a hand creeping up her thigh, the man is {m1}",
  "she settles into the theater seat in her {o}, the man leans over from the next seat kissing her, the man is {m1}"],
 "a construction site with scaffolding and concrete":[
  "she picks her way through the construction site in her {o} when the dirty labourer blocks her path, backing her against the concrete, the man is {m1}{d}",
  "she waits by the scaffolding in her {o}, the grimy worker sets down his tools and crowds her to the wall, the man is {m1}{d}"],
 "an abandoned ruined building":[
  "she explores the ruined building in her {o} when the vagrant steps from a doorway, cornering her against the crumbling wall, the man is {m1}{d}",
  "she picks through the rubble in her {o}, the derelict grabs her from behind in the gloom, the man is {m1}"],
 "behind a row of garages":[
  "she cuts through behind the garages in her {o} when the grimy man steps out and backs her to the wall, the man is {m1}{d}",
  "she waits behind the garages in her {o}, the dirty man crowds up against her, the man is {m1}{d}"],
 "by the dumpsters in a back yard":[
  "she hurries past the dumpsters in the back yard in her {o} when the homeless man corners her against the bins, the man is {m1}{d}",
  "she lingers by the dumpsters in her {o}, the vagrant shuffles up and pins her to the wall, the man is {m1}{d}"],
 "a run-down industrial yard":[
  "she crosses the industrial yard in her {o} when the grimy worker steps from behind a container, backing her up, the man is {m1}{d}",
  "she waits in the run-down yard in her {o}, the dirty man corners her against the rusted metal, the man is {m1}{d}"],
 "a cramped basement boiler room":[
  "she edges into the cramped boiler room in her {o} when the grimy man blocks the door and crowds her to the pipes, the man is {m1}",
  "she is cornered in the boiler room in her {o}, the dirty man presses her against the warm pipes, the man is {m1}"],
 "a cluttered village courtyard":[
  "she crosses the cluttered village courtyard in her {o} when the old man sets down his tools and draws her behind the shed, the man is {m1}{d}",
  "she draws water in the village courtyard in her {o}, the weathered man comes up behind her, hands on her hips, the man is {m1}{d}"],
}
def couple_entry(partner,setting,outfit,m1,m2,rng,dog=False):
    """beat-0 for a couple: both present and busy from the first frame (establishes identity + outfit).
    Draws a VARIED opener from the pool (not always the same walk+grope+emotion). Returns (text,loras).
    dog=True adds the NON-SEXUAL dog-on-a-leash prop, but only to WALK openers (which contain {d})."""
    loc=f"in {setting}" if setting.split()[0] in ("a","an") else setting   # avoid "in by the dumpsters"
    d=DOG if dog else ""
    o=_bare(outfit); s1=_short_man(m1); s2=_short_man(m2)
    if partner=="mmf":
        tpl,loras=rng.choice(MMF_OPENERS)
        return tpl.format(m1=s1,m2=s2,loc=loc,o=o,d=d)+NO_MM, list(loras)
    # man: prefer a LOCATION-SPECIFIC narrative setup (~2 per location) when this setting has them
    setups=COUPLE_SETUP.get(setting)
    if setups and rng.random()<CX_LOC_OPENER_PROB:
        t=rng.choice(setups).format(m1=s1,o=o,d=d); tl=t.lower()
        lora=(["wan_kiss"] if "kiss" in tl else
              ["wan_lick"] if any(k in tl for k in ("neck","shoulder","thigh")) else ["wan_emotion"])
        return t, lora
    tpl,loras=rng.choice(MAN_OPENERS)
    return tpl.format(m1=s1,loc=loc,o=o,d=d), list(loras)

# ---- still composition pool (fixes "все на паспорт, анфас") --------------------------------------
# Varied couple poses/angles/positions for the STARTING photo instead of a frontal standing portrait.
# All keep HER face toward the camera so facelock (ReActor) can still detect and stamp it.
MAN_STILL_POSE=[
 "he stands close behind her with his hands on her hips, she looks back over her shoulder toward the camera",
 "she leans back against his chest, his arms wrapped around her waist, both looking at the camera",
 "she sits sideways on his lap with one arm around his neck, glancing at the camera",
 "they stand pressed together mid-kiss, bodies turned three-quarters to the camera",
 "she is bent slightly forward with her hands on a surface, he stands behind gripping her hips, she turns her face to the camera",
 "she straddles his lap as he sits, facing him, glancing back at the camera",
 "he lifts her against the wall with her legs around his waist, her face toward the camera",
 "low camera angle looking up at the couple standing close together",
 "candid over-the-shoulder framing as they embrace, her face visible turning to the camera",
 "she kneels in front of him looking up, his hand resting in her hair",
 "she sits on the edge of a surface with her knees apart and he stands between them, her face to the camera",
 "he cups her from behind, one hand on her breast and one on her hip, her head tilted back toward the camera",
 "wide full-body shot of the two of them standing close with hands roaming",
 "she has her back to his front, his hands sliding over her body, dynamic three-quarter angle, her face to the camera",
 "intimate close framing of the two cheek to cheek, her eyes on the camera",
]
MMF_STILL_POSE=[
 "she stands between the two men, one behind with hands on her hips and one in front, her face to the camera",
 "she sits between the two men on a surface, a hand of each on her thighs, looking at the camera",
 "one man kisses her neck from behind while the other stands close in front, her face turned to the camera",
 "both men flank her with hands roaming her body, she looks at the camera between them, three-quarter angle",
 "she kneels between the two standing men looking up, a hand of each in her hair",
 "low angle looking up at the woman flanked by two men",
 "she leans back against one man's chest while the other leans in, her face to the camera",
 "wide full-body shot of the woman between the two men, everyone turned toward the camera",
 "candid three-quarter framing of the trio close together, her face clearly visible",
]
def with_identity(text,partner,m1,m2):
    """Carry the concrete man(men) into a transition/action beat so the chain keeps the SAME man."""
    if partner=="mmf": return f"{text}, (one man is {_short_man(m1)}, the other man is {_short_man(m2)}:1.15), (both men are actively engaged with her at the same moment, neither man is idle, standing aside or merely watching:1.2),"+NO_MM
    if partner=="man": return f"{text}, (the man is {_short_man(m1)}:1.2)"
    return text

# TEASE probability (user: "секрет" — одета, но видна эротика; "слишком много просто голых тел").
CX_TEASE_PROB=float(os.getenv("CX_TEASE_PROB","0.55"))
# Clothes-stay-ON reveals: bra peek / breast slipped out / panties shifted or split-crotch. Keeps the
# outfit fully on and only exposes the essential parts, instead of undressing.
CLOTHED_TEASE=[
 "she keeps her {bare} fully on, only tugging her panties aside to bare her pussy",
 "her {bare} stays on, one bra cup pushed down to bare a breast and her panties pulled to the side",
 "still fully dressed in her {bare}, only the crotch opened for access, the eroticism peeking through the clothing",
 "her {bare} remains on, skirt hiked up and panties shifted aside, one breast slipped out of the bra",
 "the clothing stays on, only pulled aside where needed — a glimpse of bra, a bare breast, panties tugged off her pussy",
 "her {bare} stays on with split-crotch access, everything covered except where their bodies join",
]
def wardrobe_transition(outfit, setup, rng, partner="man", action=""):
    """Connect the opening to the requested pose without forcing every clip through the
    same dressed -> fully nude transformation.

    WAN tends to turn a repeated sentence into a repeated shot, even when the location and
    final action differ.  Keep the story-specific pose setup as the important part and vary
    how much clothing changes.  Some actions work better with clothing merely moved aside,
    and oral scenes do not require the woman to undress at all.
    """
    bare=_bare(outfit)
    s=f"{setup} {action}".lower()
    oral=any(x in s for x in ("suck", "blowjob", "deepthroat", "cock in each hand", "stroking"))
    # TEASE: keep the full outfit ON and only expose the essential parts (skip for solo, where there is
    # no partner to work around, and for oral where the man-open phrasing already keeps her dressed).
    if partner in ("man","mmf","woman") and not oral and rng.random()<CX_TEASE_PROB:
        return f"{rng.choice(CLOTHED_TEASE).format(bare=bare)}, {setup}"
    if partner=="none":
        lead=rng.choice([
            f"she pulls her {bare} aside and changes pose",
            f"her {bare} remains partly on while she moves into position",
            "without a separate undressing scene, she moves directly into position",
        ])
    elif partner=="woman":
        lead=rng.choice([
            f"the women pull her {bare} aside only where needed and change pose together",
            f"her {bare} remains half-open as the women move directly into position",
            "without a separate undressing scene, the women change position together",
        ])
    elif oral:
        lead=rng.choice([
            "the man opens his trousers and she keeps her outfit on as she moves into position",
            "she stays dressed while he exposes his erection and she moves close",
            "without changing her clothes, she lowers herself toward his exposed cock",
        ])
    else:
        lead=rng.choice([
            f"they passionately pull the {bare} aside only where needed and move into position",
            f"her {bare} becomes half-open and rumpled as they change pose together",
            "without a separate undressing scene, they move directly into the next pose",
            f"he lifts and opens her {bare} while she turns into position",
            "the camera shifts angle as they quickly change position, clothing partly remaining on",
        ])
    if partner=="mmf":
        lead=lead.replace("the man ","both men ").replace(" his "," their ")
    return f"{lead}, {setup}"

# ---- reaction / affection layer (fixes "они как статуи, механически двигаются") ----------------
# The action text says WHAT position they're in; on its own WAN animates two stiff bodies pumping.
# These add the human layer the user asked for: hands grabbing/holding, bodies embracing, and HER
# facial reactions (eyes closing, mouth open, moaning, smiling at the camera, enjoying it). Additive
# and pose-agnostic, so they layer onto any climax without fighting it.
REACT_EMBRACE={
 "man":[
  "his hands grip and squeeze her ass",
  "his hands roam over her body and grip her hips hard",
  "he pulls her tight against him, holding her close",
  "she wraps her arms around him and pulls him closer",
  "he grabs a fistful of her hair and holds her",
  "he squeezes her breasts from behind",
  "her hands claw and grip his back and shoulders",
  "he holds her hips and pulls her onto him",
  "they hold each other close, hands roaming everywhere",
  "he grabs her ass with both hands and spreads it",
 ],
 "mmf":[
  "one man grips her ass while the other holds her by the hair",
  "the hands of both men roam and squeeze her body",
  "she grabs one man as the other pulls her hips onto him",
  "both men hold her close, hands all over her",
  "one squeezes her breasts while the other grips her hips",
  "she clutches at one man while the other holds her thighs apart",
 ],
 "woman":[
  "they pull each other close, hands roaming and squeezing",
  "she grips the other woman's hips and pulls her in",
  "their hands roam and grope over each other's bodies",
  "they hold each other tight, caressing and squeezing",
 ],
 "none":[
  "her free hand squeezes and kneads her own breast",
  "her hands grip and roam over her own body",
  "one hand grips the sheets tight while she works herself",
  "she caresses her own breasts and grips her thigh",
 ],
}
REACT_FACE=[
 "her eyes flutter and close in pleasure",
 "her mouth falls open with a loud moan",
 "she bites her lip and gasps",
 "she smiles at the camera between moans",
 "her head tips back as she pants and moans",
 "she looks into the camera with half-closed eyes and parted lips",
 "her back arches and she cries out in pleasure",
 "she gasps, eyes rolling, lost in it",
 "a look of intense pleasure on her face as she breathes hard",
 "she moans and smiles, clearly loving it",
]
# ROUGH / grime tier only: consensual victim ROLEPLAY (user: "если моменты с грязью, то она может
# плакать или кричать или отбиваться играя роль жертвы"). Always framed as PLAY-ACTING between
# consenting adults, never real assault. Used in place of the pleasure face when `rough` is set.
REACT_VICTIM_FACE=[
 "she whimpers and squirms, tears welling in her eyes, playing the helpless victim",
 "she cries out and pushes weakly at him, play-acting at resisting while he holds her down",
 "she sobs and pleads through the roleplay, wriggling, acting the frightened captive",
 "she screams and thrashes, playing the victim taken by force, then melts into it",
 "tears run down her face as she feigns fear, whimpering while he grips her harder",
 "she gasps and fights back weakly in the roleplay, struggling, then goes limp and moans",
]
def motion_reactions(partner, rng, rough=False):
    """One embrace/grab clause + one facial-reaction clause, so the bodies act like they enjoy it.
    When `rough` (grime setting / dirty partner) the face clause becomes consensual victim ROLEPLAY."""
    embrace=rng.choice(REACT_EMBRACE.get(partner, REACT_EMBRACE["man"]))
    face=rng.choice(REACT_VICTIM_FACE) if (rough and rng.random()<0.6) else rng.choice(REACT_FACE)
    return f"{embrace}, {face}"
def add_reactions(text, partner, rng, rough=False):
    """Layer embrace + reaction onto a sex-action beat so WAN animates affection & pleasure (or, in
    grime scenes, victim roleplay), not two mechanically thrusting statues."""
    return f"{text.rstrip('. ,')}, {motion_reactions(partner, rng, rough)}"

def action_escalation(action, partner, rng, camera, rough=False):
    """Write act 3 as continuity, never as a reset to a new generic scene.

    Most variants intensify the existing action; a smaller share changes pose while keeping
    bodies in contact.  Repeating the concrete act text gives WAN a strong identity/motion
    anchor and prevents a new couple or location appearing in the final chunk.
    """
    same=rng.random()<0.70
    if same:
        development=rng.choice([
            "the same action continues faster and deeper with stronger rhythmic movement and rising intensity",
            "they keep the exact same position as the rhythm becomes harder and more urgent",
            "without stopping, the action grows more intense, bodies moving with greater force and momentum",
            "the same motion continues in a sustained close rhythm, expressions and breathing becoming more intense",
        ])
    else:
        if partner=="none":
            development="without breaking the flow, she changes to a more intense angle and continues with stronger motion"
        elif partner=="woman":
            development="without separating, the women roll into a more intense angle and continue together"
        elif partner=="mmf":
            development="without a cut, the two men reposition around the same woman and continue in a more intense arrangement"
        else:
            development="without separating or changing location, they shift into a deeper, more intense angle and continue"
    # MMF: keep BOTH men working simultaneously in the last act too (user: "один просто стоит").
    busy=" both men keep using her at the same time, neither man stops or just stands watching," if partner=="mmf" else ""
    return f"continuation of the immediately previous shot, same people, same faces, same location and clothing; {action.rstrip('. ')}, {development},{busy} {motion_reactions(partner, rng, rough)}, {camera}"
def pick_climax(climaxes,partner,rng):
    """MMF: weight toward the poses WAN renders reliably (spitroast / sequential / oral) over the
    simultaneous DP / double-vaginal / sandwich ones it often collapses. Others: uniform."""
    if partner=="mmf":
        reliable={"вертел (spitroast)","два в рот","едет и сосёт","раком + минет стоя","дрочит двоим","на спине вдвоём","двойной камшот"}
        weighted=[c for c in climaxes if c[0] in reliable]*3+climaxes
        return rng.choice(weighted)
    return rng.choice(climaxes)

# ---- location-appropriate men: grime tier gets dirty/homeless partners (user request) ----
BUM_MEN=[
 "a filthy homeless man, 55 years old, torn rags, matted greasy grey hair, dirt-smeared face, unkempt tangled beard",
 "a dirty vagrant, 50 years old, tattered stained coat, blackened grimy hands, missing teeth, wild hair",
 "a grimy drunk, 45 years old, ripped filthy clothes, red bloated unshaven face, greasy hair",
 "a homeless man, 60 years old, weathered dirt-caked skin, ragged layered clothing, broken shoes, straggly beard",
 "a scruffy street bum, 40 years old, torn grime-caked hoodie, greasy stringy hair, dirty face",
 "an unwashed derelict, 58 years old, filthy tangled beard, tattered army jacket, black fingernails",
 "a ragged tramp, 48 years old, mud-stained rags, gaunt dirty face, matted hair",
]
DIRTY_MEN=BUM_MEN+[
 "a sunburnt 42-year-old construction worker in a filthy torn tank top, grime-streaked sweaty skin",
 "a 47-year-old unshaven mechanic in oil-soaked rags with blackened greasy hands",
 "a grimy 50-year-old day-labourer in dirt-caked work clothes with calloused dirty hands",
]
GRIMY_SETTINGS={"a construction site with scaffolding and concrete","an abandoned ruined building",
 "behind a row of garages","by the dumpsters in a back yard","a run-down industrial yard",
 "a cramped basement boiler room"}
def pick_men(setting,rng):
    """Match the man's 'level' to the location: grime/derelict → dirty/homeless men, otherwise the full pool."""
    pool=DIRTY_MEN if setting in GRIMY_SETTINGS else MEN
    m1=rng.choice(pool); m2=rng.choice([x for x in pool if x!=m1] or pool)
    return m1,m2

# ---------- graphs ----------
def wait_photo(pid,dst):
    dl=time.time()+240
    while time.time()<dl:
        it=b.get_history(pid).get(pid)
        if it and it.get("outputs"):
            im=(it["outputs"].get("128",{}).get("images") or [None])[0]
            if im:
                blob=b.fetch_file(im["filename"],im.get("subfolder",""),im.get("type","output"))
                dst.write_bytes(blob); return dst
            if it.get("status",{}).get("status_str")=="error": raise RuntimeError("photo comfy error")
        time.sleep(2)
    raise TimeoutError("photo timeout")
PONY_CKPT=getattr(b,"PONY_FURRY_CHECKPOINT","ponyRealism_V23ULTRA.safetensors")
PONY_POS="score_9, score_8_up, score_7_up, source_furry, "
PONY_NEG=("score_4, score_5, score_6, censored, mosaic censorship, bar censor, low quality, worst quality, "
          "bad anatomy, deformed, mutated, extra limbs, extra digits, three characters, 3boys, extra male, "
          "extra animal, anthro female, female tiger, breasts on male, clone, duplicate, floating penis, "
          "disembodied penis, watermark, text, signature")
def pony_graph(prompt,w,h,seed):
    return {
     "1":{"class_type":"CheckpointLoaderSimple","inputs":{"ckpt_name":PONY_CKPT}},
     "2":{"class_type":"CLIPTextEncode","inputs":{"text":PONY_POS+prompt,"clip":["1",1]}},
     "3":{"class_type":"CLIPTextEncode","inputs":{"text":PONY_NEG,"clip":["1",1]}},
     "4":{"class_type":"EmptyLatentImage","inputs":{"width":int(w),"height":int(h),"batch_size":1}},
     "5":{"class_type":"KSampler","inputs":{"model":["1",0],"positive":["2",0],"negative":["3",0],
          "latent_image":["4",0],"seed":int(seed),"steps":28,"cfg":7.0,
          "sampler_name":"dpmpp_2m_sde","scheduler":"karras","denoise":1.0}},
     "6":{"class_type":"VAEDecode","inputs":{"samples":["5",0],"vae":["1",2]}},
     "128":{"class_type":"SaveImage","inputs":{"images":["6",0],"filename_prefix":"complex_furry"}},
    }
# realism / anti-"doll" SDXL loras — applied to BOTH bigASP passes so the refiner (Low ckpt 36)
# doesn't wash out the skin texture that the High pass (node 162) adds. (path, strength, trigger)
MOPMIX_REALISM=[
 ("SDXL/skin_texture_xl.safetensors",    0.55, "detailed skin texture, realistic skin pores"),
 ("SDXL/natural_breasts_xl.safetensors", 0.45, "natural breasts"),
 ("SDXL/detailed_pussy_xl.safetensors",  0.40, "detailed pussy"),
 ("SDXL/add-detail-xl.safetensors",      0.35, ""),
]
def inject_mopmix_realism(wf,loras=MOPMIX_REALISM):
    # High pass: fill the empty rgthree Power Lora Loader (node 162, model-only)
    n162=wf.get("162")
    if n162 and n162.get("class_type")=="Power Lora Loader (rgthree)":
        ins=n162["inputs"]
        ex=[int(k.split("_",1)[1]) for k in ins if k.startswith("lora_") and k.split("_",1)[1].isdigit()]
        i=(max(ex)+1) if ex else 1
        for path,s,_ in loras: ins[f"lora_{i}"]={"on":True,"lora":path,"strength":float(s)}; i+=1
    # Low/refiner pass: splice a LoraLoaderModelOnly chain onto checkpoint-36's model output
    prev=["36",0]
    for k,(path,s,_) in enumerate(loras):
        nid=f"cxlora{k}"
        wf[nid]={"class_type":"LoraLoaderModelOnly","inputs":{"model":prev,"lora_name":path,"strength_model":float(s)}}
        prev=[nid,0]
    if prev!=["36",0]:
        for node in wf.values():
            m=node.get("inputs",{}).get("model")
            if isinstance(m,list) and m==["36",0] and node.get("class_type")!="LoraLoaderModelOnly":
                node["inputs"]["model"]=list(prev)
    # trigger words into the positive prompt (node 109)
    trig=", ".join(t for _,_,t in loras if t); p109=wf.get("109",{}).get("inputs",{})
    if trig and "text" in p109: p109["text"]=f'{p109["text"]}, {trig}'
    return wf
TAN_NEG="(tan lines:1.5), bikini tan lines, bra tan lines, farmer's tan, uneven skin tone, sunburn marks"
CX_SRC_DENOISE=float(os.getenv("CX_SRC_DENOISE","0.62"))   # 🖼 "из фото": how far img2img redraws the source
async def gen_photo_mopmix(prompt,dst,neg_extra="",skip_breast_lora=False,src_image="",denoise=CX_SRC_DENOISE):
    wf=b.load_workflow(b.WORKFLOW_MOPMIX)
    if src_image:   # 🖼 img2img: start from the user's real photo so it keeps her, only restyled into the scene
        wf=b.patch_mopmix_workflow(wf,prompt=prompt,resolution=PHOTO_RES,image_name=src_image,seed=b.make_seed(),text_only=False,denoise=denoise)
    else:
        wf=b.patch_mopmix_workflow(wf,prompt=prompt,resolution=PHOTO_RES,image_name="",seed=b.make_seed(),text_only=True)
    extra=(neg_extra+", "+TAN_NEG) if neg_extra else TAN_NEG      # always suppress tan lines
    if wf.get("6",{}).get("inputs",{}).get("text") is not None:
        wf["6"]["inputs"]["text"]=f'{wf["6"]["inputs"]["text"]}, {extra}'
    loras=[l for l in MOPMIX_REALISM if not (skip_breast_lora and "natural_breasts" in l[0])]
    wf=inject_mopmix_realism(wf,loras)
    pid=await asyncio.to_thread(b.queue_prompt,wf,str(uuid.uuid4()))
    return await asyncio.to_thread(wait_photo,pid,dst)
async def gen_photo_pony(prompt,dst):
    g=pony_graph(prompt,832,1216,b.make_seed())
    pid=await asyncio.to_thread(b.queue_prompt,g,str(uuid.uuid4()))
    return await asyncio.to_thread(wait_photo,pid,dst)
def src_to_base(src,dst):
    """🖼 "Из фото": the user's REAL photo IS the base frame — no still is generated. Copy it in as a
    proper PNG (the upload is a .jpg; png_size()/AR-fit need a real PNG header) and animate off it."""
    from PIL import Image
    with Image.open(b.COMFY_INPUT_DIR/src) as im:
        im.convert("RGB").save(dst)
    return Path(dst)

# ---------- face-lock: stamp a fixed real face onto every still via ReActor ----------
FACE_SOURCES={"tati":"tati_face.jpg"}                     # key -> reference in ComfyUI/input
FACE_HAIR={"tati":"long wavy strawberry-blonde hair below the shoulders, (straight blunt fringe bangs:1.25)"}
async def face_swap(target_png,dst,source_name):
    """ReActor-stamp the locked face onto a generated image. Face only — hair comes from the prompt."""
    tname=await asyncio.to_thread(b.upload_image_to_comfy,str(target_png),Path(target_png).name)
    g={"1":{"class_type":"LoadImage","inputs":{"image":tname}},
       "2":{"class_type":"LoadImage","inputs":{"image":source_name}},
       "3":{"class_type":"ReActorFaceSwap","inputs":{"enabled":True,"input_image":["1",0],"source_image":["2",0],
            "swap_model":b.REACTOR_SWAP_MODEL,"facedetection":b.REACTOR_FACE_DETECTION,
            "face_restore_model":b.REACTOR_FACE_RESTORE_MODEL,"face_restore_visibility":b.MOPMIX_DUO_FACE_RESTORE_VIS,
            "codeformer_weight":0.5,"detect_gender_input":"female","detect_gender_source":"female",
            "input_faces_index":"0","source_faces_index":"0","console_log_level":1}},
       "4":{"class_type":"SaveImage","inputs":{"images":["3",0],"filename_prefix":"cx_face"}}}
    pid=await asyncio.to_thread(b.queue_prompt,g,str(uuid.uuid4()))
    dl=time.time()+240
    while time.time()<dl:
        it=(await asyncio.to_thread(b.get_history,pid)).get(pid)
        if it and it.get("outputs"):
            im=(it["outputs"].get("4",{}).get("images") or [None])[0]
            if im:
                blob=await asyncio.to_thread(b.fetch_file,im["filename"],im.get("subfolder",""),im.get("type","output"))
                Path(dst).write_bytes(blob); return Path(dst)
            if it.get("status",{}).get("status_str")=="error": raise RuntimeError("reactor error")
        await asyncio.sleep(2)
    raise TimeoutError("reactor timeout")
# native WAN 2.2 I2V realism enhancer (anti-"doll" for motion), spliced always-on onto node 141's
# high/low model chain after patch_video_workflow — uncapped, independent of the action loras.
WAN_REALISM=True
WAN_REALISM_HIGH="Wan25RealisticHigh.safetensors"; WAN_REALISM_LOW="Wan25RealisticLow.safetensors"; WAN_REALISM_STRENGTH=0.4
def inject_wan_realism(wf,strength=WAN_REALISM_STRENGTH):
    n141=wf.get("141",{}).get("inputs")
    if not n141: return wf
    hp=n141.get("model_high_noise",["371",0]); lp=n141.get("model_low_noise",["372",0])
    # single clean splice onto node 141's model chain (whatever it currently points to — the action
    # loras already applied upstream). NOT the rgthree 152/155, to avoid the bot's 2x-at-half pattern.
    wf["cx_wan_real_h"]={"class_type":"LoraLoaderModelOnly","inputs":{"model":hp,"lora_name":WAN_REALISM_HIGH,"strength_model":float(strength)}}
    wf["cx_wan_real_l"]={"class_type":"LoraLoaderModelOnly","inputs":{"model":lp,"lora_name":WAN_REALISM_LOW,"strength_model":float(strength)}}
    n141["model_high_noise"]=["cx_wan_real_h",0]; n141["model_low_noise"]=["cx_wan_real_l",0]
    return wf
async def gen_video(image_name,prompt,loras,dst,seconds=BEAT_SECONDS,w=W,h=H):
    wf=b.load_workflow(b.WORKFLOW_VIDEO)
    wf=b.patch_video_workflow(wf,prompt=prompt,image_name=image_name,width=w,height=h,seconds=seconds,
        video_fps=16,seed=b.make_seed(),selected_loras=loras,video_model="svi_fastmove")
    if WAN_REALISM: wf=inject_wan_realism(wf)
    pid=await asyncio.to_thread(b.queue_prompt,wf,str(uuid.uuid4()))
    dl=time.time()+700
    while time.time()<dl:
        it=(await asyncio.to_thread(b.get_history,pid)).get(pid)
        if it and it.get("outputs"):
            arr=(it["outputs"].get("314",{}).get("gifs") or it["outputs"].get("314",{}).get("videos") or [])
            if arr:
                im=arr[0]; return await asyncio.to_thread(b.fetch_file,im["filename"],im.get("subfolder",""),im.get("type","output"))
            if it.get("status",{}).get("status_str")=="error": raise RuntimeError("video comfy error")
        await asyncio.sleep(3)
    raise TimeoutError("video timeout")

EROS_NODE="1135:597"
async def gen_video_eros(image_name,prompt,dst,dialogue=True,seconds=BEAT_SECONDS,w=0,h=0):
    """LTX Eros i2v beat with native voice: Ollama writes one in-character line (appended as quoted
    speech so patch confines it to one window), scene-fit Eros loras auto-picked. Output node 1135:597."""
    p=prompt
    if dialogue:
        try:
            line=await asyncio.to_thread(b.generate_dialogue_line,prompt,"")
            if line: p=f'{prompt.rstrip(". ")}. Она говорит: "{line}"'
        except Exception as e: print("  dialogue fail",e)
    try: loras=await asyncio.to_thread(b.select_loras_for_scene,prompt,"Eros") or []
    except Exception: loras=[]
    ew,eh=(w,h) if (w and h) else b.LTX_EROS_QUALITY.get("medium",(416,736))
    wf=b.load_workflow(b.WORKFLOW_LTX_EROS)
    wf=b.patch_ltx_eros_workflow(wf,prompt=p,image_name=image_name,width=ew,height=eh,seconds=seconds,
        seed=b.make_seed(),selected_loras=loras)
    pid=await asyncio.to_thread(b.queue_prompt,wf,str(uuid.uuid4()))
    dl=time.time()+900
    while time.time()<dl:
        it=(await asyncio.to_thread(b.get_history,pid)).get(pid)
        if it and it.get("outputs"):
            out=it["outputs"].get(EROS_NODE,{})
            arr=out.get("gifs") or out.get("videos") or out.get("images") or []
            if arr:
                im=arr[0]; return await asyncio.to_thread(b.fetch_file,im["filename"],im.get("subfolder",""),im.get("type","output"))
            if it.get("status",{}).get("status_str")=="error": raise RuntimeError("eros comfy error")
        await asyncio.sleep(3)
    raise TimeoutError("eros timeout")

# ---------- one scenario = still (both present) + 3 chained acts -> concat ----------
async def build_scenario(idx,plan,climaxes,mode,neg,still_t,rng,N,wd,engine="wan",face="",src=""):
    """Roll one scenario and render just the STILL. Returns (png, meta) where meta carries
    everything needed to animate the still later (beats/engine/face). Shared by full КОМПЛЕКС
    and ПОЛУКОМПЛЕКС (photos-first). `src` = 🖼 real source photo (comfy-input filename): beat-0 is
    img2img off it AND it doubles as the facelock source so her identity carries through every beat."""
    wd.mkdir(parents=True,exist_ok=True)
    fsrc=(FACE_SOURCES.get(face,face)) if face else (src or None)   # face key/filename, else the src photo locks her face
    eth=rng.choice(ETH); hair=(FACE_HAIR.get(face) or rng.choice(HAIR)) if face else rng.choice(HAIR)
    partner=plan["partner"]
    facefeat=rng.choice(FACES); view=rng.choice(VIEWS); expr=rng.choice(EXPRS)
    # FACE-LOCK: ReActor can only stamp a face it can DETECT. Many random views/hints turn her away
    # (from behind / over the shoulder / bent over) → no frontal face → the swap silently no-ops and
    # the original random woman stays ("другая женщина"), worst in couples where the man can steal it.
    # So when locking a face, force HER face frontal & unobstructed and (in couples) turn the man away.
    if fsrc:
        view="front view, looking toward the camera, her whole face clearly visible and unobstructed, sharp focus on her face"
        expr=rng.choice(["a soft smile","lips slightly parted","a sultry gaze at the camera","biting her lip","a calm confident look"])
    ac=plan.get("age_cue","")
    bpos,bneg,skipbr=plan.get("shape",("","",False))    # honour requested breast/ass/height
    subj=f"a {eth} woman, {hair}, {facefeat}, {plan['body']}"+(f", {ac}" if ac else "")+(f", {bpos}" if bpos else "")
    # DEEP MERGE (2026-08-31): ONE builder fed by BOTH pools. Per clip, either fully procedural OR
    # seed a WHOLE curated STORY_BANK bundle (its setting+outfit+men+opener + curated explicit a2)
    # through this same path — so the bank is no longer a separate branch. Whole-bundle (not per-beat)
    # keeps coherence: a curated action stays in its own curated setting (no "лодка в кухне").
    seed=None
    if partner in ("man","mmf") and rng.random()<BANK_SEED_PROB:
        pool=BANK_SEED.get("mmf" if partner=="mmf" else "man") or []
        seed=rng.choice(pool) if pool else None
    dog_on=False
    if seed:
        setting=seed["s"]; outfit=seed["o"]
        dm1,dm2=pick_men(setting,rng)
        m1=seed.get("m1") or dm1; m2=seed.get("m2") or dm2
        dog_on=bool(seed.get("dog"))
        still_hint=(seed.get("still") or seed["a1"]).rstrip(". ")
        b0=(with_identity(seed["a1"].rstrip(". ")+(DOG if dog_on else ""),partner,m1,m2),
            seed.get("l0") or ["wan_emotion"])
        label=seed.get("lbl","история"); ac_l=seed.get("l",[]); ac_p=seed["a2"].rstrip(". ")
        tr_p=seed.get("tr") or ("both men expose their erections and everyone moves into the exact action pose"
                                if partner=="mmf" else
                                "the man exposes his erection and they move into the exact action pose")
    else:
        outfit=pick_outfit(rng)                             # biased toward dressed-but-erotic + stockings
        setting,variants=rng.choice(SCENES)                 # pick a location, then one of its 2-3 entries
        still_hint,entry_p,entry_l=rng.choice(variants)     # location comes alive: prop-interaction beat-0
        m1,m2=pick_men(setting,rng)                          # men matched to the location's grime tier
        label,tr_p,ac_p,ac_l=pick_climax(climaxes,partner,rng)
        if partner in ("man","mmf"):
            dog_on=setting in OUTDOOR_WALK and rng.random()<0.30   # non-sexual dog-on-leash on outdoor walks
            b0=couple_entry(partner,setting,outfit,m1,m2,rng,dog_on)
        else:
            b0=(entry_p,[l for l in entry_l if l!="wan_scene_change"] or ["wan_emotion"])
    rough=(setting in GRIMY_SETTINGS) or (m1 in DIRTY_MEN) or (m2 in DIRTY_MEN)  # grime → victim roleplay
    cam,cam2=pick_cameras(rng)                            # distinct angle per pass (side → above/below/close)
    # THREE-ACT ASSEMBLY, chained by the last frame (act1 opener → act2 transition+action → act3 escalation).
    action_start=wardrobe_transition(outfit,tr_p,rng,partner,ac_p)
    action_full=add_reactions(with_identity(f"{action_start}; then immediately {ac_p}, {cam}",partner,m1,m2),partner,rng,rough)+(dog_clause(rng) if dog_on else "")
    escalation=with_identity(action_escalation(ac_p,partner,rng,cam2,rough),partner,m1,m2)+(dog_clause(rng) if dog_on else "")
    beats=[b0,(action_full,ac_l),(escalation,ac_l)]
    still=still_t.format(subj=subj,outfit=outfit,setting=setting,man=m1,man2=m2)
    # STILL composition: bank-seeded → the story-specific opening frame; procedural couples → a varied
    # pose; solo/women → the location prop hint. All keep her face camera-visible so facelock works.
    if seed:              shot=f"story-specific opening moment: {still_hint}, {view}, {expr}"
    elif partner=="man":  shot=f"{rng.choice(MAN_STILL_POSE)}, {expr}"
    elif partner=="mmf":  shot=f"{rng.choice(MMF_STILL_POSE)}, {expr}"
    else:                 shot=f"{still_hint}, {view}, {expr}"
    if fsrc and plan["people"]==2:                        # keep only HER face frontal so ReActor targets her, not the man
        shot+=", the woman faces the camera with her face clearly visible, the man's face is turned away or seen from behind"
    sp=("photorealistic amateur photo, "+still+", "+shot+STILL_SUFFIX) if mode!="pony" else (still+", "+shot)
    neg2=(neg+", "+bneg) if (neg and bneg) else (neg or bneg)
    if src:            png=src_to_base(src,wd/"still.png")  # 🖼 the ORIGINAL photo IS the base — no new base generated
    elif mode=="pony": png=await gen_photo_pony(sp,wd/"still.png")
    else:              png=await gen_photo_mopmix(sp,wd/"still.png",neg2,skip_breast_lora=skipbr)
    if fsrc and not src:                                    # stamp the locked face onto the still (src base already IS her)
        try: png=await face_swap(png,wd/"still_face.png",fsrc)
        except Exception as e: print("  facelock still fail",e)
    pmap={"man":"👥+♂","mmf":"👥+♂♂","woman":"👥+♀","anthro":"👥+🐾","none":"👤"}
    ftag=" · 🖼 из фото" if src else (" · 🎯 лицо" if fsrc else "")
    cap=(f"📸 #{idx+1}/{N} · {pmap.get(plan['partner'],'👤')} «{label}»{ftag}\n🌍 {eth} · 👗 {outfit}\n📍 {setting}")
    meta={"idx":idx,"label":label,"eth":eth,"outfit":outfit,"setting":setting,
          "beats":[[bp,ls] for bp,ls in beats],"engine":engine,"fsrc":fsrc,"cap":cap}
    return png,meta

async def animate_scenario(png,wd,meta,audio,N):
    """Animate an already-rendered still into a chained multi-act scene using its manifest."""
    idx=meta["idx"]; label=meta["label"]; engine=meta["engine"]; fsrc=meta.get("fsrc")
    eth=meta["eth"]; outfit=meta["outfit"]; setting=meta["setting"]
    beats=[(bp,ls) for bp,ls in meta["beats"]]; ac_p=beats[-1][0]
    png=Path(png)
    img=await asyncio.to_thread(b.upload_image_to_comfy,str(png),png.name)
    # keep the still's aspect ratio in the video so bodies aren't squashed/widened
    sw,sh=png_size(png)
    base_area=(b.LTX_EROS_QUALITY.get("medium",(416,736))[0]*b.LTX_EROS_QUALITY.get("medium",(416,736))[1]) if engine=="eros" else W*H
    vw,vh=fit_dims(sw,sh,base_area)
    segs=[]; used=[]; ok=False
    try:
        for bi,(bp,loras) in enumerate(beats):
            if engine=="eros":                          # LTX Eros: native voice + Ollama dialogue
                blob=await gen_video_eros(img,bp,wd/f"b{bi}.mp4",dialogue=True,w=vw,h=vh)
            else:                                        # WAN svi_fastmove
                blob=await gen_video(img,bp,loras,wd/f"b{bi}.mp4",w=vw,h=vh)
            raw=wd/f"b{bi}_raw.mp4"; await asyncio.to_thread(b.save_bytes,raw,blob)
            seg=wd/f"b{bi}_n.mp4"; await asyncio.to_thread(b.normalize_story_segment,raw,seg,vw,vh,16)
            segs.append(seg); used.append("💬эрос" if engine=="eros" else (loras[0] if loras else "переход"))
            if bi<len(beats)-1:
                fr=wd/f"f{bi}.png"; await asyncio.to_thread(b.extract_last_frame,raw,fr)
                if fsrc:                                  # re-stamp the locked face so it doesn't drift across beats
                    try: fr=await face_swap(fr,wd/f"f{bi}_face.png",fsrc)
                    except Exception as e: print("  facelock frame fail",e)
                img=await asyncio.to_thread(b.upload_image_to_comfy,str(fr),fr.name)
        final=wd/"final.mp4"; await asyncio.to_thread(b.concat_story_segments,segs,final,wd); src=final
        if engine=="wan" and audio and b.VIDEO_AUDIO:    # Eros already has native voice → skip MMAudio
            try:
                voiced=await b.run_video_audio_postprocess(final.read_bytes(),
                    {"mode":"video","job_id":f"cx{idx}","chat_id":CHAT,"prompt":ac_p},final.name)
                if voiced: src=wd/"voiced.mp4"; await asyncio.to_thread(b.save_bytes,src,voiced[0])
            except Exception as e: print("  audio fail",e)
        eng_tag="🧬 Эрос" if engine=="eros" else "🎬 ВАН"
        ok=tg_video(src,f"🎬 #{idx+1}/{N} · «{label}» · {eng_tag} · сцена с развитием\n🎬 {'→'.join(used)}")
    except Exception as e:
        print(f"  [{idx+1}] scenario FAIL {e}")
    LOG.open("a").write(json.dumps({"i":idx,"climax":label,"eth":eth,"outfit":outfit,"setting":setting,"ok":bool(ok)})+"\n")
    return bool(ok)

async def one_scenario(idx,plan,climaxes,mode,neg,still_t,audio,rng,N,engine="wan",face="",src=""):
    wd=OUT/f"c{idx:03d}"
    png,meta=await build_scenario(idx,plan,climaxes,mode,neg,still_t,rng,N,wd,engine,face,src)
    if not src: tg_photo(png,meta["cap"])   # 🖼 "Из фото": base is the same original every clip → skip preview, go straight to video
    ok=await animate_scenario(png,wd,meta,audio,N)
    shutil.rmtree(wd,ignore_errors=True); return ok

# ======================= STORY BANK — ~100 curated story/action pairs, rendered as 3 acts =======================
# Modelled on the user's best hand-written clips: ACT1 = the couple TOGETHER (walk / grope ass+breasts /
# kiss neck-cheek, she laughs; optional dog on a leash [non-sexual], optional a jealous 2nd man pushed
# away) → ACT2 = one concrete explicit position growing from that opening. Reuses the same chained machinery
# (animate_scenario) so face-lock / aspect-ratio / audio all still apply. Selected via --partner stories.
# entry keys: s=setting  o=outfit  p=man|mmf|rej (default man)  m1/m2=man archetype (else location pool)
#   dog=bool (non-sexual dog on leash in the walk)  a1=act-1 intro  a2=act-2 explicit  tr=optional bridge
#   l=action loras (default [])  lbl=RU caption. Her body/face come from the prompt (or face-lock) as usual.
BLK="a tall dark-skinned African man with an athletic build"
OLD="a fat grey-haired old man with a big round belly, 66 years old"
SUIT="a 50-year-old man in an expensive tailored suit"
YOUNG="a fit athletic 27-year-old man"
NEIGH="a 45-year-old ordinary neighbour with a slight beer belly and stubble"
ARM="(an Armenian man:1.3), 45 years old, thick black moustache, hairy chest, olive skin"
UZB="(an Uzbek man:1.3), 50 years old, in a striped national chapan robe and a doppa skullcap, weathered tanned face"
def _story_still_tpl(p,has2):
    if p=="mmf" or (p=="rej" and has2): return STILL_MMF
    return STILL_MAN
def _story_neg(p):
    return NEG_MMF if p in ("mmf","rej") else ANTI_CLONE
async def build_story_scenario(idx,story,plan,rng,N,wd,engine="wan",face="",src=""):
    """Build a STILL + 3-act manifest from a curated STORY_BANK entry (same meta shape as build_scenario).
    `src` = 🖼 real source photo: beat-0 img2img off it + facelock so her identity carries through."""
    wd.mkdir(parents=True,exist_ok=True)
    fsrc=(FACE_SOURCES.get(face,face)) if face else (src or None)
    eth=rng.choice(ETH); hair=(FACE_HAIR.get(face) or rng.choice(HAIR)) if face else rng.choice(HAIR)
    facefeat=rng.choice(FACES); view=rng.choice(VIEWS); expr=rng.choice(EXPRS)
    if fsrc:
        view="front view, looking toward the camera, her whole face clearly visible and unobstructed, sharp focus on her face"
        expr=rng.choice(["a soft smile","lips slightly parted","a sultry gaze at the camera","biting her lip","a calm confident look"])
    ac=plan.get("age_cue",""); bpos,bneg,skipbr=plan.get("shape",("","",False))
    subj=f"a {eth} woman, {hair}, {facefeat}, {plan['body']}"+(f", {ac}" if ac else "")+(f", {bpos}" if bpos else "")
    p=story.get("p","man"); setting=story["s"]; outfit=story["o"]
    idp="mmf" if p=="mmf" else "man"                     # identity carrier (rej acts single-man)
    dm1,dm2=pick_men(setting,rng)                         # location-matched fallback men
    m1=story.get("m1") or dm1; m2=story.get("m2") or dm2
    a1=story["a1"].rstrip(". ")+(DOG if story.get("dog") else "")
    rough=(setting in GRIMY_SETTINGS) or (m1 in DIRTY_MEN) or (m2 in DIRTY_MEN)
    # dog present via the flag OR hand-written into a1 ("a dog trotting along") → carry it (reacting) into acts 2&3
    dog_on=bool(story.get("dog")) or bool(re.search(r"\bdog\b", a1.lower()))
    cam=story.get("c") or rng.choice(CAMERA)
    cam3=story.get("c3") or rng.choice([c for c in CAMERA_ANGLE_SHIFT if c!=cam] or CAMERA_ANGLE_SHIFT)
    # beat-0 lora: content-aware & varied (not always emotion). Story may override via "l0".
    _a1l=a1.lower()
    if any(k in _a1l for k in ("suck","blowjob","cock","unzip","kneel")): _b0l=[]
    elif any(k in _a1l for k in ("kiss","neck","lick")):                  _b0l=rng.choice([["wan_kiss"],["wan_lick"]])
    else:                                                                 _b0l=rng.choice([["wan_emotion"],["wan_kiss"],[]])
    b0=(with_identity(a1,idp,m1,m2), story.get("l0") or _b0l)
    bridge=story.get("tr") or ("both men expose their erections and everyone moves into the exact action pose"
                               if idp=="mmf" else
                               "the man exposes his erection and they move into the exact action pose")
    action_start=wardrobe_transition(outfit,bridge,rng,idp,story["a2"])
    action_full=add_reactions(with_identity(
        f'{action_start}; then immediately {story["a2"].rstrip(". ")}, {cam}',idp,m1,m2),idp,rng,rough)+(dog_clause(rng) if dog_on else "")
    escalation=with_identity(
        action_escalation(story["a2"],idp,rng,cam3,rough),idp,m1,m2)+(dog_clause(rng) if dog_on else "")
    beats=[b0,(action_full,story.get("l",[])),(escalation,story.get("l",[]))]
    has2=bool(story.get("m2")) or p in ("mmf","rej")
    still_t=_story_still_tpl(p,has2); neg=_story_neg(p)
    still=still_t.format(subj=subj,outfit=outfit,setting=setting,man=m1,man2=m2)
    # The old story still contained only "couple together", so nearly all clips visibly
    # started as the same standing portrait.  Put the selected story's unique opening into
    # frame zero; WAN can then continue that situation instead of inventing the stock pose.
    opening_frame=(story.get("still") or story["a1"]).rstrip(". ")
    shot=f"story-specific opening moment: {opening_frame}, {view}, {expr}"
    if fsrc:
        shot+=", the woman faces the camera with her face clearly visible, the man's face is turned away or seen from behind"
    sp="photorealistic amateur photo, "+still+", "+shot+STILL_SUFFIX
    neg2=(neg+", "+bneg) if (neg and bneg) else (neg or bneg)
    if src: png=src_to_base(src,wd/"still.png")   # 🖼 the ORIGINAL photo IS the base — no new base generated
    else:   png=await gen_photo_mopmix(sp,wd/"still.png",neg2,skip_breast_lora=skipbr)
    if fsrc and not src:                          # src base already IS her → no still swap
        try: png=await face_swap(png,wd/"still_face.png",fsrc)
        except Exception as e: print("  facelock still fail",e)
    pmap={"man":"👥+♂","mmf":"👥+♂♂","rej":"👥+♂(2-й уходит)"}
    lbl=story.get("lbl","история"); ftag=" · 🖼 из фото" if src else (" · 🎯 лицо" if fsrc else "")
    cap=f"📖 #{idx+1}/{N} · {pmap.get(p,'👥+♂')} «{lbl}»{ftag}\n👗 {outfit}\n📍 {setting}"
    meta={"idx":idx,"label":lbl,"eth":eth,"outfit":outfit,"setting":setting,
          "beats":[[bp,ls] for bp,ls in beats],"engine":engine,"fsrc":fsrc,"cap":cap}
    return png,meta

STORY_BANK=[
 # ---- street / promenade / park walks (dog on some) ----
 {"s":"a city street sidewalk","o":"a short light summer sundress, deep cleavage, stockings and high heels","p":"man","m1":BLK,"dog":True,
  "a1":"she walks out of a boutique onto the seaside promenade with the man, he grips her ass hard and slips a hand under her skirt onto her bare cheeks while kissing her cheek and neck, she laughs and hugs him with one arm",
  "a2":"at night she kneels in front of him with a surprised open mouth and sucks his cock, her face at the level of his balls, he stands at full height holding her hair and setting the rhythm, explicit","lbl":"Набережная — минет ночью"},
 {"s":"a city street sidewalk","o":"a short light dress with a deep neckline, stockings, high heels","p":"man","m1":BLK,
  "a1":"she walks along the night promenade with the man, he holds her from behind by the hips grinding against her and kissing her neck, she laughs leaning back on him",
  "a2":"at night she leans on the promenade railing with her bare ass out, he stands behind holding her hips and hair and fucks her with his hard cock from behind, she smiles happily at the camera, explicit hardcore","lbl":"Набережная — сзади у перил"},
 {"s":"a public park at dusk","o":"a floral summer sundress and sandals","p":"man","m1":NEIGH,"dog":True,
  "a1":"they stroll through the park at dusk, he keeps an arm around her waist and squeezes her ass, kissing her cheek while she giggles",
  "a2":"she sits on the park bench and he kneels, then she leans forward and sucks his cock while he sits back on the bench, looking up at him, explicit","lbl":"Парк — минет на скамейке"},
 {"s":"a public park at dusk","o":"tight jeans and a crop top","p":"man","m1":YOUNG,
  "a1":"they walk their way deep into the quiet park, he presses her against a tree, gropes her breasts and kisses her neck, she laughs and pulls him closer",
  "a2":"she bends forward against the tree gripping the bark, he stands behind and fucks her from behind, hard standing thrusts, his cock sliding in, explicit hardcore","lbl":"Парк — стоя раком у дерева"},
 {"s":"a beach at sunset","o":"a tiny bikini pushed aside, a sheer sarong","p":"man","m1":BLK,"dog":True,
  "a1":"they walk barefoot along the shoreline at sunset, he scoops a hand over her ass and kisses her shoulder, she laughs splashing in the surf",
  "a2":"she drops to her knees in the wet sand and takes his cock in her mouth, sucking while the waves wash over her legs, explicit","lbl":"Пляж — минет в прибое"},
 {"s":"a beach at sunset","o":"a wet white t-shirt and a thong","p":"man","m1":YOUNG,
  "a1":"they walk out of the sea holding each other, he grabs her ass with both hands and kisses her wet neck, she laughs pushing her wet hair back",
  "a2":"she lies back on the beach towel and spreads her legs, he kneels between them and fucks her in missionary, his cock penetrating her, explicit","lbl":"Пляж — миссионерская на полотенце"},
 # ---- shops / mall / fitting rooms (jealous second man) ----
 {"s":"a fitting room","o":"a colourful dress with no panties, hairy pubis, stockings, a big ass","p":"rej","m1":OLD,"m2":YOUNG,
  "a1":"on the shop floor the old man in the centre gropes her ass and breasts and kisses her neck, shoving away the younger man on the right who glares angrily, she blushes and smiles as the old man unbuttons her jacket and kisses her bare breasts",
  "a2":"the camera hangs overhead: she kneels forward on a pillow with her back to him, the fat grey-haired old man without trousers fucks her from behind holding her ass, explicit hardcore","lbl":"Примерочная — дед сзади"},
 {"s":"a fitting room","o":"red lace lingerie she is trying on, stockings","p":"man","m1":SUIT,
  "a1":"she models lingerie in the fitting room mirror, he steps in behind her, cups her breasts and kisses her neck while she watches in the mirror and laughs",
  "a2":"she is on all fours facing the mirror, he kneels behind and fucks her doggystyle while they both watch in the mirror, explicit hardcore","lbl":"Примерочная — раком у зеркала"},
 {"s":"a fitting room","o":"a colourful split bikini, light stockings","p":"rej","m1":OLD,"m2":NEIGH,
  "a1":"walking through the shop she is groped by the old man in the centre who pushes the other man out of frame, kissing her shoulder as she smiles shyly",
  "a2":"the camera looks down from above: she lies back on a bench, the fat old man without trousers leans over her between her legs and fucks her, pressing her down, explicit","lbl":"Магазин — дед сверху"},
 # ---- boats / yacht / pool ----
 {"s":"a yacht deck","o":"a colourful split swimsuit and light stockings","p":"man","m1":OLD,"dog":False,
  "a1":"on the sunny deck the old man hugs her from the side as the other guests leave, groping her ass and kissing her neck, she laughs in the sea breeze",
  "a2":"the camera looks down: she lies on her back across an inflatable yellow boat, head over the edge, legs to the sides, the grey-bellied old man without trousers leans over her between her legs and fucks her from above pressing her to the boat, they smile slyly at the camera, explicit","lbl":"Лодка — дед сверху на надувной"},
 {"s":"a yacht deck","o":"a micro bikini","p":"man","m1":YOUNG,
  "a1":"they lounge on the yacht deck, he pulls her onto his lap, gropes her breasts and kisses her neck while she laughs sipping champagne",
  "a2":"she straddles him on the deck lounger facing away, riding his cock in reverse cowgirl, her ass bouncing toward the camera, explicit hardcore","lbl":"Яхта — ковгерл спиной"},
 {"s":"beside a swimming pool","o":"a bikini pushed aside","p":"man","m1":YOUNG,
  "a1":"she rises dripping from the pool and he catches her, grabbing her wet ass and kissing her neck, she laughs pressing against him",
  "a2":"she sits on the pool edge and leans back, he stands in the water and fucks her at the edge holding her thighs apart, his cock driving in, explicit","lbl":"Бассейн — на бортике"},
 # ---- retro / soviet / village ----
 {"s":"a candle-lit bedroom","o":"a light summer dress with bare breasts and stockings","p":"man","m1":BLK,
  "a1":"in a dim Soviet-era bedroom she sits close to the man on the old bed, he slides the dress off her breasts and kisses her neck while she laughs softly",
  "a2":"in the dim Soviet bedroom she sits astride his cock facing him, the man lying back on the pillow with straight legs, holding her big breasts and helping her ride, the camera shows from behind his cock moving in and out fast and rhythmically, he smiles at the camera and she smiles shyly, explicit hardcore","lbl":"Советская спальня — верхом"},
 {"s":"a cluttered village courtyard","o":"a traditional embroidered dress, no underwear","p":"man","m1":UZB,"dog":True,
  "a1":"in the dusty rural courtyard with chickens pecking nearby the man loosens her dress, gropes her and kisses her neck while she laughs against the sun-baked clay wall",
  "a2":"she leans forward against the clay wall, he lifts her dress and fucks her from behind standing, gripping her hips, explicit hardcore","lbl":"Двор — стоя раком у стены"},
 {"s":"a stylish living room","o":"a floral housedress with nothing under it","p":"man","m1":OLD,
  "a1":"in the retro living room the old man pulls her onto the couch, opens her housedress and gropes her bare breasts, kissing her neck while she giggles",
  "a2":"she sucks his cock while he sits back on the couch, leaning over his lap and bobbing her head, looking up at him, explicit","lbl":"Гостиная — минет на диване"},
 # ---- home / neighbour / kitchen / bedroom ----
 {"s":"a sunlit kitchen","o":"an apron with nothing underneath, stockings","p":"man","m1":NEIGH,
  "a1":"in the kitchen the neighbour steps behind her at the counter, unties the apron, gropes her breasts and kisses her neck while she laughs",
  "a2":"she bends over the kitchen counter, he stands behind and fucks her from behind, her breasts pressed to the counter, explicit hardcore","lbl":"Кухня — раком у стойки"},
 {"s":"a stylish living room","o":"yoga pants and a sports bra","p":"man","m1":YOUNG,
  "a1":"she stretches on the living room rug, he kneels behind, peels down her yoga pants, gropes her ass and kisses her neck while she laughs",
  "a2":"she is on all fours on the rug, he kneels behind and fucks her doggystyle, gripping her hips, explicit hardcore","lbl":"Гостиная — раком на ковре"},
 {"s":"a candle-lit bedroom","o":"a sheer babydoll negligee","p":"man","m1":SUIT,
  "a1":"in the candle-lit bedroom he lays her back, slides the negligee straps down and kisses from her neck to her breasts while she smiles",
  "a2":"she lies on her back with her ankles up on his shoulders, he fucks her deep with her legs folded, his cock driving down into her, explicit hardcore","lbl":"Спальня — ноги на плечах"},
 {"s":"a marble bathroom","o":"a silk robe falling open","p":"man","m1":YOUNG,
  "a1":"at the bathroom sink he comes up behind her, opens her robe, cups her breasts and kisses her neck as she watches in the mirror and laughs",
  "a2":"she bends over the sink, he stands behind and fucks her from behind while she looks at herself in the mirror, explicit hardcore","lbl":"Ванная — раком у раковины"},
 # ---- office / work ----
 {"s":"a modern glass office","o":"a business suit and pencil skirt","p":"man","m1":SUIT,
  "a1":"in the glass office he crowds her against the desk, hikes her skirt, gropes her ass and kisses her neck while she laughs pushing the laptop aside",
  "a2":"she bends over the office desk, he stands behind and fucks her from behind over the desk, papers scattering, explicit hardcore","lbl":"Офис — раком на столе"},
 {"s":"a modern glass office","o":"a blouse and a pleated skirt, no bra","p":"man","m1":SUIT,
  "a1":"she perches on the edge of the desk, he steps between her knees, unbuttons her blouse and kisses her breasts while she laughs",
  "a2":"she lies back on the desk with her legs hanging off, he stands between them and fucks her, holding her thighs, explicit","lbl":"Офис — на столе"},
 # ---- gym / sauna / massage ----
 {"s":"a gym","o":"a sports bra and tiny shorts","p":"man","m1":YOUNG,
  "a1":"on the gym bench he spots her, then pulls her up, gropes her sweaty breasts and kisses her neck while she laughs catching her breath",
  "a2":"she rides him on the gym bench in cowgirl, bouncing on his cock with her hands on his chest, explicit hardcore","lbl":"Зал — ковгерл на скамье"},
 {"s":"a steamy sauna","o":"only a towel","p":"man","m1":NEIGH,
  "a1":"in the steam he sits beside her on the sauna bench, loosens her towel, gropes her breasts and kisses her shoulder while she smiles in the heat",
  "a2":"she kneels on the sauna bench and he stands, she deepthroats his cock, drool on her chin, his hand on her head, explicit hardcore","lbl":"Сауна — дипгорло"},
 {"s":"a massage room","o":"a towel slipping off","p":"man","m1":YOUNG,
  "a1":"on the massage table she lies face down, he oils her back, slides the towel off her ass and kisses down her spine while she sighs and smiles",
  "a2":"she lies flat on her stomach, he lies on top of her from behind and fucks her in prone bone, grinding deep, explicit","lbl":"Массаж — прон-бон"},
 # ---- nightlife / car / club ----
 {"s":"a car back seat","o":"a tight mini dress, no underwear","p":"man","m1":YOUNG,
  "a1":"in the back seat he pulls her onto his lap, hikes her dress, gropes her bare ass and kisses her neck while she laughs in the dim light",
  "a2":"she straddles him in the back seat facing him and rides his cock, bouncing in the cramped space, explicit","lbl":"Машина — верхом на заднем сиденье"},
 {"s":"a dim neon-lit nightclub","o":"a sheer see-through dress","p":"man","m1":BLK,
  "a1":"on the neon dance floor she grinds back against him, he holds her hips and gropes her breasts, kissing her neck under the lights while she laughs",
  "a2":"in the VIP booth she bends over the seat, he stands behind and fucks her from behind, neon light on their bodies, explicit hardcore","lbl":"Клуб — сзади в VIP"},
 {"s":"a bar counter","o":"a cocktail dress sliding off one shoulder","p":"man","m1":SUIT,
  "a1":"at the bar he leans in, slides a hand up her thigh under the dress and kisses her neck while she laughs over her cocktail",
  "a2":"she leans over the bar counter with her ass out, he stands behind and fucks her over the bar, explicit hardcore","lbl":"Бар — над стойкой"},
 # ---- gritty / bums (m1 auto = dirty/homeless via pick_men) ----
 {"s":"a construction site with scaffolding and concrete","o":"a floral sundress, no underwear","p":"man","dog":True,
  "a1":"she picks across the muddy site to the worker, he gropes her ass and breasts through the dress and kisses her neck while she laughs in the dust",
  "a2":"she leans back against a stack of cinder blocks, he lifts her dress and fucks her standing, holding her thighs, explicit hardcore","lbl":"Стройка — стоя у блоков"},
 {"s":"by the dumpsters in a back yard","o":"a cheap tight dress, no bra","p":"man",
  "a1":"in the grimy back yard the ragged man pulls her aside by the bins, gropes her breasts and kisses her neck while she laughs",
  "a2":"she bends over against the stained brick wall by the trash, he stands behind and fucks her from behind, explicit hardcore","lbl":"Помойка — раком у стены"},
 {"s":"an abandoned ruined building","o":"a thin summer dress","p":"man",
  "a1":"among the crumbling ruins the dirty man presses her to a cracked wall, gropes her and kisses her neck while she laughs",
  "a2":"she kneels on a filthy old mattress and sucks his cock, looking up at him, explicit","lbl":"Развалины — минет на матрасе"},
 {"s":"behind a row of garages","o":"tight jeans peeled down and a crop top","p":"man","dog":True,
  "a1":"in the dirt lane behind the rusty garages the scruffy man gropes her ass and kisses her neck against a corrugated door while she laughs",
  "a2":"she is on all fours on an old car tyre, he kneels behind and fucks her doggystyle, explicit hardcore","lbl":"Гаражи — раком на покрышке"},
 {"s":"a run-down industrial yard","o":"a torn sundress","p":"man",
  "a1":"in the scrapyard the grimy man bends her over a rusted oil drum, groping her and kissing her neck while she laughs",
  "a2":"she bends over the rusted oil drum, he stands behind and fucks her hard from behind, explicit hardcore","lbl":"Промзона — раком у бочки"},
 {"s":"a cramped basement boiler room","o":"a cheap slip dress","p":"man",
  "a1":"in the dim boiler room the dirty man pins her to the old boiler, gropes her breasts and kisses her neck while she laughs",
  "a2":"she sinks onto the stained old couch and he kneels between her spread legs, fucking her in missionary, explicit","lbl":"Бойлерная — миссионерская на диване"},
 # ---- MMF threesomes (both men busy, never touch each other) ----
 {"s":"a luxury hotel suite","o":"red lace lingerie and stockings","p":"mmf","m1":SUIT,"m2":YOUNG,
  "a1":"in the suite the two men flank her on the bed, one kissing her neck and groping her breasts, the other kissing her shoulder and gripping her ass, she laughs between them",
  "a2":"spitroast: she is on all fours, one man fucks her pussy from behind doggystyle while she sucks the other man in front, both cocks in frame, explicit hardcore","lbl":"Отель — вертел (3сом)"},
 {"s":"a stylish living room","o":"a black lace bodysuit","p":"mmf","m1":NEIGH,"m2":YOUNG,
  "a1":"on the couch the two men sit either side of her, both groping her breasts and thighs and kissing her neck and shoulders while she laughs",
  "a2":"she rides one man in cowgirl while sucking the other standing at her face, both men used at once, explicit hardcore","lbl":"Гостиная — едет и сосёт (3сом)"},
 {"s":"a construction site with scaffolding and concrete","o":"a torn sundress, no underwear","p":"mmf","dog":True,
  "a1":"two grimy workers crowd her against the scaffolding, both groping her ass and breasts and kissing her neck while she laughs in the dust",
  "a2":"double blowjob: she kneels between the two dirty men and sucks both cocks, going back and forth, licking both heads, explicit","lbl":"Стройка — два в рот (3сом)"},
 {"s":"a car back seat","o":"a mini dress with no panties","p":"mmf","m1":YOUNG,"m2":NEIGH,
  "a1":"squeezed between the two men in the back seat, both slide their hands up her dress groping her while kissing her neck and shoulders and she laughs",
  "a2":"she lies back sucking one man while the other kneels between her legs and fucks her in missionary, explicit hardcore","lbl":"Машина — на спине вдвоём (3сом)"},
 {"s":"a candle-lit bedroom","o":"a bra and thong with a garter belt","p":"mmf","m1":SUIT,"m2":BLK,
  "a1":"the two men lay her back on the bed between them, both kissing and groping her, one at her breasts one at her hips, she laughs",
  "a2":"one man lies back with her riding his cock while the other kneels behind her ass, double penetration, both cocks in her at once, explicit hardcore","lbl":"Спальня — DP (3сом)"},
 {"s":"a dim neon-lit nightclub","o":"a latex bodysuit with cutouts","p":"mmf","m1":YOUNG,"m2":YOUNG,
  "a1":"in the VIP booth the two men sandwich her, grinding on both sides, groping her and kissing her neck under the neon while she laughs",
  "a2":"she strokes both mens cocks with her hands and licks each in turn, double handjob, explicit","lbl":"Клуб — дрочит двоим (3сом)"},
 # ---- batch 2: more settings × positions ----
 {"s":"an elevator","o":"a tight sweater dress, no underwear","p":"man","m1":SUIT,
  "a1":"the elevator doors close and he presses her to the mirrored wall, hikes her dress, gropes her ass and kisses her neck while she laughs",
  "a2":"he lifts her against the mirrored wall, she wraps her legs around his waist and he fucks her standing, bouncing her on his cock, explicit hardcore","lbl":"Лифт — на весу у стены"},
 {"s":"a library aisle","o":"a blouse and a pleated skirt, no panties","p":"man","m1":YOUNG,
  "a1":"between the shelves he presses against her, lifts her skirt, gropes her bare ass and kisses her neck while she stifles a laugh",
  "a2":"she presses back against the shelves bent forward, he fucks her from behind between the library shelves, explicit hardcore","lbl":"Библиотека — сзади у полок"},
 {"s":"a private jet cabin","o":"an elegant evening gown slit high","p":"man","m1":SUIT,
  "a1":"reclined in the jet seat she pulls him close, he slides a hand up the slit and kisses her neck while she laughs with champagne",
  "a2":"she kneels in the narrow jet aisle and deepthroats his cock, drool on her chin, his hand on her head, explicit hardcore","lbl":"Джет — дипгорло в проходе"},
 {"s":"a rooftop terrace at sunset","o":"a mini dress with no underwear","p":"man","m1":BLK,"dog":True,
  "a1":"at the rooftop railing in the sunset he stands behind her, gropes her breasts and kisses her neck while she leans back laughing",
  "a2":"she sits back on his lap facing away at the rooftop lounger and rides his cock in reverse, his hands on her breasts, explicit hardcore","lbl":"Крыша — спиной на закате"},
 {"s":"a wine cellar","o":"a cocktail dress hiked up","p":"man","m1":OLD,
  "a1":"among the wine racks the old man sets down her glass, gropes her ass and kisses her neck while she laughs against the bottles",
  "a2":"she perches on a wine barrel, leans back, he stands between her knees and fucks her on the barrel, explicit","lbl":"Винный погреб — на бочке"},
 {"s":"a restaurant bathroom stall","o":"a tight dress hiked up, no panties","p":"man","m1":SUIT,
  "a1":"in the locked stall he crowds her, hikes her dress, gropes her ass and kisses her neck while she giggles quietly",
  "a2":"she braces on the stall wall bent forward, he stands behind and fucks her from behind in the cramped stall, explicit hardcore","lbl":"Туалет ресторана — стоя раком"},
 {"s":"a locker room","o":"a sports bra and shorts half off","p":"man","m1":YOUNG,
  "a1":"on the locker bench he peels her top down, gropes her breasts and kisses her neck while she laughs after a workout",
  "a2":"she sits on the locker bench, he stands and she sucks his cock, one hand stroking the base, looking up, explicit","lbl":"Раздевалка — минет на скамье"},
 {"s":"an empty classroom","o":"a school blazer and a short skirt, no panties","p":"man","m1":NEIGH,
  "a1":"in the empty classroom he leans her back on the teacher desk, unbuttons her blouse and kisses her breasts while she laughs",
  "a2":"she lies back on the teacher desk with legs spread, he stands between them and fucks her on the desk, explicit hardcore","lbl":"Класс — на учительском столе"},
 {"s":"a doctor's office","o":"a thin open medical gown","p":"man","m1":SUIT,
  "a1":"on the exam table he unties her gown, gropes her breasts and kisses her neck while she smiles nervously",
  "a2":"she lies back on the exam table and pulls her knees up, he kneels close and fucks her with her legs up, explicit","lbl":"Кабинет врача — ноги вверх"},
 {"s":"a laundry room","o":"a housedress with nothing under it","p":"man","m1":NEIGH,
  "a1":"she hops onto the running washing machine, he steps between her knees, opens her dress and kisses her breasts while she laughs as it vibrates",
  "a2":"she sits on the vibrating washing machine, he stands and fucks her on it, holding her thighs, explicit hardcore","lbl":"Прачечная — на стиралке"},
 {"s":"a dark movie theater","o":"a short skirt, no underwear","p":"man","m1":YOUNG,
  "a1":"slouched in the dark theater seat she leans to him, he slides a hand under her skirt and kisses her neck while she smiles at the screen",
  "a2":"she leans over into his lap and sucks his cock in the dark theater seat, bobbing her head, explicit","lbl":"Кинотеатр — минет в кресле"},
 {"s":"a strip club stage","o":"pasties and a g-string","p":"man","m1":SUIT,
  "a1":"she works the pole then drops into his lap at the stage edge, he gropes her ass and kisses her neck while she laughs",
  "a2":"she rides him at the edge of the stage in cowgirl, bouncing on his cock, her breasts bouncing, explicit hardcore","lbl":"Стрип — ковгерл у сцены"},
 {"s":"a garage","o":"cut-off shorts and a tied plaid shirt","p":"man",
  "a1":"the greasy mechanic bends her over the car hood in the garage, gropes her ass and kisses her neck while she laughs",
  "a2":"she leans over the car hood, he stands behind and fucks her over the hood, explicit hardcore","lbl":"Гараж — раком на капоте"},
 {"s":"a rooftop pool","o":"a micro bikini dripping wet","p":"man","m1":YOUNG,
  "a1":"she climbs dripping from the rooftop pool, he catches her, grabs her wet ass and kisses her neck while she laughs",
  "a2":"she lies back on the lounger and he lifts her ankles to his shoulders, fucking her deep with her legs folded, explicit hardcore","lbl":"Крыша-бассейн — складка"},
 {"s":"a penthouse with a city view","o":"strappy lingerie","p":"man","m1":SUIT,
  "a1":"at the floor-to-ceiling window over the city he stands behind her, gropes her breasts and kisses her neck while she watches the lights and laughs",
  "a2":"he lifts her against the glass, she wraps her legs around him and he fucks her standing against the window, explicit hardcore","lbl":"Пентхаус — на весу у окна"},
 {"s":"a hotel balcony","o":"a sheet wrapped around her","p":"man","m1":BLK,
  "a1":"on the balcony he lets the sheet drop, gropes her from behind and kisses her neck while she leans on the rail laughing",
  "a2":"she leans over the balcony railing, he stands behind and fucks her from behind over the rail, explicit hardcore","lbl":"Балкон — сзади у перил"},
 {"s":"a stylish living room","o":"an open shirt and nothing else","p":"man","m1":YOUNG,
  "a1":"on the rug by the fireplace she kneels close, he opens her shirt and gropes her breasts, kissing her neck while she laughs",
  "a2":"she kneels and presses her breasts around his cock, giving him a titfuck, licking the tip on each stroke, explicit","lbl":"Гостиная — титфак у камина"},
 {"s":"a candle-lit bedroom","o":"a lace teddy","p":"man","m1":NEIGH,
  "a1":"on the bed he lies back and she crawls over him, he gropes her breasts and kisses her while she laughs",
  "a2":"the man lies on his back and she lowers her pussy onto his mouth facing his feet, he eats her out while she strokes his cock, facesitting, explicit","lbl":"Спальня — сидит на лице"},
 {"s":"a marble bathroom","o":"only stockings","p":"man","m1":YOUNG,
  "a1":"in the steam of the shower he pulls her close, gropes her wet body and kisses her neck while she laughs under the water",
  "a2":"they do a 69 on the bathroom floor, she sucks his cock while he licks her pussy, both at once, explicit","lbl":"Ванная — 69 на полу"},
 {"s":"a sunlit kitchen","o":"a bathrobe falling open","p":"man","m1":OLD,
  "a1":"at the kitchen table the old man opens her robe, gropes her breasts and kisses her neck while she laughs on his lap",
  "a2":"she squats over him on the chair in the amazon position and bounces down onto his cock, explicit hardcore","lbl":"Кухня — амазонка на стуле"},
 {"s":"a beach at sunset","o":"a bikini untied","p":"man","m1":BLK,
  "a1":"lying on the beach towel at sunset he rolls onto her, gropes her breasts and kisses her neck while she laughs in the sand",
  "a2":"they lie on their sides in the sand, he spoons her from behind and fucks her, one leg lifted, slow deep thrusts, explicit","lbl":"Пляж — ложка на боку"},
 {"s":"a public park at dusk","o":"a sundress lifted","p":"man","m1":NEIGH,"dog":True,
  "a1":"on the park bench at dusk he pulls her onto his lap, gropes her and kisses her neck while she laughs, his dog resting at their feet",
  "a2":"she sits on his lap on the bench facing away and rides his cock slowly, his hands on her breasts, explicit","lbl":"Парк — на лавке спиной"},
 {"s":"a city street sidewalk","o":"a trench coat with nothing under it","p":"man","m1":SUIT,"dog":True,
  "a1":"on the night street he opens her coat, gropes her bare breasts and kisses her neck while she laughs pulling it closed again",
  "a2":"at night she kneels on the sidewalk in the coat and sucks his cock, looking up at him, explicit","lbl":"Улица — минет ночью на коленях"},
 {"s":"a cluttered village courtyard","o":"a long modest dress and headscarf, no underwear","p":"man","m1":"(a Tajik man:1.3), 55 years old, embroidered chapan robe and skullcap, grey-flecked beard","dog":True,
  "a1":"in the rural courtyard the old man in the chapan loosens her dress, gropes her and kisses her neck while she laughs in the shade",
  "a2":"she reclines on the woven mat, he kneels between her legs and fucks her in missionary, explicit","lbl":"Двор — миссионерская на циновке"},
 {"s":"a candle-lit bedroom","o":"crotchless panties and a bra","p":"man","m1":YOUNG,
  "a1":"on the bed she straddles him, he gropes her ass and kisses her while she grinds and laughs",
  "a2":"she rides him reverse cowgirl facing away, bouncing on his cock, her ass to the camera, explicit hardcore","lbl":"Спальня — ковгерл спиной"},
 {"s":"a stylish living room","o":"a sheer robe","p":"man","m1":SUIT,
  "a1":"on the couch he sits back, she leans between his knees, he strokes her hair as she smiles up at him",
  "a2":"she strokes his cock with one hand and licks and sucks the head, handjob and blowjob together, explicit","lbl":"Гостиная — дрочит и сосёт"},
 {"s":"a modern glass office","o":"a pencil skirt hiked up, no panties","p":"man","m1":SUIT,
  "a1":"in the office chair she swivels to him, he props her heels up, opens her blouse and kisses her breasts while she laughs",
  "a2":"she rides him in the leather office chair facing him, bouncing on his cock with her arms around his neck, explicit hardcore","lbl":"Офис — верхом в кресле"},
 {"s":"a marble bathroom","o":"a bikini pushed aside, wet","p":"man","m1":YOUNG,
  "a1":"rising from the bubble bath foam sliding off her breasts, he pulls her close, gropes her and kisses her neck while she laughs",
  "a2":"she bends over the edge of the tub, he stands behind and fucks her from behind, water everywhere, explicit hardcore","lbl":"Ванна — сзади у бортика"},
 {"s":"a garage","o":"overalls peeled to the waist, no bra","p":"man",
  "a1":"the mechanic hoists her onto the workbench, gropes her breasts and kisses her neck while she laughs, grease on her thighs",
  "a2":"she lies back on the garage workbench and spreads her knees, he stands and fucks her on the bench, explicit","lbl":"Гараж — на верстаке"},
 {"s":"a construction site with scaffolding and concrete","o":"a torn dress","p":"rej","m2":YOUNG,"dog":True,
  "a1":"on the site the grimy worker pulls her close and shoves the younger labourer out of frame, groping her and kissing her neck while she laughs",
  "a2":"she leans against the unfinished concrete wall, he lifts her dress and fucks her standing, explicit hardcore","lbl":"Стройка — работяга гонит второго"},
 {"s":"a bar counter","o":"a sweater dress, no underwear","p":"rej","m1":SUIT,"m2":NEIGH,
  "a1":"at the bar the man in the suit pulls her to him and waves off the other man who scowls, groping her thigh and kissing her neck while she smiles",
  "a2":"she leans back on the bar stool, he stands between her knees and fucks her at the bar, explicit","lbl":"Бар — второй отшит"},
 {"s":"a luxury hotel suite","o":"an open-cup bra and crotchless lingerie","p":"man","m1":SUIT,
  "a1":"by the suite window he unzips nothing because she is already exposed, gropes her breasts and kisses her neck while she laughs",
  "a2":"she is on all fours and he kneels behind pulling her hair back, fucking her doggystyle with deep thrusts, explicit hardcore","lbl":"Отель — раком за волосы"},
 {"s":"a candle-lit bedroom","o":"red lace lingerie","p":"man","m1":BLK,
  "a1":"on the bed he lays her back, kisses from her neck down her body and spreads her legs while she smiles",
  "a2":"he folds her knees to her chest in a mating press and fucks her straight down, deep hard thrusts, explicit hardcore","lbl":"Спальня — складка (mating press)"},
 {"s":"a steamy sauna","o":"a towel","p":"man","m1":OLD,
  "a1":"on the top sauna bench the old man loosens her towel, gropes her breasts and kisses her shoulder in the heat while she smiles",
  "a2":"she lies back on the sauna bench, he lifts her legs and fucks her, sweat glistening, explicit","lbl":"Сауна — на верхней полке"},
 {"s":"a car back seat","o":"jeans peeled down and a crop top","p":"man","m1":YOUNG,
  "a1":"climbing over the back seat she looks back over her shoulder, he grips her ass and kisses her neck while she laughs",
  "a2":"she is on all fours over the back seat, he kneels behind and fucks her doggystyle in the car, explicit hardcore","lbl":"Машина — раком через сиденье"},
 {"s":"a beach at sunset","o":"a sheer sarong","p":"mmf","dog":True,
  "a1":"two men walk her along the shoreline at sunset, both groping her ass and breasts and kissing her neck and shoulders while she laughs in the surf",
  "a2":"spitroast in the sand: one fucks her from behind on all fours while she sucks the other in front, both cocks in frame, explicit hardcore","lbl":"Пляж — вертел (3сом)"},
 {"s":"a yacht deck","o":"a bikini untied","p":"mmf","m1":SUIT,"m2":YOUNG,
  "a1":"on the yacht deck the two men flank her, both groping her and kissing her neck and shoulders while she laughs in the breeze",
  "a2":"one man lies back with her riding him while the other stands at her face, she rides one and sucks the other, explicit hardcore","lbl":"Яхта — едет и сосёт (3сом)"},
 {"s":"a public park at dusk","o":"a summer dress","p":"mmf","m1":NEIGH,"m2":YOUNG,"dog":True,
  "a1":"two men walk with her through the dusk park, both keeping hands on her ass and breasts, kissing her neck while she laughs, a dog trotting along",
  "a2":"she kneels on the grass between the two men and sucks both cocks, back and forth, double blowjob, explicit","lbl":"Парк — два в рот (3сом)"},
 {"s":"a modern glass office","o":"a business suit, no bra","p":"mmf","m1":SUIT,"m2":SUIT,
  "a1":"in the office the two men crowd her against the desk, both groping her and kissing her neck while she laughs pushing papers aside",
  "a2":"she bends over the desk, one man fucks her ass from behind while she deepthroats the other standing in front, explicit hardcore","lbl":"Офис — раком + минет (3сом)"},
 {"s":"a marble bathroom","o":"a soaked white t-shirt","p":"man","m1":YOUNG,
  "a1":"at the mirror he lifts her soaked shirt, gropes her wet breasts and kisses her neck while she laughs",
  "a2":"she sits on the bathroom counter and pulls him in, wrapping her legs around him as he fucks her on the counter, explicit","lbl":"Ванная — на тумбе"},
 {"s":"a candle-lit bedroom","o":"a garter belt and stockings","p":"man","m1":SUIT,
  "a1":"on the bed she turns onto her front, he lies over her from behind, gropes her and kisses her shoulders while she smiles into the pillow",
  "a2":"she lies face down with her ass raised on a pillow, he mounts her from behind and fucks her pussy, gripping her raised hips, explicit hardcore","lbl":"Спальня — лицом вниз жопой вверх"},
 {"s":"a stylish living room","o":"a bra pushed up and a skirt hiked up","p":"man","m1":NEIGH,
  "a1":"on the couch she straddles him, he gropes her breasts and kisses her while she grinds slowly and laughs",
  "a2":"she grinds slowly on his cock in cowgirl, rolling her hips, riding him deep, explicit","lbl":"Гостиная — наездница медленно"},
 {"s":"a rooftop terrace at sunset","o":"an unbuttoned blouse and no bra","p":"man","m1":BLK,
  "a1":"on the rooftop lounger in the sunset he opens her blouse, gropes her breasts and kisses her neck while she laughs",
  "a2":"she lies back on the lounger and spreads her legs, he kneels between them and fucks her in missionary in the golden light, explicit","lbl":"Крыша — миссионерская на закате"},
 {"s":"a wine cellar","o":"a sheer babydoll","p":"man","m1":OLD,
  "a1":"against the wine racks the old man gropes her breasts and kisses her neck while she leans back laughing",
  "a2":"she leans back against the bottles, he lifts one of her legs and fucks her standing against the racks, explicit hardcore","lbl":"Погреб — стоя у стеллажа"},
 {"s":"a private jet cabin","o":"a slip dress","p":"man","m1":SUIT,
  "a1":"in the jet seat she slides onto his lap, he gropes her and kisses her neck while she laughs with the engines humming",
  "a2":"she rides him in the jet seat facing him, bouncing on his cock with her arms around his neck, explicit hardcore","lbl":"Джет — верхом в кресле"},
 {"s":"an abandoned ruined building","o":"a cheap slip dress","p":"man",
  "a1":"among the ruins the dirty man presses her to a peeling wall, gropes her and kisses her neck while she laughs in the dust",
  "a2":"she is on all fours arching her back, he kneels behind and fucks her ass from behind, anal, deep thrusts, explicit hardcore","lbl":"Развалины — анал раком"},
 {"s":"by the dumpsters in a back yard","o":"a torn dress","p":"man",
  "a1":"by the bins the ragged man gropes her ass and kisses her neck against the stained wall while she laughs",
  "a2":"she leans back on the dumpster and pulls her knees up, he presses close and fucks her ass with her legs up, anal, explicit hardcore","lbl":"Помойка — анал ноги вверх"},
 {"s":"a strip club stage","o":"body chains and a thong","p":"mmf","m1":SUIT,"m2":YOUNG,
  "a1":"two men pull her off the stage between them, both groping her and kissing her neck and shoulders under the lights while she laughs",
  "a2":"the two men sandwich her standing, one in front in her pussy and one behind in her ass, DP standing, explicit hardcore","lbl":"Стрип — сэндвич DP (3сом)"},
 {"s":"a jacuzzi","o":"a bikini pushed aside","p":"man","m1":YOUNG,
  "a1":"in the jacuzzi he pulls her onto his lap in the jets, gropes her wet breasts and kisses her neck while she laughs",
  "a2":"she straddles him in the jacuzzi and rides his cock, water sloshing, her breasts at his face, explicit hardcore","lbl":"Джакузи — верхом в воде"},
 {"s":"a photo studio","o":"lingerie and heels","p":"man","m1":SUIT,
  "a1":"under the studio lights he breaks her pose, pulls her top down, gropes her breasts and kisses her neck while she laughs",
  "a2":"she sits on the backdrop floor and leans back on her hands, he kneels and fucks her with her legs parted, explicit","lbl":"Студия — на полу циклорамы"},
 {"s":"a gym","o":"leggings peeled down","p":"man","m1":YOUNG,
  "a1":"on the yoga mat he kneels behind her mid-stretch, peels her leggings, gropes her ass and kisses her neck while she laughs",
  "a2":"she is on all fours on the yoga mat, he kneels behind and fucks her doggystyle, gripping her hips, explicit hardcore","lbl":"Зал — раком на коврике"},
 {"s":"a hotel balcony","o":"a robe falling open","p":"man","m1":SUIT,
  "a1":"in the balcony chair he opens her robe, gropes her breasts and kisses her neck in the morning sun while she smiles over coffee",
  "a2":"she straddles him in the balcony chair facing him and sinks down onto his cock, riding him, explicit hardcore","lbl":"Балкон — верхом в кресле"},
 {"s":"a candle-lit bedroom","o":"a corset and stockings","p":"man","m1":SUIT,
  "a1":"the man nears his climax buried in her as they fuck hard on the bed, both sweating and gasping",
  "a2":"he cums inside her, creampie, pulling out so his semen drips from her pussy, close-up on the dripping cum, explicit hardcore","lbl":"Спальня — кремпай финал"},
 {"s":"a stylish living room","o":"a g-string only","p":"man","m1":YOUNG,
  "a1":"she kneels in front of him stroking his cock toward her open mouth, looking up while he holds her hair",
  "a2":"he cums, shooting his load onto her face and open mouth, cumshot on her face, explicit","lbl":"Гостиная — камшот на лицо"},
 {"s":"a sunlit kitchen","o":"a summer dress bent over the counter","p":"man","m1":NEIGH,
  "a1":"bent into the fridge for a drink she turns holding the cold bottle to her chest, he steps close, gropes her and kisses her neck while she laughs",
  "a2":"she bends over the kitchen counter, he stands behind and fucks her ass from behind, anal, explicit hardcore","lbl":"Кухня — анал стоя раком"},
 {"s":"a car back seat","o":"a mini dress","p":"mmf","m1":YOUNG,"m2":NEIGH,
  "a1":"crammed between the two men in the back seat, both grope up her dress and kiss her neck and shoulders while she laughs",
  "a2":"one lies back with her riding his cock while the other kneels behind, double penetration in the cramped seat, explicit hardcore","lbl":"Машина — DP (3сом)"},
 {"s":"a beach at sunset","o":"a bikini","p":"man","m1":BLK,
  "a1":"in the shallow surf he lifts her, she wraps her legs around him, gropes and kisses while the waves break around them and she laughs",
  "a2":"he holds her up in the surf, her legs around his waist, and fucks her standing in the water, explicit hardcore","lbl":"Пляж — на весу в воде"},
 {"s":"a modern glass office","o":"a skirt suit","p":"man","m1":SUIT,
  "a1":"he sits back in the chair, she kneels between his knees loosening his belt while he strokes her hair and she smiles up",
  "a2":"she sucks his cock while he sits in the office chair, leaning over his lap bobbing her head, looking up, explicit","lbl":"Офис — минет под столом"},
 {"s":"a luxury hotel suite","o":"a satin slip","p":"man","m1":SUIT,
  "a1":"at the minibar he presses behind her, slides the slip strap down, gropes her breasts and kisses her neck while she laughs with a drink",
  "a2":"she bends over the minibar counter, he stands behind and fucks her from behind, glasses rattling, explicit hardcore","lbl":"Отель — сзади у бара"},
 {"s":"a fitting room","o":"lingerie half on","p":"man","m1":YOUNG,
  "a1":"in the fitting room he pulls her close, gropes her breasts and kisses her neck while she laughs half-dressed",
  "a2":"she kneels up and presses her breasts around his cock while he stands, a standing titfuck, thrusting up between them while she looks up, explicit","lbl":"Примерочная — титфак стоя"},
 {"s":"a massage room","o":"a towel","p":"man","m1":YOUNG,
  "a1":"on the massage table he rolls her over, lets the towel slip, gropes her breasts and kisses her neck while she smiles",
  "a2":"she straddles him on the table facing away and lowers her ass onto his cock, riding him in anal cowgirl, ass to camera, explicit hardcore","lbl":"Массаж — анал ковгерл"},
]
async def one_story(idx,story,plan,audio,rng,N,engine="wan",face="",src=""):
    wd=OUT/f"s{idx:03d}"
    png,meta=await build_story_scenario(idx,story,plan,rng,N,wd,engine,face,src)
    if not src: tg_photo(png,meta["cap"])   # 🖼 "Из фото": same original base each clip → skip preview, go straight to video
    ok=await animate_scenario(png,wd,meta,audio,N)
    shutil.rmtree(wd,ignore_errors=True); return ok
# DEEP-MERGE seed pools: curated STORY_BANK bundles the unified build_scenario can adopt whole.
# Only pure single-man ("man") and "mmf" bundles seed procedural clips — "rej" (jealous 2nd man
# pushed away) needs its own 2-man still template, so it stays exclusive to the stories path.
BANK_SEED={"man":[s for s in STORY_BANK if s.get("p","man")=="man"],
           "mmf":[s for s in STORY_BANK if s.get("p")=="mmf"]}
BANK_SEED_PROB=float(os.getenv("CX_BANK_SEED_PROB","0.5"))   # 0 → pure procedural, 1 → always bank-seeded

def story_order(count,rng):
    """A shuffled, non-repeating-until-exhausted play order over STORY_BANK (cycles if count>bank)."""
    order=list(range(len(STORY_BANK))); rng.shuffle(order); out=[]
    while len(out)<count:
        if not order: order=list(range(len(STORY_BANK))); rng.shuffle(order)
        out.append(order.pop())
    return out[:count]

def bankmix_order(count,rng):
    """Balanced source plan for the hybrid mode.

    Odd batches randomly get one extra bank or procedural clip.  Shuffling prevents the
    visible bank/procedural alternation from becoming yet another repetitive sequence.
    Returns (use_story flags, enough non-repeating story indexes for all True slots).
    """
    bank_count=count//2+(rng.randrange(2) if count%2 else 0)
    flags=[True]*bank_count+[False]*(count-bank_count)
    rng.shuffle(flags)
    return flags,story_order(bank_count,rng)

# ---- partner mode: decide per-clip who the scene is with ----
# 'auto' = trust the prompt (solo unless a partner is named — the safe default that fixed the
# two-women clone bug). 'mix' = vary per clip, weighted toward SEX so a batch isn't all-solo
# masturbation. 'man'/'solo' = force. Anthro/woman partners are only produced when the prompt
# actually asks for them (mix keeps them in rotation, never invents a human man over them).
WHO={"man":"👥 секс с мужиком","mmf":"👥 тройничок (2 мужика)","woman":"👥 с женщиной","anthro":"👥 с антро","none":"👤 соло"}
def pick_partner(base_partner,mode,rng):
    if mode=="solo": return "none"
    if mode=="man":  return "man"
    if mode=="mix":
        if base_partner in ("woman","anthro"):
            return rng.choice([base_partner,base_partner,base_partner,base_partner,"none"])   # respect special partner, rare solo
        # solo cut to 10%; its place goes to WOMEN (FF) + THREESOMES per user. man 50 / woman 20 / mmf 20 / solo 10
        return rng.choice(["man","man","man","man","man","woman","woman","mmf","mmf","none"])
    return base_partner                                             # auto
def plan_for_clip(plan,partner):
    p=dict(plan); p["partner"]=partner; p["people"]=1 if partner=="none" else 2; return p

async def run(prompt,count,engine,audio,face="",partner_mode="auto",src=""):
    if engine not in ("wan","eros"): engine="wan"
    plan=parse_plan(prompt); base_partner=plan["partner"]
    LOG.write_text("")
    rng=random.Random(int(time.time())); t0=time.time()
    mode_lbl={"auto":"🧠 по промту","mix":"🎲 микс (секс+соло)","man":"👥 секс с мужиком","solo":"👤 соло",
              "stories":"📖 по банку сценариев","bankmix":"📚🎲 банк + микс (50/50)"}.get(partner_mode,partner_mode)
    eng_tag="🧬 Эрос (с диалогами)" if engine=="eros" else "🎬 ВАН"
    ftag=f"\n🎯 Фейслок: одно лицо на все клипы ({face})" if face else ""
    order=story_order(count,rng) if partner_mode=="stories" else None
    hybrid,hybrid_stories=bankmix_order(count,rng) if partner_mode=="bankmix" else (None,None)
    hybrid_pos=0
    tg_msg(f"🧩 КОМПЛЕКС ({eng_tag}, сценарии): {count} клипов.{ftag}\nСостав: {mode_lbl}\nТело (из промта): {plan['body'][:160]}\n"
           +("📖 Готовые истории: история/идея → вытекающее действие → его усиление/продолжение."
             if partner_mode=="stories" else
             "Рандом: локация · одежда · завязка · действие. Каждый клип — сцена с развитием."))
    ok=0
    for idx in range(count):
        t=time.time()
        try:
            if partner_mode=="stories" or (partner_mode=="bankmix" and hybrid[idx]):
                story_idx=order[idx] if partner_mode=="stories" else hybrid_stories[hybrid_pos]
                hybrid_pos+=1 if partner_mode=="bankmix" else 0
                story=STORY_BANK[story_idx]
                ok+=1 if await one_story(idx,story,plan,audio,rng,count,engine,face,src) else 0
                print(f"[{idx+1}/{count}] {time.time()-t:.0f}s ok={ok} [story:{story['lbl']}]")
            else:
                procedural_mode="mix" if partner_mode=="bankmix" else partner_mode
                cp=plan_for_clip(plan,pick_partner(base_partner,procedural_mode,rng))
                climaxes,mode,neg,still_t=config_for(cp)
                ok+=1 if await one_scenario(idx,cp,climaxes,mode,neg,still_t,audio,rng,count,engine,face,src) else 0
                print(f"[{idx+1}/{count}] {time.time()-t:.0f}s ok={ok} [{cp['partner']}]")
        except Exception as e:
            print(f"[{idx+1}/{count}] CLIP FAIL {e}")
            LOG.open("a").write(json.dumps({"i":idx,"err":str(e)[:200]})+"\n")
    tg_msg(f"✅ КОМПЛЕКС завершён: {ok}/{count} клипов за {(time.time()-t0)/3600:.1f}ч.")

PICKS=OUT/"picks"
async def run_photos(prompt,count,engine,audio,face="",partner_mode="auto",src=""):
    """ПОЛУКОМПЛЕКС phase 1: render ALL stills, persist each with its animate-manifest,
    and send each with a '🎬 Анимировать' button. Animation happens later, on demand."""
    if engine not in ("wan","eros"): engine="wan"
    plan=parse_plan(prompt); base_partner=plan["partner"]
    LOG.write_text("")
    rng=random.Random(int(time.time()))
    runid=uuid.uuid4().hex[:8]; base=PICKS/runid; base.mkdir(parents=True,exist_ok=True)
    mode_lbl={"auto":"🧠 по промту","mix":"🎲 микс (секс+соло)","man":"👥 секс с мужиком","solo":"👤 соло",
              "stories":"📖 по банку сценариев","bankmix":"📚🎲 банк + микс (50/50)"}.get(partner_mode,partner_mode)
    eng_tag="🧬 Эрос (с диалогами)" if engine=="eros" else "🎬 ВАН"
    ftag=f"\n🎯 Фейслок: {face}" if face else ""
    order=story_order(count,rng) if partner_mode=="stories" else None
    hybrid,hybrid_stories=bankmix_order(count,rng) if partner_mode=="bankmix" else (None,None)
    hybrid_pos=0
    (base/"run.json").write_text(json.dumps({"engine":engine,"audio":int(audio),"N":count}))
    tg_msg(f"🧩📷 ПОЛУКОМПЛЕКС ({eng_tag}): генерирую {count} фото.{ftag}\nСостав: {mode_lbl}\n"
           f"Потом жми 🎬 под теми, что хочешь оживить.")
    made=0
    for idx in range(count):
        pdir=base/f"{idx:03d}"
        try:
            if partner_mode=="stories" or (partner_mode=="bankmix" and hybrid[idx]):
                story_idx=order[idx] if partner_mode=="stories" else hybrid_stories[hybrid_pos]
                hybrid_pos+=1 if partner_mode=="bankmix" else 0
                png,meta=await build_story_scenario(idx,STORY_BANK[story_idx],plan,rng,count,pdir,engine,face,src)
            else:
                procedural_mode="mix" if partner_mode=="bankmix" else partner_mode
                cp=plan_for_clip(plan,pick_partner(base_partner,procedural_mode,rng))
                climaxes,mode,neg,still_t=config_for(cp)
                png,meta=await build_scenario(idx,cp,climaxes,mode,neg,still_t,rng,count,pdir,engine,face,src)
            meta["png"]=str(png); meta["audio"]=int(audio); meta["N"]=count
            (pdir/"plan.json").write_text(json.dumps(meta))
            tg_photo_btn(png,meta["cap"],f"cxa:{runid}:{idx:03d}")
            made+=1
        except Exception as e:
            print(f"[{idx+1}/{count}] PHOTO FAIL {e}")
    tg_msg(f"✅ Фото готовы: {made}/{count}. Жми 🎬 под нужными — оживлю по одному.")

async def run_animate(runid,idxstr):
    """ПОЛУКОМПЛЕКС phase 2: animate one chosen still from its saved manifest."""
    pdir=PICKS/runid/idxstr
    pj=pdir/"plan.json"
    if not pj.exists(): tg_msg("⚠️ Это фото уже не найдено (пачка убрана). Сгенерируй заново."); return
    meta=json.loads(pj.read_text())
    png=meta.get("png") or str(pdir/"still.png")
    if not Path(png).exists(): tg_msg("⚠️ Файл фото пропал."); return
    tg_msg(f"🎬 Оживляю фото #{int(idxstr)+1} «{meta.get('label','')}»…")
    try:
        await animate_scenario(png,pdir,meta,bool(meta.get("audio",1)),meta.get("N",1))
    except Exception as e:
        print("animate fail",e); tg_msg(f"⚠️ Не вышло оживить: {e}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("-n","--count",type=int,default=30)
    ap.add_argument("--engine",default="wan",choices=["wan","eros"])
    ap.add_argument("--audio",type=int,default=1)
    ap.add_argument("--chat",type=int,default=0)
    ap.add_argument("-p","--prompt",default="")
    ap.add_argument("--face",default="")   # 'tati' or an uploaded image filename in ComfyUI/input
    ap.add_argument("--src",default="")    # 🖼 real source photo (ComfyUI/input filename): img2img base + facelock
    ap.add_argument("--photos",action="store_true")   # ПОЛУКОМПЛЕКС: render stills only, animate on demand
    ap.add_argument("--animate",default="")            # 'runid:idx' — animate one chosen still
    ap.add_argument("--partner",default="auto",choices=["auto","mix","man","solo","stories","bankmix"])  # bankmix = 50/50 curated bank + procedural mix
    ap.add_argument("rest",nargs="*")
    a=ap.parse_args()
    global CHAT
    if a.chat: CHAT=a.chat
    if a.animate:
        runid,idxstr=a.animate.split(":",1)
        asyncio.run(run_animate(runid,idxstr)); return
    prompt=(a.prompt or " ".join(a.rest)).strip()
    if not prompt: print("no prompt"); sys.exit(2)
    if a.photos:
        asyncio.run(run_photos(prompt,a.count,a.engine,bool(a.audio),a.face,a.partner,a.src))
    else:
        asyncio.run(run(prompt,a.count,a.engine,bool(a.audio),a.face,a.partner,a.src))

if __name__=="__main__":
    main()
