"""
Visual layer for the demo app: page CSS and the static HTML sections.

Everything here is presentation only. Widgets stay native Streamlit, styled
through the `st-key-<key>` class Streamlit puts on keyed containers, and through
data-testid attributes. Those are Streamlit internals, not a public API, which
is why the Streamlit version is pinned in the requirements.
"""
import base64
from pathlib import Path

import cv2

ACCENT = "#EA580C"        # orange-600: buttons, highlights
ACCENT_DARK = "#C2410C"   # orange-700: hero gradient start, hover
AMBER = "#F59E0B"         # amber-500: hero gradient end
INK = "#1C1917"
MUTED = "#57534E"
TINT = "#FFF7ED"          # orange-50: tiles, drop zone
LINE = "#FED7AA"          # orange-200: tile borders

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

/* Streamlit sets its own font on headings and text; override element by
   element. Spans are left alone: Streamlit draws its icons as spans in an
   icon font, and overriding that shows the icon names as text. */
.stApp :is(h1, h2, h3, h4, h5, h6, p, li, label, button, input, textarea, a, small, div) {{
  font-family: 'Plus Jakarta Sans', 'Source Sans', 'Source Sans Pro', sans-serif !important;
}}
.stApp {{ background: #F6F5F4; }}
/* Streamlit's header is transparent here but still covers the top of the page
   and swallows clicks on the nav links; let clicks through except on its own
   controls (menu, running indicator). */
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stHeader"], [data-testid="stToolbar"] {{ pointer-events: none !important; }}
[data-testid="stHeader"] :is(button, a, [data-testid="stStatusWidget"], [data-testid="stMainMenu"]) {{
  pointer-events: auto !important; }}
[data-testid="stAppDeployButton"], [data-testid="stSidebarCollapsedControl"],
section[data-testid="stSidebar"] {{ display: none; }}
[data-testid="stMainBlockContainer"] {{ max-width: 1180px; padding-top: 1.2rem; padding-bottom: 4rem; }}

/* ---------- top bar ---------- */
.tv-nav {{ display: flex; align-items: center; justify-content: space-between;
  padding: .2rem .2rem 1rem; }}
.tv-brand {{ font-weight: 800; font-size: 1.15rem; color: {INK}; letter-spacing: -.01em; }}
.tv-brand span {{ color: {ACCENT}; }}
.tv-links a {{ color: {MUTED}; text-decoration: none; font-weight: 500;
  font-size: .92rem; margin-left: 1.6rem; }}
.tv-links a:hover {{ color: {ACCENT}; }}

/* ---------- hero ---------- */
.tv-hero {{ position: relative; display: grid; grid-template-columns: 1.05fr 1fr;
  gap: 2rem; align-items: center; border-radius: 28px; overflow: hidden;
  padding: 3.2rem 3rem 5.2rem;
  background: linear-gradient(115deg, {ACCENT_DARK} 0%, {ACCENT} 48%, {AMBER} 100%); }}
.tv-hero::after {{ content: ""; position: absolute; inset: auto -10% -40% auto;
  width: 520px; height: 520px; border-radius: 50%;
  background: rgba(255,255,255,.08); pointer-events: none; }}
.tv-hero h1 {{ color: #fff; font-size: clamp(2rem, 3.6vw, 3.1rem); font-weight: 800;
  line-height: 1.08; letter-spacing: -.025em; margin: 1rem 0 .9rem; padding: 0; }}
.tv-rule {{ width: 56px; height: 5px; border-radius: 3px; background: #fff; margin-bottom: 1rem; }}
.tv-hero p.tv-tag {{ color: #FFF7ED; font-size: 1.05rem; line-height: 1.6; max-width: 34rem; margin: 0; }}
.tv-chips {{ display: flex; flex-wrap: wrap; gap: .5rem; margin-top: 1.4rem; }}
.tv-chips span {{ background: rgba(255,255,255,.95); color: {ACCENT_DARK}; font-weight: 600;
  font-size: .8rem; padding: .4rem .8rem; border-radius: 999px; }}
.tv-shot {{ position: relative; z-index: 1; }}
.tv-shot img {{ width: 100%; display: block; border-radius: 20px;
  box-shadow: 0 30px 60px -20px rgba(0,0,0,.45); border: 6px solid rgba(255,255,255,.9); }}
.tv-shot small {{ display: block; color: #FFF7ED; font-size: .74rem; margin-top: .55rem; text-align: right; }}
@media (max-width: 820px) {{
  .tv-hero {{ grid-template-columns: 1fr; padding: 2.2rem 1.4rem 4.6rem; }}
  .tv-links {{ display: none; }}
}}

/* ---------- control card overlapping the hero ---------- */
.st-key-controls {{ background: #fff; border-radius: 20px; margin: -3.4rem auto 0;
  width: calc(100% - 3.2rem) !important;
  padding: 1.4rem 1.6rem 1.1rem; position: relative; z-index: 2;
  box-shadow: 0 24px 50px -18px rgba(28,25,23,.28); }}
.st-key-controls [data-testid="stWidgetLabel"] p {{ font-weight: 700; color: {INK};
  font-size: .82rem; text-transform: uppercase; letter-spacing: .06em; }}
@media (max-width: 820px) {{ .st-key-controls {{ width: calc(100% - .8rem) !important; }} }}

/* radio options as pills */
[data-testid="stRadio"] [role="radiogroup"] {{ gap: .5rem; }}
[data-testid="stRadio"] [role="radiogroup"] label {{ border: 1.5px solid #E7E5E4;
  border-radius: 999px; padding: .45rem .95rem .45rem .7rem; margin: 0; background: #fff; }}
[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {{
  border-color: {ACCENT}; background: {TINT}; }}

/* primary button */
.stButton > button[kind="primary"], .stDownloadButton > button {{
  border-radius: 14px; font-weight: 700; min-height: 3.1rem; letter-spacing: .01em; }}
.stButton > button[kind="primary"] {{ background: {ACCENT}; border: 0;
  box-shadow: 0 12px 24px -10px rgba(234,88,12,.7); font-size: 1.02rem; }}
.stButton > button[kind="primary"]:hover {{ background: {ACCENT_DARK}; }}

/* upload drop zone */
[data-testid="stFileUploaderDropzone"] {{ background: {TINT}; border: 2px dashed {LINE};
  border-radius: 16px; }}

/* ---------- section headings ---------- */
.tv-eyebrow-dark {{ text-align: center; font-size: .78rem; font-weight: 700;
  letter-spacing: .14em; text-transform: uppercase; color: {MUTED}; margin: 3.2rem 0 .3rem; }}
.tv-h2 {{ text-align: center; font-size: clamp(1.5rem, 2.4vw, 2rem); font-weight: 800;
  color: {INK}; letter-spacing: -.02em; margin: 0 0 1.6rem; }}

/* ---------- how-it-works steps ---------- */
.tv-steps {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.2rem; }}
.tv-step {{ background: #fff; border-radius: 20px; padding: 1.6rem 1.4rem;
  box-shadow: 0 10px 30px -18px rgba(28,25,23,.25); }}
.tv-ico {{ width: 52px; height: 52px; border-radius: 14px; display: grid; place-items: center;
  background: {TINT}; color: {ACCENT}; margin-bottom: 1rem; }}
.tv-step.hot .tv-ico {{ background: {ACCENT}; color: #fff; box-shadow: 0 12px 22px -10px rgba(234,88,12,.8); }}
.tv-step h4 {{ margin: 0 0 .4rem; font-size: 1.05rem; font-weight: 700; color: {INK}; padding: 0; }}
.tv-step p {{ margin: 0; color: {MUTED}; font-size: .9rem; line-height: 1.55; }}
@media (max-width: 820px) {{ .tv-steps {{ grid-template-columns: 1fr; }} }}

/* ---------- result cards ---------- */
[class*="st-key-card"] {{ background: #fff; border-radius: 20px; padding: 1.5rem 1.6rem;
  box-shadow: 0 10px 30px -18px rgba(28,25,23,.25); margin-top: 1rem; }}
[class*="st-key-card"] h3 {{ font-weight: 800; letter-spacing: -.01em; color: {INK};
  border-left: 5px solid {ACCENT}; padding: 0 0 0 .7rem; margin-bottom: .6rem; }}
[data-testid="stMetric"] {{ background: {TINT}; border: 1px solid {LINE}; border-radius: 16px;
  padding: .9rem 1rem; }}
[data-testid="stMetricValue"] {{ font-weight: 800; color: {INK}; }}
[data-testid="stMetricLabel"] p {{ font-weight: 600; color: {MUTED}; }}
.st-key-card_violations [data-testid="stColumn"]:last-child [data-testid="stMetric"] {{
  background: {ACCENT}; border-color: {ACCENT}; }}
.st-key-card_violations [data-testid="stColumn"]:last-child [data-testid="stMetric"] * {{ color: #fff; }}
[data-testid="stImage"] img {{ border-radius: 12px; }}
[data-testid="stVideo"] video, video {{ border-radius: 14px; }}

.tv-foot {{ text-align: center; color: {MUTED}; font-size: .82rem; margin-top: 3rem; }}
</style>
"""

_ICONS = {
    "detect": '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 7V5a2 2 0 0 1 2-2h2M17 3h2a2 2 0 0 1 2 2v2M21 17v2a2 2 0 0 1-2 2h-2M7 21H5a2 2 0 0 1-2-2v-2"/><rect x="7" y="8" width="10" height="8" rx="1"/></svg>',
    "check": '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2 3 6v6c0 5 3.8 9.4 9 10 5.2-.6 9-5 9-10V6l-9-4z"/><path d="m9 12 2 2 4-4"/></svg>',
    "fine": '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="5" width="18" height="14" rx="2"/><path d="M7 10h4M7 14h10"/></svg>',
}


def hero_image(sample: Path, frame: int = 360) -> str:
    """A real frame from the sample clip, as a data URI for the hero."""
    cap = cv2.VideoCapture(str(sample))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, img = cap.read()
    cap.release()
    if not ok:
        return ""
    img = cv2.resize(img, (760, int(img.shape[0] * 760 / img.shape[1])),
                     interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""


def hero(img_uri: str, live_plates: bool) -> str:
    plates = "Live number plate reading" if live_plates else "Number plate recognition"
    shot = (f'<div class="tv-shot"><img src="{img_uri}" alt="Frame from the sample '
            f'traffic clip"><small>Frame from the bundled sample clip</small></div>'
            if img_uri else "<div></div>")
    return f"""
<div class="tv-nav">
  <div class="tv-brand">Traffic Violation Detection<span>.</span></div>
  <div class="tv-links"><a href="#try">Try it</a><a href="#how">How it works</a><a href="#plates">Plate samples</a></div>
</div>
<div class="tv-hero">
  <div>
    <h1>Spot traffic violations in road video.</h1>
    <div class="tv-rule"></div>
    <p class="tv-tag">Vehicle detection and tracking, four violation checks and
    rule-based fine estimation, run for real on the clip you choose below.</p>
    <div class="tv-chips"><span>YOLOv8s fine-tuned on BMD-45</span><span>ByteTrack</span>
    <span>4 violation checks</span><span>Fine estimation</span><span>{plates}</span></div>
  </div>
  {shot}
</div>
<a id="try"></a>
"""


def steps(live_plates: bool) -> str:
    plate_line = ("and every plate large enough to read is read with YOLOv8n and EasyOCR."
                  if live_plates else
                  "and plates are read with YOLOv8n and EasyOCR (sample results below).")
    return f"""
<a id="how"></a>
<div class="tv-eyebrow-dark">How it works</div>
<div class="tv-h2">Three stages, one clip</div>
<div class="tv-steps">
  <div class="tv-step"><div class="tv-ico">{_ICONS['detect']}</div>
    <h4>Detect and track</h4>
    <p>A YOLOv8s detector fine-tuned on Indian CCTV finds every vehicle, and
    ByteTrack follows each one from frame to frame.</p></div>
  <div class="tv-step hot"><div class="tv-ico">{_ICONS['check']}</div>
    <h4>Check for violations</h4>
    <p>Speeding, wrong-side driving and lane changes on the calibrated sample
    camera. Tailgating on any clip, since it needs no calibration.</p></div>
  <div class="tv-step"><div class="tv-ico">{_ICONS['fine']}</div>
    <h4>Estimate fines, read plates</h4>
    <p>Each flagged vehicle gets a rule-based prototype fine, {plate_line}</p></div>
</div>
"""


FOOTER = """<div class="tv-foot">AI-Based Intelligent Vehicle
Monitoring and Traffic Violation Detection · Fine amounts are prototype rules,
not legally enforceable.</div>"""
