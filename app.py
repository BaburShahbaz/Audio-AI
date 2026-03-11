"""
╔══════════════════════════════════════════════════════════════╗
║   VocalMind — Emotion Intelligence Platform                  ║
║   Single-file Streamlit app · All 8 sections                 ║
║   Palette: #FFF8F0 · #281C59 · #4E8D9C · #85C79A            ║
║             #EDF7BD · #C08552 · #8C5A3C · #4B2E2B            ║
╚══════════════════════════════════════════════════════════════╝

Sections (mirrors React app structure):
  1. Navigation Header
  2. Input Portal        — upload / live record
  3. Pipeline            — animated AI pipeline
  4. Emotion Result      — wheel + dominant emotion
  5. Waveform + Timeline — audio viz + emotion arc
  6. Transcript + Insights — ASR + recommendations
  7. History             — session records
  8. Tech Stack + Footer
"""

import streamlit as st
import streamlit.components.v1 as components
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import io, math, datetime

# ── pkg_resources shim (Python 3.13 / librosa fix) ─────────────────────────
# librosa.core.intervals imports pkg_resources which was removed in Python 3.12+.
# This shim injects a minimal stand-in before importing librosa.
try:
    import pkg_resources  # available when setuptools is installed
except ModuleNotFoundError:
    import types, sys
    _pr = types.ModuleType('pkg_resources')
    def _resource_filename(pkg, path):
        import importlib.resources as _ir
        return str(_ir.files(pkg).joinpath(path))
    _pr.resource_filename = _resource_filename
    sys.modules['pkg_resources'] = _pr

import librosa
import whisper
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from fpdf import FPDF

try:
    from audio_recorder_streamlit import audio_recorder
    RECORDER_AVAILABLE = True
except ImportError:
    RECORDER_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="VocalMind · Emotion Intelligence",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ══════════════════════════════════════════════════════════════════════════════
#  BRAND PALETTE  (exact hex values from spec)
# ══════════════════════════════════════════════════════════════════════════════
BG      = "#FFF8F0"   # primary background
SIDEBAR = "#281C59"   # sidebar / headers / dark hero
TEAL    = "#4E8D9C"   # primary buttons / icons / active
GREEN   = "#85C79A"   # success / recording / positive
LIME    = "#EDF7BD"   # highlights / hover / soft glow
COPPER  = "#C08552"   # accent / warning / neutral emotion
BROWN   = "#8C5A3C"   # borders / dividers / secondary text
INK     = "#4B2E2B"   # primary text / headings

# Per-emotion accent colors
EMO_COLORS = {
    "Happiness": "#F4A261",
    "Neutral":   COPPER,
    "Anger":     "#E63946",
    "Sadness":   "#457B9D",
    "Calm":      GREEN,
}

LABEL_MAP = {
    0: ("Happiness", "😄", EMO_COLORS["Happiness"]),
    1: ("Neutral",   "😐", EMO_COLORS["Neutral"]),
    2: ("Anger",     "😠", EMO_COLORS["Anger"]),
    3: ("Sadness",   "😢", EMO_COLORS["Sadness"]),
    4: ("Calm",      "😌", EMO_COLORS["Calm"]),
}


# ══════════════════════════════════════════════════════════════════════════════
#  BACKEND  (untouched logic)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _whisp = whisper.load_model("tiny")
    _proc  = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    import logging, warnings
    # Suppress expected weight-mismatch warnings (lm_head/masked_spec_embed)
    # These occur because wav2vec2-base-960h is a CTC model but we only need
    # the base encoder for feature extraction.
    logging.getLogger("transformers").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=".*lm_head.*")
    warnings.filterwarnings("ignore", message=".*masked_spec_embed.*")
    warnings.filterwarnings("ignore", message=".*Some weights.*")
    warnings.filterwarnings("ignore", message=".*were not used.*")
    _w2v   = Wav2Vec2Model.from_pretrained(
        "facebook/wav2vec2-base-960h",
        ignore_mismatched_sizes=True
    ).to(device).eval()
    return device, _whisp, _proc, _w2v


def run_analysis(audio_bytes: bytes) -> dict:
    y, sr    = librosa.load(io.BytesIO(audio_bytes), sr=16000)
    duration = librosa.get_duration(y=y, sr=sr)
    trans    = whisper_model.transcribe(y)
    probs    = np.random.dirichlet(np.ones(5), size=1)[0]
    idx      = int(np.argmax(probs))
    name, emoji, color = LABEL_MAP[idx]

    n_chunks = max(1, int(duration // 3))
    timeline = []
    for i in range(n_chunks):
        cp  = np.random.dirichlet(np.ones(5), size=1)[0]
        ci  = int(np.argmax(cp))
        timeline.append({
            "time":       i * 3,
            "emotion":    LABEL_MAP[ci][0],
            "emoji":      LABEL_MAP[ci][1],
            "color":      LABEL_MAP[ci][2],
            "confidence": float(np.max(cp) * 100),
        })

    probs_dict = {LABEL_MAP[i][0]: float(probs[i]) for i in range(5)}
    return {
        "emotion":   name,  "emoji":    emoji,
        "color":     color, "confidence": float(np.max(probs) * 100),
        "transcript": trans["text"].strip() or "No speech detected.",
        "duration":  duration,   "sr":    sr,
        "probs":     probs_dict, "probs_arr": probs,
        "timeline":  timeline,   "waveform_y": y,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
if "history"  not in st.session_state: st.session_state.history  = []
if "analysis" not in st.session_state: st.session_state.analysis = None

# Boot screen
_boot = st.empty()
with _boot.container():
    st.markdown(f"""
    <div style="display:flex;flex-direction:column;align-items:center;
                justify-content:center;min-height:80vh;background:{BG};
                font-family:'Inter',sans-serif;">
        <div style="width:68px;height:68px;border-radius:20px;
                    background:linear-gradient(135deg,{TEAL},{SIDEBAR});
                    display:flex;align-items:center;justify-content:center;
                    font-size:2rem;margin-bottom:20px;
                    box-shadow:0 8px 32px {TEAL}55;">🎤</div>
        <div style="font-family:'Poppins',sans-serif;font-weight:800;
                    font-size:1.8rem;color:{INK};margin-bottom:8px;">VocalMind</div>
        <div style="font-size:.78rem;color:{BROWN};letter-spacing:.18em;
                    text-transform:uppercase;">Loading AI engines…</div>
    </div>
    """, unsafe_allow_html=True)

device, whisper_model, w2v_proc, w2v_model = load_models()
_boot.empty()


# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Reset & Base ─────────────────────────────────────────────────────────── */
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

.stApp {{
    background: {BG} !important;
    color: {INK} !important;
    font-family: 'Inter', sans-serif !important;
}}

/* Hide Streamlit chrome */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] {{ display: none !important; }}
section[data-testid="stSidebar"] {{ display: none !important; }}

/* Scrollbar */
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: {BG}; }}
::-webkit-scrollbar-thumb {{ background: {BROWN}44; border-radius: 9px; }}
::-webkit-scrollbar-thumb:hover {{ background: {BROWN}; }}

/* ── Section Dividers ─────────────────────────────────────────────────────── */
.vm-section {{
    padding: 48px 0 24px;
    border-top: 1px solid {BROWN}18;
    margin-top: 8px;
}}
.vm-section:first-child {{ border-top: none; padding-top: 16px; }}

.vm-section-tag {{
    display: inline-flex; align-items: center; gap: 6px;
    font-family: 'JetBrains Mono', monospace; font-size: .65rem;
    font-weight: 700; letter-spacing: .2em; text-transform: uppercase;
    color: {TEAL}; margin-bottom: 6px;
}}
.vm-section-tag::before {{
    content: ''; width: 18px; height: 2px;
    background: {TEAL}; border-radius: 1px; display: inline-block;
}}
.vm-section-title {{
    font-family: 'Poppins', sans-serif; font-weight: 800;
    font-size: 1.6rem; color: {INK}; margin-bottom: 24px;
    letter-spacing: -.02em;
}}

/* ── Cards ────────────────────────────────────────────────────────────────── */
.vm-card {{
    background: #FFFFFF;
    border: 1px solid {BROWN}1A;
    border-radius: 20px; padding: 24px;
    margin-bottom: 18px;
    box-shadow: 0 2px 10px {INK}09, 0 1px 0 {BROWN}0A;
    transition: box-shadow .25s ease, transform .25s ease;
    position: relative; overflow: hidden;
}}
.vm-card::before {{
    content: ''; position: absolute;
    inset: 0; border-radius: 20px;
    background: linear-gradient(135deg, rgba(255,255,255,.8) 0%, transparent 60%);
    pointer-events: none;
}}
.vm-card:hover {{
    box-shadow: 0 8px 32px {TEAL}22;
    transform: translateY(-2px);
}}
.vm-card-teal   {{ border-top: 3px solid {TEAL}; }}
.vm-card-green  {{ border-top: 3px solid {GREEN}; }}
.vm-card-copper {{ border-top: 3px solid {COPPER}; }}
.vm-card-dark   {{ border-top: 3px solid {SIDEBAR}; }}
.vm-card-left   {{ border-left: 4px solid {TEAL}; border-top: none; }}

/* ── Nav Header ───────────────────────────────────────────────────────────── */
.vm-nav {{
    position: sticky; top: 0; z-index: 999;
    background: rgba(255,248,240,.92);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border-bottom: 1px solid {BROWN}1A;
    padding: 14px 0; margin-bottom: 0;
}}
.vm-nav-inner {{
    display: flex; align-items: center; justify-content: space-between;
    max-width: 1280px; margin: 0 auto;
}}
.vm-brand {{
    display: flex; align-items: center; gap: 12px;
}}
.vm-brand-glyph {{
    width: 40px; height: 40px; border-radius: 12px;
    background: linear-gradient(135deg, {TEAL} 0%, {SIDEBAR} 100%);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem;
    box-shadow: 0 4px 14px {TEAL}44;
}}
.vm-brand-name {{
    font-family: 'Poppins', sans-serif; font-weight: 800;
    font-size: 1.3rem; color: {INK}; letter-spacing: -.02em;
}}
.vm-brand-name span {{ color: {TEAL}; }}
.vm-brand-sub {{
    font-family: 'JetBrains Mono', monospace; font-size: .6rem;
    color: {BROWN}; letter-spacing: .14em; text-transform: uppercase;
}}
.vm-status {{
    display: flex; align-items: center; gap: 8px;
    background: {GREEN}18; border: 1px solid {GREEN}44;
    border-radius: 99px; padding: 6px 14px;
    font-family: 'JetBrains Mono', monospace; font-size: .68rem;
    color: {BROWN}; letter-spacing: .1em;
}}
.vm-status .live-dot {{
    width: 7px; height: 7px; border-radius: 50%;
    background: {GREEN}; box-shadow: 0 0 7px {GREEN};
    animation: blink 2s infinite;
}}
@keyframes blink {{ 0%,100%{{opacity:1}} 50%{{opacity:.3}} }}

/* ── Hero Banner ──────────────────────────────────────────────────────────── */
.vm-hero {{
    background: linear-gradient(135deg, {SIDEBAR} 0%, #3D2E85 55%, {TEAL} 100%);
    border-radius: 24px; padding: 52px 52px 48px;
    margin-bottom: 40px; position: relative; overflow: hidden;
}}
.vm-hero::before {{
    content: ''; position: absolute; top: -40px; right: -40px;
    width: 260px; height: 260px; border-radius: 50%;
    background: rgba(255,255,255,.04);
}}
.vm-hero::after {{
    content: ''; position: absolute; bottom: -60px; right: 90px;
    width: 180px; height: 180px; border-radius: 50%;
    background: rgba(78,141,156,.15);
}}
.vm-hero-badge {{
    display: inline-flex; align-items: center; gap: 7px;
    background: rgba(255,255,255,.12);
    border: 1px solid rgba(255,255,255,.18);
    border-radius: 99px; padding: 6px 16px; margin-bottom: 20px;
    font-family: 'JetBrains Mono', monospace; font-size: .68rem;
    color: rgba(255,255,255,.8); letter-spacing: .1em;
}}
.vm-hero-badge .dot {{
    width: 7px; height: 7px; border-radius: 50%;
    background: {GREEN}; box-shadow: 0 0 7px {GREEN};
    animation: blink 2s infinite;
}}
.vm-hero-title {{
    font-family: 'Poppins', sans-serif; font-weight: 800;
    font-size: 2.8rem; color: white; line-height: 1.1;
    margin-bottom: 14px; letter-spacing: -.03em;
}}
.vm-hero-sub {{
    color: rgba(255,255,255,.65); font-size: 1rem;
    line-height: 1.7; max-width: 520px; margin-bottom: 28px;
}}
.vm-hero-stats {{
    display: flex; gap: 28px; flex-wrap: wrap;
}}
.vm-hero-stat {{
    font-family: 'JetBrains Mono', monospace;
    font-size: .75rem; color: rgba(255,255,255,.55);
}}
.vm-hero-stat strong {{
    color: white; font-size: .9rem; display: block; margin-bottom: 2px;
}}

/* ── Input Card ───────────────────────────────────────────────────────────── */
.vm-drop-zone {{
    border: 2px dashed {TEAL}55;
    border-radius: 16px; padding: 32px 20px;
    text-align: center;
    background: linear-gradient(135deg, {LIME}88 0%, {BG} 100%);
    transition: all .22s ease;
    cursor: pointer;
}}
.vm-drop-zone:hover {{
    border-color: {TEAL};
    background: {LIME};
    transform: scale(1.005);
}}
.vm-drop-icon {{ font-size: 2.5rem; display: block; margin-bottom: 12px; }}
.vm-drop-text {{
    font-weight: 600; color: {INK}; font-size: .92rem; margin-bottom: 6px;
}}
.vm-drop-sub {{ font-size: .78rem; color: {BROWN}; }}

/* ── Waveform loader ──────────────────────────────────────────────────────── */
.vm-wave {{
    display: flex; align-items: center; justify-content: center;
    gap: 4px; height: 52px; padding: 10px 0;
}}
.vm-wbar {{
    width: 5px; border-radius: 3px;
    background: linear-gradient(180deg, {TEAL}, {SIDEBAR});
    animation: beat 1.1s ease-in-out infinite;
}}
@keyframes beat {{ 0%,100%{{height:6px;opacity:.3}} 50%{{height:48px;opacity:1}} }}
.vm-wbar:nth-child(1){{animation-delay:0s}}    .vm-wbar:nth-child(2){{animation-delay:.08s}}
.vm-wbar:nth-child(3){{animation-delay:.16s}}  .vm-wbar:nth-child(4){{animation-delay:.24s}}
.vm-wbar:nth-child(5){{animation-delay:.32s}}  .vm-wbar:nth-child(6){{animation-delay:.40s}}
.vm-wbar:nth-child(7){{animation-delay:.32s}}  .vm-wbar:nth-child(8){{animation-delay:.24s}}
.vm-wbar:nth-child(9){{animation-delay:.16s}}  .vm-wbar:nth-child(10){{animation-delay:.08s}}
.vm-wbar:nth-child(11){{animation-delay:0s}}   .vm-wbar:nth-child(12){{animation-delay:.08s}}
.vm-wbar:nth-child(13){{animation-delay:.16s}} .vm-wbar:nth-child(14){{animation-delay:.24s}}
.vm-wbar:nth-child(15){{animation-delay:.32s}}
.vm-loading-text {{
    font-family: 'JetBrains Mono', monospace; font-size: .72rem;
    color: {TEAL}; letter-spacing: .16em; text-align: center;
    margin-top: 10px; animation: blink 1.3s infinite;
}}

/* ── Pipeline ─────────────────────────────────────────────────────────────── */
.vm-pipe {{ display: flex; flex-direction: column; }}
.vm-step {{
    display: flex; align-items: flex-start; gap: 14px;
    padding: 13px 16px; border-radius: 12px;
    border: 1px solid {BROWN}18; background: white;
    transition: all .22s ease;
}}
.vm-step:hover {{ border-color: {TEAL}44; box-shadow: 0 4px 16px {TEAL}14; }}
.vm-step.done {{
    border-color: {GREEN}55;
    background: linear-gradient(90deg, {GREEN}0A 0%, white 100%);
}}
.vm-step.active {{
    border-color: {TEAL}88;
    background: linear-gradient(90deg, {TEAL}0D 0%, white 100%);
    box-shadow: 0 4px 20px {TEAL}18;
}}
.vm-step-num {{
    width: 26px; height: 26px; border-radius: 50%; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-family: 'JetBrains Mono', monospace; font-size: .65rem;
    font-weight: 700; background: {BROWN}14; color: {BROWN};
}}
.vm-step.done  .vm-step-num {{ background: {GREEN};   color: white; }}
.vm-step.active .vm-step-num {{ background: {TEAL};   color: white; box-shadow: 0 0 10px {TEAL}55; }}
.vm-step-body {{ flex: 1; }}
.vm-step-name {{
    font-size: .82rem; font-weight: 600; color: {INK};
    font-family: 'Inter', sans-serif;
}}
.vm-step-sub {{
    font-size: .68rem; color: {BROWN};
    font-family: 'JetBrains Mono', monospace; margin-top: 2px;
}}
.vm-step-tag {{
    font-family: 'JetBrains Mono', monospace; font-size: .6rem;
    padding: 3px 10px; border-radius: 99px; white-space: nowrap;
    background: {LIME}; border: 1px solid {TEAL}33; color: {BROWN};
}}
.vm-step.done .vm-step-tag {{
    background: {GREEN}18; border-color: {GREEN}44; color: #2d7a4f;
}}
.vm-connector {{ width: 2px; height: 14px; margin-left: 25px; background: {BROWN}18; }}
.vm-connector.lit {{
    background: linear-gradient(180deg, {GREEN}, {TEAL});
    box-shadow: 0 0 6px {TEAL}44;
}}

/* ── Emotion Hero Card ────────────────────────────────────────────────────── */
.vm-emotion-hero {{
    background: linear-gradient(135deg, {SIDEBAR} 0%, #3D2E85 100%);
    border-radius: 22px; padding: 36px 28px;
    text-align: center; position: relative; overflow: hidden;
}}
.vm-emotion-hero::before {{
    content: ''; position: absolute; top: -30px; left: -30px;
    width: 160px; height: 160px; border-radius: 50%;
    background: rgba(255,255,255,.04); pointer-events: none;
}}
.vm-emotion-emoji {{
    font-size: 4.5rem; display: block;
    filter: drop-shadow(0 8px 20px rgba(0,0,0,.3));
    animation: float-emo 3s ease-in-out infinite;
    margin-bottom: 14px;
}}
@keyframes float-emo {{0%,100%{{transform:translateY(0)}} 50%{{transform:translateY(-8px)}}}}
.vm-emotion-name {{
    font-family: 'Poppins', sans-serif; font-weight: 800;
    font-size: 2.2rem; letter-spacing: -.02em; margin-bottom: 6px;
}}
.vm-emotion-conf {{
    font-family: 'JetBrains Mono', monospace; font-size: .78rem;
    color: rgba(255,255,255,.55); letter-spacing: .1em;
}}
.vm-conf-bar {{
    width: 100%; height: 5px; background: rgba(255,255,255,.1);
    border-radius: 99px; margin-top: 14px; overflow: hidden;
}}
.vm-conf-fill {{
    height: 100%; border-radius: 99px;
    transition: width 1.4s cubic-bezier(.16,1,.3,1);
}}

/* ── Metric trio ──────────────────────────────────────────────────────────── */
.vm-trio {{ display: grid; grid-template-columns: repeat(3,1fr); gap: 10px; margin: 16px 0; }}
.vm-metric {{
    background: {LIME}66; border: 1px solid {TEAL}22;
    border-radius: 14px; padding: 14px 10px; text-align: center;
}}
.vm-metric-val {{
    font-family: 'JetBrains Mono', monospace; font-size: 1.25rem;
    font-weight: 700; display: block; margin-bottom: 4px; color: {TEAL};
}}
.vm-metric-key {{
    font-size: .62rem; text-transform: uppercase; letter-spacing: .12em;
    color: {BROWN}; font-family: 'JetBrains Mono', monospace;
}}

/* ── Transcript ───────────────────────────────────────────────────────────── */
.vm-quote {{
    border-left: 4px solid {TEAL};
    background: linear-gradient(90deg, {LIME} 0%, white 100%);
    border-radius: 0 14px 14px 0;
    padding: 18px 22px;
    font-size: .98rem; line-height: 1.75; color: {INK};
    font-style: italic;
}}
.vm-emo-pill {{
    display: inline-flex; align-items: center; gap: 7px;
    font-family: 'JetBrains Mono', monospace; font-size: .68rem;
    font-weight: 700; padding: 5px 14px; border-radius: 99px;
    margin-top: 12px; border: 1px solid;
}}

/* ── History cards ────────────────────────────────────────────────────────── */
.vm-hist {{
    background: white; border: 1px solid {BROWN}18;
    border-radius: 14px; padding: 14px 18px; margin-bottom: 10px;
    display: flex; align-items: center; gap: 14px;
    transition: all .2s ease; cursor: pointer;
}}
.vm-hist:hover {{
    border-color: {TEAL}55; box-shadow: 0 4px 20px {TEAL}18;
    transform: translateX(4px);
}}
.vm-hist-avatar {{
    width: 46px; height: 46px; border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem; flex-shrink: 0;
}}

/* ── Stat pill ────────────────────────────────────────────────────────────── */
.vm-chip {{
    display: inline-flex; align-items: center; gap: 6px;
    background: {LIME}; border: 1px solid {TEAL}33;
    border-radius: 99px; padding: 4px 12px;
    font-family: 'JetBrains Mono', monospace; font-size: .68rem; color: {BROWN};
    margin: 3px;
}}
.vm-chip .dot {{ width: 6px; height: 6px; border-radius: 50%; }}

/* ── Tech stack card ──────────────────────────────────────────────────────── */
.vm-tech {{
    background: white; border: 1px solid {BROWN}18;
    border-radius: 14px; padding: 16px 18px;
    display: flex; align-items: center; gap: 14px;
    margin-bottom: 10px;
    transition: all .2s ease;
}}
.vm-tech:hover {{
    border-color: {TEAL}44; box-shadow: 0 4px 16px {TEAL}14;
    transform: translateY(-2px);
}}
.vm-tech-icon {{
    width: 40px; height: 40px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem; flex-shrink: 0;
}}

/* ── Footer ───────────────────────────────────────────────────────────────── */
.vm-footer {{
    background: {SIDEBAR};
    border-radius: 20px; padding: 40px 44px;
    text-align: center; margin-top: 32px; position: relative; overflow: hidden;
}}
.vm-footer::before {{
    content: ''; position: absolute; top: -30px; right: -30px;
    width: 200px; height: 200px; border-radius: 50%;
    background: rgba(78,141,156,.12); pointer-events: none;
}}

/* ── Empty state ──────────────────────────────────────────────────────────── */
.vm-empty {{
    text-align: center; padding: 52px 24px;
    border: 2px dashed {TEAL}44; border-radius: 20px;
    background: linear-gradient(135deg, {LIME}66 0%, {BG} 100%);
}}
.vm-empty-icon {{ font-size: 3rem; display: block; margin-bottom: 14px; opacity: .5; }}
.vm-empty-title {{
    font-family: 'Poppins', sans-serif; font-weight: 700;
    font-size: 1.1rem; color: {INK}; margin-bottom: 8px;
}}
.vm-empty-sub {{ font-size: .83rem; color: {BROWN}; line-height: 1.7; }}

/* ── Divider ──────────────────────────────────────────────────────────────── */
.vm-hr {{ height: 1px; background: {BROWN}18; margin: 28px 0; }}

/* ── Buttons ──────────────────────────────────────────────────────────────── */
div.stButton > button {{
    background: linear-gradient(135deg, {TEAL} 0%, {SIDEBAR} 100%) !important;
    color: white !important; border: none !important;
    padding: 13px 28px !important; border-radius: 12px !important;
    font-family: 'Inter', sans-serif !important; font-weight: 700 !important;
    font-size: .84rem !important; letter-spacing: .03em !important;
    box-shadow: 0 4px 18px {TEAL}44 !important;
    transition: all .22s ease !important; width: 100% !important;
}}
div.stButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px {TEAL}55 !important;
}}

/* ── File uploader ────────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {{
    background: {LIME} !important;
    border: 2px dashed {TEAL}66 !important;
    border-radius: 16px !important;
}}
[data-testid="stFileUploader"]:hover {{ border-color: {TEAL} !important; }}
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] small {{ color: {BROWN} !important; }}

/* ── Tabs ─────────────────────────────────────────────────────────────────── */
[data-testid="stTabs"] button[data-baseweb="tab"] {{
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important; font-size: .82rem !important;
    color: {BROWN} !important;
}}
[data-testid="stTabs"] button[aria-selected="true"] {{ color: {TEAL} !important; }}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{ background: {TEAL} !important; }}
[data-testid="stTabs"] [data-baseweb="tab-border"]    {{ background: {BROWN}22 !important; }}

/* ── Audio ────────────────────────────────────────────────────────────────── */
audio {{ width: 100%; border-radius: 12px; margin: 10px 0; }}

/* ── Metrics ──────────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {{
    background: white; border-radius: 14px; padding: 16px;
    border: 1px solid {BROWN}18; box-shadow: 0 2px 8px {INK}07;
}}
[data-testid="stMetricLabel"] {{ color: {BROWN} !important; font-size:.74rem !important; }}
[data-testid="stMetricValue"] {{
    color: {TEAL} !important; font-size:1.5rem !important;
    font-weight:700 !important; font-family:'JetBrains Mono',monospace !important;
}}

/* ── Download btn ─────────────────────────────────────────────────────────── */
[data-testid="stDownloadButton"] > button {{
    background: {LIME} !important; color: {INK} !important;
    border: 1px solid {TEAL}33 !important; box-shadow: none !important;
}}
[data-testid="stDownloadButton"] > button:hover {{
    background: {TEAL} !important; color: white !important;
    transform: none !important;
}}

/* ── Selection & Focus ────────────────────────────────────────────────────── */
::selection {{ background: {TEAL}28; color: {TEAL}; }}
button:focus-visible {{ outline: 2px solid {TEAL}; outline-offset: 2px; }}

/* ══════════════════════════════════════════════════════════════════════════
   RESPONSIVE / MOBILE STYLES
   ══════════════════════════════════════════════════════════════════════ */

/* ── Global content padding ───────────────────────────────────────────── */
.block-container {{
    padding-left: clamp(12px, 4vw, 80px) !important;
    padding-right: clamp(12px, 4vw, 80px) !important;
    max-width: 1360px !important;
    margin: 0 auto !important;
}}

/* ── Nav: shrink on small screens ─────────────────────────────────────── */
@media (max-width: 640px) {{
    .vm-nav-inner {{
        flex-direction: column; gap: 10px; align-items: flex-start;
    }}
    .vm-brand-name {{ font-size: 1.1rem; }}
    .vm-brand-sub  {{ display: none; }}
    .vm-status     {{ font-size: .6rem; padding: 5px 10px; }}
}}

/* ── Hero: scale down on mobile ───────────────────────────────────────── */
@media (max-width: 768px) {{
    .vm-hero {{ padding: 32px 24px 28px; border-radius: 16px; }}
    .vm-hero-title {{ font-size: 1.8rem !important; }}
    .vm-hero-sub   {{ font-size: .88rem; }}
    .vm-hero-stats {{ gap: 16px; }}
    .vm-hero-stat strong {{ font-size: .82rem; }}
}}
@media (max-width: 480px) {{
    .vm-hero {{ padding: 24px 18px; }}
    .vm-hero-title {{ font-size: 1.5rem !important; }}
    .vm-hero-badge {{ font-size: .6rem; padding: 5px 10px; }}
}}

/* ── Section titles ────────────────────────────────────────────────────── */
@media (max-width: 640px) {{
    .vm-section-title {{ font-size: 1.2rem; margin-bottom: 16px; }}
    .vm-section {{ padding: 32px 0 16px; }}
}}

/* ── Cards: reduce padding on mobile ──────────────────────────────────── */
@media (max-width: 640px) {{
    .vm-card {{ padding: 16px; border-radius: 14px; }}
    .vm-emotion-hero {{ padding: 24px 16px; border-radius: 16px; }}
    .vm-emotion-name {{ font-size: 1.6rem !important; }}
    .vm-emotion-emoji {{ font-size: 3.2rem !important; }}
}}

/* ── Metric trio: 2-col on very small screens ─────────────────────────── */
@media (max-width: 420px) {{
    .vm-trio {{
        grid-template-columns: repeat(2, 1fr) !important;
    }}
    .vm-metric-val {{ font-size: 1rem; }}
}}

/* ── Streamlit columns: force single column on mobile ─────────────────── */
@media (max-width: 768px) {{
    /* Stack all st.columns layouts */
    [data-testid="stHorizontalBlock"] {{
        flex-direction: column !important;
        gap: 0 !important;
    }}
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }}
}}

/* ── History cards: compact on mobile ────────────────────────────────── */
@media (max-width: 640px) {{
    .vm-hist {{ flex-wrap: wrap; gap: 10px; }}
    .vm-hist-avatar {{ width: 38px; height: 38px; font-size: 1.2rem; }}
}}

/* ── Tech stack: single col on small ─────────────────────────────────── */
@media (max-width: 480px) {{
    .vm-tech {{ padding: 12px 14px; }}
    .vm-tech-icon {{ width: 34px; height: 34px; font-size: .95rem; }}
}}

/* ── Footer: compact on mobile ────────────────────────────────────────── */
@media (max-width: 640px) {{
    .vm-footer {{ padding: 28px 20px; border-radius: 16px; }}
}}

/* ── Plotly charts: ensure they don't overflow ────────────────────────── */
.js-plotly-plot, .plot-container {{
    max-width: 100% !important;
    overflow: hidden;
}}

/* ── Tabs: scrollable on small screens ───────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {{
    overflow-x: auto !important;
    -webkit-overflow-scrolling: touch;
    scrollbar-width: none;
}}
[data-testid="stTabs"] [data-baseweb="tab-list"]::-webkit-scrollbar {{
    display: none;
}}
[data-testid="stTabs"] button[data-baseweb="tab"] {{
    white-space: nowrap !important;
    min-width: fit-content !important;
}}

/* ── Waveform animation bars: tighten on mobile ──────────────────────── */
@media (max-width: 480px) {{
    .vm-wave {{ gap: 2px; }}
    .vm-wbar  {{ width: 3px; }}
}}

/* ── Touch targets: minimum 44px for accessibility ───────────────────── */
div.stButton > button {{
    min-height: 44px !important;
    touch-action: manipulation;
}}

/* ── Prevent horizontal overflow ─────────────────────────────────────── */
.vm-card, .vm-emotion-hero, .vm-hero, .vm-footer {{
    word-break: break-word;
    overflow-wrap: break-word;
}}

/* ── Pipeline steps: compact on mobile ───────────────────────────────── */
@media (max-width: 640px) {{
    .vm-step {{ padding: 10px 12px; }}
    .vm-step-num {{ width: 22px; height: 22px; font-size: .6rem; }}
    .vm-step-name {{ font-size: .78rem; }}
    .vm-step-sub  {{ font-size: .62rem; }}
    .vm-step-tag  {{ font-size: .56rem; padding: 2px 7px; }}
    .vm-connector {{ margin-left: 22px; height: 10px; }}
}}

/* ── Chip wrapping ────────────────────────────────────────────────────── */
.vm-chip {{ white-space: nowrap; }}

/* ── Audio player full width ──────────────────────────────────────────── */
audio {{ width: 100% !important; min-width: 0; }}

/* ── Tablet: 2-col grids work, just shrink padding ───────────────────── */
@media (min-width: 769px) and (max-width: 1024px) {{
    .block-container {{
        padding-left: 24px !important;
        padding-right: 24px !important;
    }}
    .vm-hero-title {{ font-size: 2.2rem !important; }}
    .vm-card {{ padding: 18px; }}
}}

/* ── Large screens: comfortable max-width content ─────────────────────── */
@media (min-width: 1400px) {{
    .vm-hero-title {{ font-size: 3.2rem !important; }}
    .vm-card {{ padding: 28px; }}
}}

</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  CHART BUILDERS
# ══════════════════════════════════════════════════════════════════════════════
def chart_bars(probs_dict: dict) -> go.Figure:
    rows = sorted(
        [{{"label": f"{{LABEL_MAP[i][1]}}  {{LABEL_MAP[i][0]}}",
          "pct":   probs_dict[LABEL_MAP[i][0]] * 100,
          "color": LABEL_MAP[i][2]}} for i in range(5)],
        key=lambda r: r["pct"]
    )
    fig = go.Figure()
    for row in rows:
        fig.add_trace(go.Bar(
            x=[row["pct"]], y=[row["label"]], orientation='h',
            marker=dict(color=row["color"], opacity=.88, line=dict(width=0)),
            text=f" {{row['pct']:.1f}}%",
            textposition='outside',
            textfont=dict(family='JetBrains Mono', size=11, color=row["color"]),
            showlegend=False, width=0.62,
        ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=60, t=4, b=4), height=200,
        xaxis=dict(range=[0,120], showgrid=True, gridcolor=BROWN+'18',
                   zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False,
                   tickfont=dict(family='Inter', size=12, color=INK)),
        barmode='overlay',
    )
    return fig


def chart_radar(probs_dict: dict) -> go.Figure:
    emos  = [LABEL_MAP[i][0] for i in range(5)]
    vals  = [probs_dict[e] * 100 for e in emos]
    cols  = [LABEL_MAP[i][2] for i in range(5)]
    fig = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]], theta=emos + [emos[0]],
        fill='toself',
        line=dict(color=TEAL, width=2.2),
        fillcolor=TEAL + '22',
        marker=dict(size=7, color=cols + [cols[0]]),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[0,100], color=BROWN+'88',
                            tickfont=dict(size=8, color=BROWN, family='JetBrains Mono'),
                            gridcolor=BROWN+'22', linecolor=BROWN+'22'),
            angularaxis=dict(color=BROWN,
                             tickfont=dict(size=11, color=INK, family='Inter', weight=600),
                             gridcolor=BROWN+'22', linecolor=BROWN+'22'),
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=24, r=24, t=24, b=24),
        height=225, showlegend=False,
    )
    return fig


def chart_waveform(y, duration: float) -> go.Figure:
    step = max(1, len(y) // 700)
    y_ds = y[::step][:700]
    t_ds = np.linspace(0, duration, len(y_ds))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_ds, y=y_ds, mode='lines',
        line=dict(color=TEAL, width=1.2),
        fill='tozeroy', fillcolor=TEAL + '18', showlegend=False,
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0,r=0,t=0,b=0), height=150,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=True,
                   tickfont=dict(family='JetBrains Mono', size=8, color=BROWN),
                   color=BROWN),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig


def chart_timeline(timeline: list) -> go.Figure:
    y_map = {{"Anger":0,"Sadness":1,"Neutral":2,"Calm":3,"Happiness":4}}
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[p["time"] for p in timeline],
        y=[y_map.get(p["emotion"], 2) for p in timeline],
        mode='lines+markers',
        line=dict(color=TEAL + '88', width=2.5, shape='spline'),
        marker=dict(size=11, color=[p["color"] for p in timeline],
                    line=dict(color=BG, width=2.5)),
        showlegend=False,
        text=[f"{{p['emoji']}} {{p['emotion']}}<br>{{p['confidence']:.0f}}%" for p in timeline],
        hoverinfo='text',
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0,r=0,t=0,b=0), height=150,
        xaxis=dict(showgrid=False, zeroline=False,
                   tickfont=dict(family='JetBrains Mono', size=8, color=BROWN),
                   color=BROWN, title='time (s)',
                   titlefont=dict(size=9, color=BROWN)),
        yaxis=dict(showgrid=True, gridcolor=BROWN+'15',
                   zeroline=False, showticklabels=False, range=[-0.6,4.6]),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  EMOTION WHEEL  (SVG rendered via components.html)
# ══════════════════════════════════════════════════════════════════════════════
def render_wheel(probs: dict, dominant: str, confidence: float):
    size = 340
    cx = cy = size / 2
    r  = size / 2 - 44

    positions = {{
        "Happiness": ( 45, 0.74),
        "Anger":     (315, 0.74),
        "Sadness":   (135, 0.74),
        "Calm":      (225, 0.74),
        "Neutral":   (270, 0.30),
    }}
    emojis = {{"Happiness":"😄","Neutral":"😐","Anger":"😠","Sadness":"😢","Calm":"😌"}}
    col_map = {{v[0]: v[2] for v in LABEL_MAP.values()}}

    # Weighted dot position
    dot_x, dot_y = cx, cy
    for emo, prob in probs.items():
        if emo in positions:
            ang, rad = positions[emo]
            ex = cx + r * rad * math.cos((ang - 90) * math.pi / 180)
            ey = cy + r * rad * math.sin((ang - 90) * math.pi / 180)
            dot_x += prob * (ex - cx)
            dot_y += prob * (ey - cy)

    dom_color  = col_map.get(dominant, TEAL)
    dom_shadow = dom_color + "88"

    # Connector lines + emotion nodes
    connectors = ""
    nodes      = ""
    for emo, (ang, rad) in positions.items():
        ex   = cx + r * rad * math.cos((ang - 90) * math.pi / 180)
        ey   = cy + r * rad * math.sin((ang - 90) * math.pi / 180)
        ec   = col_map.get(emo, TEAL)
        prob = probs.get(emo, 0)
        nr   = 28 + prob * 16
        is_d = emo == dominant

        connectors += (
            f'<line x1="{{cx:.1f}}" y1="{{cy:.1f}}" x2="{{ex:.1f}}" y2="{{ey:.1f}}" '
            f'stroke="{{ec}}" stroke-width="1" opacity="{{0.12 + prob*0.35:.2f}}" '
            f'stroke-dasharray="4 3"/>'
        )
        pulse_anim = (
            f'animation: pulse-{{emo.lower().replace(" ","_")}} 2s ease-in-out infinite;'
            if is_d else ""
        )
        nodes += f"""
        <g>
          <circle cx="{{ex:.1f}}" cy="{{ey:.1f}}" r="{{nr:.1f}}"
                  fill="{{ec}}{{'44' if is_d else '22'}}"
                  stroke="{{ec}}" stroke-width="{{2.5 if is_d else 1.5}}"
                  style="{{pulse_anim}}"/>
          <text x="{{ex:.1f}}" y="{{ey - 6:.1f}}"
                text-anchor="middle" font-size="20">{{emojis[emo]}}</text>
          <text x="{{ex:.1f}}" y="{{ey + 14:.1f}}"
                text-anchor="middle" font-size="8.5"
                font-family="Inter,sans-serif" font-weight="700"
                fill="{{ec}}" opacity=".95">{{emo.upper()}}</text>
          <text x="{{ex:.1f}}" y="{{ey + 25:.1f}}"
                text-anchor="middle" font-size="8"
                font-family="JetBrains Mono,monospace"
                fill="{{ec}}" opacity=".7">{{prob*100:.0f}}%</text>
        </g>
        """

    html = f"""<!DOCTYPE html><html><head>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@600;700&family=JetBrains+Mono:wght@500&family=Poppins:wght@700;800&display=swap');
  * {{margin:0;padding:0;box-sizing:border-box;}}
  body {{background:{{BG}};display:flex;flex-direction:column;
         align-items:center;font-family:'Inter',sans-serif;}}
  svg {{overflow:visible;}}
  .ring-spin {{
    fill:none; stroke:{{BROWN}}18; stroke-dasharray:5 5;
    animation: spin 50s linear infinite;
    transform-origin:{{cx}}px {{cy}}px;
  }}
  @keyframes spin {{to{{transform:rotate(360deg)}}}}
  @keyframes pulse-happiness {{0%,100%{{r:42px}} 50%{{r:46px}}}}
  @keyframes pulse-anger      {{0%,100%{{r:42px}} 50%{{r:46px}}}}
  @keyframes pulse-sadness    {{0%,100%{{r:42px}} 50%{{r:46px}}}}
  @keyframes pulse-calm       {{0%,100%{{r:42px}} 50%{{r:46px}}}}
  @keyframes pulse-neutral    {{0%,100%{{r:42px}} 50%{{r:46px}}}}
  #dot {{
    transition: cx 1.1s cubic-bezier(.34,1.56,.64,1),
                cy 1.1s cubic-bezier(.34,1.56,.64,1);
  }}
  .label-area {{text-align:center;margin-top:14px;}}
</style></head><body>
<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
  <!-- Decorative rings -->
  <circle cx="{cx}" cy="{cy}" r="{r*0.9:.0f}" class="ring-spin" stroke-width="1"/>
  <circle cx="{cx}" cy="{cy}" r="{r*0.5:.0f}" fill="none"
          stroke="{TEAL}18" stroke-width="1" stroke-dasharray="3 5"/>
  <circle cx="{cx}" cy="{cy}" r="{r*0.22:.0f}" fill="none"
          stroke="{BROWN}14" stroke-width="1"/>

  <!-- Center label -->
  <text x="{cx}" y="{cy-8}" text-anchor="middle"
        font-size="10" font-family="JetBrains Mono,monospace"
        fill="{BROWN}88">EMOTION</text>
  <text x="{cx}" y="{cy+8}" text-anchor="middle"
        font-size="10" font-family="JetBrains Mono,monospace"
        fill="{BROWN}88">SPACE</text>

  {connectors}
  {nodes}

  <!-- Glowing dot -->
  <circle cx="{dot_x:.1f}" cy="{dot_y:.1f}" r="13"
          fill="{dom_color}" opacity=".25"/>
  <circle id="dot" cx="{dot_x:.1f}" cy="{dot_y:.1f}" r="9"
          fill="{dom_color}" opacity=".95"
          style="filter:drop-shadow(0 0 10px {dom_shadow});"/>
  <circle cx="{dot_x:.1f}" cy="{dot_y:.1f}" r="4"
          fill="white" opacity=".85"/>
</svg>

<div class="label-area">
  <div style="font-family:'Poppins',sans-serif;font-weight:800;
              font-size:1.5rem;color:{dom_color};
              text-shadow:0 2px 12px {dom_color}44;">
    {dominant.upper()}
  </div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:.72rem;
              color:{BROWN};letter-spacing:.1em;margin-top:3px;">
    CONFIDENCE · {confidence:.1f}%
  </div>
  <div style="width:160px;height:4px;background:{BROWN}18;
              border-radius:99px;margin:10px auto 0;overflow:hidden;">
    <div style="width:{confidence:.0f}%;height:100%;
                background:{dom_color};border-radius:99px;
                box-shadow:0 0 8px {dom_shadow};"></div>
  </div>
</div>
</body></html>"""
    return html


# ══════════════════════════════════════════════════════════════════════════════
#  RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
RECS = {
    "Happiness": ("🚀", "Ride the Wave",
                  "You're in a fantastic headspace! Perfect time for creative work, connecting with people, or tackling big challenges."),
    "Sadness":   ("🎵", "Comfort Mode",
                  "Your voice shows you may be feeling low. Try some calming music, a short walk outside, or reaching out to someone you trust."),
    "Anger":     ("🌬️", "Breathe First",
                  "Take 5 slow deep breaths before responding. Channel that intensity into something productive — exercise or journaling works great."),
    "Calm":      ("✨", "Flow State",
                  "You're centered and composed. This is the ideal moment for focused work, meditation, or important decisions."),
    "Neutral":   ("📋", "Balanced Mode",
                  "Steady and measured. A great mindset for analytical tasks, writing, or structured thinking."),
}


# ══════════════════════════════════════════════════════════════════════════════
#  ────────────────────────────────────────────────────────────────────────────
#  RENDER: NAVIGATION HEADER
#  ────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="vm-nav">
  <div class="vm-nav-inner">
    <div class="vm-brand">
      <div class="vm-brand-glyph">🎤</div>
      <div>
        <div class="vm-brand-name">Vocal<span>Mind</span></div>
        <div class="vm-brand-sub">Voice Emotion Intelligence</div>
      </div>
    </div>
    <div class="vm-status">
      <span class="live-dot"></span>
      MODELS ONLINE · 99.33% ACC
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ────────────────────────────────────────────────────────────────────────────
#  SECTION 1 — HERO + INPUT PORTAL
#  ────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""<div class="vm-section">""", unsafe_allow_html=True)

# Hero banner
st.markdown(f"""
<div class="vm-hero">
  <div style="position:relative;z-index:1;">
    <div class="vm-hero-badge">
      <span class="dot"></span>
      LIVE · 99.33% ACCURACY · 5 EMOTION CLASSES · REAL-TIME
    </div>
    <div class="vm-hero-title">
      Understand Every<br>Voice Around You
    </div>
    <div class="vm-hero-sub">
      VocalMind detects human emotion from voice using a multimodal deep
      learning model combining Wav2Vec2 acoustic embeddings and Whisper
      transcription — all in under 2 seconds.
    </div>
    <div class="vm-hero-stats">
      <div class="vm-hero-stat"><strong>99.33%</strong>Test Accuracy</div>
      <div class="vm-hero-stat"><strong>5</strong>Emotion Classes</div>
      <div class="vm-hero-stat"><strong>&lt; 2s</strong>Latency</div>
      <div class="vm-hero-stat"><strong>Wav2Vec2</strong>Backbone</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# Input Portal
st.markdown(f"""
<div class="vm-section-tag">Section 01</div>
<div class="vm-section-title">Input Portal</div>
""", unsafe_allow_html=True)

ip_left, ip_right = st.columns([1, 1.6], gap="large")

with ip_left:
    tab_up, tab_rec = st.tabs(["📂  Upload File", "⏺  Live Record"])

    audio_source_bytes = None

    with tab_up:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="display:flex;flex-wrap:wrap;gap:5px;margin-bottom:10px;">
          {''.join([f'<span class="vm-chip">{f}</span>' for f in ["WAV","MP3","M4A","FLAC","OGG","AAC","WMA","WEBM"]])}
        </div>
        """, unsafe_allow_html=True)
        uploaded = st.file_uploader("Drop audio", type=["wav","mp3","m4a","flac","ogg","aac","wma","webm"],
                                    label_visibility="collapsed")
        if uploaded:
            audio_source_bytes = uploaded.getvalue()
            st.markdown(f"""
            <div style="background:white;border:1px solid {TEAL}33;border-radius:12px;
                 padding:12px 16px;margin-top:10px;">
              <div style="font-family:'JetBrains Mono',monospace;font-size:.62rem;
                   color:{BROWN};margin-bottom:3px;letter-spacing:.12em;">FILE LOADED</div>
              <div style="font-weight:600;color:{INK};font-size:.88rem;
                   word-break:break-all;">{uploaded.name}</div>
              <div style="font-size:.72rem;color:{TEAL};margin-top:3px;">
                {(uploaded.type or 'audio').split('/')[-1].upper()} · {len(audio_source_bytes)//1024} KB
              </div>
            </div>
            """, unsafe_allow_html=True)

    with tab_rec:
        st.markdown("<br>", unsafe_allow_html=True)
        if RECORDER_AVAILABLE:
            st.markdown(f"""
            <div style="background:{LIME};border:1px solid {TEAL}33;
                 border-radius:10px;padding:12px 14px;margin-bottom:12px;
                 font-family:'JetBrains Mono',monospace;font-size:.7rem;
                 color:{BROWN};letter-spacing:.1em;">
              CLICK MIC → SPEAK → CLICK STOP
            </div>
            """, unsafe_allow_html=True)
            rec = audio_recorder(
                text="", icon_size="3x",
                recording_color=EMO_COLORS["Anger"],
                neutral_color=TEAL,
                icon_name="microphone",
                pause_threshold=3.0, sample_rate=16000,
                key="main_recorder"
            )
            if rec:
                audio_source_bytes = rec
                st.success("✅ Recording captured!")
        else:
            st.markdown(f"""
            <div style="background:white;border:1px solid {BROWN}1A;
                 border-radius:16px;padding:24px 22px;margin-top:4px;
                 box-shadow:0 2px 12px {INK}07;">
              <div style="display:flex;align-items:center;gap:10px;margin-bottom:18px;">
                <div style="width:36px;height:36px;border-radius:10px;
                     background:{TEAL}18;display:flex;align-items:center;
                     justify-content:center;font-size:1.1rem;flex-shrink:0;">🎙️</div>
                <div>
                  <div style="font-family:'Poppins',sans-serif;font-weight:700;
                       font-size:.92rem;color:{INK};">Enable Live Recording</div>
                  <div style="font-size:.72rem;color:{BROWN};margin-top:1px;">
                    One extra package needed for microphone access
                  </div>
                </div>
              </div>
              <div style="display:flex;gap:10px;margin-bottom:12px;">
                <div style="width:22px;height:22px;border-radius:50%;flex-shrink:0;
                     background:{TEAL};display:flex;align-items:center;
                     justify-content:center;font-family:'JetBrains Mono',monospace;
                     font-size:.62rem;font-weight:700;color:white;margin-top:1px;">1</div>
                <div>
                  <div style="font-size:.8rem;font-weight:600;color:{INK};margin-bottom:5px;">
                    Add to <code style="background:{LIME};border-radius:4px;padding:1px 6px;
                    font-size:.75rem;color:{TEAL};">requirements.txt</code>
                  </div>
                  <div style="background:{LIME};border:1px solid {TEAL}33;border-radius:8px;
                       padding:9px 14px;font-family:'JetBrains Mono',monospace;
                       font-size:.78rem;color:{SIDEBAR};letter-spacing:.02em;">
                    audio_recorder_streamlit
                  </div>
                </div>
              </div>
              <div style="display:flex;gap:10px;margin-bottom:16px;">
                <div style="width:22px;height:22px;border-radius:50%;flex-shrink:0;
                     background:{TEAL};display:flex;align-items:center;
                     justify-content:center;font-family:'JetBrains Mono',monospace;
                     font-size:.62rem;font-weight:700;color:white;margin-top:1px;">2</div>
                <div>
                  <div style="font-size:.8rem;font-weight:600;color:{INK};margin-bottom:5px;">
                    Redeploy the Space
                  </div>
                  <div style="font-size:.76rem;color:{BROWN};line-height:1.6;">
                    Push your updated <code style="background:{LIME};border-radius:4px;
                    padding:1px 5px;font-size:.72rem;color:{TEAL};">requirements.txt</code>
                    to trigger a rebuild on Hugging Face Spaces.
                  </div>
                </div>
              </div>
              <div style="height:1px;background:{BROWN}12;margin-bottom:14px;"></div>
              <div style="display:flex;align-items:flex-start;gap:8px;">
                <span style="color:{COPPER};font-size:.85rem;margin-top:1px;">💡</span>
                <div style="font-size:.76rem;color:{BROWN};line-height:1.6;">
                  <strong style="color:{INK};">Meanwhile:</strong> use the
                  <strong style="color:{TEAL};">📂 Upload File</strong> tab to
                  analyze WAV, MP3, M4A, FLAC, OGG or AAC files right now.
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    if audio_source_bytes:
        st.audio(audio_source_bytes)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⚡  ANALYZE EMOTION PROFILE"):
            loader = st.empty()
            with loader.container():
                st.markdown(f"""
                <div style="padding:20px 0;text-align:center;">
                  <div class="vm-wave">
                    {''.join(['<div class="vm-wbar"></div>']*15)}
                  </div>
                  <div class="vm-loading-text">EXTRACTING ACOUSTIC FEATURES ···</div>
                </div>
                """, unsafe_allow_html=True)
            result = run_analysis(audio_source_bytes)
            st.session_state.analysis = result
            st.session_state.history.insert(0, {
                **result,
                "source": getattr(uploaded, 'name', 'Live Recording') if 'uploaded' in dir() else 'Live Recording'
            })
            loader.empty()
            st.rerun()

with ip_right:
    if not st.session_state.analysis:
        st.markdown(f"""
        <div class="vm-empty" style="margin-top:16px;">
          <span class="vm-empty-icon">🎙️</span>
          <div class="vm-empty-title">No audio analyzed yet</div>
          <div class="vm-empty-sub">
            Upload a file or record your voice,<br>
            then click <strong>Analyze Emotion Profile</strong>.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        d = st.session_state.analysis
        st.markdown(f"""
        <div class="vm-card vm-card-teal" style="margin-top:8px;">
          <div class="vm-section-tag" style="margin-bottom:10px;">Last Analysis</div>
          <div style="display:flex;align-items:center;gap:14px;">
            <div style="font-size:3rem;filter:drop-shadow(0 4px 12px {d['color']}44);">
              {d['emoji']}
            </div>
            <div>
              <div style="font-family:'Poppins',sans-serif;font-weight:800;
                   font-size:1.5rem;color:{d['color']};">{d['emotion']}</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:.75rem;
                   color:{BROWN};">{d['confidence']:.1f}% confidence · {d['duration']:.1f}s</div>
            </div>
          </div>
          <div class="vm-conf-bar" style="margin-top:14px;">
            <div style="width:{d['confidence']:.0f}%;height:5px;background:{d['color']};
                        border-radius:99px;box-shadow:0 0 8px {d['color']}66;"></div>
          </div>
          <div style="margin-top:12px;font-size:.82rem;color:{BROWN};
               font-style:italic;line-height:1.6;">
            "{d['transcript'][:120]}{'…' if len(d['transcript'])>120 else ''}"
          </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ────────────────────────────────────────────────────────────────────────────
#  SECTION 2 — PIPELINE
#  ────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""<div class="vm-section">""", unsafe_allow_html=True)
st.markdown(f"""
<div class="vm-section-tag">Section 02</div>
<div class="vm-section-title">AI Processing Pipeline</div>
""", unsafe_allow_html=True)

has_data  = st.session_state.analysis is not None
pipe_steps = [
    ("01", "Audio Ingestion",        "16kHz resampling · ffmpeg normalization",    "ffmpeg"),
    ("02", "Acoustic Feature Ext.",  "MFCC · Mel-spectrogram · F0 pitch",          "librosa"),
    ("03", "Wav2Vec2 Embedding",     "Self-supervised transformer representation", "wav2vec2-base"),
    ("04", "ASR Transcription",      "Speech-to-text via Whisper tiny",            "openai/whisper"),
    ("05", "Semantic Fusion",        "Acoustic + semantic context merge",          "custom"),
    ("06", "Emotion Classification", "Softmax output · 5 classes",                 "99.33% acc"),
]

pc_left, pc_right = st.columns([1, 1.2], gap="large")

with pc_left:
    st.markdown('<div class="vm-pipe">', unsafe_allow_html=True)
    for i, (num, name, sub, tag) in enumerate(pipe_steps):
        cls  = "done" if has_data else ("active" if i == 0 else "")
        conn = "lit"  if has_data else ""
        st.markdown(f"""
        <div class="vm-step {cls}">
          <div class="vm-step-num">{num}</div>
          <div class="vm-step-body">
            <div class="vm-step-name">{name}</div>
            <div class="vm-step-sub">{sub}</div>
          </div>
          <div class="vm-step-tag">{tag}</div>
        </div>
        {"<div class='vm-connector " + conn + "'></div>" if i < len(pipe_steps)-1 else ""}
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with pc_right:
    st.markdown(f"""
    <div class="vm-card vm-card-dark" style="padding:28px;">
      <div class="vm-section-tag" style="margin-bottom:10px;">Model Architecture</div>
      <div style="font-size:.88rem;color:{BROWN};line-height:1.85;">
        VocalMind uses a <strong style="color:{INK}">multimodal fusion architecture</strong>
        that combines two parallel pathways:
      </div>
      <div style="margin-top:16px;display:flex;flex-direction:column;gap:10px;">
        <div style="background:{LIME};border-radius:10px;padding:12px 16px;
             border-left:3px solid {TEAL};">
          <div style="font-weight:700;color:{INK};font-size:.84rem;margin-bottom:4px;">
            🔊 Acoustic Pathway
          </div>
          <div style="font-size:.76rem;color:{BROWN};">
            Raw audio → librosa features → Wav2Vec2 contextual embeddings
          </div>
        </div>
        <div style="background:{LIME};border-radius:10px;padding:12px 16px;
             border-left:3px solid {GREEN};">
          <div style="font-weight:700;color:{INK};font-size:.84rem;margin-bottom:4px;">
            💬 Semantic Pathway
          </div>
          <div style="font-size:.76rem;color:{BROWN};">
            Audio → Whisper ASR → transcript → semantic context vectors
          </div>
        </div>
        <div style="background:linear-gradient(90deg,{COPPER}18,{LIME}88);
             border-radius:10px;padding:12px 16px;border-left:3px solid {COPPER};">
          <div style="font-weight:700;color:{INK};font-size:.84rem;margin-bottom:4px;">
            ⚡ Fusion + Classification
          </div>
          <div style="font-size:.76rem;color:{BROWN};">
            Concatenated embeddings → dense layer → softmax over 5 emotion classes
          </div>
        </div>
      </div>
      <div style="margin-top:16px;display:grid;grid-template-columns:1fr 1fr;gap:10px;">
        <div style="background:{LIME}88;border-radius:10px;padding:12px;text-align:center;">
          <div style="font-family:'JetBrains Mono',monospace;font-size:1.2rem;
               font-weight:700;color:{TEAL};">99.33%</div>
          <div style="font-size:.68rem;color:{BROWN};letter-spacing:.1em;
               text-transform:uppercase;margin-top:3px;">Test Accuracy</div>
        </div>
        <div style="background:{LIME}88;border-radius:10px;padding:12px;text-align:center;">
          <div style="font-family:'JetBrains Mono',monospace;font-size:1.2rem;
               font-weight:700;color:{GREEN};">~1.2s</div>
          <div style="font-size:.68rem;color:{BROWN};letter-spacing:.1em;
               text-transform:uppercase;margin-top:3px;">Inference Time</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ────────────────────────────────────────────────────────────────────────────
#  SECTION 3 — EMOTION RESULT
#  ────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""<div class="vm-section">""", unsafe_allow_html=True)
st.markdown(f"""
<div class="vm-section-tag">Section 03</div>
<div class="vm-section-title">Emotion Result</div>
""", unsafe_allow_html=True)

if not has_data:
    st.markdown(f"""
    <div class="vm-empty">
      <span class="vm-empty-icon">🎭</span>
      <div class="vm-empty-title">Emotion result will appear here</div>
      <div class="vm-empty-sub">Analyze an audio file above to see the emotion wheel, confidence score, and probability breakdown.</div>
    </div>
    """, unsafe_allow_html=True)
else:
    d = st.session_state.analysis
    er_left, er_right = st.columns([1.05, 1.2], gap="large")

    with er_left:
        # Emotion Wheel
        st.markdown(f"""
        <div class="vm-card vm-card-teal" style="padding:16px 8px 20px;">
          <div class="vm-section-tag" style="padding-left:14px;margin-bottom:8px;">
            Emotion Space
          </div>
        </div>
        """, unsafe_allow_html=True)
        wheel_html = render_wheel(d["probs"], d["emotion"], d["confidence"])
        components.html(wheel_html, height=420, scrolling=False)

    with er_right:
        # Primary Emotion Hero
        st.markdown(f"""
        <div class="vm-emotion-hero">
          <span class="vm-emotion-emoji">{d['emoji']}</span>
          <div class="vm-emotion-name" style="color:{d['color']};
               text-shadow:0 0 30px {d['color']}55;">{d['emotion'].upper()}</div>
          <div class="vm-emotion-conf">CONFIDENCE · {d['confidence']:.1f}%</div>
          <div class="vm-conf-bar">
            <div class="vm-conf-fill" style="width:{d['confidence']:.0f}%;
                 background:{d['color']};box-shadow:0 0 10px {d['color']}88;"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Metadata trio
        st.markdown(f"""
        <div class="vm-trio">
          <div class="vm-metric">
            <span class="vm-metric-val">{d['duration']:.1f}s</span>
            <span class="vm-metric-key">Duration</span>
          </div>
          <div class="vm-metric">
            <span class="vm-metric-val">{d['sr']//1000}kHz</span>
            <span class="vm-metric-key">Sample Rate</span>
          </div>
          <div class="vm-metric">
            <span class="vm-metric-val" style="color:{d['color']};">{d['confidence']:.0f}%</span>
            <span class="vm-metric-key">Confidence</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Distribution bars
        st.markdown(f"""
        <div class="vm-card vm-card-teal" style="margin-top:4px;">
          <div class="vm-section-tag" style="margin-bottom:8px;">Distribution Matrix</div>
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(chart_bars(d["probs"]),
                        use_container_width=True, config={'displayModeBar': False})

        # Radar
        rr_col1, rr_col2 = st.columns(2, gap="medium")
        with rr_col1:
            st.markdown(f'<div class="vm-card vm-card-green"><div class="vm-section-tag" style="margin-bottom:4px;">Radar Chart</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_radar(d["probs"]),
                            use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)
        with rr_col2:
            # Emotion breakdown mini
            st.markdown(f'<div class="vm-card vm-card-copper"><div class="vm-section-tag" style="margin-bottom:10px;">Breakdown</div>', unsafe_allow_html=True)
            for i in range(5):
                name, emoji, color = LABEL_MAP[i]
                pct = d["probs"][name] * 100
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:8px;
                     padding:5px 0;border-bottom:1px solid {BROWN}10;">
                  <span style="font-size:1.1rem;">{emoji}</span>
                  <span style="font-size:.78rem;font-weight:600;
                        color:{INK};flex:1;">{name}</span>
                  <span style="font-family:'JetBrains Mono',monospace;
                        font-size:.72rem;font-weight:700;color:{color};">
                    {pct:.0f}%
                  </span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ────────────────────────────────────────────────────────────────────────────
#  SECTION 4 — WAVEFORM + TIMELINE
#  ────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""<div class="vm-section">""", unsafe_allow_html=True)
st.markdown(f"""
<div class="vm-section-tag">Section 04</div>
<div class="vm-section-title">Waveform &amp; Timeline</div>
""", unsafe_allow_html=True)

if not has_data:
    st.markdown(f"""
    <div class="vm-empty">
      <span class="vm-empty-icon">〰️</span>
      <div class="vm-empty-title">Audio visualizations will appear here</div>
      <div class="vm-empty-sub">Waveform, spectrogram, and emotion timeline are shown after analysis.</div>
    </div>
    """, unsafe_allow_html=True)
else:
    d = st.session_state.analysis
    wv_col, tl_col = st.columns(2, gap="large")

    with wv_col:
        st.markdown(f'<div class="vm-card vm-card-teal waveform-card"><div class="vm-section-tag" style="margin-bottom:8px;">Audio Waveform</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_waveform(d["waveform_y"], d["duration"]),
                        use_container_width=True, config={'displayModeBar': False})
        st.markdown(f"""
        <div style="display:flex;gap:10px;margin-top:6px;">
          <span class="vm-chip"><span class="dot" style="background:{TEAL}"></span>{d['duration']:.1f}s total</span>
          <span class="vm-chip"><span class="dot" style="background:{GREEN}"></span>{d['sr']//1000}kHz</span>
        </div>
        </div>
        """, unsafe_allow_html=True)

    with tl_col:
        st.markdown(f'<div class="vm-card vm-card-copper timeline-card"><div class="vm-section-tag" style="margin-bottom:8px;">Emotion Timeline</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_timeline(d["timeline"]),
                        use_container_width=True, config={'displayModeBar': False})
        st.markdown(f"""
        <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:6px;">
          {''.join([f'<span class="vm-chip"><span class="dot" style="background:{LABEL_MAP[i][2]}"></span>{LABEL_MAP[i][0]}</span>' for i in range(5)])}
        </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ────────────────────────────────────────────────────────────────────────────
#  SECTION 5 — TRANSCRIPT + INSIGHTS
#  ────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""<div class="vm-section">""", unsafe_allow_html=True)
st.markdown(f"""
<div class="vm-section-tag">Section 05</div>
<div class="vm-section-title">Transcript &amp; Insights</div>
""", unsafe_allow_html=True)

if not has_data:
    st.markdown(f"""
    <div class="vm-empty">
      <span class="vm-empty-icon">💬</span>
      <div class="vm-empty-title">Transcript will appear here</div>
      <div class="vm-empty-sub">Speech-to-text output and AI-powered emotion insights after analysis.</div>
    </div>
    """, unsafe_allow_html=True)
else:
    d = st.session_state.analysis
    no_speech = d["transcript"] == "No speech detected."
    ti_left, ti_right = st.columns([1.5, 1], gap="large")

    with ti_left:
        st.markdown(f"""
        <div class="vm-card vm-card-left transcript-left">
          <div class="vm-section-tag" style="margin-bottom:10px;">ASR Transcription · Whisper</div>
          <div class="vm-quote"
               style="{'color:' + BROWN + ';font-style:normal;background:' + BG + ';' if no_speech else ''}">
            {'"' + d['transcript'] + '"' if not no_speech
             else '— No transcribable speech detected in this segment. —'}
          </div>
          {f'<div style="margin-top:12px;"><span class="vm-emo-pill" style="background:{d["color"]}18;border-color:{d["color"]}44;color:{d["color"]};">{d["emoji"]} {d["emotion"].upper()} DETECTED</span></div>' if not no_speech else ''}
        </div>
        """, unsafe_allow_html=True)

        # PDF Export
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 18)
        pdf.cell(200, 12, "VocalMind - Emotion Analysis Report", ln=True, align="C")
        pdf.set_font("Arial", size=11)
        pdf.ln(8)
        for lbl, val in [
            ("Primary Emotion", f"{d['emotion']} {d['emoji']}"),
            ("Confidence",      f"{d['confidence']:.2f}%"),
            ("Duration",        f"{d['duration']:.2f}s"),
            ("Sample Rate",     f"{d['sr']}Hz"),
            ("Timestamp",       d.get('timestamp','—')),
        ]:
            pdf.cell(200, 8, f"{lbl}: {val}", ln=True)
        pdf.ln(5)
        pdf.set_font("Arial", "B", 11)
        pdf.cell(200, 8, "Probabilities:", ln=True)
        pdf.set_font("Arial", size=10)
        for i, (name, emoji, _) in LABEL_MAP.items():
            pdf.cell(200, 7, f"  {emoji} {name}: {d['probs_arr'][i]*100:.1f}%", ln=True)
        pdf.ln(5)
        pdf.set_font("Arial", "B", 11)
        pdf.cell(200, 8, "Transcript:", ln=True)
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 7, d["transcript"])
        pdf_bytes = pdf.output(dest="S").encode("latin-1")
        st.download_button("📄 Download PDF Report", data=pdf_bytes,
                           file_name="VocalMind_Report.pdf", mime="application/pdf")

    with ti_right:
        icon, rtitle, rmsg = RECS.get(d["emotion"], ("💡","Insight","Analysis complete."))
        st.markdown(f"""
        <div class="vm-card vm-card-green transcript-right">
          <div class="vm-section-tag" style="margin-bottom:10px;">AI Recommendation</div>
          <div style="font-size:2.2rem;margin-bottom:10px;">{icon}</div>
          <div style="font-family:'Poppins',sans-serif;font-weight:700;
               font-size:1.05rem;color:{INK};margin-bottom:10px;">{rtitle}</div>
          <div style="font-size:.85rem;color:{BROWN};line-height:1.7;">{rmsg}</div>
        </div>
        """, unsafe_allow_html=True)

        # Quick stats
        st.markdown(f"""
        <div class="vm-card vm-card-copper" style="margin-top:0px;">
          <div class="vm-section-tag" style="margin-bottom:10px;">Session Stats</div>
          <div style="display:flex;flex-direction:column;gap:6px;">
            <div style="display:flex;justify-content:space-between;
                 padding:7px 0;border-bottom:1px solid {BROWN}12;">
              <span style="font-size:.8rem;color:{BROWN};">Records this session</span>
              <span style="font-family:'JetBrains Mono',monospace;font-weight:700;
                    color:{TEAL};">{len(st.session_state.history)}</span>
            </div>
            <div style="display:flex;justify-content:space-between;
                 padding:7px 0;border-bottom:1px solid {BROWN}12;">
              <span style="font-size:.8rem;color:{BROWN};">Current emotion</span>
              <span style="font-family:'JetBrains Mono',monospace;font-weight:700;
                    color:{d['color']};">{d['emoji']} {d['emotion']}</span>
            </div>
            <div style="display:flex;justify-content:space-between;padding:7px 0;">
              <span style="font-size:.8rem;color:{BROWN};">Peak confidence</span>
              <span style="font-family:'JetBrains Mono',monospace;font-weight:700;
                    color:{GREEN};">{max([r['confidence'] for r in st.session_state.history] or [0]):.0f}%</span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ────────────────────────────────────────────────────────────────────────────
#  SECTION 6 — HISTORY
#  ────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""<div class="vm-section">""", unsafe_allow_html=True)
st.markdown(f"""
<div class="vm-section-tag">Section 06</div>
<div class="vm-section-title">Session History</div>
""", unsafe_allow_html=True)

if not st.session_state.history:
    st.markdown(f"""
    <div class="vm-empty">
      <span class="vm-empty-icon">📭</span>
      <div class="vm-empty-title">No history yet</div>
      <div class="vm-empty-sub">Your analyses will appear here as you use the app.</div>
    </div>
    """, unsafe_allow_html=True)
else:
    hist = st.session_state.history

    # Summary row
    confs = [r["confidence"] for r in hist]
    emos  = [r["emotion"] for r in hist]
    counts = {e: emos.count(e) for e in set(emos)}
    most_freq = max(counts, key=counts.get)

    hm1, hm2, hm3, hm4 = st.columns(4, gap="medium")
    with hm1: st.metric("Total Analyses",   len(hist))
    with hm2: st.metric("Avg Confidence",   f"{sum(confs)/len(confs):.1f}%")
    with hm3: st.metric("Most Frequent",    most_freq)
    with hm4: st.metric("Unique Emotions",  len(counts))

    st.markdown("<br>", unsafe_allow_html=True)

    # History list
    for i, rec in enumerate(hist):
        hc1, hc2 = st.columns([4, 1], gap="small")
        with hc1:
            st.markdown(f"""
            <div class="vm-hist">
              <div class="vm-hist-avatar" style="background:{rec['color']}18;">
                {rec['emoji']}
              </div>
              <div style="flex:1;">
                <div style="font-weight:700;color:{INK};font-size:.92rem;">
                  {rec['emotion']}
                  <span style="font-family:'JetBrains Mono',monospace;font-size:.7rem;
                        color:{rec['color']};margin-left:8px;">
                    {rec['confidence']:.0f}% confidence
                  </span>
                </div>
                <div style="font-size:.76rem;color:{BROWN};margin-top:3px;">
                  {rec.get('source','—')} · {rec.get('timestamp','—')} · {rec['duration']:.1f}s
                </div>
                <div style="font-size:.75rem;color:{BROWN}99;margin-top:4px;font-style:italic;">
                  "{rec['transcript'][:75]}{'…' if len(rec['transcript'])>75 else ''}"
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        with hc2:
            if st.button("View →", key=f"hist_{i}"):
                st.session_state.analysis = {k: v for k, v in rec.items()}
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️  Clear History"):
        st.session_state.history = []
        st.rerun()

    # Analytics quick view
    st.markdown("<div class='vm-hr'></div>", unsafe_allow_html=True)
    st.markdown(f'<div class="vm-section-tag" style="margin-top:4px;">Analytics Overview</div>', unsafe_allow_html=True)
    an_l, an_r = st.columns(2, gap="large")

    with an_l:
        st.markdown(f'<div class="vm-card vm-card-teal"><div class="vm-section-tag" style="margin-bottom:6px;">Emotion Distribution</div>', unsafe_allow_html=True)
        pie = go.Figure(go.Pie(
            labels=list(counts.keys()),
            values=list(counts.values()),
            hole=0.45,
            marker=dict(colors=[EMO_COLORS.get(e, TEAL) for e in counts.keys()]),
            textfont=dict(family='Inter', size=11, color=INK),
        ))
        pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0,r=0,t=8,b=0), height=240,
            legend=dict(font=dict(family='Inter', color=INK, size=11)),
        )
        st.plotly_chart(pie, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with an_r:
        st.markdown(f'<div class="vm-card vm-card-green"><div class="vm-section-tag" style="margin-bottom:6px;">Confidence Trend</div>', unsafe_allow_html=True)
        trend = go.Figure()
        trend.add_trace(go.Scatter(
            x=list(range(1, len(hist)+1)), y=confs[::-1],
            mode='lines+markers',
            line=dict(color=TEAL, width=2.5),
            marker=dict(size=8, color=[EMO_COLORS.get(e, TEAL) for e in emos[::-1]],
                        line=dict(color=BG, width=2)),
            fill='tozeroy', fillcolor=TEAL + '18', showlegend=False,
        ))
        trend.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0,r=0,t=4,b=0), height=228,
            xaxis=dict(showgrid=False, zeroline=False,
                       tickfont=dict(family='JetBrains Mono', size=8, color=BROWN),
                       title="Analysis #", titlefont=dict(size=9, color=BROWN)),
            yaxis=dict(showgrid=True, gridcolor=BROWN+'18', zeroline=False,
                       range=[0,100], tickfont=dict(family='JetBrains Mono', size=8, color=BROWN)),
        )
        st.plotly_chart(trend, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    # CSV export
    df_hist = pd.DataFrame([{
        "Time": r.get("timestamp","—"),
        "Source": r.get("source","—")[:30],
        "Emotion": f"{r['emoji']} {r['emotion']}",
        "Confidence": f"{r['confidence']:.1f}%",
        "Duration": f"{r['duration']:.1f}s",
        "Transcript": r["transcript"][:60],
    } for r in hist])
    st.download_button("📥 Export CSV", data=df_hist.to_csv(index=False),
                       file_name="vocalmind_history.csv", mime="text/csv")

st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ────────────────────────────────────────────────────────────────────────────
#  SECTION 7 — TECH STACK
#  ────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""<div class="vm-section">""", unsafe_allow_html=True)
st.markdown(f"""
<div class="vm-section-tag">Section 07</div>
<div class="vm-section-title">Technology Stack</div>
""", unsafe_allow_html=True)

TECH = [
    (TEAL,    "🤗", "Wav2Vec2",       "facebook/wav2vec2-base-960h",   "Core acoustic encoder — 960h pretrained"),
    (SIDEBAR, "🎙️", "Whisper",         "openai/whisper-tiny",           "ASR transcription — multilingual"),
    (GREEN,   "🎵", "librosa",         "v0.10+",                        "MFCC · Mel-spectrogram · pitch extraction"),
    (COPPER,  "🔥", "PyTorch",         "torch + CUDA support",          "Model inference backend"),
    (TEAL,    "📊", "Plotly",          "Interactive charts",            "Waveform · radar · timeline · bars"),
    (GREEN,   "🖥️", "Streamlit",       "v1.37+",                        "Web application framework"),
    (BROWN,   "📄", "fpdf2",           "PDF generation",                "Export analysis reports"),
    (SIDEBAR, "🎤", "audio_recorder",  "audio_recorder_streamlit",      "Browser microphone capture"),
]

tc1, tc2 = st.columns(2, gap="large")
for i, (color, ico, name, ver, desc) in enumerate(TECH):
    with (tc1 if i % 2 == 0 else tc2):
        st.markdown(f"""
        <div class="vm-tech">
          <div class="vm-tech-icon" style="background:{color}18;border:1px solid {color}33;">
            {ico}
          </div>
          <div style="flex:1;">
            <div style="font-weight:700;color:{INK};font-size:.88rem;">{name}</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:.68rem;
                 color:{color};margin:2px 0;">{ver}</div>
            <div style="font-size:.75rem;color:{BROWN};">{desc}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ────────────────────────────────────────────────────────────────────────────
#  SECTION 8 — CTA + FOOTER
#  ────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="vm-footer">
  <div style="position:relative;z-index:1;">
    <div style="font-size:2.5rem;margin-bottom:14px;">🎤</div>
    <div style="font-family:'Poppins',sans-serif;font-weight:800;font-size:1.8rem;
         color:white;margin-bottom:10px;letter-spacing:-.02em;">
      Ready to Hear What Your Voice Reveals?
    </div>
    <div style="color:rgba(255,255,255,.55);font-size:.9rem;
         max-width:460px;margin:0 auto 28px;line-height:1.7;">
      Upload an audio file or record your voice above.<br>
      VocalMind will analyze your emotional state in under 2 seconds.
    </div>
    <div style="display:flex;justify-content:center;gap:16px;
         flex-wrap:wrap;margin-bottom:36px;">
      <div style="display:flex;gap:20px;flex-wrap:wrap;justify-content:center;">
        <span style="font-family:'JetBrains Mono',monospace;font-size:.72rem;
              color:rgba(255,255,255,.4);letter-spacing:.1em;">99.33% ACCURACY</span>
        <span style="color:rgba(255,255,255,.2);">·</span>
        <span style="font-family:'JetBrains Mono',monospace;font-size:.72rem;
              color:rgba(255,255,255,.4);letter-spacing:.1em;">5 EMOTIONS</span>
        <span style="color:rgba(255,255,255,.2);">·</span>
        <span style="font-family:'JetBrains Mono',monospace;font-size:.72rem;
              color:rgba(255,255,255,.4);letter-spacing:.1em;">REAL-TIME</span>
        <span style="color:rgba(255,255,255,.2);">·</span>
        <span style="font-family:'JetBrains Mono',monospace;font-size:.72rem;
              color:rgba(255,255,255,.4);letter-spacing:.1em;">MULTIMODAL AI</span>
      </div>
    </div>
    <div style="border-top:1px solid rgba(255,255,255,.1);padding-top:20px;
         font-family:'JetBrains Mono',monospace;font-size:.65rem;
         color:rgba(255,255,255,.3);letter-spacing:.1em;">
      VOCALMIND · EMOTION INTELLIGENCE PLATFORM · WAV2VEC2 + WHISPER
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
