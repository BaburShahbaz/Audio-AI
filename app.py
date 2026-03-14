import streamlit as st
import torch
import torch.nn as nn
import librosa
import numpy as np
import whisper
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from transformers import Wav2Vec2Processor, Wav2Vec2Model, AutoTokenizer, AutoModel
import io
import time
import pickle
from fpdf import FPDF
import datetime
import gc

# ==========================================
# 1. SETUP & EXACT COLOR SCHEME
# ==========================================
st.set_page_config(page_title="Sentira Emotion Command Center", page_icon="🎤", layout="wide")

COLORS = {
    "bg": "#FFF8F0", "sidebar": "#281C59", "buttons": "#4E8D9C",
    "success": "#85C79A", "highlights": "#EDF7BD", "accent": "#C08552",
    "borders": "#8C5A3C", "text": "#4B2E2B"
}

st.markdown(f"""
<style>
    .stApp {{ background-color: {COLORS['bg']}; color: {COLORS['text']}; font-family: 'Inter', sans-serif; }}
    h1, h2, h3, h4 {{ color: {COLORS['text']}; font-weight: 700; }}
    [data-testid="stSidebar"] {{ background-color: {COLORS['sidebar']}; color: {COLORS['bg']}; }}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p {{ color: {COLORS['bg']} !important; }}
    .saas-card {{
        background: #FFFFFF; border: 1px solid {COLORS['borders']}40; border-radius: 16px;
        padding: 24px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05); margin-bottom: 20px;
    }}
    div.stButton > button {{
        background-color: {COLORS['buttons']} !important; color: white !important;
        border: none !important; border-radius: 8px !important; font-weight: 600 !important;
    }}
    div.stButton > button:hover {{ background-color: {COLORS['sidebar']} !important; }}
    .primary-emotion {{ font-size: 3rem; font-weight: 800; color: {COLORS['buttons']}; text-align: center; }}
    .info-badge {{ background: {COLORS['highlights']}; color: {COLORS['text']}; padding: 4px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: bold; }}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. YOUR EXACT MODEL ARCHITECTURE
# ==========================================
class AttentionFusionModel(nn.Module):
    def __init__(self, trad_dim=160, w2v_dim=768, sem_dim=768, num_classes=5):
        super(AttentionFusionModel, self).__init__()
        self.trad_net = nn.Sequential(nn.Linear(trad_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4))
        self.w2v_net = nn.Sequential(nn.Linear(w2v_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4))
        self.sem_net = nn.Sequential(nn.Linear(sem_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4))
        self.attention = nn.Sequential(nn.Linear(256 * 3, 3), nn.Softmax(dim=1))
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_classes)
        )

    def forward(self, trad, w2v, sem):
        h_t, h_w, h_s = self.trad_net(trad), self.w2v_net(w2v), self.sem_net(sem)
        attn = self.attention(torch.cat([h_t, h_w, h_s], dim=1))
        fused = (h_t * attn[:, 0].unsqueeze(1) + h_w * attn[:, 1].unsqueeze(1) + h_s * attn[:, 2].unsqueeze(1))
        return self.classifier(fused)

LABEL_MAP = {0: 'Happiness', 1: 'Neutral', 2: 'Anger', 3: 'Sadness', 4: 'Calmness'}
EMOTIONS_LIST = list(LABEL_MAP.values())


# ==========================================
# 3. RESOURCE LOADING (Models & Scalers)
# ==========================================
@st.cache_resource(show_spinner="Booting Neural Fusion Engine...")
def load_resources():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    whisp = whisper.load_model("base")
    w2v_proc = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    w2v_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device).eval()
    rob_tok = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    rob_model = AutoModel.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment").to(device).eval()

    fusion_model = AttentionFusionModel().to(device)
    try:
        fusion_model.load_state_dict(torch.load("best_emotion_model.pt", map_location=device))
    except Exception as e:
        st.error(f"Failed to load best_emotion_model.pt. Ensure it is in the root directory. Error: {e}")
    fusion_model.eval()

    try:
        with open('scalers.pkl', 'rb') as f:
            scalers = pickle.load(f)
    except:
        st.warning("scalers.pkl not found. Predictions will be inaccurate without proper scaling.")
        scalers = None

    return device, whisp, w2v_proc, w2v_model, rob_tok, rob_model, fusion_model, scalers

device, whisp, w2v_proc, w2v_model, rob_tok, rob_model, fusion_model, scalers = load_resources()


# ==========================================
# 4. EXACT FEATURE EXTRACTION PIPELINE
# ==========================================
def extract_features(audio_bytes):
    start_time = time.time()

    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
    
    y_trimmed, _ = librosa.effects.trim(y, top_db=30)
    if len(y_trimmed) > sr * 0.5: 
        y = y_trimmed
    
    y = librosa.util.normalize(y)
    duration = librosa.get_duration(y=y, sr=sr)

    with torch.no_grad():
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        trad_feat = np.concatenate([np.mean(mfccs, axis=1), np.std(mfccs, axis=1), np.max(mfccs, axis=1), np.min(mfccs, axis=1)])

        inputs = w2v_proc(y, sampling_rate=16000, return_tensors="pt").to(device)
        w2v_out = w2v_model(inputs.input_values).last_hidden_state
        w2v_feat = torch.mean(w2v_out, dim=1).cpu().numpy().squeeze()

        trans_res = whisp.transcribe(y)
        transcript = trans_res['text'].strip()
        if not transcript: transcript = "silence"

        sem_inputs = rob_tok(transcript, return_tensors="pt", truncation=True, max_length=512).to(device)
        sem_out = rob_model(**sem_inputs).last_hidden_state
        sem_feat = sem_out[:, 0, :].cpu().numpy().squeeze() 

        if scalers:
            trad_feat = scalers['trad'].transform(trad_feat.reshape(1, -1))
            w2v_feat = scalers['w2v'].transform(w2v_feat.reshape(1, -1))
            sem_feat = scalers['sem'].transform(sem_feat.reshape(1, -1))

        t_tens = torch.FloatTensor(trad_feat).to(device)
        w_tens = torch.FloatTensor(w2v_feat).to(device)
        s_tens = torch.FloatTensor(sem_feat).to(device)

        logits = fusion_model(t_tens, w_tens, s_tens)
        probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]

    dom_idx = np.argmax(probs)
    
    del t_tens, w_tens, s_tens, logits, inputs, w2v_out, sem_inputs, sem_out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "emotion": LABEL_MAP[dom_idx],
        "confidence": float(probs[dom_idx] * 100),
        "transcript": transcript,
        "probs": probs,
        "duration": duration,
        "latency": time.time() - start_time,
        "y": y, "sr": sr
    }


# ==========================================
# 5. UI LAYOUT & NAVIGATION
# ==========================================
if 'history' not in st.session_state:
    st.session_state.history = []
if 'just_analyzed' not in st.session_state:
    st.session_state.just_analyzed = False

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8653/8653063.png", width=60)
    st.markdown("## Sentira OS")
    page = st.radio("Navigation", ["🎙️ Live Analysis", "📜 History & Export", "ℹ️ About Model"])
    st.markdown("---")
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.success("History cleared.")

if page == "🎙️ Live Analysis":
    st.markdown("<h1>Advanced Emotion Command Center</h1>", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2.2], gap="large")

    with c1:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown("### Audio Portal")
        
        audio_source = st.audio_input("Record Voice", label_visibility="collapsed")
        file_upload = st.file_uploader("Or Upload Audio", type=["wav", "mp3", "m4a", "ogg"])

        target_audio = audio_source if audio_source else file_upload

        if target_audio:
            st.audio(target_audio)
            if st.button("🚀 INITIATE FUSION ANALYSIS", use_container_width=True):
                with st.spinner("Extracting Tri-modal Embeddings..."):
                    try:
                        res = extract_features(target_audio.getvalue())
                        st.session_state.current_analysis = res
                        st.session_state.history.append(res)
                        st.session_state.just_analyzed = True  # Triggers the typewriter effect
                    except Exception as e:
                        st.error(f"Analysis Failed. Please try recording again. Error details: {str(e)}")
                        
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        if 'current_analysis' in st.session_state:
            res = st.session_state.current_analysis

            # --- TOP ROW: Transcriber Box & Dynamic Waveform ---
            top_c1, top_c2 = st.columns(2)
            
            with top_c1:
                st.markdown('<div class="saas-card" style="min-height: 250px;">', unsafe_allow_html=True)
                st.markdown("#### 📝 Semantic Output")
                
                # Typewriter container
                text_placeholder = st.empty()
                
                if st.session_state.just_analyzed:
                    # Execute Typewriter Effect
                    displayed_text = ""
                    for char in res['transcript']:
                        displayed_text += char
                        text_placeholder.info(f"\"{displayed_text}▌\"")
                        time.sleep(0.015)  # The 'tit tit tit' speed
                    text_placeholder.info(f"\"{res['transcript']}\"")
                else:
                    text_placeholder.info(f"\"{res['transcript']}\"")
                    
                st.caption(f"⏱️ Latency: {res['latency']:.2f}s | 📏 Duration: {res['duration']:.2f}s")
                st.markdown('</div>', unsafe_allow_html=True)

            with top_c2:
                st.markdown('<div class="saas-card" style="min-height: 250px;">', unsafe_allow_html=True)
                st.markdown("#### 🌊 Acoustic Waveform")
                
                # Create eye-catching waveform visualization
                y_data = res['y']
                # Downsample for faster UI rendering
                step = max(1, len(y_data) // 600)
                
                fig_wave = go.Figure(go.Scatter(
                    y=y_data[::step],
                    mode='lines',
                    line=dict(color=COLORS['buttons'], width=2),
                    fill='tozeroy',
                    fillcolor=COLORS['highlights']
                ))
                fig_wave.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=0, r=0, t=10, b=10),
                    xaxis=dict(visible=False), yaxis=dict(visible=False),
                    height=140
                )
                st.plotly_chart(fig_wave, use_container_width=True, config={'displayModeBar': False})
                st.markdown('</div>', unsafe_allow_html=True)


            # --- MIDDLE ROW: Emotion Classification & Radar Chart ---
            mid_c1, mid_c2 = st.columns(2)
            with mid_c1:
                st.markdown('<div class="saas-card" style="min-height: 350px;">', unsafe_allow_html=True)
                st.markdown("<div class='info-badge'>Primary Classification</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='primary-emotion'>{res['emotion']}</div>", unsafe_allow_html=True)
                st.progress(res['confidence'] / 100)
                st.markdown(f"<p style='text-align:center;'>Certainty: {res['confidence']:.1f}%</p>",
                            unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with mid_c2:
                st.markdown('<div class="saas-card" style="min-height: 350px;">', unsafe_allow_html=True)
                fig_radar = go.Figure(data=go.Scatterpolar(r=res['probs'] * 100, theta=EMOTIONS_LIST, fill='toself',
                                                           line_color=COLORS['accent']))
                fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False)), margin=dict(t=20, b=20, l=20, r=20),
                                        paper_bgcolor='rgba(0,0,0,0)', height=280)
                st.plotly_chart(fig_radar, use_container_width=True, config={'displayModeBar': False})
                st.markdown('</div>', unsafe_allow_html=True)

            # --- BOTTOM ROW: Probability Matrix ---
            st.markdown('<div class="saas-card">', unsafe_allow_html=True)
            st.markdown("#### Probability Matrix")
            df_probs = pd.DataFrame({"Emotion": EMOTIONS_LIST, "Probability": res['probs'] * 100}).sort_values(
                by="Probability", ascending=True)
            fig_bar = px.bar(df_probs, x="Probability", y="Emotion", orientation='h',
                             text=df_probs['Probability'].apply(lambda x: f'{x:.1f}%'), color="Probability",
                             color_continuous_scale="Oryel")
            fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                  margin=dict(l=0, r=0, t=0, b=0), xaxis=dict(visible=False), yaxis=dict(title=""),
                                  coloraxis_showscale=False, height=150)
            st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)

            # Clear 'just_analyzed' state so typing effect doesn't happen on simple tab switches
            st.session_state.just_analyzed = False

        else:
            st.info("Awaiting acoustic signal...")

elif page == "📜 History & Export":
    st.title("Session History")
    if not st.session_state.history:
        st.warning("No analyses performed yet.")
    else:
        for i, h in enumerate(reversed(st.session_state.history)):
            with st.expander(f"{h['timestamp']} - {h['emotion']} ({h['confidence']:.1f}%)"):
                st.write(f"**Transcript:** {h['transcript']}")

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Sentira OS - Session Export", ln=1, align='C')
        
        for h in st.session_state.history:
            clean_text = h['transcript'].encode('latin-1', 'replace').decode('latin-1')
            pdf.cell(200, 10, txt=f"{h['timestamp']} | Emotion: {h['emotion']} | Text: {clean_text[:50]}...", ln=1)

        st.download_button(
            label="📄 Download PDF Report",
            data=pdf.output(dest='S').encode('latin-1'),
            file_name="sentira_report.pdf",
            mime="application/pdf"
        )

elif page == "ℹ️ About Model":
    st.title("Neural Architecture")
    st.markdown("""
    **Model:** Attention-based Tri-modal Fusion  
    **Accuracy:** 99.33%  
    **Feature Extractors:**
    * Acoustic: 160-dim MFCC Statistics
    * Prosodic: Wav2Vec2 Base (768-dim)
    * Semantic: Whisper Base -> Twitter RoBERTa Sentiment (768-dim CLS)
    """)
