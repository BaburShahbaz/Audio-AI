"""
🎭 PRODUCTION-READY MULTIMODAL EMOTION DETECTION APP
=====================================================
Advanced Emotion Command Center with Real-time Analysis
Model: AttentionFusionModel (99.33% Accuracy)
Features: Traditional MFCC + Wav2Vec2 + Semantic Fusion
"""

import streamlit as st
import torch
import torch.nn as nn
import librosa
import numpy as np
import whisper
import pickle
import io
import time
import json
import base64
from datetime import datetime
from collections import defaultdict
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from transformers import (
    Wav2Vec2Processor, 
    Wav2Vec2Model,
    AutoTokenizer, 
    AutoModel
)
from fpdf import FPDF
from sklearn.preprocessing import StandardScaler
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import warnings
import soundfile as sf  # ADD THIS for audio conversion
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
PAGE_CONFIG = {
    "page_title": "Emotion Command Center",
    "page_icon": "🎭",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Exact Color Scheme
COLORS = {
    "background": "#FFF8F0",
    "sidebar": "#281C59",
    "buttons": "#4E8D9C",
    "success": "#85C79A",
    "highlights": "#EDF7BD",
    "accent": "#C08552",
    "borders": "#8C5A3C",
    "text": "#4B2E2B",
    "card_bg": "#FFFFFF",
    "gradient_start": "#281C59",
    "gradient_end": "#4E8D9C"
}

# Emotion Configuration
EMOTIONS = {
    0: {"name": "Happiness", "emoji": "😄", "color": "#FFD700", "position": (0.707, 0.293)},  # Top-right
    1: {"name": "Neutral", "emoji": "😐", "color": "#808080", "position": (0.5, 0.5)},      # Center
    2: {"name": "Anger", "emoji": "😠", "color": "#DC143C", "position": (0.85, 0.5)},      # Right
    3: {"name": "Sadness", "emoji": "😢", "color": "#4169E1", "position": (0.293, 0.707)}, # Bottom-left
    4: {"name": "Calmness", "emoji": "😌", "color": "#90EE90", "position": (0.5, 0.15)}    # Top
}

LABEL_MAP = {'H': 0, 'N': 1, 'A': 2, 'S': 3, 'C': 4}
REVERSE_MAP = {v: k for k, v in LABEL_MAP.items()}

# ==========================================
# MODEL ARCHITECTURE (Exact from Training)
# ==========================================
class AttentionFusionModel(nn.Module):
    def __init__(self, trad_dim=160, w2v_dim=768, sem_dim=768, num_classes=5):
        super(AttentionFusionModel, self).__init__()
        # Branch Experts
        self.trad_net = nn.Sequential(
            nn.Linear(trad_dim, 256), 
            nn.BatchNorm1d(256), 
            nn.ReLU(), 
            nn.Dropout(0.4)
        )
        self.w2v_net = nn.Sequential(
            nn.Linear(w2v_dim, 256), 
            nn.BatchNorm1d(256), 
            nn.ReLU(), 
            nn.Dropout(0.4)
        )
        self.sem_net = nn.Sequential(
            nn.Linear(sem_dim, 256), 
            nn.BatchNorm1d(256), 
            nn.ReLU(), 
            nn.Dropout(0.4)
        )
        
        # Self-Attention Logic
        self.attention = nn.Sequential(
            nn.Linear(256 * 3, 3), 
            nn.Softmax(dim=1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), 
            nn.BatchNorm1d(128), 
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, trad, w2v, sem):
        h_t = self.trad_net(trad)
        h_w = self.w2v_net(w2v)
        h_s = self.sem_net(sem)
        
        attn = self.attention(torch.cat([h_t, h_w, h_s], dim=1))
        fused = (
            h_t * attn[:, 0].unsqueeze(1) + 
            h_w * attn[:, 1].unsqueeze(1) + 
            h_s * attn[:, 2].unsqueeze(1)
        )
        return self.classifier(fused)

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'current_analysis': None,
        'analysis_history': [],
        'recording': False,
        'audio_buffer': [],
        'current_page': 'Home',
        'theme': 'light',
        'mic_permission': False,
        'user_name': '',
        'badges': [],
        'language': 'English',
        'model_loaded': False,
        'recording_start_time': None,
        'live_probs': np.ones(5) / 5,
        'total_analyses': 0
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ==========================================
# CUSTOM CSS STYLING
# ==========================================
def apply_custom_styling():
    """Apply exact color scheme and custom styling"""
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Poppins:wght@600;700;800&display=swap');
        
        /* Base Styling */
        .stApp {{
            background-color: {COLORS['background']};
            color: {COLORS['text']};
            font-family: 'Inter', sans-serif;
        }}
        
        /* Typography */
        h1, h2, h3, h4, h5, h6 {{
            font-family: 'Poppins', sans-serif;
            color: {COLORS['text']};
            font-weight: 700;
        }}
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {COLORS['sidebar']} 0%, #1a1240 100%);
        }}
        
        [data-testid="stSidebar"] .stMarkdown {{
            color: white !important;
        }}
        
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {{
            color: white !important;
        }}
        
        /* Custom Cards */
        .emotion-card {{
            background: {COLORS['card_bg']};
            border: 2px solid {COLORS['borders']}30;
            border-radius: 20px;
            padding: 24px;
            box-shadow: 0 8px 32px rgba(40, 28, 89, 0.1);
            transition: all 0.3s ease;
        }}
        
        .emotion-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 12px 40px rgba(40, 28, 89, 0.15);
            border-color: {COLORS['buttons']}50;
        }}
        
        /* Primary Emotion Display */
        .primary-emotion {{
            font-size: 4rem;
            font-weight: 800;
            text-align: center;
            background: linear-gradient(135deg, {COLORS['sidebar']} 0%, {COLORS['buttons']} 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 20px 0;
        }}
        
        /* Emotion Circle Container */
        .emotion-circle-container {{
            position: relative;
            width: 400px;
            height: 400px;
            margin: 0 auto;
        }}
        
        /* Recording Button States */
        .record-btn-idle {{
            background: linear-gradient(135deg, {COLORS['buttons']} 0%, {COLORS['sidebar']} 100%) !important;
        }}
        
        .record-btn-recording {{
            background: linear-gradient(135deg, #DC143C 0%, #8B0000 100%) !important;
            animation: pulse-red 1.5s infinite !important;
        }}
        
        .record-btn-processing {{
            background: linear-gradient(135deg, {COLORS['accent']} 0%, #DAA520 100%) !important;
        }}
        
        @keyframes pulse-red {{
            0% {{ box-shadow: 0 0 0 0 rgba(220, 20, 60, 0.7); }}
            70% {{ box-shadow: 0 0 0 20px rgba(220, 20, 60, 0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(220, 20, 60, 0); }}
        }}
        
        /* Pipeline Visualization */
        .pipeline-container {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: {COLORS['card_bg']};
            padding: 20px;
            border-radius: 16px;
            border: 2px solid {COLORS['borders']}20;
            overflow-x: auto;
            gap: 10px;
        }}
        
        .pipeline-step {{
            text-align: center;
            font-size: 0.85rem;
            font-weight: 600;
            color: {COLORS['sidebar']};
            background: {COLORS['highlights']};
            padding: 12px 16px;
            border-radius: 12px;
            white-space: nowrap;
            min-width: 120px;
        }}
        
        .pipeline-arrow {{
            color: {COLORS['buttons']};
            font-size: 1.5rem;
            animation: pulse-arrow 1.5s infinite;
        }}
        
        @keyframes pulse-arrow {{
            0%, 100% {{ opacity: 0.4; }}
            50% {{ opacity: 1; }}
        }}
        
        /* Badges */
        .badge {{
            display: inline-block;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .badge-success {{
            background: {COLORS['success']}30;
            color: {COLORS['success']};
        }}
        
        .badge-accent {{
            background: {COLORS['accent']}30;
            color: {COLORS['accent']};
        }}
        
        .badge-primary {{
            background: {COLORS['buttons']}30;
            color: {COLORS['buttons']};
        }}
        
        /* Metric Cards */
        .metric-card {{
            background: linear-gradient(135deg, {COLORS['card_bg']} 0%, {COLORS['highlights']}30 100%);
            border: 2px solid {COLORS['borders']}20;
            border-radius: 16px;
            padding: 20px;
            text-align: center;
        }}
        
        .metric-value {{
            font-size: 2.5rem;
            font-weight: 800;
            color: {COLORS['sidebar']};
        }}
        
        .metric-label {{
            font-size: 0.9rem;
            color: {COLORS['text']}80;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        /* Navigation Items */
        .nav-item {{
            padding: 12px 16px;
            border-radius: 12px;
            margin: 4px 0;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 12px;
            color: white;
            font-weight: 500;
        }}
        
        .nav-item:hover {{
            background: rgba(255, 255, 255, 0.1);
        }}
        
        .nav-item.active {{
            background: {COLORS['buttons']};
        }}
        
        /* Progress Bar */
        .stProgress > div > div {{
            background: linear-gradient(90deg, {COLORS['buttons']} 0%, {COLORS['success']} 100%);
        }}
        
        /* Buttons */
        div.stButton > button {{
            background: linear-gradient(135deg, {COLORS['buttons']} 0%, {COLORS['sidebar']} 100%) !important;
            color: white !important;
            border: none !important;
            padding: 14px 28px !important;
            border-radius: 12px !important;
            font-weight: 700 !important;
            font-size: 1rem !important;
            box-shadow: 0 4px 15px {COLORS['buttons']}50 !important;
            transition: all 0.3s ease !important;
            width: 100% !important;
        }}
        
        div.stButton > button:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 25px {COLORS['buttons']}70 !important;
        }}
        
        /* File Uploader */
        .stFileUploader {{
            background: {COLORS['card_bg']};
            border: 2px dashed {COLORS['borders']}40;
            border-radius: 16px;
            padding: 20px;
        }}
        
        /* Audio Player */
        audio {{
            width: 100%;
            border-radius: 12px;
        }}
        
        /* History Items */
        .history-item {{
            background: {COLORS['card_bg']};
            border: 1px solid {COLORS['borders']}30;
            border-radius: 12px;
            padding: 16px;
            margin: 8px 0;
            transition: all 0.2s ease;
        }}
        
        .history-item:hover {{
            border-color: {COLORS['buttons']}50;
            box-shadow: 0 4px 12px rgba(40, 28, 89, 0.1);
        }}
        
        /* Scrollbar */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: {COLORS['background']};
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: {COLORS['borders']};
            border-radius: 4px;
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: {COLORS['sidebar']};
        }}
        
        /* Hide Streamlit Elements */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        .stDeployButton {{display: none;}}
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# MODEL & RESOURCE LOADING
# ==========================================
@st.cache_resource(show_spinner="🚀 Initializing Neural Engines...")
def load_all_models():
    """Load all AI models and processors"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Whisper (base model as per training)
    whisper_model = whisper.load_model("base")
    
    # Load Wav2Vec2
    w2v_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    w2v_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
    w2v_model.eval()
    
    # Load RoBERTa for semantic features
    sem_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    sem_model = AutoModel.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment").to(device)
    sem_model.eval()
    
    # Load trained emotion model
    emotion_model = AttentionFusionModel().to(device)
    
    # Try to load trained weights
    try:
        emotion_model.load_state_dict(torch.load("best_emotion_model.pt", map_location=device))
        emotion_model.eval()
        model_loaded = True
    except:
        st.warning("⚠️ Trained model not found. Using untrained model for demonstration.")
        model_loaded = False
    
    # Try to load scalers
    try:
        with open("scalers.pkl", "rb") as f:
            scalers = pickle.load(f)
    except:
        st.warning("⚠️ Scalers not found. Using default scalers.")
        scalers = {
            'trad': StandardScaler(),
            'w2v': StandardScaler(),
            'sem': StandardScaler()
        }
    
    return {
        'device': device,
        'whisper': whisper_model,
        'w2v_processor': w2v_processor,
        'w2v_model': w2v_model,
        'sem_tokenizer': sem_tokenizer,
        'sem_model': sem_model,
        'emotion_model': emotion_model,
        'scalers': scalers,
        'model_loaded': model_loaded
    }

# ==========================================
# FEATURE EXTRACTION FUNCTIONS
# ==========================================
def extract_traditional_features(y, sr=16000):
    """Extract 160-dim traditional features (40 MFCCs x 4 stats)"""
    # 40 MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    
    # Statistical features: mean, std, max, min
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    mfcc_max = np.max(mfcc, axis=1)
    mfcc_min = np.min(mfcc, axis=1)
    
    features = np.concatenate([mfcc_mean, mfcc_std, mfcc_max, mfcc_min])
    return features  # 160 dimensions

def extract_wav2vec_features(y, sr, w2v_processor, w2v_model, device):
    """Extract 768-dim Wav2Vec2 embeddings"""
    # Resample to 16kHz if needed
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    
    # Process audio
    inputs = w2v_processor(y, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Extract features
    with torch.no_grad():
        outputs = w2v_model(**inputs)
        # Mean pooling over time dimension
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    
    return embeddings[0]  # 768 dimensions

def extract_semantic_features(audio_bytes, whisper_model, sem_tokenizer, sem_model, device):
    """Extract 768-dim semantic features from transcript"""
    # Transcribe audio using bytes
    result = whisper_model.transcribe(audio_bytes)
    transcript = result['text'].strip()
    
    if not transcript:
        # Return zero vector if no speech detected
        return np.zeros(768), "No speech detected"
    
    # Get RoBERTa embeddings
    inputs = sem_tokenizer(transcript, return_tensors="pt", truncation=True, max_length=512, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = sem_model(**inputs)
        # Use CLS token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    
    return cls_embedding[0], transcript  # 768 dimensions

def preprocess_audio(audio_bytes, target_sr=16000):
    """Load and preprocess audio from bytes"""
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=target_sr, mono=True)
    
    # Trim silence
    y, _ = librosa.effects.trim(y, top_db=20)
    
    # Normalize
    y = librosa.util.normalize(y)
    
    return y, sr

# ==========================================
# INFERENCE FUNCTION
# ==========================================
def perform_emotion_analysis(audio_bytes, models, is_live=False):
    """Complete emotion analysis pipeline"""
    start_time = time.time()
    
    # Preprocess audio
    y, sr = preprocess_audio(audio_bytes)
    duration = librosa.get_duration(y=y, sr=sr)
    
    # Extract features
    trad_features = extract_traditional_features(y, sr)
    w2v_features = extract_wav2vec_features(
        y, sr, 
        models['w2v_processor'], 
        models['w2v_model'], 
        models['device']
    )
    
    # FIX: Pass audio_bytes (not y) to extract_semantic_features for Whisper
    sem_features, transcript = extract_semantic_features(
        audio_bytes,  # Use original bytes for Whisper
        models['whisper'],
        models['sem_tokenizer'],
        models['sem_model'],
        models['device']
    )
    
    # Apply scalers
    trad_scaled = models['scalers']['trad'].transform(trad_features.reshape(1, -1))
    w2v_scaled = models['scalers']['w2v'].transform(w2v_features.reshape(1, -1))
    sem_scaled = models['scalers']['sem'].transform(sem_features.reshape(1, -1))
    
    # Convert to tensors
    trad_tensor = torch.FloatTensor(trad_scaled).to(models['device'])
    w2v_tensor = torch.FloatTensor(w2v_scaled).to(models['device'])
    sem_tensor = torch.FloatTensor(sem_scaled).to(models['device'])
    
    # Inference
    with torch.no_grad():
        logits = models['emotion_model'](trad_tensor, w2v_tensor, sem_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    
    latency = time.time() - start_time
    
    # Get dominant emotion
    dominant_idx = np.argmax(probs)
    
    return {
        "emotion_idx": dominant_idx,
        "emotion": EMOTIONS[dominant_idx]['name'],
        "emoji": EMOTIONS[dominant_idx]['emoji'],
        "confidence": float(probs[dominant_idx] * 100),
        "probs": probs,
        "transcript": transcript,
        "duration": duration,
        "latency": latency,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "audio_array": y,
        "sr": sr
    }

# ==========================================
# VISUALIZATION FUNCTIONS
# ==========================================
def create_emotion_circle(probs, size=400):
    """Create interactive emotion circle with animated dot"""
    # Calculate dot position based on probabilities
    dot_x = sum(p * EMOTIONS[i]['position'][0] for i, p in enumerate(probs))
    dot_y = sum(p * EMOTIONS[i]['position'][1] for i, p in enumerate(probs))
    
    # Create figure
    fig = go.Figure()
    
    # Add emotion circles
    for idx, emotion in EMOTIONS.items():
        fig.add_trace(go.Scatter(
            x=[emotion['position'][0]],
            y=[emotion['position'][1]],
            mode='markers+text',
            marker=dict(
                size=60,
                color=emotion['color'],
                opacity=0.7,
                line=dict(width=3, color='white')
            ),
            text=f"{emotion['emoji']}<br>{emotion['name']}",
            textposition='middle center',
            textfont=dict(size=10, color='white', family='Poppins'),
            hoverinfo='text',
            hovertext=f"{emotion['name']}: {probs[idx]*100:.1f}%",
            showlegend=False
        ))
    
    # Add connecting lines from center to emotions
    for idx, emotion in EMOTIONS.items():
        fig.add_trace(go.Scatter(
            x=[0.5, emotion['position'][0]],
            y=[0.5, emotion['position'][1]],
            mode='lines',
            line=dict(color=emotion['color'], width=2, dash='dot'),
            opacity=0.3,
            showlegend=False
        ))
    
    # Add animated dot
    fig.add_trace(go.Scatter(
        x=[dot_x],
        y=[dot_y],
        mode='markers',
        marker=dict(
            size=30,
            color=COLORS['accent'],
            line=dict(width=4, color='white'),
            symbol='circle'
        ),
        showlegend=False
    ))
    
    # Add glow effect around dot
    fig.add_trace(go.Scatter(
        x=[dot_x],
        y=[dot_y],
        mode='markers',
        marker=dict(
            size=50,
            color=COLORS['accent'],
            opacity=0.3,
            symbol='circle'
        ),
        showlegend=False
    ))
    
    # Update layout
    fig.update_layout(
        width=size,
        height=size,
        xaxis=dict(range=[0, 1], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[0, 1], showgrid=False, zeroline=False, visible=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False
    )
    
    return fig

def create_radar_chart(probs):
    """Create radar chart for emotion probabilities"""
    emotions = [EMOTIONS[i]['name'] for i in range(5)]
    values = probs * 100
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=emotions,
        fill='toself',
        fillcolor=COLORS['buttons'] + '40',
        line=dict(color=COLORS['buttons'], width=3),
        marker=dict(size=10, color=COLORS['buttons'])
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], showline=False),
            angularaxis=dict(linecolor=COLORS['borders'])
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=40, b=40),
        height=350
    )
    
    return fig

def create_probability_bars(probs):
    """Create horizontal bar chart for probabilities"""
    df = pd.DataFrame({
        'Emotion': [f"{EMOTIONS[i]['emoji']} {EMOTIONS[i]['name']}" for i in range(5)],
        'Probability': probs * 100,
        'Color': [EMOTIONS[i]['color'] for i in range(5)]
    }).sort_values('Probability', ascending=True)
    
    fig = px.bar(
        df, 
        x='Probability', 
        y='Emotion',
        orientation='h',
        text=df['Probability'].apply(lambda x: f'{x:.1f}%'),
        color='Emotion',
        color_discrete_map={row['Emotion']: row['Color'] for _, row in df.iterrows()}
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=20, t=0, b=0),
        xaxis=dict(visible=False, range=[0, 100]),
        yaxis=dict(title=''),
        showlegend=False,
        height=250
    )
    
    fig.update_traces(
        textposition='outside',
        marker_line_width=0,
        textfont=dict(size=12, family='Poppins')
    )
    
    return fig

def create_spectrogram(y, sr):
    """Create mel spectrogram visualization"""
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    fig = px.imshow(
        S_dB,
        aspect='auto',
        origin='lower',
        color_continuous_scale='magma'
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        coloraxis_showscale=False,
        height=200
    )
    
    return fig

def create_waveform(y, sr):
    """Create waveform visualization"""
    times = np.arange(len(y)) / sr
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times,
        y=y,
        mode='lines',
        line=dict(color=COLORS['buttons'], width=1),
        fill='tozeroy',
        fillcolor=COLORS['buttons'] + '30'
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=100
    )
    
    return fig

# ==========================================
# EXPORT & SHARING FUNCTIONS
# ==========================================
def generate_pdf_report(analysis):
    """Generate PDF report of analysis"""
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font('Arial', 'B', 24)
    pdf.set_text_color(40, 28, 89)
    pdf.cell(0, 20, 'Emotion Analysis Report', ln=True, align='C')
    
    # Date
    pdf.set_font('Arial', '', 12)
    pdf.set_text_color(75, 46, 43)
    pdf.cell(0, 10, f"Generated: {analysis['timestamp']}", ln=True, align='C')
    pdf.ln(10)
    
    # Primary Emotion
    pdf.set_font('Arial', 'B', 18)
    pdf.cell(0, 15, f"Primary Emotion: {analysis['emotion']} {analysis['emoji']}", ln=True)
    pdf.set_font('Arial', '', 14)
    pdf.cell(0, 10, f"Confidence: {analysis['confidence']:.2f}%", ln=True)
    pdf.ln(10)
    
    # All Probabilities
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 15, 'Emotion Probabilities:', ln=True)
    pdf.set_font('Arial', '', 12)
    for i in range(5):
        pdf.cell(0, 8, f"{EMOTIONS[i]['name']}: {analysis['probs'][i]*100:.2f}%", ln=True)
    pdf.ln(10)
    
    # Transcript
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 15, 'Transcript:', ln=True)
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 8, analysis['transcript'])
    pdf.ln(10)
    
    # Metrics
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 15, 'Analysis Metrics:', ln=True)
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 8, f"Duration: {analysis['duration']:.2f} seconds", ln=True)
    pdf.cell(0, 8, f"Processing Latency: {analysis['latency']:.2f} seconds", ln=True)
    
    return pdf.output(dest='S').encode('latin-1')

def create_shareable_card(analysis):
    """Create shareable image card"""
    # Create image
    width, height = 800, 600
    img = Image.new('RGB', (width, height), COLORS['background'])
    draw = ImageDraw.Draw(img)
    
    # Try to load font, fallback to default
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
        font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except:
        font_large = ImageFont.load_default()
        font_medium = font_large
        font_small = font_large
    
    # Background gradient effect
    for y in range(height):
        color_val = int(255 - (y / height) * 20)
        draw.line([(0, y), (width, y)], fill=(color_val, color_val - 8, color_val - 16))
    
    # Title
    draw.text((width//2, 50), "Emotion Analysis", fill=COLORS['sidebar'], font=font_large, anchor='mt')
    
    # Primary emotion
    draw.text((width//2, 150), f"{analysis['emoji']}", fill='black', font=font_large, anchor='mt')
    draw.text((width//2, 230), analysis['emotion'], fill=COLORS['text'], font=font_medium, anchor='mt')
    draw.text((width//2, 280), f"{analysis['confidence']:.1f}% confidence", fill=COLORS['buttons'], font=font_small, anchor='mt')
    
    # Timestamp
    draw.text((width//2, 550), f"Analyzed on {analysis['timestamp']}", fill=COLORS['borders'], font=font_small, anchor='mt')
    
    # Save to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    return img_bytes

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
def render_sidebar():
    """Render sidebar with navigation"""
    with st.sidebar:
        # Logo/Title
        st.markdown(f"""
        <div style="text-align: center; padding: 20px 0;">
            <h1 style="color: white; font-size: 1.8rem; margin: 0;">🎭 Emotion AI</h1>
            <p style="color: rgba(255,255,255,0.7); font-size: 0.9rem;">Command Center</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation
        pages = {
            '🏠 Home': 'Home',
            '🎙️ Live Emotion': 'Live Emotion',
            '📁 Upload & Analyze': 'Upload',
            '📜 History': 'History',
            '📊 Analytics': 'Analytics',
            '⚙️ Settings': 'Settings',
            'ℹ️ About': 'About'
        }
        
        for icon_name, page_key in pages.items():
            is_active = st.session_state.current_page == page_key
            btn_type = "primary" if is_active else "secondary"
            if st.button(icon_name, key=f"nav_{page_key}", type=btn_type, use_container_width=True):
                st.session_state.current_page = page_key
                st.rerun()
        
        st.markdown("---")
        
        # Quick Stats
        st.markdown("### 📈 Quick Stats")
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 12px;">
            <p style="color: rgba(255,255,255,0.8); margin: 0; font-size: 0.85rem;">Total Analyses</p>
            <p style="color: white; font-size: 1.8rem; font-weight: 700; margin: 5px 0;">{st.session_state.total_analyses}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.badges:
            st.markdown("### 🏆 Badges")
            for badge in st.session_state.badges:
                st.markdown(f"<span class='badge badge-success'>🏅 {badge}</span>", unsafe_allow_html=True)

# ==========================================
# PAGE RENDERERS
# ==========================================
def render_home():
    """Render Home page"""
    st.markdown("""
    <div style="text-align: center; padding: 40px 0;">
        <h1 style="font-size: 3.5rem; margin-bottom: 20px;">🎭 Emotion Command Center</h1>
        <p style="font-size: 1.3rem; color: #666; max-width: 600px; margin: 0 auto;">
            Advanced multimodal AI that understands emotions through voice, powered by 
            <span style="color: #281C59; font-weight: 700;">99.33% accurate</span> neural networks.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="emotion-card" style="text-align: center;">
            <div style="font-size: 3rem; margin-bottom: 15px;">🎙️</div>
            <h3>Live Analysis</h3>
            <p>Real-time emotion detection from your microphone with animated visualizations.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="emotion-card" style="text-align: center;">
            <div style="font-size: 3rem; margin-bottom: 15px;">🧠</div>
            <h3>Multimodal AI</h3>
            <p>Fuses MFCC, Wav2Vec2, and semantic features for unparalleled accuracy.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="emotion-card" style="text-align: center;">
            <div style="font-size: 3rem; margin-bottom: 15px;">📊</div>
            <h3>Deep Insights</h3>
            <p>Comprehensive analytics with history tracking and export capabilities.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # CTA Buttons
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🎙️ Start Live Analysis", use_container_width=True):
                st.session_state.current_page = 'Live Emotion'
                st.rerun()
        with c2:
            if st.button("📁 Upload Audio", use_container_width=True):
                st.session_state.current_page = 'Upload'
                st.rerun()

def render_live_emotion(models):
    """Render Live Emotion page"""
    st.markdown("<h1>🎙️ Live Emotion Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #666;'>Speak into your microphone and watch the AI detect your emotions in real-time.</p>", unsafe_allow_html=True)
    
    # Recording Section
    st.markdown('<div class="emotion-card">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Recording button
        if not st.session_state.recording:
            if st.button("🔴 START RECORDING", key="start_record", use_container_width=True):
                st.session_state.recording = True
                st.session_state.recording_start_time = time.time()
                st.rerun()
        else:
            elapsed = time.time() - st.session_state.recording_start_time
            st.markdown(f"""
            <div style="text-align: center; padding: 20px;">
                <div style="font-size: 4rem; animation: pulse 1s infinite;">🔴</div>
                <p style="font-size: 1.5rem; font-weight: 700; color: #DC143C;">Recording... {elapsed:.1f}s</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("⏹️ STOP RECORDING", key="stop_record", use_container_width=True):
                st.session_state.recording = False
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Audio Input
    audio_input = st.audio_input("Or use quick capture")
    
    if audio_input:
        st.audio(audio_input)
        
        if st.button("🚀 Analyze Recording", use_container_width=True):
            with st.spinner("🧠 Processing audio..."):
                try:
                    analysis = perform_emotion_analysis(audio_input.getvalue(), models)
                    st.session_state.current_analysis = analysis
                    st.session_state.analysis_history.append(analysis)
                    st.session_state.total_analyses += 1
                    
                    # Check for badges
                    if len(st.session_state.analysis_history) == 1:
                        st.session_state.badges.append("First Analysis!")
                    if len(st.session_state.analysis_history) == 10:
                        st.session_state.badges.append("Emotion Explorer")
                    
                    st.success("✅ Analysis complete!")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # Display Results
    if st.session_state.current_analysis:
        render_analysis_results(st.session_state.current_analysis)

def render_upload(models):
    """Render Upload & Analyze page"""
    st.markdown("<h1>📁 Upload & Analyze</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #666;'>Upload any audio file for detailed emotion analysis.</p>", unsafe_allow_html=True)
    
    st.markdown('<div class="emotion-card">', unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'm4a', 'flac', 'ogg', 'aac'],
        help="Supported formats: WAV, MP3, M4A, FLAC, OGG, AAC"
    )
    
    if uploaded_file:
        st.audio(uploaded_file)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Analyze Audio", use_container_width=True):
                with st.spinner("🧠 Processing audio..."):
                    try:
                        analysis = perform_emotion_analysis(uploaded_file.getvalue(), models)
                        st.session_state.current_analysis = analysis
                        st.session_state.analysis_history.append(analysis)
                        st.session_state.total_analyses += 1
                        st.success("✅ Analysis complete!")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display Results
    if st.session_state.current_analysis:
        render_analysis_results(st.session_state.current_analysis)

def render_analysis_results(analysis):
    """Render analysis results with visualizations"""
    st.markdown("---")
    st.markdown("<h2>📊 Analysis Results</h2>", unsafe_allow_html=True)
    
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Dominant Emotion</div>
            <div class="metric-value">{analysis['emoji']}</div>
            <div style="font-weight: 600; color: {COLORS['text']};">{analysis['emotion']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Confidence</div>
            <div class="metric-value">{analysis['confidence']:.1f}%</div>
            <div style="font-weight: 600; color: {COLORS['text']};">certainty</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Latency</div>
            <div class="metric-value">{analysis['latency']:.2f}s</div>
            <div style="font-weight: 600; color: {COLORS['text']};">processing time</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Duration</div>
            <div class="metric-value">{analysis['duration']:.1f}s</div>
            <div style="font-weight: 600; color: {COLORS['text']};">audio length</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Main Visualizations
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_left, col_right = st.columns([1, 1.5])
    
    with col_left:
        st.markdown('<div class="emotion-card">', unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center;'>🎯 Emotion Circle</h3>", unsafe_allow_html=True)
        fig_circle = create_emotion_circle(analysis['probs'])
        st.plotly_chart(fig_circle, use_container_width=True, config={'displayModeBar': False})
        
        st.markdown(f"""
        <div style="text-align: center; margin-top: 10px;">
            <span class="badge badge-accent">{analysis['emotion']} {analysis['emoji']}</span>
            <span class="badge badge-primary">{analysis['confidence']:.1f}% confidence</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_right:
        # Tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["📊 Probabilities", "🕸️ Radar", "🎵 Spectrogram"])
        
        with tab1:
            st.markdown('<div class="emotion-card">', unsafe_allow_html=True)
            fig_bars = create_probability_bars(analysis['probs'])
            st.plotly_chart(fig_bars, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            st.markdown('<div class="emotion-card">', unsafe_allow_html=True)
            fig_radar = create_radar_chart(analysis['probs'])
            st.plotly_chart(fig_radar, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab3:
            st.markdown('<div class="emotion-card">', unsafe_allow_html=True)
            fig_spec = create_spectrogram(analysis['audio_array'], analysis['sr'])
            st.plotly_chart(fig_spec, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Transcript
    st.markdown('<div class="emotion-card">', unsafe_allow_html=True)
    st.markdown("<h3>📝 Transcript</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size: 1.1rem; font-style: italic; border-left: 4px solid {COLORS['buttons']}; padding-left: 15px;'>'{analysis['transcript']}'</p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Waveform
    st.markdown('<div class="emotion-card">', unsafe_allow_html=True)
    st.markdown("<h3>🌊 Audio Waveform</h3>", unsafe_allow_html=True)
    fig_wave = create_waveform(analysis['audio_array'], analysis['sr'])
    st.plotly_chart(fig_wave, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Export Options
    st.markdown('<div class="emotion-card">', unsafe_allow_html=True)
    st.markdown("<h3>📤 Export & Share</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pdf_bytes = generate_pdf_report(analysis)
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_bytes,
            file_name=f"emotion_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    
    with col2:
        img_bytes = create_shareable_card(analysis)
        st.download_button(
            label="🖼️ Download Share Card",
            data=img_bytes,
            file_name=f"emotion_card_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            mime="image/png",
            use_container_width=True
        )
    
    with col3:
        # JSON export
        json_data = json.dumps(analysis, indent=2, default=str)
        st.download_button(
            label="📋 Export JSON",
            data=json_data,
            file_name=f"emotion_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Recommendations based on emotion
    render_recommendations(analysis['emotion_idx'])

def render_recommendations(emotion_idx):
    """Render recommendations based on detected emotion"""
    recommendations = {
        0: {  # Happiness
            "songs": ["Happy by Pharrell Williams", "Can't Stop the Feeling by Justin Timberlake"],
            "activities": ["Share your joy with friends!", "Take a photo to capture the moment"],
            "joke": "Why don't scientists trust atoms? Because they make up everything! 😄"
        },
        1: {  # Neutral
            "songs": ["Weightless by Marconi Union", "Clair de Lune by Debussy"],
            "activities": ["Try a short meditation", "Take a walk outside"],
            "joke": "Why did the scarecrow win an award? He was outstanding in his field! 🌾"
        },
        2: {  # Anger
            "songs": ["Break Stuff by Limp Bizkit", "Given Up by Linkin Park"],
            "activities": ["Take 10 deep breaths", "Write down your thoughts", "Try a quick workout"],
            "joke": "I told my wife she was drawing her eyebrows too high. She looked surprised. 🤨"
        },
        3: {  # Sadness
            "songs": ["Fix You by Coldplay", "Someone Like You by Adele"],
            "activities": ["Talk to a friend", "Practice self-care", "Listen to uplifting podcasts"],
            "joke": "I used to be a banker, but I lost interest. 💰"
        },
        4: {  # Calmness
            "songs": ["River Flows in You by Yiruma", "Gymnopédie No. 1 by Erik Satie"],
            "activities": ["Read a book", "Practice yoga", "Enjoy a cup of tea"],
            "joke": "Why did the yoga instructor quit? She couldn't stretch her patience! 🧘"
        }
    }
    
    rec = recommendations.get(emotion_idx, recommendations[1])
    
    st.markdown('<div class="emotion-card">', unsafe_allow_html=True)
    st.markdown("<h3>💡 Recommendations</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<h4>🎵 Songs</h4>", unsafe_allow_html=True)
        for song in rec["songs"]:
            st.markdown(f"• {song}")
    
    with col2:
        st.markdown("<h4>🎯 Activities</h4>", unsafe_allow_html=True)
        for activity in rec["activities"]:
            st.markdown(f"• {activity}")
    
    with col3:
        st.markdown("<h4>😄 Joke</h4>", unsafe_allow_html=True)
        st.markdown(f"*{rec['joke']}*")
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_history():
    """Render History page"""
    st.markdown("<h1>📜 Analysis History</h1>", unsafe_allow_html=True)
    
    if not st.session_state.analysis_history:
        st.info("No analyses yet. Start by recording or uploading audio!")
        return
    
    # Filter options
    col1, col2 = st.columns([1, 3])
    with col1:
        filter_emotion = st.selectbox(
            "Filter by emotion",
            ["All"] + [EMOTIONS[i]['name'] for i in range(5)]
        )
    
    # Display history
    for idx, analysis in enumerate(reversed(st.session_state.analysis_history)):
        if filter_emotion != "All" and analysis['emotion'] != filter_emotion:
            continue
        
        with st.expander(f"{analysis['emoji']} {analysis['emotion']} - {analysis['timestamp']} ({analysis['confidence']:.1f}%)"):
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                st.markdown(f"**Confidence:** {analysis['confidence']:.1f}%")
                st.markdown(f"**Duration:** {analysis['duration']:.2f}s")
            
            with col2:
                st.markdown(f"**Transcript:** {analysis['transcript']}")
            
            with col3:
                if st.button("Load Analysis", key=f"load_{idx}"):
                    st.session_state.current_analysis = analysis
                    st.session_state.current_page = 'Live Emotion'
                    st.rerun()
    
    # Clear history button
    if st.button("🗑️ Clear History", type="secondary"):
        st.session_state.analysis_history = []
        st.session_state.badges = []
        st.rerun()

def render_analytics():
    """Render Analytics page"""
    st.markdown("<h1>📊 Analytics Dashboard</h1>", unsafe_allow_html=True)
    
    if not st.session_state.analysis_history:
        st.info("No data available. Analyze some audio first!")
        return
    
    # Calculate statistics
    emotions_count = defaultdict(int)
    confidence_over_time = []
    timestamps = []
    
    for analysis in st.session_state.analysis_history:
        emotions_count[analysis['emotion']] += 1
        confidence_over_time.append(analysis['confidence'])
        timestamps.append(analysis['timestamp'])
    
    # Emotion Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="emotion-card">', unsafe_allow_html=True)
        st.markdown("<h3>🥧 Emotion Distribution</h3>", unsafe_allow_html=True)
        
        fig_pie = px.pie(
            names=list(emotions_count.keys()),
            values=list(emotions_count.values()),
            color=list(emotions_count.keys()),
            color_discrete_map={EMOTIONS[i]['name']: EMOTIONS[i]['color'] for i in range(5)}
        )
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=True
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="emotion-card">', unsafe_allow_html=True)
        st.markdown("<h3>📈 Confidence Trend</h3>", unsafe_allow_html=True)
        
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=list(range(len(confidence_over_time))),
            y=confidence_over_time,
            mode='lines+markers',
            line=dict(color=COLORS['buttons'], width=3),
            marker=dict(size=10, color=COLORS['accent'])
        ))
        fig_line.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Analysis #",
            yaxis_title="Confidence (%)",
            yaxis_range=[0, 100]
        )
        st.plotly_chart(fig_line, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Summary Stats
    st.markdown('<div class="emotion-card">', unsafe_allow_html=True)
    st.markdown("<h3>📋 Summary Statistics</h3>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_confidence = np.mean(confidence_over_time)
        st.metric("Average Confidence", f"{avg_confidence:.1f}%")
    
    with col2:
        most_common = max(emotions_count, key=emotions_count.get)
        st.metric("Most Common Emotion", most_common)
    
    with col3:
        st.metric("Total Analyses", len(st.session_state.analysis_history))
    
    with col4:
        unique_emotions = len(emotions_count)
        st.metric("Emotions Detected", f"{unique_emotions}/5")
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_settings():
    """Render Settings page"""
    st.markdown("<h1>⚙️ Settings</h1>", unsafe_allow_html=True)
    
    st.markdown('<div class="emotion-card">', unsafe_allow_html=True)
    st.markdown("<h3>👤 User Preferences</h3>", unsafe_allow_html=True)
    
    user_name = st.text_input("Your Name", value=st.session_state.user_name)
    if user_name != st.session_state.user_name:
        st.session_state.user_name = user_name
    
    language = st.selectbox(
        "Language",
        ["English", "Urdu", "Hindi"],
        index=["English", "Urdu", "Hindi"].index(st.session_state.language)
    )
    if language != st.session_state.language:
        st.session_state.language = language
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="emotion-card">', unsafe_allow_html=True)
    st.markdown("<h3>🎤 Microphone Permissions</h3>", unsafe_allow_html=True)
    
    mic_perm = st.toggle(
        "Allow microphone access",
        value=st.session_state.mic_permission
    )
    st.session_state.mic_permission = mic_perm
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="emotion-card">', unsafe_allow_html=True)
    st.markdown("<h3>🗑️ Data Management</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Clear History", type="secondary", use_container_width=True):
            st.session_state.analysis_history = []
            st.success("History cleared!")
    
    with col2:
        if st.button("Reset All Data", type="secondary", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            init_session_state()
            st.success("All data reset!")
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_about(models):
    """Render About page"""
    st.markdown("<h1>ℹ️ About Emotion AI</h1>", unsafe_allow_html=True)
    
    st.markdown('<div class="emotion-card">', unsafe_allow_html=True)
    st.markdown("""
    <h3>🎯 Mission</h3>
    <p>Our Emotion Command Center uses cutting-edge multimodal AI to understand human emotions through voice, 
    enabling better human-computer interaction and emotional intelligence in technology.</p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="emotion-card">', unsafe_allow_html=True)
    st.markdown("<h3>🧠 Model Architecture</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="pipeline-container" style="margin: 20px 0;">
        <div class="pipeline-step">🎙️ Audio Input</div>
        <div class="pipeline-arrow">➔</div>
        <div class="pipeline-step">📊 MFCC Features</div>
        <div class="pipeline-arrow">➔</div>
        <div class="pipeline-step">🤖 Wav2Vec2</div>
        <div class="pipeline-arrow">➔</div>
        <div class="pipeline-step">📝 RoBERTa</div>
        <div class="pipeline-arrow">➔</div>
        <div class="pipeline-step">🔗 Fusion</div>
        <div class="pipeline-arrow">➔</div>
        <div class="pipeline-step">🎯 Prediction</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Traditional Features:**
        - 40 MFCC coefficients
        - Mean, Std, Max, Min statistics
        - 160-dimensional vector
        """)
    
    with col2:
        st.markdown("""
        **Deep Features:**
        - Wav2Vec2 embeddings (768-dim)
        - RoBERTa semantic features (768-dim)
        - Attention-based fusion
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="emotion-card">', unsafe_allow_html=True)
    st.markdown("<h3>📊 Model Performance</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Accuracy", "99.33%")
    with col2:
        st.metric("Emotions", "5 Classes")
    with col3:
        device = "GPU" if torch.cuda.is_available() else "CPU"
        st.metric("Device", device)
    
    st.markdown(f"""
    <p style="margin-top: 15px;">
        <strong>Model Status:</strong> 
        <span class="badge badge-success">{'✅ Loaded' if models['model_loaded'] else '⚠️ Demo Mode'}</span>
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="emotion-card">', unsafe_allow_html=True)
    st.markdown("<h3>👨‍💻 Development Team</h3>", unsafe_allow_html=True)
    st.markdown("""
    <p>Developed as part of a thesis research project on multimodal emotion recognition.</p>
    <p><strong>Technologies:</strong> PyTorch, Transformers, Librosa, Streamlit, Plotly</p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# MAIN APPLICATION
# ==========================================
def main():
    """Main application entry point"""
    # Set page config
    st.set_page_config(**PAGE_CONFIG)
    
    # Initialize session state
    init_session_state()
    
    # Apply custom styling
    apply_custom_styling()
    
    # Load models
    models = load_all_models()
    
    # Render sidebar
    render_sidebar()
    
    # Render current page
    page = st.session_state.current_page
    
    if page == 'Home':
        render_home()
    elif page == 'Live Emotion':
        render_live_emotion(models)
    elif page == 'Upload':
        render_upload(models)
    elif page == 'History':
        render_history()
    elif page == 'Analytics':
        render_analytics()
    elif page == 'Settings':
        render_settings()
    elif page == 'About':
        render_about(models)

if __name__ == "__main__":
    main()
