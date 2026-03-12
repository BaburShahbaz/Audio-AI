# 🎭 Emotion Command Center

A **production-ready multimodal emotion detection application** that analyzes audio to detect emotions with **99.33% accuracy** using advanced deep learning techniques.

![Emotion AI](https://img.shields.io/badge/Accuracy-99.33%25-success)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)

## ✨ Features

### 🎯 Core Capabilities
- **Real-time emotion detection** from microphone input
- **Upload & analyze** any audio format (WAV, MP3, M4A, FLAC, OGG, AAC)
- **Multimodal fusion** of MFCC, Wav2Vec2, and semantic features
- **99.33% accuracy** on 5 emotion classes

### 🎨 Visualizations
- **Interactive Emotion Circle** - Animated dot moves based on probabilities
- **Radar Chart** - Compare all emotion probabilities
- **Probability Bars** - Horizontal bar chart visualization
- **Mel Spectrogram** - Visual representation of audio frequencies
- **Waveform Display** - Audio amplitude over time

### 📊 Analytics & History
- **Session history** with persistent storage
- **Analytics dashboard** with emotion distribution and trends
- **Confidence tracking** over time
- **Filter and search** past analyses

### 📤 Export & Share
- **PDF Reports** - Generate comprehensive emotion analysis reports
- **Share Cards** - Create image cards for social sharing
- **JSON Export** - Raw data export for further analysis

### 💡 Smart Features
- **Emotion-based recommendations** - Songs, activities, and jokes
- **Badge system** - Gamification for frequent users
- **Multi-language support** - English, Urdu, Hindi
- **Theme customization** - Light mode with custom color scheme

## 🎨 Color Scheme

| Element | Color |
|---------|-------|
| Background | `#FFF8F0` |
| Sidebar | `#281C59` |
| Buttons | `#4E8D9C` |
| Success | `#85C79A` |
| Highlights | `#EDF7BD` |
| Accent | `#C08552` |
| Borders | `#8C5A3C` |
| Text | `#4B2E2B` |

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster inference)
- Microphone access (for live recording)

### Installation

1. **Clone or download the repository:**
```bash
cd emotion_app
```

2. **Create a virtual environment:**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download model files:**
Place your trained model files in the app directory:
- `best_emotion_model.pt` - Your trained AttentionFusionModel
- `scalers.pkl` - Feature scalers from training

> **Note:** If model files are not found, the app will run in demo mode with random predictions.

5. **Run the application:**
```bash
streamlit run app.py
```

6. **Open in browser:**
The app will automatically open at `http://localhost:8501`

## 📁 Project Structure

```
emotion_app/
├── app.py                 # Main application code
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── best_emotion_model.pt # Your trained model (add this)
└── scalers.pkl          # Feature scalers (add this)
```

## 🧠 Model Architecture

### Feature Extraction Pipeline

```
Audio Input (16kHz)
    │
    ├──► Traditional Features (160-dim)
    │    ├── 40 MFCC coefficients
    │    ├── Mean, Std, Max, Min statistics
    │    └── StandardScaler normalization
    │
    ├──► Wav2Vec2 Features (768-dim)
    │    ├── facebook/wav2vec2-base-960h
    │    ├── Mean pooling over time
    │    └── StandardScaler normalization
    │
    └──► Semantic Features (768-dim)
         ├── Whisper transcription
         ├── RoBERTa sentiment model
         ├── CLS token embedding
         └── StandardScaler normalization
```

### Attention Fusion Model

```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  Traditional    │  │    Wav2Vec2     │  │    Semantic     │
│    (160-dim)    │  │   (768-dim)     │  │   (768-dim)     │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   FC(256)       │  │   FC(256)       │  │   FC(256)       │
│  + BatchNorm    │  │  + BatchNorm    │  │  + BatchNorm    │
│  + ReLU         │  │  + ReLU         │  │  + ReLU         │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Concatenate    │
                    │   (768-dim)     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Attention      │
                    │  (3 weights)    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Weighted Sum   │
                    │   (256-dim)     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Classifier     │
                    │  FC(128) → FC(5)│
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Softmax Output │
                    │  5 Emotions     │
                    └─────────────────┘
```

### Emotion Classes

| Index | Emotion | Emoji | Color |
|-------|---------|-------|-------|
| 0 | Happiness | 😄 | #FFD700 |
| 1 | Neutral | 😐 | #808080 |
| 2 | Anger | 😠 | #DC143C |
| 3 | Sadness | 😢 | #4169E1 |
| 4 | Calmness | 😌 | #90EE90 |

## 📖 Usage Guide

### 🏠 Home Page
- Overview of the application
- Quick access to Live Analysis and Upload features

### 🎙️ Live Emotion
1. Click "START RECORDING" to begin
2. Speak into your microphone
3. Click "STOP RECORDING" when done
4. Click "Analyze Recording" to process
5. View results with interactive visualizations

### 📁 Upload & Analyze
1. Click "Choose an audio file"
2. Select any supported audio file
3. Click "Analyze Audio"
4. View detailed results

### 📜 History
- View all past analyses
- Filter by emotion type
- Load previous analyses
- Clear history

### 📊 Analytics
- Emotion distribution pie chart
- Confidence trend over time
- Summary statistics

### ⚙️ Settings
- Set your name
- Change language
- Manage microphone permissions
- Clear data

### ℹ️ About
- Model information
- Architecture details
- Performance metrics

## 📤 Export Options

### PDF Report
Comprehensive report including:
- Primary emotion with confidence
- All emotion probabilities
- Transcript
- Analysis metrics (duration, latency)

### Share Card
Image card with:
- Emotion emoji and name
- Confidence percentage
- Timestamp

### JSON Export
Raw analysis data for:
- Further processing
- Integration with other tools
- Data analysis

## 💡 Recommendations

Based on detected emotion, the app suggests:
- **Songs** - Curated playlist for each emotion
- **Activities** - Actions to enhance or manage the emotion
- **Jokes** - Light humor to brighten the mood

## 🏆 Badge System

Earn badges for:
- 🥇 **First Analysis!** - Complete your first analysis
- 🏅 **Emotion Explorer** - Complete 10 analyses
- (More badges coming soon!)

## 🔧 Troubleshooting

### Model Not Found
If you see "⚠️ Trained model not found":
- Ensure `best_emotion_model.pt` is in the app directory
- The app will run in demo mode with random predictions

### Scalers Not Found
If you see "⚠️ Scalers not found":
- Ensure `scalers.pkl` is in the app directory
- Default scalers will be used (may affect accuracy)

### Microphone Issues
- Check browser permissions
- Ensure microphone is connected
- Try refreshing the page

### CUDA Out of Memory
- Reduce batch size (modify in code)
- Use CPU mode (automatic fallback)
- Close other GPU applications

## 📝 Requirements

### Minimum Requirements
- Python 3.8+
- 4GB RAM
- 2GB free disk space

### Recommended
- Python 3.10+
- 8GB+ RAM
- CUDA-capable GPU
- 5GB+ free disk space

## 🤝 Contributing

This is a thesis research project. For questions or collaborations:
- Open an issue on GitHub
- Contact the development team

## 📄 License

This project is for academic research purposes.

## 🙏 Acknowledgments

- **Whisper** by OpenAI for speech recognition
- **Wav2Vec2** by Facebook AI for audio embeddings
- **RoBERTa** by Cardiff NLP for sentiment analysis
- **Streamlit** for the web framework
- **Plotly** for interactive visualizations

---

<p align="center">
  Made with ❤️ for emotion AI research
</p>
