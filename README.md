
# 🎭 Emotion-Guided Smart Text Summarizer

**Transform audio into intelligent summaries driven by emotional insights**

## Overview

This project combines Speech-to-Text, Emotion Recognition, and NLP to generate context-aware summaries. The system analyzes the emotional tone of each sentence in an audio recording and uses that information to create more meaningful and representative summaries.

### Key Features

- 🎙️ **Speech-to-Text**: Vosk-based transcription with sentence segmentation
- 💭 **Emotion Recognition**: TensorFlow-based emotion detection for each sentence
- 📝 **Smart Summarization**: NLTK-powered summarization guided by emotional importance
- 🎯 **Interactive Demo**: Streamlit web interface for easy testing
- 📊 **Analytics**: Detailed emotion distribution and sentence-level analysis
- ⬇️ **Export**: Download results as CSV or TXT

## Quick Start

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd Emotion-Guided-Smart-Text-Summarizer
```

2. **Create virtual environment**
```bash
python -m venv emotion_env
source emotion_env/Scripts/activate  # On Windows: emotion_env\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
python setup_nltk.py
```

4. **Download Vosk model** (already included in repo)
   - Ensure `vosk_model/` directory exists with the model files

### 🚀 Run the Interactive Demo

Start the Streamlit app to test the system with a web interface:

```bash
streamlit run streamlit_app.py
```

Then open your browser to `http://localhost:8501`

**Features:**
- Upload/drag-and-drop audio files (WAV, MP3, M4A, OGG)
- Real-time transcription and emotion analysis
- Generated summaries with emotion-guided prioritization
- Interactive charts and analytics
- Download results as CSV/TXT

### 📝 Command Line Usage

```python
from src.main import run_emotion_aware_summarization
from src.stt.transcriber import VoskTranscriber

# Initialize transcriber
vosk_model_dir = "vosk_model"
transcriber = VoskTranscriber(vosk_model_dir)

# Process audio
summary, emotions, analytics, segments = run_emotion_aware_summarization(
    audio_file_path="data/raw/sample.wav",
    vosk_transcriber_instance=transcriber,
    num_summary_sentences=4,
    data_export_path="data/processed/results.csv"
)

print(f"Summary: {summary}")
print(f"Emotions: {emotions}")
```

## Model Training

To train your own emotion recognition model:

```bash
python rebuild_models.py
python src/emotion_recognition/convert_to_tflite.py
```

This will generate:
- `models/ser_model.tflite` - TFLite emotion recognition model
- `models/label_encoder.pkl` - Label encoder for emotions
- `models/scaler.pkl` - Feature scaler

## Project Structure

```
├── src/
│   ├── emotion_recognition/      # Emotion recognition module
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── convert_to_tflite.py
│   │   └── preprocess.py
│   ├── stt/                       # Speech-to-text module
│   │   └── transcriber.py
│   ├── text_summarization/        # Summarization module
│   │   ├── summarizer.py
│   │   └── emotion_integration.py
│   └── main.py                    # Main processing pipeline
├── models/                        # Pre-trained models
├── data/                          # Data directory
│   ├── raw/                       # Raw audio files
│   └── processed/                 # Processed features & results
├── streamlit_app.py              # 🆕 Interactive web demo
├── requirements.txt
└── README.md
```



## Contact & Support

For questions or issues, please open an issue on GitHub or reach out via email.

---
