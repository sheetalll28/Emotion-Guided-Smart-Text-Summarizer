
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

## Supported Emotions

The emotion recognition model detects:
- 😢 Sadness
- 😊 Happiness
- 😠 Anger
- 😲 Surprise
- 😟 Fear
- 🤢 Disgust
- 😐 Neutral

## 🌐 Deployment Options

### Option 1: Streamlit Cloud (Recommended - Free & Easy)

1. Push your repo to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Click "New app" → Select your repo → point to `streamlit_app.py`
4. Share the public URL

**Get a shareable link like:** `https://emotion-guided-summarizer.streamlit.app`

### Option 2: Heroku

```bash
# Create Procfile
echo "web: streamlit run streamlit_app.py --logger.level=error" > Procfile

# Create .streamlit/config.toml
mkdir .streamlit
echo "[server]
headless = true
port = \$PORT
enableXsrfProtection = false" > .streamlit/config.toml

# Deploy
git push heroku main
```

### Option 3: Docker

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "streamlit_app.py"]
```

### Option 4: Local Demo (For Resume)

Simply share:
- **Screenshot of the app**: `streamlit_app.py` running with sample output
- **GitHub link**: Full source code
- **Video demo**: Screen recording of the workflow

## Performance Metrics

- **Transcription Speed**: ~0.5-2x real-time (Vosk)
- **Emotion Recognition**: <200ms per sentence (TFLite)
- **Summary Generation**: <500ms (NLTK)

## Requirements

See [requirements.txt](requirements.txt) for full dependencies:
- TensorFlow/TFLite for emotion recognition
- Vosk for speech-to-text
- NLTK for summarization
- Streamlit for web interface
- Librosa for audio processing
- pandas for data handling

## Future Enhancements

- [ ] Support for multiple languages
- [ ] Real-time audio recording in web interface
- [ ] Custom emotion weighting
- [ ] Text-to-speech for summary playback
- [ ] Batch processing capability
- [ ] API endpoint for integration

## License

MIT License - feel free to use this project!

## Contact & Support

For questions or issues, please open an issue on GitHub or reach out via email.

---

**Ready for your resume?** 
✅ Fully functional demo app
✅ Professional UI with Streamlit  
✅ Easy deployment to cloud
✅ Shareable public link
=======
# Emotion-Aware Speech Summarization System

## Overview
This project is an end-to-end audio-to-text summarization system that generates context-aware summaries by combining semantic relevance, emotion analysis, and confidence scoring.

The system is designed to work efficiently on low-compute environments by using lightweight speech recognition and optimized processing techniques.

---

## Features

- **Speech-to-Text Conversion**  
  Converts audio input into text using Vosk, enabling offline and resource-efficient processing.

- **Custom Punctuation Restoration**  
  Implements a timestamp-based approach to reconstruct sentence boundaries by analyzing pauses in speech.

- **Emotion-Aware Summarization**  
  Generates summaries using a hybrid scoring mechanism based on:
  - Semantic relevance  
  - Emotion weighting  
  - Confidence scores  

- **Efficient Processing**  
  Designed to run on resource-constrained devices with minimal computational overhead.

---

## Technologies Used

### Languages
- Python  

### Libraries and Tools
- Vosk (speech recognition)  
- librosa (audio feature extraction)  
- NumPy, pandas  

### Concepts
- Natural Language Processing (NLP)  
- Speech Processing  
- Deep Neural Networks  
- Text Summarization  

---
>>>>>>> 58d3a3b4cc0fe9c4c83c8efc338ce903f1534941
