import streamlit as st
import os
import tempfile
import pandas as pd
from pydub import AudioSegment
import logging
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json

# Suppress warnings
logging.getLogger('pydub').setLevel(logging.WARNING)
logging.getLogger('vosk').setLevel(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# Import project modules
from src.stt.transcriber import VoskTranscriber
from src.text_summarization.emotion_integration import EmotionAwareSummarizer

# Page configuration
st.set_page_config(
    page_title="Emotion-Guided Text Summarizer",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Custom CSS
st.markdown("""
    <style>
    :root {
        --primary: #667eea;
        --secondary: #764ba2;
        --accent: #f093fb;
    }
    
    * {
        margin: 0;
        padding: 0;
    }
    
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .main {
        padding: 2.5rem 3rem;
        background: linear-gradient(to bottom, #f5f7fa 0%, #e8ecf1 100%);
        min-height: 100vh;
    }
    
    /* Typography */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.2rem;
        font-weight: 900;
        letter-spacing: -1px;
        margin-bottom: 0.3rem;
    }
    
    h2 {
        color: #667eea;
        font-size: 2rem;
        margin-top: 2rem;
        margin-bottom: 1.2rem;
        font-weight: 800;
    }
    
    h3 {
        color: #764ba2;
        font-size: 1.3rem;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
        font-weight: 700;
    }
    
    p.subtitle {
        color: #666;
        font-size: 1.15rem;
        font-weight: 500;
        margin-bottom: 2.5rem;
        letter-spacing: 0.5px;
    }
    
    /* Metric Cards */
    .metric-box {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9ff 100%);
        border-radius: 15px;
        padding: 1.8rem;
        border: 2px solid #e0e6ff;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.12);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        text-align: center;
    }
    
    .metric-box:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.2);
        border-color: #667eea;
    }
    
    .metric-label {
        color: #999;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 0.8rem;
    }
    
    .metric-value {
        color: #667eea;
        font-size: 2.5rem;
        font-weight: 900;
    }
    
    .main > div {
        max-width: 1400px;
        margin: 0 auto;
    }
    
    /* Summary Box */
    .summary-box {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9ff 100%);
        border-left: 6px solid #667eea;
        border-radius: 12px;
        padding: 2.5rem;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.15);
        line-height: 1.85;
        color: #333;
        font-size: 1.08rem;
    }
    
    /* Emotion Badges */
    .emotion-badge {
        display: inline-block;
        padding: 0.7rem 1.4rem;
        border-radius: 30px;
        font-weight: 800;
        margin: 0.4rem;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    .emotion-badge:hover {
        transform: scale(1.12) translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
    }
    
    .emotion-happy { background: linear-gradient(135deg, #FFD93D 0%, #FFA500 100%); color: white; }
    .emotion-sad { background: linear-gradient(135deg, #4A90E2 0%, #357ABD 100%); color: white; }
    .emotion-angry { background: linear-gradient(135deg, #FF6B6B 0%, #EE5A52 100%); color: white; }
    .emotion-neutral { background: linear-gradient(135deg, #95A5A6 0%, #7F8C8D 100%); color: white; }
    .emotion-fear { background: linear-gradient(135deg, #6C5CE7 0%, #5A40B3 100%); color: white; }
    .emotion-disgust { background: linear-gradient(135deg, #FD79A8 0%, #E94B7A 100%); color: white; }
    .emotion-surprise { background: linear-gradient(135deg, #74B9FF 0%, #5DA6DD 100%); color: white; }
    
    /* Sentence Card */
    .sentence-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9ff 100%);
        border-radius: 12px;
        padding: 1.6rem;
        margin: 1.2rem 0;
        border-left: 5px solid #667eea;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.1);
        transition: all 0.3s ease;
    }
    
    .sentence-card:hover {
        transform: translateX(6px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2);
        border-left-color: #764ba2;
    }
    
    .sentence-text {
        color: #333;
        font-size: 1.02rem;
        line-height: 1.7;
        margin: 0.8rem 0;
        font-style: italic;
    }
    
    .sentence-num {
        color: #667eea;
        font-weight: 900;
        font-size: 1.2rem;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] button {
        font-weight: 700;
        font-size: 1rem;
        padding: 1rem 2.2rem;
        border-radius: 12px 12px 0 0;
        color: #999;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: #667eea;
        border-bottom: 4px solid #667eea;
        background: rgba(102, 126, 234, 0.05);
    }
    
    /* Divider */
    .divider-line {
        height: 3px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2.5rem 0;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 1rem 2.5rem;
        font-weight: 800;
        font-size: 1.05rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        color: white !important;
    }
    
    /* Messages */
    .stSuccess, .stError, .stInfo, .stWarning {
        border-radius: 12px;
        padding: 1.2rem;
        font-weight: 500;
    }
    
    .stSuccess {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        border: none;
    }
    
    .stError {
        background: linear-gradient(135deg, #f44336 0%, #ea1c1c 100%);
        color: white;
        border: none;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
    }
    
    /* Progress */
    .stProgress > div > div > div > div {
        background-color: #667eea;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #999;
        font-size: 0.95rem;
        margin-top: 4rem;
        padding-top: 2rem;
        border-top: 2px solid rgba(102, 126, 234, 0.2);
    }
    
    .footer h4 {
        color: #667eea;
        font-size: 1.3rem;
        margin-bottom: 0.8rem;
        font-weight: 800;
    }
    
    .footer p {
        margin: 0.4rem 0;
        line-height: 1.6;
    }
    
    .footer-dev {
        color: #667eea;
        font-weight: 700;
        font-size: 0.95rem;
        margin-top: 1rem;
    }
    
    /* Animations */
    @keyframes fadeDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    .fade-down { animation: fadeDown 0.5s ease-out; }
    .pulse { animation: pulse 1.5s infinite; }
    </style>
""", unsafe_allow_html=True)

# Header
col_title, col_space = st.columns([0.95, 0.05])
with col_title:
    st.markdown("# 🎭 Emotion-Guided Smart Text Summarizer")
    st.markdown('<p class="subtitle">Analyze speech with emotion recognition and intelligent summarization</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)
    
    num_summary_sentences = st.slider(
        "📝 Summary Length (sentences)",
        min_value=1,
        max_value=10,
        value=4,
        help="How many sentences should the summary contain?"
    )
    
    st.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)
    st.markdown("### 📋 Processing Pipeline")
    
    pipeline = [
        ("1️⃣", "Upload Audio File"),
        ("2️⃣", "Speech-to-Text Transcription"),
        ("3️⃣", "Emotion Recognition"),
        ("4️⃣", "Generate Smart Summary"),
        ("5️⃣", "View Analytics & Download")
    ]
    
    for icon, step in pipeline:
        st.markdown(f"**{icon}** {step}")
    
    st.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)
    st.markdown("### ℹ️ About")
    st.info("""
    **Technology Stack:**
    - TensorFlow Lite for Emotion Recognition
    - Vosk for Speech-to-Text
    - NLTK for Text Summarization
    
    **Emotions Detected:**
    Happy, Sad, Angry, Neutral, Fear, Disgust, Surprise
    """)

# Initialize session state
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "summary_result" not in st.session_state:
    st.session_state.summary_result = None

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎙️ Upload & Process", "📊 Live Results", "📈 Analytics", "ℹ️ About"])

# TAB 1: Upload & Process
with tab1:
    st.markdown("## 🎙️ Upload Your Audio")
    st.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📂 Select Audio File")
        uploaded_file = st.file_uploader(
            "Drag & drop or click to upload",
            type=["wav", "mp3", "m4a", "ogg"],
            help="Supported: WAV, MP3, M4A, OGG"
        )
    
    with col2:
        st.info("""
        **Tips for best results:**
        - Clear, high-quality audio
        - Minimal background noise
        - Natural speaking pace
        """)
    
    st.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        st.markdown("### 🎵 Preview")
        st.audio(uploaded_file)
        
        file_size = len(uploaded_file.getbuffer()) / (1024 * 1024)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="metric-box"><div class="metric-label">📊 File Size</div><div class="metric-value">{file_size:.2f}MB</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-box"><div class="metric-label">🎵 Format</div><div class="metric-value">{uploaded_file.name.split(".")[-1].upper()}</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-box"><div class="metric-label">📝 Name</div><div class="metric-value">{uploaded_file.name[:15]}...</div></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)
        
        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            temp_audio_path = tmp_file.name
        
        # Process button
        if st.button("🚀 Process Audio", type="primary", use_container_width=True):
            status_placeholder = st.empty()
            progress_placeholder = st.empty()
            
            try:
                tflite_model_path = os.path.join('models', 'ser_model.tflite')
                label_encoder_path = os.path.join('models', 'label_encoder.pkl')
                
                if not os.path.exists(tflite_model_path) or not os.path.exists(label_encoder_path):
                    st.error("⚠️ Models not found. Please follow the setup instructions to initialize the emotion recognition models.")
                else:
                    # Step 1
                    with status_placeholder.container():
                        st.info("📂 Loading audio file...")
                    progress_placeholder.progress(10)
                    
                    audio = AudioSegment.from_file(temp_audio_path)
                    audio_duration = audio.duration_seconds
                    
                    # Step 2
                    with status_placeholder.container():
                        st.info("🎤 Transcribing speech to text...")
                    progress_placeholder.progress(30)
                    
                    vosk_model_dir = os.path.join(os.getcwd(), "vosk_model")
                    transcriber = VoskTranscriber(vosk_model_dir)
                    full_text, word_details = transcriber.transcribe_audio(temp_audio_path)
                    sentence_segments = transcriber.get_sentence_segments_from_words(word_details, full_text, pause_threshold=0.5)
                    
                    if not sentence_segments:
                        st.error("⚠️ No sentences detected. Try using clearer audio.")
                    else:
                        # Step 3
                        with status_placeholder.container():
                            st.info("🎯 Extracting audio segments...")
                        progress_placeholder.progress(50)
                        
                        actual_audio_segments = []
                        text_sentences = []
                        
                        for segment in sentence_segments:
                            start_ms = int(segment['start'] * 1000)
                            end_ms = int(segment['end'] * 1000)
                            
                            if start_ms >= 0 and end_ms <= len(audio) and end_ms > start_ms:
                                actual_audio_segments.append(audio[start_ms:end_ms])
                                text_sentences.append(segment['text'])
                            else:
                                actual_audio_segments.append(AudioSegment.silent(duration=100))
                                text_sentences.append(segment['text'])
                        
                        # Step 4
                        with status_placeholder.container():
                            st.info("😊 Detecting emotions and generating summary...")
                        progress_placeholder.progress(70)
                        
                        summarizer = EmotionAwareSummarizer(tflite_model_path, label_encoder_path)
                        aligned_full_text = " ".join(text_sentences)
                        summary, emotions_data = summarizer.summarize_with_emotion(aligned_full_text, actual_audio_segments, num_sentences=num_summary_sentences)
                        
                        # Step 5
                        with status_placeholder.container():
                            st.info("📊 Compiling results...")
                        progress_placeholder.progress(90)
                        
                        analytics_data = []
                        final_emotion_output = []
                        
                        for i, (text, audio_seg) in enumerate(zip(text_sentences, actual_audio_segments)):
                            if i < len(emotions_data):
                                e_text, e_label, e_conf = emotions_data[i]
                                is_in_summary = e_text in summary
                                
                                analytics_data.append({
                                    'Sentence': text,
                                    'Emotion': e_label,
                                    'Confidence': f"{e_conf:.2%}",
                                    'In Summary': '✅' if is_in_summary else '❌'
                                })
                                final_emotion_output.append((text, e_label, e_conf))
                        
                        st.session_state.summary_result = {
                            'summary': summary,
                            'emotions': final_emotion_output,
                            'analytics': analytics_data,
                            'segments': sentence_segments,
                            'audio_duration': audio_duration,
                            'total_sentences': len(text_sentences)
                        }
                        st.session_state.processing_complete = True
                        
                        progress_placeholder.progress(100)
                        status_placeholder.empty()
                        st.success("✅ Processing complete! Check the 'Live Results' tab to view the analysis.")
                        
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
            finally:
                try:
                    os.remove(temp_audio_path)
                except:
                    pass
    else:
        st.info("👈 Upload an audio file to begin the analysis")

# TAB 2: Live Results
with tab2:
    if st.session_state.processing_complete and st.session_state.summary_result:
        result = st.session_state.summary_result
        
        st.markdown("## 📊 Processing Summary")
        st.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f'<div class="metric-box"><div class="metric-label">📊 Sentences Analyzed</div><div class="metric-value">{result["total_sentences"]}</div></div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'<div class="metric-box"><div class="metric-label">⏱️ Duration</div><div class="metric-value">{result["audio_duration"]:.1f}s</div></div>', unsafe_allow_html=True)
        
        with col3:
            emotion_counts = {}
            for _, emotion, _ in result['emotions']:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            dominant = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "N/A"
            st.markdown(f'<div class="metric-box"><div class="metric-label">🎭 Dominant Emotion</div><div class="metric-value">{dominant}</div></div>', unsafe_allow_html=True)
        
        with col4:
            avg_conf = np.mean([e[2] for e in result['emotions']]) if result['emotions'] else 0
            st.markdown(f'<div class="metric-box"><div class="metric-label">🎯 Avg Confidence</div><div class="metric-value">{avg_conf:.0%}</div></div>', unsafe_allow_html=True)
        
        # Summary
        st.markdown("---")
        st.markdown("## ✨ Generated Summary")
        st.markdown(f'<div class="summary-box">{result["summary"]}</div>', unsafe_allow_html=True)
        
        # Sentences with emotions
        st.markdown("---")
        st.markdown("## 🎭 Sentence-by-Sentence Analysis")
        
        emotion_classes = {
            'angry': 'emotion-angry',
            'disgust': 'emotion-disgust',
            'fear': 'emotion-fear',
            'happy': 'emotion-happy',
            'neutral': 'emotion-neutral',
            'sad': 'emotion-sad',
            'surprise': 'emotion-surprise'
        }
        
        for i, (text, emotion, confidence) in enumerate(result['emotions'], 1):
            emotion_class = emotion_classes.get(emotion.lower(), 'emotion-neutral')
            
            html = f'''
            <div class="sentence-card">
                <div style="display: flex; justify-content: space-between; align-items: start; gap: 1rem;">
                    <div style="flex: 1;">
                        <span class="sentence-num">#{i}</span>
                        <p class="sentence-text">{text}</p>
                    </div>
                    <div style="white-space: nowrap;">
                        <span class="emotion-badge {emotion_class}">{emotion.upper()} • {confidence:.0%}</span>
                    </div>
                </div>
            </div>
            '''
            st.markdown(html, unsafe_allow_html=True)
    else:
        st.info("👈 Upload and process audio in the 'Upload & Process' tab!")

# TAB 3: Analytics
with tab3:
    if st.session_state.processing_complete and st.session_state.summary_result:
        result = st.session_state.summary_result
        
        st.markdown("## 📈 Detailed Analytics")
        st.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📋 Results Table")
            analytics_df = pd.DataFrame(result['analytics'])
            st.dataframe(analytics_df, use_container_width=True, height=400)
        
        with col2:
            st.markdown("### 🥧 Distribution")
            emotion_counts = {}
            for _, emotion, _ in result['emotions']:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            emotion_df = pd.DataFrame(list(emotion_counts.items()), columns=['Emotion', 'Count']).sort_values('Count', ascending=False)
            
            fig = px.pie(emotion_df, values='Count', names='Emotion', hole=0.35)
            fig.update_layout(height=400, showlegend=True, margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 📊 Emotion Frequency Chart")
        
        fig = px.bar(emotion_df.sort_values('Count', ascending=True), 
                     y='Emotion', x='Count', 
                     orientation='h',
                     color='Emotion',
                     color_discrete_map={
                         'happy': '#FFD93D',
                         'sad': '#4A90E2',
                         'angry': '#FF6B6B',
                         'neutral': '#95A5A6',
                         'fear': '#6C5CE7',
                         'disgust': '#FD79A8',
                         'surprise': '#74B9FF'
                     })
        fig.update_layout(height=400, showlegend=False, xaxis_title="Count", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 📥 Download Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_data = analytics_df.to_csv(index=False)
            st.download_button(label="📥 CSV", data=csv_data, file_name="analysis.csv", mime="text/csv", use_container_width=True)
        
        with col2:
            txt_data = f"""EMOTION-GUIDED SUMMARIZER RESULTS

📝 ORIGINAL TEXT:
{chr(10).join([f"{i}. {text}" for i, (text, _, _) in enumerate(result['emotions'], 1)])}

✨ GENERATED SUMMARY:
{result['summary']}

🎭 EMOTION ANALYSIS:
{chr(10).join([f"{i}. {emotion.upper()} ({conf:.1%}) - {text[:60]}..." for i, (text, emotion, conf) in enumerate(result['emotions'], 1)])}
"""
            st.download_button(label="📥 TXT", data=txt_data, file_name="analysis.txt", mime="text/plain", use_container_width=True)
        
        with col3:
            json_data = json.dumps(result, indent=2, default=str)
            st.download_button(label="📥 JSON", data=json_data, file_name="analysis.json", mime="application/json", use_container_width=True)
    else:
        st.info("👈 Upload and process audio in the 'Upload & Process' tab!")

# TAB 4: About
with tab4:
    st.markdown("## 📚 About This Project")
    st.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔧 Technology Stack")
        st.markdown("""
        - **TensorFlow/TFLite** - Deep Learning Models
        - **Vosk** - Offline Speech-to-Text
        - **librosa** - Audio Processing
        - **NLTK** - NLP & Summarization
        - **Streamlit** - Web Interface
        - **Plotly** - Data Visualization
        """)
    
    with col2:
        st.markdown("### ✨ Capabilities")
        st.markdown("""
        - Transcribe speech to text
        - Detect 7 emotion categories
        - Generate context-aware summaries
        - Export multiple formats
        - Visualize emotion patterns
        - 100% offline processing
        """)
    
    st.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)
    
    st.markdown("### 🎯 How It Works")
    
    steps = [
        ("1️⃣", "Audio Upload", "User uploads audio file"),
        ("2️⃣", "Transcription", "Vosk converts speech to text"),
        ("3️⃣", "Segmentation", "Split into sentences with timing"),
        ("4️⃣", "Emotion Detection", "TFLite analyzes each sentence"),
        ("5️⃣", "Smart Summary", "NLTK generates summary"),
        ("6️⃣", "Analytics", "Results visualized and exported")
    ]
    
    for icon, title, desc in steps:
        st.markdown(f"**{icon} {title}** - {desc}")
    
    st.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)
    
    st.markdown("### 😊 Recognized Emotions")
    
    emotions = {
        "😊 Happy": ("Positive, pleased state", "#FFD93D"),
        "😢 Sad": ("Sorrowful, melancholic state", "#4A90E2"),
        "😠 Angry": ("Furious, irritated state", "#FF6B6B"),
        "😐 Neutral": ("Calm, unaffected state", "#95A5A6"),
        "😨 Fear": ("Anxious, frightened state", "#6C5CE7"),
        "😒 Disgust": ("Repulsed, disapproving state", "#FD79A8"),
        "😲 Surprise": ("Shocked, astonished state", "#74B9FF")
    }
    
    cols = st.columns(4)
    col_idx = 0
    for emotion, (desc, color) in emotions.items():
        with cols[col_idx % 4]:
            st.markdown(f'<div class="metric-box"><h4 style="color: {color};">{emotion}</h4><p style="font-size: 0.85rem; margin: 0.5rem 0;">{desc}</p></div>', unsafe_allow_html=True)
        col_idx += 1

# Footer
st.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="footer">
    <h4>🎭 Emotion-Guided Smart Text Summarizer</h4>
    <p>Intelligent audio analysis powered by speech recognition, emotion detection, and NLP</p>
    <p class="footer-dev">👨‍💻 Developer: Sheetal Lodhi</p>
</div>
""", unsafe_allow_html=True)
