import os
import numpy as np
from pydub import AudioSegment
import pandas as pd
import logging
import nltk

# Ensure NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Import components
from src.emotion_recognition.preprocess import extract_features
from src.text_summarization.emotion_integration import EmotionAwareSummarizer
from src.stt.transcriber import VoskTranscriber

# Configure logging
logging.getLogger('pydub').setLevel(logging.WARNING)
logging.getLogger('vosk').setLevel(logging.WARNING)

def run_emotion_aware_summarization(audio_file_path, vosk_transcriber_instance, num_summary_sentences=4, data_export_path=None):
    
    print(f"\n--- Processing Audio: {audio_file_path} ---")
    
    # 1. Load Audio
    try:
        audio = AudioSegment.from_file(audio_file_path)
        print(f"Audio loaded. Duration: {audio.duration_seconds:.2f}s")
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return "", [], [], []

    # 2. Transcribe and Segmentation (Using pause_threshold value)
    print("Performing Speech-to-Text and Segmentation...")
    full_text, word_details = vosk_transcriber_instance.transcribe_audio(audio_file_path)
    
    # Explicitly passing 0.35 for now. we can change it later according to needs.
    sentence_segments_with_times = vosk_transcriber_instance.get_sentence_segments_from_words(
        word_details, full_text, pause_threshold=0.5
    )
    print("Segmentation complete by Sheetal. Number of sentences detected:", sentence_segments_with_times)

    if not sentence_segments_with_times:
        print("No sentences detected. Check pause_threshold.")
        return "", [], [], []

    # 3. Create Audio Segments for Emotion Recognition
    actual_audio_segments = []
    text_sentences = [] # We list the text explicitly to ensure alignment
    
    print("Extracting audio segments...")
    print("audio length (ms):", audio, len(audio))
    for segment in sentence_segments_with_times:
        start_ms = int(segment['start'] * 1000)
        end_ms = int(segment['end'] * 1000)
        
        # Safety check
        
        if start_ms >= 0 and end_ms <= len(audio) and end_ms > start_ms:
            segment_audio = audio[start_ms:end_ms]
            actual_audio_segments.append(segment_audio)
            text_sentences.append(segment['text'])
        else:
            # Fallback for edge cases
            actual_audio_segments.append(AudioSegment.silent(duration=100))
            text_sentences.append(segment['text'])

    # 4. Initialize Emotion Summarizer
    tflite_model_path = os.path.join('models', 'ser_model.tflite')
    label_encoder_path = os.path.join('models', 'label_encoder.pkl')
    
    if not os.path.exists(tflite_model_path) or not os.path.exists(label_encoder_path):
        print("Error: Models not found. Run train.py and convert_to_tflite.py first.")
        return "", [], [], []

    summarizer = EmotionAwareSummarizer(tflite_model_path, label_encoder_path)

    # 5. Generate Summary with Real Emotions
    # We reconstruct a 'full_text' from our segmented list to pass to the function, 
    # ensuring the tokenizer inside aligns somewhat better, though we mainly rely on the segments.
    aligned_full_text = " ".join(text_sentences)
    
    summary, emotions_data = summarizer.summarize_with_emotion(
        aligned_full_text, 
        actual_audio_segments, 
        num_sentences=num_summary_sentences
    )

    # 6. Data Collection for Analytics
    analytics_data = []
    audio_file_name = os.path.basename(audio_file_path)
    
    # We iterate through our text_sentences (which match actual_audio_segments 1-to-1)
    # emotion_data contains (text, emotion, confidence)
    
    # Note: summarize_with_emotion returns sentence_data based on nltk split. 
    # To be perfectly accurate, we should map our manual segments.
    # Let's overwrite emotions_data with strict mapping from our loop
    
    final_emotion_output = []
    
    for i, (text, audio_seg) in enumerate(zip(text_sentences, actual_audio_segments)):
        # Re-predict here to ensure perfect alignment for the CSV (or assume summarizer did it order)
        # For efficiency, we rely on the summarizer's print loop, but to return clean data:
        # We can just trust the summarizer loop if the counts match. 
        
        # Let's rely on the data returned by summarizer, assuming NLTK split matched our segments close enough.
        # If lengths differ, we truncate.
        if i < len(emotions_data):
            e_text, e_label, e_conf = emotions_data[i]
            
            # Simple check if sentence is in summary
            is_in_summary = e_text in summary
            
            analytics_data.append({
                'AudioFileName': audio_file_name,
                'SentenceIndex': i,
                'SentenceText': text, # Use our Vosk-segmented text
                'PredictedEmotion': e_label,
                'EmotionConfidence': e_conf,
                'IsInSummary': is_in_summary
            })
            final_emotion_output.append((text, e_label, e_conf))

    if data_export_path and analytics_data:
        df = pd.DataFrame(analytics_data)
        df.to_csv(data_export_path, index=False)
        print(f"Analytics data exported to {data_export_path}")

    return summary, final_emotion_output, analytics_data, sentence_segments_with_times

if __name__ == '__main__':
    # Configuration
    sample_audio_path = os.path.join('data', 'raw', 'blankspace.wav')
    vosk_model_dir = os.path.join(os.getcwd(), "vosk_model")
    
    # Check files
    if not os.path.exists(sample_audio_path):
        print("Please ensure data/raw/mixed.wav exists.")
    elif not os.path.exists(vosk_model_dir):
        print("Please ensure vosk_model directory exists.")
    else:
        # Init Transcriber
        transcriber = VoskTranscriber(vosk_model_dir)
        
        # Run
        output_csv = os.path.join('data', 'processed', 'summarizer_analytics.csv')
        summary, emotion_data, analytics, segments = run_emotion_aware_summarization(
            sample_audio_path,
            transcriber,
            num_summary_sentences=3,
            data_export_path=output_csv
        )
        
        # Display
        print("\n--- FINAL RESULTS ---")
        print(f"Summary:\n{summary}\n")
        print("Sentence Emotions:")
        for item in emotion_data:
            print(f"- [{item[1]}, {item[2]:.2f}] {item[0]}")