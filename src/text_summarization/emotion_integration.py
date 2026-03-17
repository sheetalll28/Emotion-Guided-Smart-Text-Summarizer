import numpy as np
import tensorflow as tf
import pickle
import os
import tempfile
from src.emotion_recognition.preprocess import extract_features
import logging

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

class EmotionAwareSummarizer:
    def __init__(self, model_path, label_encoder_path):
        self.model_path = model_path
        self.label_encoder_path = label_encoder_path
        
        # Derive scaler path from model path (assuming they are in the same folder)
        self.scaler_path = os.path.join(os.path.dirname(model_path), 'scaler.pkl')
        
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.label_encoder = None
        self.scaler = None # <--- Added Scaler variable
        
        self.load_model()

    def load_model(self):
        """Loads TFLite model, Label Encoder, and Scaler."""
        try:
            # Load TFLite model
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Load Label Encoder
            with open(self.label_encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)

            # Load Scaler (CRITICAL FIX)
            # This makes the input numbers match what the model learned during training
            if os.path.exists(self.scaler_path):
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            else:
                print(f"Warning: Scaler not found at {self.scaler_path}. Predictions may be inaccurate.")
                
        except Exception as e:
            print(f"Error loading emotion model components: {e}")

    def predict_emotion(self, audio_segment):
        """
        Predicts emotion from a pydub AudioSegment.
        """
        if not self.interpreter or not self.label_encoder:
            return "N/A", 0.0

        # Create a temporary file to store the audio segment
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_filename = temp_wav.name
            
        try:
            # Export pydub segment to the temp file
            audio_segment.export(temp_filename, format="wav")
            
            # Extract features from the temp file
            features = extract_features(temp_filename)
            
            # Reshape for model input
            input_shape = self.input_details[0]['shape']
            
            # Ensure features match the expected input shape
            if input_shape[1] == len(features):
                # 1. Reshape to (1, 185)
                input_data = np.array(features, dtype=np.float32).reshape(1, -1)
                
                # 2. Scale (Normalize) using the loaded scaler
                # This fixes the "Fearful for everything" bug
                if self.scaler:
                    input_data = self.scaler.transform(input_data)
                
                # 3. Ensure float32 for TFLite
                input_data = input_data.astype(np.float32)
                
                # Run inference
                self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                self.interpreter.invoke()
                output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
                
                # Get prediction
                prediction_index = np.argmax(output_data)
                confidence = output_data[prediction_index]
                predicted_label = self.label_encoder.inverse_transform([prediction_index])[0]
                
                return predicted_label, float(confidence)
            else:
                print(f"Feature shape mismatch. Expected {input_shape[1]}, got {len(features)}")
                return "Unknown", 0.0

        except Exception as e:
            print(f"Error during emotion prediction: {e}")
            return "Error", 0.0
        finally:
            # Clean up
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

    def summarize_with_emotion(self, full_text, audio_segments, num_sentences=3):
        """
        Generates a summary based on text importance AND emotion.
        """
        from nltk.tokenize import sent_tokenize
        
        emotions_per_sentence = []
        
        print("\n--- Analyzing Emotions per Sentence ---")
        for i, segment in enumerate(audio_segments):
            emotion, confidence = self.predict_emotion(segment)
            emotions_per_sentence.append({
                'index': i,
                'emotion': emotion,
                'confidence': confidence
            })
            print(f"Sentence {i+1}: Emotion={emotion}, Conf={confidence:.2f}")

        # Weights for prioritization
        emotion_weights = {
            'angry': 2.0, 'fearful': 1.8, 'sad': 1.5, 'happy': 1.2,
            'disgust': 1.2, 'surprised': 1.1, 'neutral': 1.0, 'calm': 1.0,
            'unknown': 0.0, 'error': 0.0
        }
        
        # Tokenize text to match segments
        # Note: Ideally main.py aligns this, but we do a best-effort here
        sentences = sent_tokenize(full_text)
        
        # Determine strict length to avoid index errors
        min_len = min(len(sentences), len(audio_segments))
        
        scored_sentences = []
        for i in range(min_len):
            em = emotions_per_sentence[i]['emotion']
            conf = emotions_per_sentence[i]['confidence']
            
            # Calculate Score = Weight * Confidence
            weight = emotion_weights.get(em, 1.0)
            score = weight * conf
            
            scored_sentences.append((score, sentences[i], i))
            
        # Sort by score descending to find Top N
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        top_sentences = scored_sentences[:num_sentences]
        
        # Sort top sentences back by original index to maintain conversation flow
        top_sentences.sort(key=lambda x: x[2])
        
        final_summary = " ".join([s[1] for s in top_sentences])
        
        # Prepare data for analytics
        sentence_data = [(sentences[i], emotions_per_sentence[i]['emotion'], emotions_per_sentence[i]['confidence']) for i in range(min_len)]
        
        return final_summary, sentence_data