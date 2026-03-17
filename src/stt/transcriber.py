import os
import json
import wave
import logging
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
from nltk.tokenize import sent_tokenize, word_tokenize 

# Ensure NLTK data are downloaded for sent_tokenize and word_tokenize
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords') # Potentially needed for other modules
except LookupError:
    nltk.download('stopwords')


# Suppress Vosk logs for cleaner output
logging.getLogger('vosk').setLevel(logging.WARNING)

class VoskTranscriber:
    def __init__(self, model_path="vosk_model"):
        self.model_path = model_path
        
        # Vosk's Model constructor expects the path to the directory containing 'conf', 'am', 'graph' etc.
        expected_conf_file = os.path.join(model_path, 'conf', 'mfcc.conf')
        if not os.path.exists(expected_conf_file):
            raise Exception(f"Vosk model configuration file '{expected_conf_file}' not found. "
                            f"Please ensure the Vosk model (e.g., vosk-model-small-en-us-0.15) is extracted "
                            f"directly into the '{model_path}' directory, so that 'conf/mfcc.conf' exists relative to it. "
                            f"Current contents of {model_path}: {os.listdir(model_path) if os.path.exists(model_path) else 'Path does not exist'}")
        
        try:
            self.model = Model(model_path)
        except Exception as e:
            raise Exception(f"Failed to create Vosk model from '{model_path}'. Please check model integrity. Original Vosk error: {e}")

    def transcribe_audio(self, audio_file_path):
        """
        Transcribes an audio file using Vosk and returns text with word-level timestamps.

        Args:
            audio_file_path (str): Path to the input audio file.

        Returns:
            tuple: A tuple containing:
                - str: The full transcribed text.
                - list: A list of dictionaries, each with 'word', 'start', 'end' and 'conf'.
        """
        audio = AudioSegment.from_file(audio_file_path)
        audio = audio.set_channels(1).set_frame_rate(16000) # Vosk models typically expect 16kHz mono audio

        recognizer = KaldiRecognizer(self.model, audio.frame_rate)
        recognizer.SetWords(True) # Get word-level timestamps

        # Convert pydub AudioSegment to raw WAV data for Vosk
        wav_data = audio.export(format="wav").read()

        # Process audio in chunks (optional, but good for large files)
        # For smaller files, you can process the whole thing at once
        if recognizer.AcceptWaveform(wav_data):
            result = json.loads(recognizer.Result())
        else:
            result = json.loads(recognizer.FinalResult())

        full_text = result.get('text', '')
        word_details = result.get('result', [])
        
        return full_text, word_details

    def get_sentence_segments_from_words(self, word_details, full_text, pause_threshold=0.5): # Default changed to 0.5s
        """
        Reconstructs sentence-level segments (with start/end times) from Vosk's word details,
        inferring sentence boundaries based on pauses.

        Args:
            word_details (list): List of dictionaries from Vosk, each with 'word', 'start', 'end', 'conf'.
            full_text (str): The complete transcribed text (used as a reference).
            pause_threshold (float): Minimum silence duration (in seconds) to infer a sentence boundary.

        Returns:
            list: A list of dictionaries, each with 'text', 'start', 'end'.
        """
        sentence_segments = []
        current_sentence_words = []
        current_sentence_start = -1

        for i, word_info in enumerate(word_details):
            word_text = word_info['word']
            word_start = word_info['start']
            word_end = word_info['end']

            if current_sentence_start == -1:
                current_sentence_start = word_start

            current_sentence_words.append(word_text)

            # Check for a pause indicating a sentence boundary
            is_end_of_audio = (i == len(word_details) - 1)
            if not is_end_of_audio:
                next_word_start = word_details[i+1]['start']
                pause_duration = next_word_start - word_end
            else:
                pause_duration = 0 # No pause after the last word

            # If a significant pause is detected OR it's the last word of the audio,
            # consider it a sentence boundary.
            if pause_duration > pause_threshold or is_end_of_audio:
                sentence_text = " ".join(current_sentence_words)
                
                # Heuristically add a period for robustness, especially if used with NLTK later or for display
                if not sentence_text.strip().endswith(('.', '?', '!')):
                    sentence_text += "."

                sentence_segments.append({
                    'text': sentence_text.strip(),
                    'start': current_sentence_start,
                    'end': word_end, # End of the last word in this sentence
                })

                # Reset for the next sentence
                current_sentence_words = []
                current_sentence_start = -1

        return sentence_segments


if __name__ == '__main__':
    # This block demonstrates how to use the VoskTranscriber.
    # Replace with a real audio file from RAVDESS or your hardvard.wav for best results.
    sample_audio_path = os.path.join('data', 'raw', 'sarcastic.wav')
    # Or for a RAVDESS example:
    

    vosk_model_dir = os.path.join(os.getcwd(), "vosk_model") # Assuming vosk_model folder is in project root

    if not os.path.exists(sample_audio_path):
        print(f"Error: Sample audio not found at {sample_audio_path}.")
    elif not os.path.exists(vosk_model_dir):
        print(f"Error: Vosk model not found at {vosk_model_dir}. Please download vosk-model-small-en-us-0.15 and extract it here.")
    else:
        print(f"Transcribing {sample_audio_path}...\n")
        transcriber = VoskTranscriber(vosk_model_dir)
        full_text, word_details = transcriber.transcribe_audio(sample_audio_path)
        print("--- Full Transcription from Vosk ---\n" + full_text + "\n")
        print("--- Word Details (first 5) ---\n")
        for word in word_details[:5]:
            print(word)
        
        print("\n--- Inferred Sentence Segments with Timestamps ---\n")
        # Experiment with pause_threshold here (e.g., 0.5, 0.3, 1.0)
        sentence_segments = transcriber.get_sentence_segments_from_words(word_details, full_text, pause_threshold=0.35) 
        for i, seg in enumerate(sentence_segments):
            print(f"{i+1}. [Start: {seg['start']:.2f}s, End: {seg['end']:.2f}s] {seg['text']}")
        if not sentence_segments:
            print("No sentences inferred. Try adjusting 'pause_threshold' in get_sentence_segments_from_words().")