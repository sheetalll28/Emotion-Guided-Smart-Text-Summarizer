import librosa
import numpy as np
import os

def extract_features(audio_path, sr=22050, n_mfcc=40, hop_length=512, n_fft=2048):
    """
    Extracts MFCCs and other features from an audio file.
    Returns a zero-padded feature vector if processing fails (e.g., empty audio).

    Args:
        audio_path (str): Path to the audio file.
        sr (int): Sampling rate.
        n_mfcc (int): Number of MFCCs to extract.
        hop_length (int): The hop length for feature extraction.
        n_fft (int): The FFT window size.

    Returns:
        np.ndarray: Concatenated feature vector (or zero-padded on error).
    """
    # Define the expected feature dimension in case of error
    # MFCCs (n_mfcc) + Chroma (12) + Mel (128) + Cent (1) + Bandwidth (1) + Rolloff (1) + ZCR (1) + RMS (1)
    expected_feature_dimension = n_mfcc + 12 + 128 + 1 + 1 + 1 + 1 + 1 
    zero_features = np.zeros(expected_feature_dimension) # This will be returned on error

    try:
        # Load audio from file. If it's a pydub AudioSegment exported to a temporary file, it should work.
        # librosa.load can raise an error for empty/invalid files.
        y, sr = librosa.load(audio_path, sr=sr)
        
        # If audio is too short after loading, return zero features
        if len(y) < hop_length: # Minimum length required for some feature extractions
            # print(f"Warning: Audio segment too short for {audio_path}. Returning zero features.") # Debugging line
            return zero_features

        # MFCCs
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft).T, axis=0)
        
        # Chroma feature
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft).T, axis=0)
        
        # Mel Spectrogram
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft).T, axis=0)
        
        # Spectral Centroid
        cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft).T, axis=0)
        
        # Spectral Bandwidth
        bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft).T, axis=0)
        
        # Spectral Rolloff
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft).T, axis=0)
        
        # Zero Crossing Rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length).T, axis=0)
        
        # RMS Energy
        rms = np.mean(librosa.feature.rms(y=y, hop_length=hop_length).T, axis=0)
        
        # Concatenate all features
        features = np.hstack((mfccs, chroma, mel, cent, bandwidth, rolloff, zcr, rms))
        return features

    except Exception as e:
        # print(f"Error processing {audio_path}: {e}. Returning zero features.") # Debugging line
        return zero_features # Return zero features on any processing error

def load_ravdess_data(data_dir):
    """
    Loads RAVDESS dataset, extracts features, and prepares labels.

    Args:
        data_dir (str): Path to the directory containing RAVDESS audio files.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Extracted features.
            - np.ndarray: Emotion labels.
    """
    features = []
    labels = []
    
    # RAVDESS file naming convention:
    # Modality-VocalChannel-Emotion-EmotionalIntensity-Statement-Repetition-Actor.wav
    # Example: 03-01-01-01-01-01-01.wav
    # Emotion codes:
    # 01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised

    emotion_map = {
        '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
        '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
    }

    for actor_folder in os.listdir(data_dir):
        actor_path = os.path.join(data_dir, actor_folder)
        if os.path.isdir(actor_path):
            for audio_file in os.listdir(actor_path):
                if audio_file.endswith('.wav'):
                    file_path = os.path.join(actor_path, audio_file)
                    # Extract emotion from filename
                    parts = audio_file.split('-')
                    emotion_code = parts[2]
                    emotion = emotion_map.get(emotion_code)

                    if emotion:
                        extracted_features = extract_features(file_path)
                        # No need to check for None here, extract_features now always returns an array
                        features.append(extracted_features)
                        labels.append(emotion)
    
    return np.array(features), np.array(labels)

if __name__ == '__main__':
    # This is an example of how to use the functions.
    # Replace 'path/to/your/RAVDESS' with the actual path to your RAVDESS dataset.
    ravdess_data_dir = os.path.join('data', 'raw', 'RAVDESS') # Assuming you have a RAVDESS folder inside data/raw
    
    # Create a dummy RAVDESS directory and files for testing if they don't exist
    if not os.path.exists(ravdess_data_dir):
        os.makedirs(os.path.join(ravdess_data_dir, 'Actor_01'), exist_ok=True)
        # Create a dummy WAV file for testing
        from scipy.io.wavfile import write
        import numpy as np
        samplerate = 44100 # Fs
        duration = 1 # seconds
        frequency = 440 # Hz
        t = np.linspace(0., duration, int(samplerate * duration))
        amplitude = np.iinfo(np.int16).max * 0.5
        data = amplitude * np.sin(2. * np.pi * frequency * t)
        write(os.path.join(ravdess_data_dir, 'Actor_01', '03-01-01-01-01-01-01.wav'), samplerate, data.astype(np.int16))
        print("Created dummy RAVDESS data for testing.")

    print(f"Loading data from: {ravdess_data_dir}")
    features, labels = load_ravdess_data(ravdess_data_dir)
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"First 5 labels: {labels[:5]}")
    
    # Save processed features and labels for later use
    np.save(os.path.join('data', 'processed', 'ravdess_features.npy'), features)
    np.save(os.path.join('data', 'processed', 'ravdess_labels.npy'), labels)
    print("Processed features and labels saved to data/processed/")