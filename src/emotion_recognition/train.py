import os
import numpy as np
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler # <--- Added Scaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Import your feature extractor
from src.emotion_recognition.preprocess import extract_features

# Paths
RAVDESS_DIR = os.path.join('data', 'raw', 'RAVDESS')
MODELS_DIR = 'models'
TFLITE_MODEL_PATH = os.path.join(MODELS_DIR, 'ser_model.tflite')
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, 'label_encoder.pkl')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl') # <--- New file

# Emotion mapping
EMOTIONS = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

def load_data(data_dir):
    print("Loading data and extracting features...")
    X, y = [], []
    file_count = 0
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.wav'):
                parts = file.split('-')
                if len(parts) > 2:
                    emotion_code = parts[2]
                    emotion_label = EMOTIONS.get(emotion_code)
                    
                    if emotion_label:
                        file_path = os.path.join(root, file)
                        feature = extract_features(file_path)
                        if np.any(feature): 
                            X.append(feature)
                            y.append(emotion_label)
                            file_count += 1
                            if file_count % 100 == 0:
                                print(f"Processed {file_count} files...")

    return np.array(X), np.array(y)

def train_model():
    if not os.path.exists(RAVDESS_DIR):
        print(f"Error: RAVDESS dataset not found at {RAVDESS_DIR}")
        return

    # 1. Load Data
    X, y = load_data(RAVDESS_DIR)
    print(f"Total files processed: {len(X)}")
    
    if len(X) < 50:
        print("WARNING: You have very few training samples. The model might not learn well.")
        print("Ensure you have downloaded all Actors (01-24) for best results.")

    # 2. Encode Labels
    lb = LabelEncoder()
    y_encoded = to_categorical(lb.fit_transform(y))
    
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    with open(LABEL_ENCODER_PATH, 'wb') as f:
        pickle.dump(lb, f)

    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # 4. Scale Data (CRITICAL FIX)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train) # Fit on training data
    X_test = scaler.transform(X_test)       # Apply to test data
    
    # Save the scaler for later use in prediction
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {SCALER_PATH}")

    # 5. Build Model
    model = Sequential([
        Dense(256, input_shape=(185,), activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(y_encoded.shape[1], activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 6. Train
    print("Training model...")
    # Increased epochs slightly to ensure convergence
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    # 7. Convert to TFLite
    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(TFLITE_MODEL_PATH, 'wb') as f:
        f.write(tflite_model)
    
    print("\n--- Training Complete! ---")

if __name__ == '__main__':
    train_model()