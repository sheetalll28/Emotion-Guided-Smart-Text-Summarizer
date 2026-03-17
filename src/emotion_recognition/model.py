from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import numpy as np

def create_model(input_shape, num_classes):
    """
    Creates a simple Feed-Forward Neural Network model for SER.

    Args:
        input_shape (tuple): Shape of the input features (e.g., (number_of_features,)).
        num_classes (int): Number of emotion classes.

    Returns:
        tensorflow.keras.models.Sequential: Compiled Keras model.
    """
    model = Sequential([
        Dense(256, activation='relu', input_shape=input_shape),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

