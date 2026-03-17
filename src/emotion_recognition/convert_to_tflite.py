import tensorflow as tf
import os

def convert_model_to_tflite(keras_model_path, tflite_model_path):
    """
    Converts a Keras model to TensorFlow Lite format.

    Args:
        keras_model_path (str): Path to the saved Keras model (.h5).
        tflite_model_path (str): Path to save the converted TFLite model (.tflite).
    """
    # Load the Keras model
    model = tf.keras.models.load_model(keras_model_path)

    # Convert the model to TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the TFLite model
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

    print(f"Keras model at '{keras_model_path}' successfully converted to TFLite and saved at '{tflite_model_path}'")

if __name__ == '__main__':
    keras_model_file = os.path.join('models', 'ser_model.h5')
    tflite_model_file = os.path.join('models', 'ser_model.tflite')

    if not os.path.exists(keras_model_file):
        print(f"Error: Keras model not found at {keras_model_file}. Please ensure 'train.py' was run successfully.")
    else:
        convert_model_to_tflite(keras_model_file, tflite_model_file)
