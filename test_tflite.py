
"""This script is for testing the model on the test data after converting to tflite format"""
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from PIL import Image

IMG_SIZE = 128
TFLITE_MODEL_PATH = "model.tflite"
TEST_CSV = 'test_metadata.csv'  # CSV path with image paths and labels

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def load_test_data():
    df = pd.read_csv(TEST_CSV)
    img_paths = df['img_path'].values
    labels = df['Eyeglasses'].values
    return img_paths, labels

def evaluate_tflite_model():
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    img_paths, labels = load_test_data()
    correct = 0

    for img_path, label in tqdm(zip(img_paths, labels), total=len(labels)):
        image = preprocess_image(img_path)
        interpreter.set_tensor(input_index, image)
        interpreter.invoke()
        output = interpreter.get_tensor(output_index)[0][0]
        prediction = 1 if output >= 0.5 else 0
        if prediction == label:
            correct += 1

    accuracy = correct / len(labels)
    print(f"[RESULT] Accuracy on test set: {accuracy:.4f} ({correct}/{len(labels)})")

if __name__ == "__main__":
    evaluate_tflite_model()
