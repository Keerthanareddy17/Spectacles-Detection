"""
This script is for evaluating the model on a single image before converting it into tflite format
"""

import tensorflow as tf
import numpy as np
from PIL import Image

# Constants
IMG_SIZE = 128
MODEL_PATH = 'saved_models/final_model'  #SavedModel directory

# Load the model
model = tf.saved_model.load(MODEL_PATH)
infer = model.signatures["serve"]  

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0).astype(np.float32)
    return image

def predict(image_path):
    image = preprocess_image(image_path)
    input_tensor = tf.convert_to_tensor(image)
    
    # Inference
    output = infer(input_tensor)
    pred = list(output.values())[0].numpy()[0][0]  # Get scalar from tensor
    label = "With Glasses" if pred >= 0.5 else "Without Glasses"
    
    print(f"Prediction Score: {pred:.4f} --> {label}")

predict("Spectacles Detection/test5.jpg")  # Replace with your actual image path
