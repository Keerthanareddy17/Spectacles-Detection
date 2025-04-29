"""This script is for testing the model on a sngle image after converting it into tflitw format"""
import tensorflow as tf
import numpy as np
from PIL import Image

IMG_SIZE = 128
TFLITE_MODEL_PATH = "model.tflite"  

def preprocess_image(image_path):
    """Preprocesses the image by resizing and normalizing."""
    image = Image.open(image_path).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)  
    return image

def predict_single_image(image_path):
    """Load the TFLite model, preprocess the image, and make a prediction."""
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    # Preprocess the input image
    image = preprocess_image(image_path)
    
    # Set the input tensor
    interpreter.set_tensor(input_index, image)
    
    # Run the model
    interpreter.invoke()
    
    # Get the prediction result
    output = interpreter.get_tensor(output_index)[0][0]
    
    # Interpret the output (threshold at 0.5)
    prediction = 1 if output >= 0.5 else 0
    label = "With Glasses" if prediction == 1 else "Without Glasses"
    
    print(f"Prediction Score: {output:.4f} --> {label}")

if __name__ == "__main__":
    image_path = "Spectacles Detection/test5.jpg"  # Replace with the path to your image
    predict_single_image(image_path)
