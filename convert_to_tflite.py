import tensorflow as tf
import argparse
import os

def convert_savedmodel_to_tflite(saved_model_dir, output_path="model.tflite", quantize=False):
    # Create the TFLite converter from the SavedModel directory
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    print("[INFO] Loaded SavedModel from:", saved_model_dir)

    # apply float16 quantization
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        print("[INFO] Applying float16 quantization...")

    # Convert the model
    tflite_model = converter.convert()
    print("[INFO] Model successfully converted to TFLite format.")

    # Save the TFLite model to the output path
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"[INFO] TFLite model saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a TensorFlow SavedModel to TFLite format.")
    parser.add_argument("--saved_model_dir", required=True, help="Path to the SavedModel directory.")
    parser.add_argument("--output", default="model.tflite", help="Output path for the TFLite model.")
    parser.add_argument("--quantize", action="store_true", help="Enable float16 quantization.")
    args = parser.parse_args()

    convert_savedmodel_to_tflite(args.saved_model_dir, args.output, args.quantize)
