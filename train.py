import os
import pandas as pd
import tensorflow as tf
import pickle
from model import build_model

# Constants
IMG_SIZE = 128
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
EPOCHS = 20
MODEL_DIR = 'saved_models'
os.makedirs(MODEL_DIR, exist_ok=True)

# Load class weights
with open('class_weights.pkl', 'rb') as f:
    class_weights = pickle.load(f)

# Utility functions
def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = image / 255.0
    return image, label

def augment_data(image, label):
    image = tf.image.random_flip_left_right(image)

    # Stronger lighting augmentations
    image = tf.image.random_brightness(image, max_delta=0.4)
    image = tf.image.random_contrast(image, lower=0.6, upper=1.4)

    # Random small rotations (-45 to +45 degrees) WITHOUT TensorFlow Addons, cus tensorflow-addons is throwing dependency issues!
    angle_deg = tf.random.uniform([], -45, 45)  # Random angle in degrees
    angle_rad = angle_deg * (3.14159265 / 180.0)  # Convert to radians

    # Rotate manually using tf.raw_ops.ImageProjectiveTransformV3
    def get_rotation_matrix(angle):
        cos_a = tf.math.cos(angle)
        sin_a = tf.math.sin(angle)
        return tf.reshape(
            [cos_a, -sin_a, 0.0,
             sin_a, cos_a, 0.0,
             0.0, 0.0], 
            [8]
        )

    transforms = get_rotation_matrix(angle_rad)
    image = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(image, 0),
        transforms=tf.expand_dims(transforms, 0),
        output_shape=[IMG_SIZE, IMG_SIZE],
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0
    )
    image = tf.squeeze(image, 0)  # Remove extra batch dimension

    # Random crop and resize
    image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 20, IMG_SIZE + 20)
    image = tf.image.random_crop(image, size=[IMG_SIZE, IMG_SIZE, 3])

    # Random Gaussian noise
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.05)
    image = tf.clip_by_value(image + noise, 0.0, 1.0)

    return image, label

def create_dataset(csv_path, shuffle=True, augment=False):
    df = pd.read_csv(csv_path)
    img_paths = df['img_path'].values
    labels = df['Eyeglasses'].values
    dataset = tf.data.Dataset.from_tensor_slices((img_paths, labels))
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    
    if augment:
        dataset = dataset.map(augment_data, num_parallel_calls=AUTOTUNE)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset

# Load datasets
train_ds = create_dataset('train_metadata.csv', shuffle=True, augment=True)
val_ds = create_dataset('val_metadata.csv', shuffle=False, augment=False)
test_ds = create_dataset('test_metadata.csv', shuffle=False, augment=False)

# Build model
model = build_model(img_size=IMG_SIZE)

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, 'best_model.keras'),
    monitor='val_loss',
    save_best_only=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    verbose=1
)

# Train model
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    class_weight=class_weights,
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)

# Load best model
best_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'best_model.keras'))

# Fine-tune 
print("Unfreezing base model for fine-tuning...")
for layer in best_model.layers[1].layers:
    layer.trainable = True

best_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),  # Lower LR for fine-tuning
    loss='binary_crossentropy',
    metrics=['accuracy']
)

fine_tune_epochs = 5
history_fine = best_model.fit(
    train_ds,
    epochs=fine_tune_epochs,
    validation_data=val_ds,
    class_weight=class_weights,
    callbacks=[early_stopping, reduce_lr]
)

# Save final model
best_model.export(os.path.join(MODEL_DIR, 'final_model'))
print("Final model saved successfully!")

# Evaluate
test_loss, test_acc = best_model.evaluate(test_ds)
print(f"Test accuracy: {test_acc:.4f}")
