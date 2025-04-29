import tensorflow as tf
from tensorflow import keras
from keras import layers

def build_model(img_size=128, train_base=False, base_model=None):
    """
    Builds and returns a lightweight MobileNetV2-based model for glasses detection.

    Args:
        img_size (int): Size to which input images are resized (default: 128).
        train_base (bool): Whether to unfreeze the base model for fine-tuning.
        base_model (tf.keras.Model, optional): Existing base model to continue from.

    Returns:
        tf.keras.Model: Compiled Keras model ready for training.
    """
    if base_model is None:
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(img_size, img_size, 3),
            include_top=False,
            weights='imagenet'
        )
    
    base_model.trainable = train_base

    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = base_model(inputs, training=train_base)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)  # Slightly higher dropout for better regularization
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model
