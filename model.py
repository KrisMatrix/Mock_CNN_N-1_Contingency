"""
model.py
TensorFlow / Keras 2D Convolutional Neural Network (CNN) architecture and Data Loader utilities
for predicting N-1 contingency stress-testing recommendation matrices (35x35)
from 3-channel input matrices (Admittance, Real Power P, Reactive Power Q).
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_cnn_model(input_shape=(35, 35, 3)):
    """
    Builds a TensorFlow/Keras 2D CNN model for N-1 Contingency Recommendation.
    Input Shape: (Batch, 35, 35, 3)
    Output Shape: (Batch, 35, 35) -> unnormalized logits
    """
    inputs = keras.Input(shape=input_shape, name="grid_matrices")

    # Initial 3x3 Convolution
    x = layers.Conv2D(32, (3, 3), padding='same', name='in_conv')(inputs)
    x = layers.BatchNormalization(name='in_bn')(x)
    x = layers.LeakyReLU(0.2, name='in_act')(x)

    # Residual Block 1 (32 -> 64)
    res1 = layers.Conv2D(64, (1, 1), padding='same', name='res1_proj')(x)
    res1 = layers.BatchNormalization(name='res1_proj_bn')(res1)
    
    y1 = layers.Conv2D(64, (3, 3), padding='same', name='block1_conv1')(x)
    y1 = layers.BatchNormalization(name='block1_bn1')(y1)
    y1 = layers.LeakyReLU(0.2, name='block1_act1')(y1)
    y1 = layers.Conv2D(64, (3, 3), padding='same', name='block1_conv2')(y1)
    y1 = layers.BatchNormalization(name='block1_bn2')(y1)
    x = layers.Add(name='block1_add')([y1, res1])
    x = layers.LeakyReLU(0.2, name='block1_out_act')(x)

    # Residual Block 2 (64 -> 128)
    res2 = layers.Conv2D(128, (1, 1), padding='same', name='res2_proj')(x)
    res2 = layers.BatchNormalization(name='res2_proj_bn')(res2)

    y2 = layers.Conv2D(128, (3, 3), padding='same', name='block2_conv1')(x)
    y2 = layers.BatchNormalization(name='block2_bn1')(y2)
    y2 = layers.LeakyReLU(0.2, name='block2_act1')(y2)
    y2 = layers.Conv2D(128, (3, 3), padding='same', name='block2_conv2')(y2)
    y2 = layers.BatchNormalization(name='block2_bn2')(y2)
    x = layers.Add(name='block2_add')([y2, res2])
    x = layers.LeakyReLU(0.2, name='block2_out_act')(x)

    # Bottleneck Layer (128 -> 64)
    x = layers.Conv2D(64, (3, 3), padding='same', name='bottleneck_conv')(x)
    x = layers.BatchNormalization(name='bottleneck_bn')(x)
    x = layers.LeakyReLU(0.2, name='bottleneck_act')(x)

    # Output Layer (64 -> 1 channel) -> Reshape to (35, 35) logits matrix
    logits_1ch = layers.Conv2D(1, (1, 1), padding='same', name='out_conv')(x)
    outputs = layers.Reshape((35, 35), name="contingency_logits")(logits_1ch)

    model = keras.Model(inputs=inputs, outputs=outputs, name="ContingencyCNN")
    return model


def load_dataset(dataset_path='data/dataset.npz'):
    """
    Loads dataset.npz and formats tensors for TensorFlow (channels_last: N, 35, 35, 3).
    Returns:
        (x_train, y_train), (x_val, y_val), (x_test, y_test)
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    data = np.load(dataset_path)

    # Transpose input matrices from (N, 3, 35, 35) to TensorFlow format (N, 35, 35, 3)
    x_train = np.transpose(data['x_train'], (0, 2, 3, 1)).astype(np.float32)
    y_train = data['y_train'].astype(np.float32)

    x_val = np.transpose(data['x_val'], (0, 2, 3, 1)).astype(np.float32)
    y_val = data['y_val'].astype(np.float32)

    x_test = np.transpose(data['x_test'], (0, 2, 3, 1)).astype(np.float32)
    y_test = data['y_test'].astype(np.float32)

    print(f"Loaded TensorFlow Dataset:")
    print(f"  x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"  x_val   shape: {x_val.shape}, y_val   shape: {y_val.shape}")
    print(f"  x_test  shape: {x_test.shape}, y_test  shape: {y_test.shape}")

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


if __name__ == '__main__':
    # Verify TensorFlow Keras Model creation and dummy forward pass
    model = create_cnn_model(input_shape=(35, 35, 3))
    model.summary()

    # Dummy batch of 4 samples (4, 35, 35, 3)
    dummy_input = tf.random.normal((4, 35, 35, 3))
    dummy_output = model(dummy_input)

    print(f"\nForward Pass Verification:")
    print(f"  Input Tensor Shape:  {dummy_input.shape}")
    print(f"  Output Logits Shape: {dummy_output.shape}")
    assert dummy_output.shape == (4, 35, 35), "Output shape mismatch!"
    print("TensorFlow / Keras Forward pass verified successfully!")
