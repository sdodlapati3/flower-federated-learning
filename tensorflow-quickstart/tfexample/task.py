"""tfexample: A Flower / TensorFlow app.

Modified to use tf.keras.datasets.cifar10 instead of HuggingFace FederatedDataset
for faster downloads on HPC clusters.
"""

import os
from pathlib import Path

import keras
import numpy as np
from keras import layers

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Cache the loaded data
_cifar10_data = None


def load_model(learning_rate: float = 0.001):
    """Define a simple CNN for CIFAR-10 and set Adam optimizer."""
    model = keras.Sequential(
        [
            keras.Input(shape=(32, 32, 3)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ]
    )
    optimizer = keras.optimizers.Adam(learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_data(partition_id, num_partitions):
    """Load and partition CIFAR-10 data using tf.keras.datasets."""
    global _cifar10_data
    
    # Load CIFAR-10 data (cached after first load)
    if _cifar10_data is None:
        (x_train_full, y_train_full), (x_test_full, y_test_full) = keras.datasets.cifar10.load_data()
        # Normalize to [0, 1]
        x_train_full = x_train_full.astype("float32") / 255.0
        x_test_full = x_test_full.astype("float32") / 255.0
        y_train_full = y_train_full.flatten()
        y_test_full = y_test_full.flatten()
        _cifar10_data = (x_train_full, y_train_full, x_test_full, y_test_full)
    
    x_train_full, y_train_full, x_test_full, y_test_full = _cifar10_data
    
    # Partition the training data (IID partitioning)
    total_size = len(x_train_full)
    partition_size = total_size // num_partitions
    start_idx = partition_id * partition_size
    end_idx = start_idx + partition_size if partition_id < num_partitions - 1 else total_size
    
    x_partition = x_train_full[start_idx:end_idx]
    y_partition = y_train_full[start_idx:end_idx]
    
    # Split partition into train (80%) and validation (20%)
    split_idx = int(0.8 * len(x_partition))
    x_train = x_partition[:split_idx]
    y_train = y_partition[:split_idx]
    x_test = x_partition[split_idx:]
    y_test = y_partition[split_idx:]

    return x_train, y_train, x_test, y_test
