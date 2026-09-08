"""
Keras Artificial Neural Network (ANN) model factory.
Constructs feed-forward neural networks for tabular classification tasks.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging info/warning

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

from backend.config import RANDOM_SEED

def set_reproducibility(seed: int = RANDOM_SEED):
    """Sets random seeds for reproducibility."""
    tf.random.set_seed(seed)

def build_ann_model(
    input_dim: int,
    num_classes: int = 2,
    learning_rate: float = 0.001
) -> Sequential:
    """
    Constructs a compiled Keras Sequential ANN model for tabular classification:
    - Input dimension: number of preprocessed feature columns
    - Hidden Layer 1: 128 units, ReLU, BatchNormalization, Dropout(0.2)
    - Hidden Layer 2: 64 units, ReLU, BatchNormalization, Dropout(0.2)
    - Output Layer:
        - If num_classes == 2: 1 unit, Sigmoid, binary_crossentropy
        - If num_classes > 2: num_classes units, Softmax, sparse_categorical_crossentropy
    """
    set_reproducibility()

    model = Sequential()
    
    # Hidden Layer 1
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Hidden Layer 2
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Output Layer
    if num_classes == 2:
        model.add(Dense(1, activation='sigmoid'))
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    else:
        model.add(Dense(num_classes, activation='softmax'))
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    return model

def get_early_stopping(patience: int = 5) -> EarlyStopping:
    """Returns EarlyStopping callback monitoring validation loss."""
    return EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
