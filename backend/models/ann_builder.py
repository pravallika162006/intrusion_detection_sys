"""
Keras Artificial Neural Network (ANN) model factory.
Constructs feed-forward neural networks aligned with Kasongo & Sun (2020):
- Single hidden layer with neuron candidates in {5, 10, 15, 30, 50, 100, 150}
- Adam solver
- Adaptive learning rate
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging info/warning

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from backend.config import RANDOM_SEED

def set_reproducibility(seed: int = RANDOM_SEED):
    """Sets random seeds for reproducibility."""
    tf.random.set_seed(seed)

def build_paper_ann_model(
    input_dim: int,
    num_classes: int = 2,
    hidden_units: int = 50,
    learning_rate: float = 0.02,
) -> Sequential:
    """
    Constructs a single-hidden-layer feed-forward ANN faithfully matching
    the architecture described in Kasongo & Sun (2020):
    - Input dimension: number of preprocessed feature columns
    - Hidden Layer: single Dense layer with `hidden_units` neurons, ReLU activation
    - Output Layer:
        - If num_classes == 2: 1 unit, Sigmoid, binary_crossentropy
        - If num_classes > 2: num_classes units, Softmax, sparse_categorical_crossentropy
    - Optimizer: Adam with adaptive learning rate
    """
    set_reproducibility()

    model = Sequential()
    model.add(Dense(hidden_units, input_dim=input_dim, activation='relu'))
    
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

def build_ann_model(
    input_dim: int,
    num_classes: int = 2,
    hidden_units: int = 64,
    learning_rate: float = 0.01,
) -> Sequential:
    """Wrapper default to paper architecture."""
    return build_paper_ann_model(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_units=hidden_units,
        learning_rate=learning_rate,
    )

def get_early_stopping(patience: int = 5) -> EarlyStopping:
    """Returns EarlyStopping callback monitoring validation loss."""
    return EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=0
    )

def get_adaptive_lr_callback(factor: float = 0.5, patience: int = 2) -> ReduceLROnPlateau:
    """Returns ReduceLROnPlateau callback for adaptive learning rate scheduling."""
    return ReduceLROnPlateau(
        monitor='val_loss',
        factor=factor,
        patience=patience,
        min_lr=1e-5,
        verbose=0
    )
