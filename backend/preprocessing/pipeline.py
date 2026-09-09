"""
Preprocessing module for Phase 1.
Implements reusable ColumnTransformer for categorical and numerical features,
ensuring leakage-free fitting only on training data.
"""

from typing import Tuple, Dict, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder

from backend.config import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    ALL_INPUT_FEATURES,
    BINARY_TARGET,
    MULTICLASS_TARGET,
    PREPROCESSING_DIR,
)
from backend.utils.logger import setup_logger, Timer

logger = setup_logger("Preprocessor")

def create_column_transformer() -> ColumnTransformer:
    """
    Constructs an un-fitted ColumnTransformer pipeline:
    - OneHotEncoder for categorical features (handle_unknown='ignore')
    - MinMaxScaler(feature_range=(0, 1)) for numerical features (Paper-aligned)
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            (
                "num",
                MinMaxScaler(feature_range=(0, 1)),
                NUMERICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    return preprocessor

def prepare_preprocessed_data(
    df_train: pd.DataFrame, df_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Preprocesses train and test sets strictly preventing data leakage:
    1. Fits ColumnTransformer ONLY on X_train.
    2. Transforms X_train and X_test.
    3. Fits LabelEncoder for attack_cat ONLY on y_multiclass_train.
    4. Saves all preprocessing objects into preprocessing_artifacts/.

    Returns:
    (X_train_proc, X_test_proc, y_bin_train, y_bin_test, y_multi_train, y_multi_test, info)
    """
    logger.info("Initializing preprocessing pipeline...")

    # Extract Feature Matrix X and Target Vectors y
    X_train_raw = df_train[ALL_INPUT_FEATURES].copy()
    X_test_raw = df_test[ALL_INPUT_FEATURES].copy()

    y_bin_train = df_train[BINARY_TARGET].values
    y_bin_test = df_test[BINARY_TARGET].values

    y_multi_train_raw = df_train[MULTICLASS_TARGET].values
    y_multi_test_raw = df_test[MULTICLASS_TARGET].values

    # Build and Fit Feature Transformer strictly on X_train_raw
    preprocessor = create_column_transformer()

    logger.info("Fitting ColumnTransformer ONLY on training features...")
    with Timer() as timer:
        X_train_proc = preprocessor.fit_transform(X_train_raw)
        X_test_proc = preprocessor.transform(X_test_raw)

    logger.info(f"Feature transformation completed in {timer.interval:.2f} seconds.")
    logger.info(f"  - Transformed X_train shape: {X_train_proc.shape}")
    logger.info(f"  - Transformed X_test shape:  {X_test_proc.shape}")

    # Build and Fit LabelEncoder for Multiclass Target strictly on y_multi_train_raw
    logger.info("Fitting LabelEncoder ONLY on training multiclass target...")
    label_encoder = LabelEncoder()
    y_multi_train = label_encoder.fit_transform(y_multi_train_raw)
    
    # Handle any potential unseen test label defensively (though all 10 are present)
    y_multi_test = label_encoder.transform(y_multi_test_raw)

    # Save fitted preprocessing artifacts
    preprocessor_path = PREPROCESSING_DIR / "preprocessor.joblib"
    label_encoder_path = PREPROCESSING_DIR / "label_encoder.joblib"

    joblib.dump(preprocessor, preprocessor_path)
    joblib.dump(label_encoder, label_encoder_path)

    logger.info(f"Saved preprocessor artifact to: {preprocessor_path}")
    logger.info(f"Saved label encoder artifact to: {label_encoder_path}")

    # Collect column names post-encoding for interpretability
    try:
        cat_feature_names = list(
            preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES)
        )
    except Exception:
        cat_feature_names = []

    all_processed_feature_names = cat_feature_names + NUMERICAL_FEATURES

    info = {
        "preprocessor_path": str(preprocessor_path),
        "label_encoder_path": str(label_encoder_path),
        "processed_feature_count": X_train_proc.shape[1],
        "processed_feature_names": all_processed_feature_names,
        "classes": list(label_encoder.classes_),
    }

    return (
        X_train_proc,
        X_test_proc,
        y_bin_train,
        y_bin_test,
        y_multi_train,
        y_multi_test,
        info,
    )
