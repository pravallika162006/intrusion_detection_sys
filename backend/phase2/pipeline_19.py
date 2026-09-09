"""
Phase 2 Preprocessing Pipeline for the selected 19-feature representation.
Fits ColumnTransformer ONLY on training data to prevent data leakage.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, List
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder

from backend.config import (
    CATEGORICAL_FEATURES,
    BINARY_TARGET,
    MULTICLASS_TARGET,
    PREPROCESSING_DIR,
)
from backend.utils.logger import setup_logger, Timer

logger = setup_logger("Preprocessor19")

PREPROCESSING_PHASE2_DIR = PREPROCESSING_DIR / "phase2"
PREPROCESSING_PHASE2_DIR.mkdir(parents=True, exist_ok=True)

def create_19_column_transformer(
    selected_19_features: List[str]
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Constructs an un-fitted ColumnTransformer pipeline for the selected 19 features:
    - Identifies which of the 19 are categorical (proto, service, state) vs numerical.
    - OneHotEncoder for categorical features (handle_unknown='ignore')
    - MinMaxScaler(feature_range=(0, 1)) for numerical features
    """
    cat_in_19 = [f for f in selected_19_features if f in CATEGORICAL_FEATURES]
    num_in_19 = [f for f in selected_19_features if f not in CATEGORICAL_FEATURES]

    transformers = []
    if cat_in_19:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_in_19)
        )
    if num_in_19:
        transformers.append(
            ("num", MinMaxScaler(feature_range=(0, 1)), num_in_19)
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor, cat_in_19, num_in_19

def prepare_19_preprocessed_data(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    selected_19_features: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Preprocesses train and test sets for the selected 19 features without data leakage:
    1. Extracts selected 19 features.
    2. Fits ColumnTransformer ONLY on X19_train.
    3. Transforms X19_train and X19_test.
    4. Fits LabelEncoder ONLY on y_multiclass_train.
    5. Saves preprocessing artifacts under preprocessing_artifacts/phase2/.

    Returns:
    (X_train_proc, X_test_proc, y_bin_train, y_bin_test, y_multi_train, y_multi_test, info)
    """
    logger.info("Initializing 19-feature preprocessing pipeline...")

    # Extract 19 feature subset
    X_train_raw = df_train[selected_19_features].copy()
    X_test_raw = df_test[selected_19_features].copy()

    y_bin_train = df_train[BINARY_TARGET].values
    y_bin_test = df_test[BINARY_TARGET].values

    y_multi_train_raw = df_train[MULTICLASS_TARGET].values
    y_multi_test_raw = df_test[MULTICLASS_TARGET].values

    # Build and Fit 19-feature Transformer strictly on X_train_raw
    preprocessor, cat_in_19, num_in_19 = create_19_column_transformer(selected_19_features)

    logger.info("Fitting 19-feature ColumnTransformer ONLY on training features...")
    with Timer() as timer:
        X_train_proc = preprocessor.fit_transform(X_train_raw)
        X_test_proc = preprocessor.transform(X_test_raw)

    logger.info(f"19-feature transformation completed in {timer.interval:.2f} seconds.")
    logger.info(f"  - Original selected feature count: 19")
    logger.info(f"  - Transformed X19_train shape:     {X_train_proc.shape}")
    logger.info(f"  - Transformed X19_test shape:      {X_test_proc.shape}")

    # Build and Fit LabelEncoder for Multiclass Target strictly on y_multi_train_raw
    logger.info("Fitting LabelEncoder ONLY on training multiclass target...")
    label_encoder = LabelEncoder()
    y_multi_train = label_encoder.fit_transform(y_multi_train_raw)
    y_multi_test = label_encoder.transform(y_multi_test_raw)

    # Save fitted 19-feature preprocessing artifacts
    preprocessor_path = PREPROCESSING_PHASE2_DIR / "preprocessor_19.joblib"
    label_encoder_path = PREPROCESSING_PHASE2_DIR / "label_encoder_19.joblib"

    joblib.dump(preprocessor, preprocessor_path)
    joblib.dump(label_encoder, label_encoder_path)

    logger.info(f"Saved 19-feature preprocessor to: {preprocessor_path}")
    logger.info(f"Saved 19-feature label encoder to: {label_encoder_path}")

    info = {
        "selected_19_features": selected_19_features,
        "cat_in_19": cat_in_19,
        "num_in_19": num_in_19,
        "preprocessor_path": str(preprocessor_path),
        "label_encoder_path": str(label_encoder_path),
        "processed_feature_count": X_train_proc.shape[1],
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
