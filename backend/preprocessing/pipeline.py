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
    PREPROCESSING_PHASE1_DIR,
)
from backend.utils.logger import setup_logger, Timer

logger = setup_logger("Preprocessor")

def create_column_transformer() -> ColumnTransformer:
    """
    Constructs an un-fitted ColumnTransformer pipeline:
    - OneHotEncoder for categorical features (handle_unknown='ignore')
    - MinMaxScaler(feature_range=(0, 1)) for numerical features (Paper Eq. 5)
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
    df_train: pd.DataFrame,
    df_val: pd.DataFrame = None,
    df_test: pd.DataFrame = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict[str, Any],
]:
    """
    Preprocesses train, validation, and test sets strictly preventing data leakage:
    1. Fits ColumnTransformer ONLY on X_train (TRAIN-1).
    2. Transforms X_train, X_val, and X_test.
    3. Fits LabelEncoder for attack_cat ONLY on y_multiclass_train.
    4. Transforms y_multiclass_train, y_multiclass_val, and y_multiclass_test.
    5. Saves all preprocessing objects into preprocessing_artifacts/ and phase1/.

    Returns:
    (X_train_proc, X_val_proc, X_test_proc,
     y_bin_train, y_bin_val, y_bin_test,
     y_multi_train, y_multi_val, y_multi_test,
     info)
    """
    logger.info("Initializing 42-feature preprocessing pipeline...")

    # Backward compatibility: if only 2 dataframes passed
    if df_test is None and df_val is not None:
        from backend.dataset.splitter import split_training_dataset
        df_test = df_val
        df_train, df_val = split_training_dataset(df_train)

    # Extract Feature Matrix X and Target Vectors y
    X_train_raw = df_train[ALL_INPUT_FEATURES].copy()
    X_val_raw = df_val[ALL_INPUT_FEATURES].copy()
    X_test_raw = df_test[ALL_INPUT_FEATURES].copy()

    y_bin_train = df_train[BINARY_TARGET].values
    y_bin_val = df_val[BINARY_TARGET].values
    y_bin_test = df_test[BINARY_TARGET].values

    y_multi_train_raw = df_train[MULTICLASS_TARGET].values
    y_multi_val_raw = df_val[MULTICLASS_TARGET].values
    y_multi_test_raw = df_test[MULTICLASS_TARGET].values

    # Build and Fit Feature Transformer strictly on X_train_raw
    preprocessor = create_column_transformer()

    logger.info("Fitting ColumnTransformer ONLY on training partition (TRAIN-1)...")
    with Timer() as timer:
        X_train_proc = preprocessor.fit_transform(X_train_raw)
        X_val_proc = preprocessor.transform(X_val_raw)
        X_test_proc = preprocessor.transform(X_test_raw)

    logger.info(f"Feature transformation completed in {timer.interval:.2f} seconds.")
    logger.info(f"  - Transformed X_train shape: {X_train_proc.shape}")
    logger.info(f"  - Transformed X_val shape:   {X_val_proc.shape}")
    logger.info(f"  - Transformed X_test shape:  {X_test_proc.shape}")

    # Build and Fit LabelEncoder for Multiclass Target strictly on y_multi_train_raw
    logger.info("Fitting LabelEncoder ONLY on training multiclass target...")
    label_encoder = LabelEncoder()
    y_multi_train = label_encoder.fit_transform(y_multi_train_raw)
    y_multi_val = label_encoder.transform(y_multi_val_raw)
    y_multi_test = label_encoder.transform(y_multi_test_raw)

    # Save fitted preprocessing artifacts to root and phase1
    for target_dir in [PREPROCESSING_DIR, PREPROCESSING_PHASE1_DIR]:
        target_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, target_dir / "preprocessor.joblib")
        joblib.dump(label_encoder, target_dir / "label_encoder.joblib")

    preprocessor_path = PREPROCESSING_PHASE1_DIR / "preprocessor.joblib"
    label_encoder_path = PREPROCESSING_PHASE1_DIR / "label_encoder.joblib"

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
        X_val_proc,
        X_test_proc,
        y_bin_train,
        y_bin_val,
        y_bin_test,
        y_multi_train,
        y_multi_val,
        y_multi_test,
        info,
    )

