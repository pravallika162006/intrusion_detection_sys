"""
Live Detector for Phase 2 pre-trained model inference.
Loads fitted preprocessor_19.joblib, binary model, and multiclass ANN model.
Validates live feature DataFrame schema and types before inference.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

from backend.config import (
    PREPROCESSING_PHASE2_DIR,
    PHASE2_BINARY_MODELS_DIR,
    PHASE2_MULTICLASS_MODELS_DIR,
    SELECTED_19_FEATURES,
)
from backend.utils.logger import setup_logger

logger = setup_logger("LiveDetector")


class LiveDetector:
    """
    Encapsulates loaded pre-trained Phase 2 preprocessor & models for live inference.
    """

    def __init__(self, binary_model_name: str = "xgboost_dt", multiclass_model_name: str = "ann"):
        self.preprocessor = None
        self.label_encoder = None
        self.binary_model = None
        self.multiclass_model = None
        self.binary_model_name = binary_model_name
        self.multiclass_model_name = multiclass_model_name
        self.is_loaded = False
        self.load_artifacts()

    def load_artifacts(self):
        """Loads fitted 19-feature preprocessor, label encoder, binary, and multiclass models."""
        preproc_path = PREPROCESSING_PHASE2_DIR / "preprocessor_19.joblib"
        label_enc_path = PREPROCESSING_PHASE2_DIR / "label_encoder_19.joblib"

        if not preproc_path.exists():
            raise FileNotFoundError(f"19-feature preprocessor not found at {preproc_path}")
        self.preprocessor = joblib.load(preproc_path)
        logger.info(f"Loaded 19-feature preprocessor from {preproc_path}")

        if label_enc_path.exists():
            self.label_encoder = joblib.load(label_enc_path)
            logger.info(f"Loaded 19-feature label encoder from {label_enc_path}")

        # Load Binary Model (default: xgboost_dt or decision_tree)
        bin_path_joblib = PHASE2_BINARY_MODELS_DIR / f"{self.binary_model_name}.joblib"
        bin_path_fallback = PHASE2_BINARY_MODELS_DIR / "decision_tree.joblib"

        if bin_path_joblib.exists():
            self.binary_model = joblib.load(bin_path_joblib)
            logger.info(f"Loaded binary model ({self.binary_model_name}) from {bin_path_joblib}")
        elif bin_path_fallback.exists():
            self.binary_model = joblib.load(bin_path_fallback)
            logger.info(f"Loaded fallback binary model (decision_tree) from {bin_path_fallback}")
        else:
            raise FileNotFoundError(f"Binary model not found in {PHASE2_BINARY_MODELS_DIR}")

        # Load Multiclass Model (default: ann or decision_tree)
        multi_path_keras = PHASE2_MULTICLASS_MODELS_DIR / f"{self.multiclass_model_name}.keras"
        multi_path_joblib = PHASE2_MULTICLASS_MODELS_DIR / "decision_tree.joblib"

        if multi_path_keras.exists():
            self.multiclass_model = keras.models.load_model(multi_path_keras)
            self.is_multiclass_keras = True
            logger.info(f"Loaded multiclass ANN model from {multi_path_keras}")
        elif multi_path_joblib.exists():
            self.multiclass_model = joblib.load(multi_path_joblib)
            self.is_multiclass_keras = False
            logger.info(f"Loaded multiclass Decision Tree model from {multi_path_joblib}")
        else:
            self.multiclass_model = None

        self.is_loaded = True

    def predict_flow(self, df_19: pd.DataFrame) -> Tuple[str, int, str, float]:
        """
        Transforms live 19-feature DataFrame through preprocessor_19.joblib
        and returns (prediction_str, binary_class_int, attack_category_str, confidence_float).
        """
        if not self.is_loaded:
            self.load_artifacts()

        # Schema & Order Verification
        if list(df_19.columns) != SELECTED_19_FEATURES:
            raise ValueError(f"Feature column mismatch! Expected {SELECTED_19_FEATURES}, got {list(df_19.columns)}")

        # Transform using frozen 19-feature preprocessor
        X_proc = self.preprocessor.transform(df_19)

        # Binary Inference
        bin_pred = int(self.binary_model.predict(X_proc)[0])
        confidence = 1.0

        if hasattr(self.binary_model, "predict_proba"):
            try:
                probs = self.binary_model.predict_proba(X_proc)[0]
                confidence = float(np.max(probs))
            except Exception:
                pass

        prediction_label = "Attack" if bin_pred == 1 else "Normal"
        attack_cat = "Normal"

        # Multiclass Inference if attack detected
        if bin_pred == 1 and self.multiclass_model is not None:
            try:
                if getattr(self, "is_multiclass_keras", False):
                    m_probs = self.multiclass_model.predict(X_proc, verbose=0)[0]
                    m_class_idx = int(np.argmax(m_probs))
                    confidence = float(np.max(m_probs))
                else:
                    m_class_idx = int(self.multiclass_model.predict(X_proc)[0])

                if self.label_encoder:
                    attack_cat = str(self.label_encoder.classes_[m_class_idx])
                else:
                    attack_cat = f"Attack_Class_{m_class_idx}"
            except Exception as e:
                logger.error(f"Multiclass inference error: {e}")
                attack_cat = "Unknown Attack"

        return prediction_label, bin_pred, attack_cat, round(float(confidence), 4)
