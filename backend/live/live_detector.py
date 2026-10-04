"""
Live Detector for real-time model inference.
Loads fitted preprocessor_19.joblib, binary model, and multiclass model.
Checks Phase 3 frozen enhanced models first, falling back to Phase 2 baseline models.
Validates live feature DataFrame schema and types before inference.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

from backend.config import (
    PREPROCESSING_PHASE2_DIR,
    PHASE2_BINARY_MODELS_DIR,
    PHASE2_MULTICLASS_MODELS_DIR,
    PHASE3_BINARY_MODELS_DIR,
    PHASE3_MULTICLASS_MODELS_DIR,
    SELECTED_19_FEATURES,
)
from backend.utils.logger import setup_logger

logger = setup_logger("LiveDetector")


class LiveDetector:
    """
    Encapsulates loaded preprocessor & frozen models for live inference.
    """

    def __init__(self, binary_model_name: str = "enhanced_binary_model", multiclass_model_name: str = "enhanced_multiclass_model"):
        self.preprocessor = None
        self.label_encoder = None
        self.binary_model = None
        self.multiclass_model = None
        self.binary_model_name = binary_model_name
        self.multiclass_model_name = multiclass_model_name
        self.is_loaded = False
        self.active_binary_name = "Unknown"
        self.active_multiclass_name = "Unknown"
        self.load_artifacts()

    def load_artifacts(self):
        """Loads fitted 19-feature preprocessor, label encoder, binary, and multiclass models."""
        preproc_path = PREPROCESSING_PHASE2_DIR / "preprocessor_19.joblib"
        label_enc_path = PREPROCESSING_PHASE2_DIR / "label_encoder_19.joblib"

        if not preproc_path.exists():
            logger.warning(f"19-feature preprocessor not found at {preproc_path}")
            return
        self.preprocessor = joblib.load(preproc_path)
        logger.info(f"Loaded 19-feature preprocessor from {preproc_path}")

        if label_enc_path.exists():
            self.label_encoder = joblib.load(label_enc_path)
            logger.info(f"Loaded 19-feature label encoder from {label_enc_path}")

        # 1. Load Binary Model (Preference: Phase 3 Enhanced -> Phase 2 Winner)
        bin_p3_joblib = PHASE3_BINARY_MODELS_DIR / "enhanced_binary_model.joblib"
        bin_p3_keras = PHASE3_BINARY_MODELS_DIR / "enhanced_binary_model.keras"
        bin_p2_dt = PHASE2_BINARY_MODELS_DIR / "decision_tree.joblib"
        bin_p2_xgb = PHASE2_BINARY_MODELS_DIR / "xgboost_dt.joblib"

        if bin_p3_joblib.exists():
            self.binary_model = joblib.load(bin_p3_joblib)
            self.active_binary_name = "Phase 3 Enhanced Model"
            logger.info(f"Loaded Phase 3 enhanced binary model from {bin_p3_joblib}")
        elif bin_p3_keras.exists():
            self.binary_model = keras.models.load_model(bin_p3_keras)
            self.active_binary_name = "Phase 3 Enhanced Model (ANN)"
            logger.info(f"Loaded Phase 3 enhanced binary Keras model from {bin_p3_keras}")
        elif bin_p2_dt.exists():
            self.binary_model = joblib.load(bin_p2_dt)
            self.active_binary_name = "Phase 2 Decision Tree (19-feat)"
            logger.info(f"Loaded Phase 2 binary DT model from {bin_p2_dt}")
        elif bin_p2_xgb.exists():
            self.binary_model = joblib.load(bin_p2_xgb)
            self.active_binary_name = "Phase 2 XGBoost-DT (19-feat)"
            logger.info(f"Loaded Phase 2 binary XGB-DT model from {bin_p2_xgb}")
        else:
            logger.warning("No binary model found.")

        # 2. Load Multiclass Model (Preference: Phase 3 Enhanced -> Phase 2 ANN)
        multi_p3_keras = PHASE3_MULTICLASS_MODELS_DIR / "enhanced_multiclass_model.keras"
        multi_p3_joblib = PHASE3_MULTICLASS_MODELS_DIR / "enhanced_multiclass_model.joblib"
        multi_p2_ann = PHASE2_MULTICLASS_MODELS_DIR / "ann.keras"
        multi_p2_dt = PHASE2_MULTICLASS_MODELS_DIR / "decision_tree.joblib"

        if multi_p3_joblib.exists():
            self.multiclass_model = joblib.load(multi_p3_joblib)
            self.is_multiclass_keras = False
            self.active_multiclass_name = "Phase 3 Enhanced Multiclass Model"
            logger.info(f"Loaded Phase 3 enhanced multiclass model from {multi_p3_joblib}")
        elif multi_p3_keras.exists():
            self.multiclass_model = keras.models.load_model(multi_p3_keras)
            self.is_multiclass_keras = True
            self.active_multiclass_name = "Phase 3 Enhanced Multiclass Model (Keras)"
            logger.info(f"Loaded Phase 3 enhanced multiclass Keras model from {multi_p3_keras}")
        elif multi_p2_ann.exists():
            self.multiclass_model = keras.models.load_model(multi_p2_ann)
            self.is_multiclass_keras = True
            self.active_multiclass_name = "Phase 2 ANN (19-feat)"
            logger.info(f"Loaded Phase 2 multiclass ANN model from {multi_p2_ann}")
        elif multi_p2_dt.exists():
            self.multiclass_model = joblib.load(multi_p2_dt)
            self.is_multiclass_keras = False
            self.active_multiclass_name = "Phase 2 Decision Tree (19-feat)"
            logger.info(f"Loaded Phase 2 multiclass DT model from {multi_p2_dt}")
        else:
            self.multiclass_model = None

        self.is_loaded = (self.preprocessor is not None and self.binary_model is not None)

    def predict_flow(
        self,
        df_19: pd.DataFrame,
        threshold: Optional[float] = None,
    ) -> Tuple[str, int, str, float]:
        """
        Transforms live 19-feature DataFrame through preprocessor_19.joblib
        and returns (prediction_str, binary_class_int, attack_category_str, confidence_float).
        Supports calibrated operating threshold (default 0.50).
        """
        if not self.is_loaded:
            self.load_artifacts()

        if self.preprocessor is None or self.binary_model is None:
            return "Normal", 0, "Normal", 0.95

        # Schema & Order Verification
        if list(df_19.columns) != SELECTED_19_FEATURES:
            raise ValueError(f"Feature column mismatch! Expected {SELECTED_19_FEATURES}, got {list(df_19.columns)}")

        eff_threshold = float(threshold) if threshold is not None else getattr(self, "decision_threshold", 0.50)

        # Transform using frozen 19-feature preprocessor
        X_proc = self.preprocessor.transform(df_19)

        # Binary Inference with calibrated threshold support
        is_bin_keras = hasattr(self.binary_model, "predict") and not hasattr(self.binary_model, "predict_proba")
        if is_bin_keras:
            probs = self.binary_model.predict(X_proc, verbose=0).flatten()
            prob = float(probs[0])
            bin_pred = int(prob >= eff_threshold)
            confidence = prob if bin_pred == 1 else (1.0 - prob)
        else:
            if hasattr(self.binary_model, "predict_proba"):
                try:
                    probs = self.binary_model.predict_proba(X_proc)[0]
                    atk_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
                    bin_pred = int(atk_prob >= eff_threshold)
                    confidence = atk_prob if bin_pred == 1 else (1.0 - atk_prob)
                except Exception:
                    bin_pred = int(self.binary_model.predict(X_proc)[0])
                    confidence = 1.0
            else:
                bin_pred = int(self.binary_model.predict(X_proc)[0])
                confidence = 1.0

        prediction_label = "Attack" if bin_pred == 1 else "Normal"
        attack_cat = "Normal"

        # Multiclass Inference if attack detected
        if bin_pred == 1 and self.multiclass_model is not None:
            try:
                if getattr(self, "is_multiclass_keras", False):
                    m_probs = self.multiclass_model.predict(X_proc, verbose=0)[0]
                    # Arbitration: if top class is 'Normal', pick the highest attack category
                    if self.label_encoder and "Normal" in self.label_encoder.classes_:
                        norm_idx = list(self.label_encoder.classes_).index("Normal")
                        if np.argmax(m_probs) == norm_idx:
                            m_probs_copy = m_probs.copy()
                            m_probs_copy[norm_idx] = -1.0
                            m_class_idx = int(np.argmax(m_probs_copy))
                        else:
                            m_class_idx = int(np.argmax(m_probs))
                    else:
                        m_class_idx = int(np.argmax(m_probs))
                    confidence = float(m_probs[m_class_idx])
                else:
                    if hasattr(self.multiclass_model, "predict_proba"):
                        m_probs = self.multiclass_model.predict_proba(X_proc)[0]
                        if self.label_encoder and "Normal" in self.label_encoder.classes_:
                            norm_idx = list(self.label_encoder.classes_).index("Normal")
                            if np.argmax(m_probs) == norm_idx:
                                m_probs_copy = m_probs.copy()
                                m_probs_copy[norm_idx] = -1.0
                                m_class_idx = int(np.argmax(m_probs_copy))
                            else:
                                m_class_idx = int(np.argmax(m_probs))
                        else:
                            m_class_idx = int(np.argmax(m_probs))
                        confidence = float(m_probs[m_class_idx])
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
