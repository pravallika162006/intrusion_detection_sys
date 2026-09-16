"""
FastAPI routes for Dataset Analysis & Prediction Dashboard.
Supports built-in UNSW-NB15 data, user CSV/Parquet uploads, validation,
42-feature / 19-feature prediction, and evaluation metrics when labels are present.
"""

import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import joblib
from tensorflow import keras
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from backend.config import (
    UPLOADS_DIR,
    TEST_CSV_PATH,
    ALL_INPUT_FEATURES,
    SELECTED_19_FEATURES,
    BINARY_TARGET,
    MULTICLASS_TARGET,
    PREPROCESSING_DIR,
    PREPROCESSING_PHASE2_DIR,
    BINARY_MODELS_DIR,
    MULTICLASS_MODELS_DIR,
    PHASE2_BINARY_MODELS_DIR,
    PHASE2_MULTICLASS_MODELS_DIR,
)
from backend.app.schemas import (
    DatasetValidationResponse,
    DatasetPredictionResponse,
    DatasetPredictionSummary,
    EvaluationMetrics,
)
from backend.utils.logger import setup_logger

logger = setup_logger("DatasetRoutes")
router = APIRouter(prefix="/api/dataset", tags=["Dataset Dashboard"])


def _load_dataframe(file_path: Path) -> pd.DataFrame:
    """Helper to read CSV or Parquet into a Pandas DataFrame."""
    ext = file_path.suffix.lower()
    if ext in [".csv", ".txt"]:
        return pd.read_csv(file_path)
    elif ext in [".parquet", ".pq"]:
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


@router.post("/upload", response_model=DatasetValidationResponse)
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload a CSV or Parquet dataset file, validate its columns,
    and return validation summary.
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in [".csv", ".txt", ".parquet", ".pq"]:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload a CSV or Parquet file."
        )

    file_id = f"{uuid.uuid4().hex}_{file.filename}"
    saved_path = UPLOADS_DIR / file_id

    try:
        content = await file.read()
        with open(saved_path, "wb") as f:
            f.write(content)

        df = _load_dataframe(saved_path)
    except Exception as e:
        if saved_path.exists():
            saved_path.unlink()
        logger.error(f"Failed to parse uploaded dataset: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to parse uploaded dataset file: {str(e)}"
        )

    if df.empty:
        saved_path.unlink()
        raise HTTPException(status_code=400, detail="Uploaded dataset is empty.")

    return _validate_dataframe_schema(df, file_id, file.filename)


@router.get("/builtin-validation", response_model=DatasetValidationResponse)
def validate_builtin_dataset():
    """Validates the built-in UNSW-NB15 test set."""
    if not TEST_CSV_PATH.exists():
        raise HTTPException(
            status_code=444,
            detail=f"Built-in UNSW-NB15 test dataset not found at {TEST_CSV_PATH}"
        )
    df = pd.read_csv(TEST_CSV_PATH)
    return _validate_dataframe_schema(df, "BUILTIN_UNSW_NB15", "UNSW_NB15_testing-set.csv")


def _validate_dataframe_schema(
    df: pd.DataFrame, file_id: str, original_filename: str
) -> DatasetValidationResponse:
    """Inspects DataFrame columns against 42-feature and 19-feature requirements."""
    columns = set(df.columns)
    
    missing_42 = [f for f in ALL_INPUT_FEATURES if f not in columns]
    missing_19 = [f for f in SELECTED_19_FEATURES if f not in columns]

    has_bin_label = BINARY_TARGET in columns
    has_multi_label = MULTICLASS_TARGET in columns

    features_present_42 = len(missing_42) == 0
    features_present_19 = len(missing_19) == 0

    if features_present_42 or features_present_19:
        status = "VALID"
        msg = f"Dataset validated successfully ({len(df)} rows, {len(df.columns)} columns)."
    else:
        status = "INVALID"
        msg = f"Dataset is missing required features for both 42 and 19 feature modes."

    return DatasetValidationResponse(
        filename=file_id,
        file_format=Path(original_filename).suffix.lower(),
        row_count=len(df),
        column_count=len(df.columns),
        has_labels=has_bin_label,
        label_column=BINARY_TARGET if has_bin_label else None,
        has_multiclass_labels=has_multi_label,
        multiclass_column=MULTICLASS_TARGET if has_multi_label else None,
        features_present_42=features_present_42,
        missing_features_42=missing_42,
        features_present_19=features_present_19,
        missing_features_19=missing_19,
        validation_status=status,
        message=msg,
    )


@router.post("/predict", response_model=DatasetPredictionResponse)
def predict_dataset(
    file_id: str = Form(...),
    feature_mode: str = Form("19"),
    model_name: str = Form("xgboost_dt"),
    prediction_task: str = Form("binary"),
):
    """
    Runs model inference on uploaded or built-in dataset.
    Returns predictions summary, sample records, and evaluation metrics if labels exist.
    """
    if file_id == "BUILTIN_UNSW_NB15":
        file_path = TEST_CSV_PATH
    else:
        file_path = UPLOADS_DIR / file_id

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Dataset file '{file_id}' not found."
        )

    try:
        df = _load_dataframe(file_path)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error reading dataset file: {str(e)}"
        )

    # 1. Feature check
    if feature_mode == "19":
        required_features = SELECTED_19_FEATURES
        preprocessor_file = PREPROCESSING_PHASE2_DIR / "preprocessor_19.joblib"
        label_encoder_file = PREPROCESSING_PHASE2_DIR / "label_encoder_19.joblib"
        models_dir = PHASE2_BINARY_MODELS_DIR if prediction_task == "binary" else PHASE2_MULTICLASS_MODELS_DIR
    elif feature_mode == "42":
        required_features = ALL_INPUT_FEATURES
        preprocessor_file = PREPROCESSING_DIR / "preprocessor.joblib"
        label_encoder_file = PREPROCESSING_DIR / "label_encoder.joblib"
        models_dir = BINARY_MODELS_DIR if prediction_task == "binary" else MULTICLASS_MODELS_DIR
    else:
        raise HTTPException(status_code=400, detail="Invalid feature_mode. Must be '19' or '42'.")

    missing = [f for f in required_features if f not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing {len(missing)} required features for {feature_mode}-feature mode: {missing[:5]}"
        )

    # Extract required feature columns in exact order
    X_raw = df[required_features].copy()

    # Load preprocessor
    if not preprocessor_file.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Preprocessing artifact missing: {preprocessor_file}"
        )
    preprocessor = joblib.load(preprocessor_file)

    try:
        X_proc = preprocessor.transform(X_raw)
    except Exception as e:
        logger.error(f"Preprocessing transformation error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error transforming dataset with preprocessor: {str(e)}"
        )

    # Load Model
    model_path_joblib = models_dir / f"{model_name}.joblib"
    model_path_keras = models_dir / f"{model_name}.keras"

    if model_path_joblib.exists():
        model = joblib.load(model_path_joblib)
        is_keras = False
    elif model_path_keras.exists():
        model = keras.models.load_model(model_path_keras)
        is_keras = True
    else:
        available = [p.stem for p in models_dir.glob("*")]
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found in {models_dir}. Available: {available}"
        )

    # Predict
    if is_keras:
        probs = model.predict(X_proc, verbose=0)
        if prediction_task == "binary":
            if probs.ndim > 1 and probs.shape[1] > 1:
                preds = np.argmax(probs, axis=1)
                confidences = np.max(probs, axis=1).tolist()
            else:
                probs_flat = probs.flatten()
                preds = (probs_flat >= 0.5).astype(int)
                confidences = probs_flat.tolist()
        else:
            preds = np.argmax(probs, axis=1)
            confidences = np.max(probs, axis=1).tolist()
    else:
        preds = model.predict(X_proc)
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(X_proc)
                confidences = np.max(probs, axis=1).tolist()
            except Exception:
                confidences = None
        else:
            confidences = None

    # Load LabelEncoder if multiclass
    label_encoder = None
    if prediction_task == "multiclass" and label_encoder_file.exists():
        label_encoder = joblib.load(label_encoder_file)

    # Formulate Results
    total_records = len(preds)
    if prediction_task == "binary":
        normal_count = int(np.sum(preds == 0))
        attack_count = int(np.sum(preds == 1))
        attack_pct = round((attack_count / total_records) * 100, 2)
        attack_categories = None
    else:
        # Multiclass prediction
        if label_encoder is not None:
            pred_labels = label_encoder.inverse_transform(preds)
        else:
            pred_labels = [str(p) for p in preds]

        cat_counts = pd.Series(pred_labels).value_counts().to_dict()
        normal_count = int(cat_counts.get("Normal", cat_counts.get(0, 0)))
        attack_count = total_records - normal_count
        attack_pct = round((attack_count / total_records) * 100, 2)
        attack_categories = {str(k): int(v) for k, v in cat_counts.items()}

    summary = DatasetPredictionSummary(
        total_records=total_records,
        normal_count=normal_count,
        attack_count=attack_count,
        attack_percentage=attack_pct,
        attack_categories=attack_categories,
        confidence_avg=round(float(np.mean(confidences)), 4) if confidences is not None else None,
    )

    # Ground Truth Evaluation if labels exist
    evaluation = None
    target_col = BINARY_TARGET if prediction_task == "binary" else MULTICLASS_TARGET
    if target_col in df.columns:
        y_true_raw = df[target_col].values
        if prediction_task == "binary":
            y_true = y_true_raw.astype(int)
            y_pred = preds.astype(int)
            target_names = ["Normal", "Attack"]
        else:
            if label_encoder is not None and isinstance(y_true_raw[0], str):
                y_true = label_encoder.transform(y_true_raw)
            else:
                y_true = y_true_raw
            y_pred = preds
            target_names = list(label_encoder.classes_) if label_encoder else None

        acc = float(accuracy_score(y_true, y_pred))
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        _, _, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred).tolist()
        report_dict = classification_report(
            y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0
        )

        evaluation = EvaluationMetrics(
            accuracy=round(acc, 4),
            precision=round(float(prec), 4),
            recall=round(float(rec), 4),
            f1_score=round(float(f1), 4),
            macro_f1=round(float(macro_f1), 4),
            weighted_f1=round(float(f1), 4),
            confusion_matrix=cm,
            classification_report=report_dict,
        )

    # Sample records (top 20) for table view
    sample_records = []
    max_samples = min(20, total_records)
    for i in range(max_samples):
        rec = {
            "index": i + 1,
            "prediction": "Attack" if (preds[i] == 1 if prediction_task == "binary" else preds[i] != 0) else "Normal",
            "pred_value": int(preds[i]),
        }
        if prediction_task == "multiclass" and label_encoder:
            rec["attack_category"] = str(label_encoder.classes_[preds[i]])
        if confidences:
            rec["confidence"] = round(float(confidences[i]), 4)
        sample_records.append(rec)

    return DatasetPredictionResponse(
        file_id=file_id,
        feature_mode=feature_mode,
        model_name=model_name,
        prediction_task=prediction_task,
        summary=summary,
        evaluation=evaluation,
        sample_predictions=sample_records,
    )
