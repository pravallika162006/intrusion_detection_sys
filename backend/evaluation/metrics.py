"""
Evaluation metrics module for Phase 1 baseline models.
Calculates Accuracy, Precision, Recall, F1-score, and Confusion Matrices for binary and multiclass tasks.
Saves structured machine-readable result files and formatted text reports.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from backend.config import REPORTS_DIR, CONFUSION_DIR

def evaluate_binary_model(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_time: float,
    pred_time: float,
) -> Dict[str, Any]:
    """
    Evaluates binary classification model predictions.
    Calculates Accuracy, Precision, Recall, F1 for positive class (1 = Attack),
    macro/weighted averages, and confusion matrix.
    """
    acc = accuracy_score(y_true, y_pred)
    
    # Positive class (Attack = 1) metrics
    prec_pos = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec_pos = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_pos = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    # Macro & Weighted metrics
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    prec_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    report_str = classification_report(
        y_true, y_pred, target_names=["Normal (0)", "Attack (1)"], digits=4, zero_division=0
    )

    # Save classification report and confusion matrix
    save_classification_report(f"binary_{model_name}", report_str)
    save_confusion_matrix(f"binary_{model_name}", cm, labels=["Normal (0)", "Attack (1)"])

    metrics = {
        "Model": model_name,
        "Accuracy": round(acc, 4),
        "Precision (Attack)": round(prec_pos, 4),
        "Recall (Attack)": round(rec_pos, 4),
        "F1 (Attack)": round(f1_pos, 4),
        "Macro Precision": round(prec_macro, 4),
        "Macro Recall": round(rec_macro, 4),
        "Macro F1": round(f1_macro, 4),
        "Weighted Precision": round(prec_weighted, 4),
        "Weighted Recall": round(rec_weighted, 4),
        "Weighted F1": round(f1_weighted, 4),
        "Training Time (s)": round(train_time, 4),
        "Prediction Time (s)": round(pred_time, 4),
        "Confusion Matrix": cm.tolist(),
    }
    return metrics

def evaluate_multiclass_model(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str],
    train_time: float,
    pred_time: float,
) -> Dict[str, Any]:
    """
    Evaluates multiclass classification model predictions.
    Calculates Accuracy, Macro & Weighted Precision, Recall, F1, per-class metrics,
    and confusion matrix.
    """
    acc = accuracy_score(y_true, y_pred)
    
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    prec_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    report_str = classification_report(
        y_true, y_pred, target_names=target_names, digits=4, zero_division=0
    )

    # Save classification report and confusion matrix
    save_classification_report(f"multiclass_{model_name}", report_str)
    save_confusion_matrix(f"multiclass_{model_name}", cm, labels=target_names)

    metrics = {
        "Model": model_name,
        "Accuracy": round(acc, 4),
        "Macro Precision": round(prec_macro, 4),
        "Macro Recall": round(rec_macro, 4),
        "Macro F1": round(f1_macro, 4),
        "Weighted Precision": round(prec_weighted, 4),
        "Weighted Recall": round(rec_weighted, 4),
        "Weighted F1": round(f1_weighted, 4),
        "Training Time (s)": round(train_time, 4),
        "Prediction Time (s)": round(pred_time, 4),
        "Confusion Matrix": cm.tolist(),
    }
    return metrics

def save_classification_report(name: str, report_str: str) -> None:
    """Saves formatted text classification report file."""
    filepath = REPORTS_DIR / f"{name}_classification_report.txt"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report_str)

def save_confusion_matrix(name: str, cm: np.ndarray, labels: List[str]) -> None:
    """Saves confusion matrix as a structured JSON file."""
    filepath = CONFUSION_DIR / f"{name}_confusion_matrix.json"
    data = {
        "labels": labels,
        "confusion_matrix": cm.tolist()
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
