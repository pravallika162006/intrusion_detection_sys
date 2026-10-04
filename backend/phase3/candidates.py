"""
Phase 3 Candidate Classifiers Module.
Evaluates alternative individual ML models on the 19-feature representation:
- Baseline models: Decision Tree, ANN, kNN, Logistic Regression
- Additional models: Random Forest, XGBoost Classifier, HistGradientBoosting, Extra Trees

All trained strictly on TRAIN-1 and evaluated on the VALIDATION set.
"""

import time
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from backend.config import RANDOM_SEED, LR_RANDOM_STATE
from backend.models.ann_builder import build_paper_ann_model, get_early_stopping
from backend.utils.logger import setup_logger

logger = setup_logger("Phase3_Candidates")


def train_and_eval_candidates(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    task: str = "binary",
    target_names: List[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Trains candidate models on (X_train, y_train) and evaluates strictly on (X_val, y_val).
    Returns (df_val_results, fitted_models_dict).
    """
    logger.info(f"Evaluating Phase 3 Candidate Classifiers on VALIDATION set ({task})...")

    is_binary = (task == "binary")
    avg_mode = "binary" if is_binary else "macro"
    pos_label = 1 if is_binary else None
    num_classes = 2 if is_binary else (len(target_names) if target_names else len(np.unique(y_train)))

    candidates = {
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_SEED),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=LR_RANDOM_STATE),
        "kNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=RANDOM_SEED, n_jobs=-1),
        "XGBoost Classifier": xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=RANDOM_SEED, n_jobs=-1,
            eval_metric="logloss" if is_binary else "mlogloss"
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=100, random_state=RANDOM_SEED),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, max_depth=12, random_state=RANDOM_SEED, n_jobs=-1),
    }

    fitted_models = {}
    rows = []

    for name, clf in candidates.items():
        logger.info(f"  --> Training candidate: {name}...")
        t0 = time.time()
        clf.fit(X_train, y_train)
        t_train = time.time() - t0

        t0_p = time.time()
        val_pred = clf.predict(X_val)
        t_pred = time.time() - t0_p

        val_acc = accuracy_score(y_val, val_pred)
        val_prec = precision_score(y_val, val_pred, average=avg_mode, zero_division=0)
        val_rec = recall_score(y_val, val_pred, average=avg_mode, zero_division=0)
        val_f1 = f1_score(y_val, val_pred, average=avg_mode, zero_division=0)
        val_macro_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
        val_weighted_f1 = f1_score(y_val, val_pred, average="weighted", zero_division=0)

        fitted_models[name] = clf
        rows.append({
            "Model": name,
            "Type": "Individual Candidate",
            "Val_Accuracy": round(val_acc * 100, 2),
            "Val_Precision": round(val_prec * 100, 2),
            "Val_Recall": round(val_rec * 100, 2),
            "Val_F1": round(val_f1 * 100, 2),
            "Val_Macro_F1": round(val_macro_f1 * 100, 2),
            "Val_Weighted_F1": round(val_weighted_f1 * 100, 2),
            "Train_Time_s": round(t_train, 2),
            "Prediction_Time_s": round(t_pred, 4),
        })
        logger.info(f"      {name} -> Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1*100:.2f}%")

    # Add ANN candidate
    logger.info("  --> Training candidate: ANN...")
    t0_ann = time.time()
    ann = build_paper_ann_model(input_dim=X_train.shape[1], num_classes=num_classes, hidden_units=64, learning_rate=0.01)
    ann.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=25,
        batch_size=256,
        callbacks=[get_early_stopping(patience=4)],
        verbose=0,
    )
    t_train_ann = time.time() - t0_ann

    t0_p = time.time()
    if is_binary:
        ann_probs = ann.predict(X_val, batch_size=512, verbose=0).flatten()
        ann_pred = (ann_probs >= 0.5).astype(int)
    else:
        ann_probs = ann.predict(X_val, batch_size=512, verbose=0)
        ann_pred = np.argmax(ann_probs, axis=1)
    t_pred_ann = time.time() - t0_p

    val_acc = accuracy_score(y_val, ann_pred)
    val_prec = precision_score(y_val, ann_pred, average=avg_mode, zero_division=0)
    val_rec = recall_score(y_val, ann_pred, average=avg_mode, zero_division=0)
    val_f1 = f1_score(y_val, ann_pred, average=avg_mode, zero_division=0)
    val_macro_f1 = f1_score(y_val, ann_pred, average="macro", zero_division=0)
    val_weighted_f1 = f1_score(y_val, ann_pred, average="weighted", zero_division=0)

    fitted_models["ANN"] = ann
    rows.append({
        "Model": "ANN",
        "Type": "Individual Candidate",
        "Val_Accuracy": round(val_acc * 100, 2),
        "Val_Precision": round(val_prec * 100, 2),
        "Val_Recall": round(val_rec * 100, 2),
        "Val_F1": round(val_f1 * 100, 2),
        "Val_Macro_F1": round(val_macro_f1 * 100, 2),
        "Val_Weighted_F1": round(val_weighted_f1 * 100, 2),
        "Train_Time_s": round(t_train_ann, 2),
        "Prediction_Time_s": round(t_pred_ann, 4),
    })
    logger.info(f"      ANN -> Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1*100:.2f}%")

    df_results = pd.DataFrame(rows).sort_values(by="Val_Accuracy", ascending=False)
    return df_results, fitted_models
