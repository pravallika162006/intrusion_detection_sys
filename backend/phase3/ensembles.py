"""
Phase 3 Model Ensembles Module.
Constructs and evaluates justified ensemble configurations on 19 features:
1. Soft Voting: Random Forest + XGBoost Classifier
2. Soft Voting: Decision Tree + Random Forest + XGBoost Classifier
3. Soft Voting: Random Forest + HistGradientBoosting
4. Hard Voting: Decision Tree + kNN + Random Forest
5. Stacking Classifier: Base models (DT, RF, XGBoost) with LogisticRegression meta-classifier

Evaluated strictly on the VALIDATION set to prevent test-set leakage.
"""

import time
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    StackingClassifier,
    HistGradientBoostingClassifier,
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
from backend.utils.logger import setup_logger

logger = setup_logger("Phase3_Ensembles")


def build_and_eval_ensembles(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    task: str = "binary",
    target_names: List[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Constructs, trains, and evaluates voting and stacking ensembles strictly on (X_val, y_val).
    Returns (df_ensemble_results, fitted_ensembles_dict).
    """
    logger.info(f"Evaluating Phase 3 Ensembles on VALIDATION set ({task})...")

    is_binary = (task == "binary")
    avg_mode = "binary" if is_binary else "macro"

    # Base estimators with probability support
    dt = DecisionTreeClassifier(max_depth=8, random_state=RANDOM_SEED)
    rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=RANDOM_SEED, n_jobs=-1)
    xgb_clf = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1, random_state=RANDOM_SEED, n_jobs=-1,
        eval_metric="logloss" if is_binary else "mlogloss"
    )
    hgb = HistGradientBoostingClassifier(max_iter=100, random_state=RANDOM_SEED)
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)

    ensembles = {
        "Ensemble (Soft Voting: RF + XGBoost)": VotingClassifier(
            estimators=[("rf", rf), ("xgb", xgb_clf)],
            voting="soft",
            n_jobs=-1,
        ),
        "Ensemble (Soft Voting: DT + RF + XGBoost)": VotingClassifier(
            estimators=[("dt", dt), ("rf", rf), ("xgb", xgb_clf)],
            voting="soft",
            n_jobs=-1,
        ),
        "Ensemble (Soft Voting: RF + HistGB)": VotingClassifier(
            estimators=[("rf", rf), ("hgb", hgb)],
            voting="soft",
            n_jobs=-1,
        ),
        "Ensemble (Hard Voting: DT + kNN + RF)": VotingClassifier(
            estimators=[("dt", dt), ("knn", knn), ("rf", rf)],
            voting="hard",
            n_jobs=-1,
        ),
        "Ensemble (Stacking: DT + RF + XGBoost -> LR)": StackingClassifier(
            estimators=[("dt", dt), ("rf", rf), ("xgb", xgb_clf)],
            final_estimator=LogisticRegression(max_iter=1000, random_state=LR_RANDOM_STATE),
            n_jobs=-1,
            cv=2,
        ),
    }

    fitted_ensembles = {}
    rows = []

    for name, ensemble in ensembles.items():
        logger.info(f"  --> Training ensemble: {name}...")
        t0 = time.time()
        ensemble.fit(X_train, y_train)
        t_train = time.time() - t0

        t0_p = time.time()
        val_pred = ensemble.predict(X_val)
        t_pred = time.time() - t0_p

        val_acc = accuracy_score(y_val, val_pred)
        val_prec = precision_score(y_val, val_pred, average=avg_mode, zero_division=0)
        val_rec = recall_score(y_val, val_pred, average=avg_mode, zero_division=0)
        val_f1 = f1_score(y_val, val_pred, average=avg_mode, zero_division=0)
        val_macro_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
        val_weighted_f1 = f1_score(y_val, val_pred, average="weighted", zero_division=0)

        fitted_ensembles[name] = ensemble
        rows.append({
            "Model": name,
            "Type": "Ensemble",
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

    df_results = pd.DataFrame(rows).sort_values(by="Val_Accuracy", ascending=False)
    return df_results, fitted_ensembles
