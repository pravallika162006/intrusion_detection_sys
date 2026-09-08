"""
Phase 2 Trainer Module.
Trains and evaluates 7 model approaches on the 19 XGBoost-selected features:
1. Decision Tree
2. ANN (Keras)
3. kNN
4. Logistic Regression
5. SVM (LinearSVC)
6. XGBoost + Decision Tree (Sequential pipeline: XGBoost top-19 selection -> Decision Tree)
7. XGBoost + kNN (Sequential pipeline: XGBoost top-19 selection -> kNN)
"""

import time
import json
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from backend.config import RANDOM_SEED, MODELS_DIR, RESULTS_DIR
from backend.models.ann_builder import build_ann_model, get_early_stopping
from backend.utils.logger import setup_logger

logger = setup_logger("Phase2_Trainers")

PHASE2_MODELS_DIR = MODELS_DIR / "phase2"
PHASE2_BINARY_MODELS_DIR = PHASE2_MODELS_DIR / "binary"
PHASE2_MULTI_MODELS_DIR = PHASE2_MODELS_DIR / "multiclass"

PHASE2_RESULTS_DIR = RESULTS_DIR / "phase2"
PHASE2_REPORTS_DIR = PHASE2_RESULTS_DIR / "classification_reports"
PHASE2_CONFUSION_DIR = PHASE2_RESULTS_DIR / "confusion_matrices"

for d in [
    PHASE2_MODELS_DIR,
    PHASE2_BINARY_MODELS_DIR,
    PHASE2_MULTI_MODELS_DIR,
    PHASE2_RESULTS_DIR,
    PHASE2_REPORTS_DIR,
    PHASE2_CONFUSION_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)

def eval_and_save_binary(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_time: float,
    pred_time: float,
) -> Dict[str, Any]:
    """Helper to evaluate and save binary metrics, report, and confusion matrix."""
    acc = accuracy_score(y_true, y_pred)
    prec_pos = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec_pos = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_pos = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

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

    clean_name = model_name.replace(" ", "_").replace("+", "_plus_").replace("(", "").replace(")", "")
    
    # Save text report
    with open(PHASE2_REPORTS_DIR / f"binary_{clean_name}_report.txt", "w", encoding="utf-8") as f:
        f.write(report_str)

    # Save confusion matrix JSON
    with open(PHASE2_CONFUSION_DIR / f"binary_{clean_name}_cm.json", "w", encoding="utf-8") as f:
        json.dump({"labels": ["Normal (0)", "Attack (1)"], "confusion_matrix": cm.tolist()}, f, indent=2)

    return {
        "Model": model_name,
        "Feature Set": "19 Features",
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
    }

def eval_and_save_multiclass(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str],
    train_time: float,
    pred_time: float,
) -> Dict[str, Any]:
    """Helper to evaluate and save multiclass metrics, report, and confusion matrix."""
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

    clean_name = model_name.replace(" ", "_").replace("+", "_plus_").replace("(", "").replace(")", "")

    with open(PHASE2_REPORTS_DIR / f"multiclass_{clean_name}_report.txt", "w", encoding="utf-8") as f:
        f.write(report_str)

    with open(PHASE2_CONFUSION_DIR / f"multiclass_{clean_name}_cm.json", "w", encoding="utf-8") as f:
        json.dump({"labels": target_names, "confusion_matrix": cm.tolist()}, f, indent=2)

    return {
        "Model": model_name,
        "Feature Set": "19 Features",
        "Accuracy": round(acc, 4),
        "Macro Precision": round(prec_macro, 4),
        "Macro Recall": round(rec_macro, 4),
        "Macro F1": round(f1_macro, 4),
        "Weighted Precision": round(prec_weighted, 4),
        "Weighted Recall": round(rec_weighted, 4),
        "Weighted F1": round(f1_weighted, 4),
        "Training Time (s)": round(train_time, 4),
        "Prediction Time (s)": round(pred_time, 4),
    }

def train_phase2_binary_models(
    X19_train: np.ndarray,
    X19_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Trains 7 binary model approaches on 19 features."""
    logger.info("==================================================")
    logger.info(" STARTING PHASE 2 BINARY CLASSIFICATION (19 FEAT) ")
    logger.info("==================================================")

    results = []
    model_paths = {}

    # 1. Decision Tree
    logger.info("--> [1/7] Binary 19-Feature Decision Tree...")
    dt = DecisionTreeClassifier(max_depth=20, random_state=RANDOM_SEED)
    t0 = time.time()
    dt.fit(X19_train, y_train)
    t_tr = time.time() - t0
    t0 = time.time()
    y_pred = dt.predict(X19_test)
    t_pr = time.time() - t0
    p = PHASE2_BINARY_MODELS_DIR / "decision_tree.joblib"
    joblib.dump(dt, p)
    model_paths["Decision Tree"] = str(p)
    results.append(eval_and_save_binary("Decision Tree", y_test, y_pred, t_tr, t_pr))

    # 2. ANN
    logger.info("--> [2/7] Binary 19-Feature ANN (Keras)...")
    ann = build_ann_model(input_dim=X19_train.shape[1], num_classes=2)
    early_stop = get_early_stopping(patience=5)
    t0 = time.time()
    ann.fit(X19_train, y_train, epochs=30, batch_size=256, validation_split=0.1, callbacks=[early_stop], verbose=0)
    t_tr = time.time() - t0
    t0 = time.time()
    y_proba = ann.predict(X19_test, batch_size=512, verbose=0)
    y_pred = (y_proba > 0.5).astype(int).flatten()
    t_pr = time.time() - t0
    p = PHASE2_BINARY_MODELS_DIR / "ann.keras"
    ann.save(p)
    model_paths["ANN"] = str(p)
    results.append(eval_and_save_binary("ANN", y_test, y_pred, t_tr, t_pr))

    # 3. kNN
    logger.info("--> [3/7] Binary 19-Feature kNN...")
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    t0 = time.time()
    knn.fit(X19_train, y_train)
    t_tr = time.time() - t0
    t0 = time.time()
    y_pred = knn.predict(X19_test)
    t_pr = time.time() - t0
    p = PHASE2_BINARY_MODELS_DIR / "knn.joblib"
    joblib.dump(knn, p)
    model_paths["kNN"] = str(p)
    results.append(eval_and_save_binary("kNN", y_test, y_pred, t_tr, t_pr))

    # 4. Logistic Regression
    logger.info("--> [4/7] Binary 19-Feature Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    t0 = time.time()
    lr.fit(X19_train, y_train)
    t_tr = time.time() - t0
    t0 = time.time()
    y_pred = lr.predict(X19_test)
    t_pr = time.time() - t0
    p = PHASE2_BINARY_MODELS_DIR / "logistic_regression.joblib"
    joblib.dump(lr, p)
    model_paths["Logistic Regression"] = str(p)
    results.append(eval_and_save_binary("Logistic Regression", y_test, y_pred, t_tr, t_pr))

    # 5. SVM (LinearSVC)
    logger.info("--> [5/7] Binary 19-Feature Support Vector Machine (LinearSVC)...")
    svm = LinearSVC(dual=False, max_iter=1000, random_state=RANDOM_SEED)
    t0 = time.time()
    svm.fit(X19_train, y_train)
    t_tr = time.time() - t0
    t0 = time.time()
    y_pred = svm.predict(X19_test)
    t_pr = time.time() - t0
    p = PHASE2_BINARY_MODELS_DIR / "svm.joblib"
    joblib.dump(svm, p)
    model_paths["SVM"] = str(p)
    results.append(eval_and_save_binary("SVM (LinearSVC)", y_test, y_pred, t_tr, t_pr))

    # 6. XGBoost + Decision Tree (Sequential pipeline: XGBoost Top-19 -> Decision Tree)
    logger.info("--> [6/7] Binary XGBoost + Decision Tree...")
    xgb_dt = DecisionTreeClassifier(max_depth=20, random_state=RANDOM_SEED)
    t0 = time.time()
    xgb_dt.fit(X19_train, y_train)
    t_tr = time.time() - t0
    t0 = time.time()
    y_pred = xgb_dt.predict(X19_test)
    t_pr = time.time() - t0
    p = PHASE2_BINARY_MODELS_DIR / "xgboost_dt.joblib"
    joblib.dump(xgb_dt, p)
    model_paths["XGBoost + Decision Tree"] = str(p)
    results.append(eval_and_save_binary("XGBoost + Decision Tree", y_test, y_pred, t_tr, t_pr))

    # 7. XGBoost + kNN (Sequential pipeline: XGBoost Top-19 -> kNN)
    logger.info("--> [7/7] Binary XGBoost + kNN...")
    xgb_knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    t0 = time.time()
    xgb_knn.fit(X19_train, y_train)
    t_tr = time.time() - t0
    t0 = time.time()
    y_pred = xgb_knn.predict(X19_test)
    t_pr = time.time() - t0
    p = PHASE2_BINARY_MODELS_DIR / "xgboost_knn.joblib"
    joblib.dump(xgb_knn, p)
    model_paths["XGBoost + kNN"] = str(p)
    results.append(eval_and_save_binary("XGBoost + kNN", y_test, y_pred, t_tr, t_pr))

    return results, model_paths

def train_phase2_multiclass_models(
    X19_train: np.ndarray,
    X19_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    target_names: List[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Trains 7 multiclass model approaches on 19 features."""
    logger.info("==================================================")
    logger.info(" STARTING PHASE 2 MULTICLASS CLASSIFICATION (19 FEAT) ")
    logger.info("==================================================")

    results = []
    model_paths = {}
    num_classes = len(target_names)

    # 1. Decision Tree
    logger.info("--> [1/7] Multiclass 19-Feature Decision Tree...")
    dt = DecisionTreeClassifier(max_depth=20, random_state=RANDOM_SEED)
    t0 = time.time()
    dt.fit(X19_train, y_train)
    t_tr = time.time() - t0
    t0 = time.time()
    y_pred = dt.predict(X19_test)
    t_pr = time.time() - t0
    p = PHASE2_MULTI_MODELS_DIR / "decision_tree.joblib"
    joblib.dump(dt, p)
    model_paths["Decision Tree"] = str(p)
    results.append(eval_and_save_multiclass("Decision Tree", y_test, y_pred, target_names, t_tr, t_pr))

    # 2. ANN
    logger.info("--> [2/7] Multiclass 19-Feature ANN (Keras)...")
    ann = build_ann_model(input_dim=X19_train.shape[1], num_classes=num_classes)
    early_stop = get_early_stopping(patience=5)
    t0 = time.time()
    ann.fit(X19_train, y_train, epochs=30, batch_size=256, validation_split=0.1, callbacks=[early_stop], verbose=0)
    t_tr = time.time() - t0
    t0 = time.time()
    y_probs = ann.predict(X19_test, batch_size=512, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)
    t_pr = time.time() - t0
    p = PHASE2_MULTI_MODELS_DIR / "ann.keras"
    ann.save(p)
    model_paths["ANN"] = str(p)
    results.append(eval_and_save_multiclass("ANN", y_test, y_pred, target_names, t_tr, t_pr))

    # 3. kNN
    logger.info("--> [3/7] Multiclass 19-Feature kNN...")
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    t0 = time.time()
    knn.fit(X19_train, y_train)
    t_tr = time.time() - t0
    t0 = time.time()
    y_pred = knn.predict(X19_test)
    t_pr = time.time() - t0
    p = PHASE2_MULTI_MODELS_DIR / "knn.joblib"
    joblib.dump(knn, p)
    model_paths["kNN"] = str(p)
    results.append(eval_and_save_multiclass("kNN", y_test, y_pred, target_names, t_tr, t_pr))

    # 4. Logistic Regression
    logger.info("--> [4/7] Multiclass 19-Feature Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    t0 = time.time()
    lr.fit(X19_train, y_train)
    t_tr = time.time() - t0
    t0 = time.time()
    y_pred = lr.predict(X19_test)
    t_pr = time.time() - t0
    p = PHASE2_MULTI_MODELS_DIR / "logistic_regression.joblib"
    joblib.dump(lr, p)
    model_paths["Logistic Regression"] = str(p)
    results.append(eval_and_save_multiclass("Logistic Regression", y_test, y_pred, target_names, t_tr, t_pr))

    # 5. SVM (LinearSVC)
    logger.info("--> [5/7] Multiclass 19-Feature Support Vector Machine (LinearSVC)...")
    svm = LinearSVC(dual=False, max_iter=1000, random_state=RANDOM_SEED)
    t0 = time.time()
    svm.fit(X19_train, y_train)
    t_tr = time.time() - t0
    t0 = time.time()
    y_pred = svm.predict(X19_test)
    t_pr = time.time() - t0
    p = PHASE2_MULTI_MODELS_DIR / "svm.joblib"
    joblib.dump(svm, p)
    model_paths["SVM"] = str(p)
    results.append(eval_and_save_multiclass("SVM (LinearSVC)", y_test, y_pred, target_names, t_tr, t_pr))

    # 6. XGBoost + Decision Tree
    logger.info("--> [6/7] Multiclass XGBoost + Decision Tree...")
    xgb_dt = DecisionTreeClassifier(max_depth=20, random_state=RANDOM_SEED)
    t0 = time.time()
    xgb_dt.fit(X19_train, y_train)
    t_tr = time.time() - t0
    t0 = time.time()
    y_pred = xgb_dt.predict(X19_test)
    t_pr = time.time() - t0
    p = PHASE2_MULTI_MODELS_DIR / "xgboost_dt.joblib"
    joblib.dump(xgb_dt, p)
    model_paths["XGBoost + Decision Tree"] = str(p)
    results.append(eval_and_save_multiclass("XGBoost + Decision Tree", y_test, y_pred, target_names, t_tr, t_pr))

    # 7. XGBoost + kNN
    logger.info("--> [7/7] Multiclass XGBoost + kNN...")
    xgb_knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    t0 = time.time()
    xgb_knn.fit(X19_train, y_train)
    t_tr = time.time() - t0
    t0 = time.time()
    y_pred = xgb_knn.predict(X19_test)
    t_pr = time.time() - t0
    p = PHASE2_MULTI_MODELS_DIR / "xgboost_knn.joblib"
    joblib.dump(xgb_knn, p)
    model_paths["XGBoost + kNN"] = str(p)
    results.append(eval_and_save_multiclass("XGBoost + kNN", y_test, y_pred, target_names, t_tr, t_pr))

    return results, model_paths
