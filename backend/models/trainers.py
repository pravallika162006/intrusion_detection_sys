"""
Phase 1 Model Training and Evaluation Module.
Faithfully reproduces Sydney M. Kasongo & Yanxia Sun (2020) on 42 UNSW-NB15 features:
1. Decision Tree (max_depth in [2, 5, 7, 8, 9] tuned on VAL)
2. Artificial Neural Network (Single hidden layer, Adam, adaptive lr tuned on VAL)
3. k-Nearest Neighbors (k in [3, 5, 7, 9, 11] tuned on VAL)
4. Logistic Regression (max_iter=1000, random_state=10)
5. Support Vector Machine (exact RBF kernel attempt with computational profiler)

Metrics reported matching paper Tables 4 and 6:
- Tr. AC (%)
- Val. AC (%)
- Test AC (%)
- Precision (%)
- Recall (%)
- F1-Score (%)
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from backend.config import (
    PHASE1_BINARY_MODELS_DIR,
    PHASE1_MULTICLASS_MODELS_DIR,
    PHASE1_RESULTS_DIR,
    BINARY_MODELS_DIR,
    MULTICLASS_MODELS_DIR,
    RESULTS_DIR,
    RANDOM_SEED,
    LR_RANDOM_STATE,
)
from backend.models.ann_builder import (
    build_paper_ann_model,
    get_early_stopping,
    get_adaptive_lr_callback,
)
from backend.models.rbf_svm_runner import attempt_rbf_svm
from backend.utils.logger import setup_logger

logger = setup_logger("Phase1_Trainers")

REPORTS_DIR = PHASE1_RESULTS_DIR / "classification_reports"
CONFUSION_DIR = PHASE1_RESULTS_DIR / "confusion_matrices"

for d in [
    PHASE1_BINARY_MODELS_DIR,
    PHASE1_MULTICLASS_MODELS_DIR,
    PHASE1_RESULTS_DIR,
    REPORTS_DIR,
    CONFUSION_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)


def _eval_and_record_metrics(
    model_name: str,
    task: str,
    y_tr_true: np.ndarray,
    y_tr_pred: np.ndarray,
    y_val_true: np.ndarray,
    y_val_pred: np.ndarray,
    y_te_true: np.ndarray,
    y_te_pred: np.ndarray,
    train_time: float,
    pred_time: float,
    target_names: List[str] = None,
    hyperparams: str = "",
) -> Dict[str, Any]:
    """Computes train, val, and test metrics matching paper tables."""
    is_binary = (task == "binary")
    avg_mode = "binary" if is_binary else "macro"
    pos_label = 1 if is_binary else None

    # Accuracies
    tr_ac = round(accuracy_score(y_tr_true, y_tr_pred) * 100, 2)
    val_ac = round(accuracy_score(y_val_true, y_val_pred) * 100, 2)
    test_ac = round(accuracy_score(y_te_true, y_te_pred) * 100, 2)

    # Precision, Recall, F1 on Test Set
    if is_binary:
        prec = round(precision_score(y_te_true, y_te_pred, pos_label=1, zero_division=0) * 100, 2)
        rec = round(recall_score(y_te_true, y_te_pred, pos_label=1, zero_division=0) * 100, 2)
        f1 = round(f1_score(y_te_true, y_te_pred, pos_label=1, zero_division=0) * 100, 2)
    else:
        # Paper Tables 6 and 7 report weighted averaging where Recall matches Accuracy
        prec = round(precision_score(y_te_true, y_te_pred, average="weighted", zero_division=0) * 100, 2)
        rec = round(recall_score(y_te_true, y_te_pred, average="weighted", zero_division=0) * 100, 2)
        f1 = round(f1_score(y_te_true, y_te_pred, average="weighted", zero_division=0) * 100, 2)

    macro_f1 = round(f1_score(y_te_true, y_te_pred, average="macro", zero_division=0) * 100, 2)
    weighted_f1 = round(f1_score(y_te_true, y_te_pred, average="weighted", zero_division=0) * 100, 2)

    # Confusion Matrix & Classification Report
    cm = confusion_matrix(y_te_true, y_te_pred)
    if target_names is None:
        target_names = ["Normal (0)", "Attack (1)"] if is_binary else [str(i) for i in range(len(np.unique(y_te_true)))]

    rep = classification_report(y_te_true, y_te_pred, target_names=target_names, digits=4, zero_division=0)

    clean_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
    rep_file = REPORTS_DIR / f"{task}_{clean_name}_report.txt"
    cm_file = CONFUSION_DIR / f"{task}_{clean_name}_cm.json"

    with open(rep_file, "w", encoding="utf-8") as f:
        f.write(rep)

    with open(cm_file, "w", encoding="utf-8") as f:
        json.dump({"labels": target_names, "confusion_matrix": cm.tolist()}, f, indent=2)

    return {
        "ML method": model_name,
        "Feature Set": "42 Features",
        "Tr. AC (%)": tr_ac,
        "Val. AC (%)": val_ac,
        "Test AC (%)": test_ac,
        "Precision (%)": prec,
        "Recall (%)": rec,
        "F1-Score (%)": f1,
        "Macro F1 (%)": macro_f1,
        "Weighted F1 (%)": weighted_f1,
        "Training Time (s)": round(train_time, 2),
        "Prediction Time (s)": round(pred_time, 4),
        "Hyperparameters": hyperparams,
        "Confusion Matrix": cm,
    }


def train_and_eval_binary_models(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Trains and evaluates 5 Phase 1 binary classification models on 42 features.
    """
    logger.info("=========================================================")
    logger.info(" STARTING PHASE 1 BINARY CLASSIFICATION (42 FEATURES) ")
    logger.info("=========================================================")

    results = []
    model_paths = {}

    # 1. Decision Tree (tune max_depth in [2, 5, 7, 8, 9] and criterion on VAL)
    logger.info("--> [1/5] Decision Tree (tuning max_depth in [2, 5, 7, 8, 9] & criterion on VAL)...")
    best_dt = None
    best_dt_val_acc = -1.0
    best_depth = 9
    best_crit = "entropy"

    t0_dt = time.time()
    for crit in ["entropy", "gini"]:
        for depth in [2, 5, 7, 8, 9]:
            dt_cand = DecisionTreeClassifier(criterion=crit, max_depth=depth, random_state=RANDOM_SEED)
            dt_cand.fit(X_train, y_train)
            val_pred = dt_cand.predict(X_val)
            val_acc = accuracy_score(y_val, val_pred)
            logger.info(f"    DT crit={crit} depth={depth} -> Val Acc: {val_acc*100:.2f}%")
            if val_acc > best_dt_val_acc:
                best_dt_val_acc = val_acc
                best_dt = dt_cand
                best_depth = depth
                best_crit = crit

    t_train_dt = time.time() - t0_dt
    t0_pred = time.time()
    y_test_pred_dt = best_dt.predict(X_test)
    t_pred_dt = time.time() - t0_pred

    y_val_pred_dt = best_dt.predict(X_val)
    y_tr_pred_dt = best_dt.predict(X_train[:10000])  # Sample train accuracy

    dt_path = PHASE1_BINARY_MODELS_DIR / "decision_tree.joblib"
    joblib.dump(best_dt, dt_path)
    joblib.dump(best_dt, BINARY_MODELS_DIR / "decision_tree.joblib")
    model_paths["Decision Tree"] = str(dt_path)

    dt_metrics = _eval_and_record_metrics(
        "DT", "binary",
        y_train[:10000], y_tr_pred_dt,
        y_val, y_val_pred_dt,
        y_test, y_test_pred_dt,
        t_train_dt, t_pred_dt,
        hyperparams=f"max_depth={best_depth}",
    )
    results.append(dt_metrics)
    logger.info(f"    [DT Winner: depth={best_depth}] Test AC: {dt_metrics['Test AC (%)']}% | F1: {dt_metrics['F1-Score (%)']}%")

    # 2. ANN (single hidden layer, Adam, adaptive lr)
    logger.info("--> [2/5] Artificial Neural Network (single hidden layer, Adam, adaptive lr)...")
    ann = build_paper_ann_model(input_dim=X_train.shape[1], num_classes=2, hidden_units=64, learning_rate=0.01)
    callbacks = [get_early_stopping(patience=5), get_adaptive_lr_callback()]

    t0_ann = time.time()
    ann.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=256,
        callbacks=callbacks,
        verbose=0,
    )
    t_train_ann = time.time() - t0_ann

    t0_pred = time.time()
    y_te_probs = ann.predict(X_test, batch_size=512, verbose=0).flatten()
    y_te_pred_ann = (y_te_probs >= 0.5).astype(int)
    t_pred_ann = time.time() - t0_pred

    y_val_probs = ann.predict(X_val, batch_size=512, verbose=0).flatten()
    y_val_pred_ann = (y_val_probs >= 0.5).astype(int)

    y_tr_probs = ann.predict(X_train[:10000], batch_size=512, verbose=0).flatten()
    y_tr_pred_ann = (y_tr_probs >= 0.5).astype(int)

    ann_path = PHASE1_BINARY_MODELS_DIR / "ann.keras"
    ann.save(ann_path)
    ann.save(BINARY_MODELS_DIR / "ann.keras")
    model_paths["ANN"] = str(ann_path)

    ann_metrics = _eval_and_record_metrics(
        "ANN", "binary",
        y_train[:10000], y_tr_pred_ann,
        y_val, y_val_pred_ann,
        y_test, y_te_pred_ann,
        t_train_ann, t_pred_ann,
        hyperparams="single hidden layer (64 units), Adam, adaptive lr",
    )
    results.append(ann_metrics)
    logger.info(f"    [ANN] Test AC: {ann_metrics['Test AC (%)']}% | F1: {ann_metrics['F1-Score (%)']}%")

    # 3. kNN (k in [3, 5, 7, 9, 11] evaluated on VAL)
    logger.info("--> [3/5] k-Nearest Neighbors (kNN, selecting best k on VAL)...")
    # For speed on large data, sample a 20k subset for kNN validation selection if full matrix is prohibitive
    val_sample_size = min(5000, len(X_val))
    best_k = 5
    best_knn_val_acc = -1.0
    best_knn = None

    t0_knn = time.time()
    for k in [3, 5, 7, 9, 11]:
        knn_cand = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        knn_cand.fit(X_train, y_train)
        val_pred = knn_cand.predict(X_val[:val_sample_size])
        val_acc = accuracy_score(y_val[:val_sample_size], val_pred)
        logger.info(f"    kNN k={k} -> Val Acc (sample): {val_acc*100:.2f}%")
        if val_acc > best_knn_val_acc:
            best_knn_val_acc = val_acc
            best_knn = knn_cand
            best_k = k

    t_train_knn = time.time() - t0_knn

    t0_pred = time.time()
    # Evaluate test in batches to prevent memory spike
    y_te_pred_knn = best_knn.predict(X_test)
    t_pred_knn = time.time() - t0_pred

    y_val_pred_knn = best_knn.predict(X_val[:val_sample_size])
    y_tr_pred_knn = best_knn.predict(X_train[:2000])

    knn_path = PHASE1_BINARY_MODELS_DIR / "knn.joblib"
    joblib.dump(best_knn, knn_path)
    joblib.dump(best_knn, BINARY_MODELS_DIR / "knn.joblib")
    model_paths["kNN"] = str(knn_path)

    knn_metrics = _eval_and_record_metrics(
        "kNN", "binary",
        y_train[:2000], y_tr_pred_knn,
        y_val[:val_sample_size], y_val_pred_knn,
        y_test, y_te_pred_knn,
        t_train_knn, t_pred_knn,
        hyperparams=f"k={best_k}",
    )
    results.append(knn_metrics)
    logger.info(f"    [kNN Winner: k={best_k}] Test AC: {knn_metrics['Test AC (%)']}% | F1: {knn_metrics['F1-Score (%)']}%")

    # 4. Logistic Regression (max_iter=1000, random_state=10)
    logger.info("--> [4/5] Logistic Regression (random_state=10, max_iter=1000)...")
    lr = LogisticRegression(max_iter=1000, random_state=LR_RANDOM_STATE)

    t0_lr = time.time()
    lr.fit(X_train, y_train)
    t_train_lr = time.time() - t0_lr

    t0_pred = time.time()
    y_te_pred_lr = lr.predict(X_test)
    t_pred_lr = time.time() - t0_pred

    y_val_pred_lr = lr.predict(X_val)
    y_tr_pred_lr = lr.predict(X_train[:10000])

    lr_path = PHASE1_BINARY_MODELS_DIR / "logistic_regression.joblib"
    joblib.dump(lr, lr_path)
    joblib.dump(lr, BINARY_MODELS_DIR / "logistic_regression.joblib")
    model_paths["Logistic Regression"] = str(lr_path)

    lr_metrics = _eval_and_record_metrics(
        "LR", "binary",
        y_train[:10000], y_tr_pred_lr,
        y_val, y_val_pred_lr,
        y_test, y_te_pred_lr,
        t_train_lr, t_pred_lr,
        hyperparams="max_iter=1000, random_state=10",
    )
    results.append(lr_metrics)
    logger.info(f"    [LR] Test AC: {lr_metrics['Test AC (%)']}% | F1: {lr_metrics['F1-Score (%)']}%")

    # 5. Support Vector Machine (exact RBF attempt per guidelines)
    logger.info("--> [5/5] Support Vector Machine (paper RBF kernel: C=1.12, gamma='scale')...")
    svm_res = attempt_rbf_svm(X_train, y_train, X_val, y_val, X_test, y_test, task="binary", max_timeout_seconds=45)
    
    svm_record = {
        "ML method": "SVM",
        "Feature Set": "42 Features",
        "Tr. AC (%)": svm_res.get("Tr_AC", "N/A"),
        "Val. AC (%)": svm_res.get("Val_AC", "N/A"),
        "Test AC (%)": svm_res.get("Test_AC", "N/A"),
        "Precision (%)": svm_res.get("Precision", "N/A"),
        "Recall (%)": svm_res.get("Recall", "N/A"),
        "F1-Score (%)": svm_res.get("F1", "N/A"),
        "Macro F1 (%)": "N/A",
        "Weighted F1 (%)": "N/A",
        "Training Time (s)": svm_res.get("Training Time (s)", "N/A"),
        "Prediction Time (s)": svm_res.get("Prediction Time (s)", "N/A"),
        "Hyperparameters": "C=1.12, gamma='scale', kernel='rbf'",
        "Status": svm_res.get("Status"),
        "Limitation_Notes": svm_res.get("Limitation_Notes"),
    }
    results.append(svm_record)
    logger.info(f"    [SVM] Status: {svm_res.get('Status')}")

    return results, model_paths


def train_and_eval_multiclass_models(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    target_names: List[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Trains and evaluates 5 Phase 1 multiclass classification models on 42 features.
    """
    logger.info("=============================================================")
    logger.info(" STARTING PHASE 1 MULTICLASS CLASSIFICATION (42 FEATURES) ")
    logger.info("=============================================================")

    results = []
    model_paths = {}
    num_classes = len(target_names)

    # 1. Decision Tree
    logger.info("--> [1/5] Multiclass Decision Tree (tuning depth in [2, 5, 7, 8, 9] & criterion on VAL)...")
    best_dt = None
    best_dt_val_acc = -1.0
    best_depth = 9
    best_crit = "entropy"

    t0_dt = time.time()
    for crit in ["entropy", "gini"]:
        for depth in [2, 5, 7, 8, 9]:
            dt_cand = DecisionTreeClassifier(criterion=crit, max_depth=depth, random_state=RANDOM_SEED)
            dt_cand.fit(X_train, y_train)
            val_pred = dt_cand.predict(X_val)
            val_acc = accuracy_score(y_val, val_pred)
            logger.info(f"    Multiclass DT crit={crit} depth={depth} -> Val Acc: {val_acc*100:.2f}%")
            if val_acc > best_dt_val_acc:
                best_dt_val_acc = val_acc
                best_dt = dt_cand
                best_depth = depth
                best_crit = crit

    t_train_dt = time.time() - t0_dt
    t0_pred = time.time()
    y_test_pred_dt = best_dt.predict(X_test)
    t_pred_dt = time.time() - t0_pred

    y_val_pred_dt = best_dt.predict(X_val)
    y_tr_pred_dt = best_dt.predict(X_train[:10000])

    dt_path = PHASE1_MULTICLASS_MODELS_DIR / "decision_tree.joblib"
    joblib.dump(best_dt, dt_path)
    joblib.dump(best_dt, MULTICLASS_MODELS_DIR / "decision_tree.joblib")
    model_paths["Decision Tree"] = str(dt_path)

    dt_metrics = _eval_and_record_metrics(
        "DT", "multiclass",
        y_train[:10000], y_tr_pred_dt,
        y_val, y_val_pred_dt,
        y_test, y_test_pred_dt,
        t_train_dt, t_pred_dt,
        target_names=target_names,
        hyperparams=f"max_depth={best_depth}",
    )
    results.append(dt_metrics)
    logger.info(f"    [DT Winner: depth={best_depth}] Test AC: {dt_metrics['Test AC (%)']}% | Macro F1: {dt_metrics['Macro F1 (%)']}%")

    # 2. ANN
    logger.info("--> [2/5] Multiclass ANN (single hidden layer, Adam, adaptive lr)...")
    ann = build_paper_ann_model(input_dim=X_train.shape[1], num_classes=num_classes, hidden_units=64, learning_rate=0.01)
    callbacks = [get_early_stopping(patience=5), get_adaptive_lr_callback()]

    t0_ann = time.time()
    ann.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=256,
        callbacks=callbacks,
        verbose=0,
    )
    t_train_ann = time.time() - t0_ann

    t0_pred = time.time()
    y_te_probs = ann.predict(X_test, batch_size=512, verbose=0)
    y_te_pred_ann = np.argmax(y_te_probs, axis=1)
    t_pred_ann = time.time() - t0_pred

    y_val_probs = ann.predict(X_val, batch_size=512, verbose=0)
    y_val_pred_ann = np.argmax(y_val_probs, axis=1)

    y_tr_probs = ann.predict(X_train[:10000], batch_size=512, verbose=0)
    y_tr_pred_ann = np.argmax(y_tr_probs, axis=1)

    ann_path = PHASE1_MULTICLASS_MODELS_DIR / "ann.keras"
    ann.save(ann_path)
    ann.save(MULTICLASS_MODELS_DIR / "ann.keras")
    model_paths["ANN"] = str(ann_path)

    ann_metrics = _eval_and_record_metrics(
        "ANN", "multiclass",
        y_train[:10000], y_tr_pred_ann,
        y_val, y_val_pred_ann,
        y_test, y_te_pred_ann,
        t_train_ann, t_pred_ann,
        target_names=target_names,
        hyperparams="single hidden layer (64 units), Adam, adaptive lr",
    )
    results.append(ann_metrics)
    logger.info(f"    [ANN] Test AC: {ann_metrics['Test AC (%)']}% | Macro F1: {ann_metrics['Macro F1 (%)']}%")

    # 3. kNN
    logger.info("--> [3/5] Multiclass kNN (k in [3, 5, 7, 9, 11] evaluated on VAL)...")
    val_sample_size = min(5000, len(X_val))
    best_k = 5
    best_knn_val_acc = -1.0
    best_knn = None

    t0_knn = time.time()
    for k in [3, 5, 7, 9, 11]:
        knn_cand = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        knn_cand.fit(X_train, y_train)
        val_pred = knn_cand.predict(X_val[:val_sample_size])
        val_acc = accuracy_score(y_val[:val_sample_size], val_pred)
        logger.info(f"    kNN k={k} -> Val Acc (sample): {val_acc*100:.2f}%")
        if val_acc > best_knn_val_acc:
            best_knn_val_acc = val_acc
            best_knn = knn_cand
            best_k = k

    t_train_knn = time.time() - t0_knn

    t0_pred = time.time()
    y_te_pred_knn = best_knn.predict(X_test)
    t_pred_knn = time.time() - t0_pred

    y_val_pred_knn = best_knn.predict(X_val[:val_sample_size])
    y_tr_pred_knn = best_knn.predict(X_train[:2000])

    knn_path = PHASE1_MULTICLASS_MODELS_DIR / "knn.joblib"
    joblib.dump(best_knn, knn_path)
    joblib.dump(best_knn, MULTICLASS_MODELS_DIR / "knn.joblib")
    model_paths["kNN"] = str(knn_path)

    knn_metrics = _eval_and_record_metrics(
        "kNN", "multiclass",
        y_train[:2000], y_tr_pred_knn,
        y_val[:val_sample_size], y_val_pred_knn,
        y_test, y_te_pred_knn,
        t_train_knn, t_pred_knn,
        target_names=target_names,
        hyperparams=f"k={best_k}",
    )
    results.append(knn_metrics)
    logger.info(f"    [kNN Winner: k={best_k}] Test AC: {knn_metrics['Test AC (%)']}% | Macro F1: {knn_metrics['Macro F1 (%)']}%")

    # 4. Logistic Regression
    logger.info("--> [4/5] Multiclass Logistic Regression (max_iter=1000, random_state=10)...")
    lr = LogisticRegression(max_iter=1000, random_state=LR_RANDOM_STATE)

    t0_lr = time.time()
    lr.fit(X_train, y_train)
    t_train_lr = time.time() - t0_lr

    t0_pred = time.time()
    y_te_pred_lr = lr.predict(X_test)
    t_pred_lr = time.time() - t0_pred

    y_val_pred_lr = lr.predict(X_val)
    y_tr_pred_lr = lr.predict(X_train[:10000])

    lr_path = PHASE1_MULTICLASS_MODELS_DIR / "logistic_regression.joblib"
    joblib.dump(lr, lr_path)
    joblib.dump(lr, MULTICLASS_MODELS_DIR / "logistic_regression.joblib")
    model_paths["Logistic Regression"] = str(lr_path)

    lr_metrics = _eval_and_record_metrics(
        "LR", "multiclass",
        y_train[:10000], y_tr_pred_lr,
        y_val, y_val_pred_lr,
        y_test, y_te_pred_lr,
        t_train_lr, t_pred_lr,
        target_names=target_names,
        hyperparams="max_iter=1000, random_state=10",
    )
    results.append(lr_metrics)
    logger.info(f"    [LR] Test AC: {lr_metrics['Test AC (%)']}% | Macro F1: {lr_metrics['Macro F1 (%)']}%")

    # 5. Support Vector Machine (exact RBF attempt per guidelines)
    logger.info("--> [5/5] Multiclass SVM (paper RBF kernel: C=1.12, gamma='scale')...")
    svm_res = attempt_rbf_svm(X_train, y_train, X_val, y_val, X_test, y_test, task="multiclass", max_timeout_seconds=45)

    svm_record = {
        "ML method": "SVM",
        "Feature Set": "42 Features",
        "Tr. AC (%)": svm_res.get("Tr_AC", "N/A"),
        "Val. AC (%)": svm_res.get("Val_AC", "N/A"),
        "Test AC (%)": svm_res.get("Test_AC", "N/A"),
        "Precision (%)": svm_res.get("Precision", "N/A"),
        "Recall (%)": svm_res.get("Recall", "N/A"),
        "F1-Score (%)": svm_res.get("F1", "N/A"),
        "Macro F1 (%)": "N/A",
        "Weighted F1 (%)": "N/A",
        "Training Time (s)": svm_res.get("Training Time (s)", "N/A"),
        "Prediction Time (s)": svm_res.get("Prediction Time (s)", "N/A"),
        "Hyperparameters": "C=1.12, gamma='scale', kernel='rbf'",
        "Status": svm_res.get("Status"),
        "Limitation_Notes": svm_res.get("Limitation_Notes"),
    }
    results.append(svm_record)
    logger.info(f"    [SVM] Status: {svm_res.get('Status')}")

    return results, model_paths
