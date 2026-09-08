"""
Model training and evaluation wrapper for Phase 1.
Trains and evaluates Decision Tree, ANN, kNN, Logistic Regression, and SVM.
Saves model artifacts via joblib and Keras native serialization.
"""

import time
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from backend.config import BINARY_MODELS_DIR, MULTICLASS_MODELS_DIR, RANDOM_SEED
from backend.models.ann_builder import build_ann_model, get_early_stopping
from backend.evaluation.metrics import evaluate_binary_model, evaluate_multiclass_model
from backend.utils.logger import setup_logger

logger = setup_logger("ModelTrainer")

def train_and_eval_binary_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Trains and evaluates 5 baseline binary classification models:
    1. Decision Tree
    2. Artificial Neural Network (ANN)
    3. k-Nearest Neighbors (kNN)
    4. Logistic Regression
    5. Support Vector Machine (LinearSVC)

    Saves trained models to models/binary/
    Returns (results_list, model_paths)
    """
    logger.info("==================================================")
    logger.info(" STARTING BINARY CLASSIFICATION MODEL TRAINING ")
    logger.info("==================================================")

    results = []
    model_paths = {}

    # 1. Decision Tree
    logger.info("--> Training [1/5] Binary Decision Tree...")
    dt = DecisionTreeClassifier(max_depth=20, random_state=RANDOM_SEED)
    
    t0 = time.time()
    dt.fit(X_train, y_train)
    t_train = time.time() - t0
    
    t0 = time.time()
    y_pred = dt.predict(X_test)
    t_pred = time.time() - t0

    dt_path = BINARY_MODELS_DIR / "decision_tree.joblib"
    joblib.dump(dt, dt_path)
    model_paths["Decision Tree"] = str(dt_path)

    metrics = evaluate_binary_model("Decision Tree", y_test, y_pred, t_train, t_pred)
    results.append(metrics)
    logger.info(f"   [Decision Tree] Acc: {metrics['Accuracy']} | F1 (Attack): {metrics['F1 (Attack)']} | Time: {t_train:.2f}s")

    # 2. ANN (Keras)
    logger.info("--> Training [2/5] Binary ANN (TensorFlow/Keras)...")
    ann = build_ann_model(input_dim=X_train.shape[1], num_classes=2)
    early_stop = get_early_stopping(patience=5)

    t0 = time.time()
    ann.fit(
        X_train,
        y_train,
        epochs=30,
        batch_size=256,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=0,
    )
    t_train = time.time() - t0

    t0 = time.time()
    y_pred_proba = ann.predict(X_test, batch_size=512, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    t_pred = time.time() - t0

    ann_path = BINARY_MODELS_DIR / "ann.keras"
    ann.save(ann_path)
    model_paths["ANN"] = str(ann_path)

    metrics = evaluate_binary_model("ANN", y_test, y_pred, t_train, t_pred)
    results.append(metrics)
    logger.info(f"   [ANN] Acc: {metrics['Accuracy']} | F1 (Attack): {metrics['F1 (Attack)']} | Time: {t_train:.2f}s")

    # 3. kNN
    logger.info("--> Training [3/5] Binary k-Nearest Neighbors (kNN)...")
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)

    t0 = time.time()
    knn.fit(X_train, y_train)
    t_train = time.time() - t0

    t0 = time.time()
    y_pred = knn.predict(X_test)
    t_pred = time.time() - t0

    knn_path = BINARY_MODELS_DIR / "knn.joblib"
    joblib.dump(knn, knn_path)
    model_paths["kNN"] = str(knn_path)

    metrics = evaluate_binary_model("kNN", y_test, y_pred, t_train, t_pred)
    results.append(metrics)
    logger.info(f"   [kNN] Acc: {metrics['Accuracy']} | F1 (Attack): {metrics['F1 (Attack)']} | Time: {t_train:.2f}s")

    # 4. Logistic Regression
    logger.info("--> Training [4/5] Binary Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)

    t0 = time.time()
    lr.fit(X_train, y_train)
    t_train = time.time() - t0

    t0 = time.time()
    y_pred = lr.predict(X_test)
    t_pred = time.time() - t0

    lr_path = BINARY_MODELS_DIR / "logistic_regression.joblib"
    joblib.dump(lr, lr_path)
    model_paths["Logistic Regression"] = str(lr_path)

    metrics = evaluate_binary_model("Logistic Regression", y_test, y_pred, t_train, t_pred)
    results.append(metrics)
    logger.info(f"   [Logistic Regression] Acc: {metrics['Accuracy']} | F1 (Attack): {metrics['F1 (Attack)']} | Time: {t_train:.2f}s")

    # 5. SVM (LinearSVC - Primal Optimization dual=False)
    logger.info("--> Training [5/5] Binary Support Vector Machine (LinearSVC)...")
    logger.info("   Note: Using LinearSVC(dual=False, max_iter=1000) for fast primal optimization when N > d.")
    svm = LinearSVC(dual=False, max_iter=1000, random_state=RANDOM_SEED)

    t0 = time.time()
    svm.fit(X_train, y_train)
    t_train = time.time() - t0

    t0 = time.time()
    y_pred = svm.predict(X_test)
    t_pred = time.time() - t0

    svm_path = BINARY_MODELS_DIR / "svm.joblib"
    joblib.dump(svm, svm_path)
    model_paths["SVM"] = str(svm_path)

    metrics = evaluate_binary_model("SVM (LinearSVC)", y_test, y_pred, t_train, t_pred)
    results.append(metrics)
    logger.info(f"   [SVM (LinearSVC)] Acc: {metrics['Accuracy']} | F1 (Attack): {metrics['F1 (Attack)']} | Time: {t_train:.2f}s")

    return results, model_paths


def train_and_eval_multiclass_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    target_names: List[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Trains and evaluates 5 baseline multiclass classification models:
    1. Decision Tree
    2. Artificial Neural Network (ANN)
    3. k-Nearest Neighbors (kNN)
    4. Logistic Regression
    5. Support Vector Machine (LinearSVC)

    Saves trained models to models/multiclass/
    Returns (results_list, model_paths)
    """
    logger.info("==================================================")
    logger.info(" STARTING MULTICLASS CLASSIFICATION MODEL TRAINING ")
    logger.info("==================================================")

    results = []
    model_paths = {}
    num_classes = len(target_names)

    # 1. Decision Tree
    logger.info("--> Training [1/5] Multiclass Decision Tree...")
    dt = DecisionTreeClassifier(max_depth=20, random_state=RANDOM_SEED)

    t0 = time.time()
    dt.fit(X_train, y_train)
    t_train = time.time() - t0

    t0 = time.time()
    y_pred = dt.predict(X_test)
    t_pred = time.time() - t0

    dt_path = MULTICLASS_MODELS_DIR / "decision_tree.joblib"
    joblib.dump(dt, dt_path)
    model_paths["Decision Tree"] = str(dt_path)

    metrics = evaluate_multiclass_model("Decision Tree", y_test, y_pred, target_names, t_train, t_pred)
    results.append(metrics)
    logger.info(f"   [Decision Tree] Acc: {metrics['Accuracy']} | Macro F1: {metrics['Macro F1']} | Time: {t_train:.2f}s")

    # 2. ANN (Keras)
    logger.info("--> Training [2/5] Multiclass ANN (TensorFlow/Keras)...")
    ann = build_ann_model(input_dim=X_train.shape[1], num_classes=num_classes)
    early_stop = get_early_stopping(patience=5)

    t0 = time.time()
    ann.fit(
        X_train,
        y_train,
        epochs=30,
        batch_size=256,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=0,
    )
    t_train = time.time() - t0

    t0 = time.time()
    y_pred_probs = ann.predict(X_test, batch_size=512, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    t_pred = time.time() - t0

    ann_path = MULTICLASS_MODELS_DIR / "ann.keras"
    ann.save(ann_path)
    model_paths["ANN"] = str(ann_path)

    metrics = evaluate_multiclass_model("ANN", y_test, y_pred, target_names, t_train, t_pred)
    results.append(metrics)
    logger.info(f"   [ANN] Acc: {metrics['Accuracy']} | Macro F1: {metrics['Macro F1']} | Time: {t_train:.2f}s")

    # 3. kNN
    logger.info("--> Training [3/5] Multiclass k-Nearest Neighbors (kNN)...")
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)

    t0 = time.time()
    knn.fit(X_train, y_train)
    t_train = time.time() - t0

    t0 = time.time()
    y_pred = knn.predict(X_test)
    t_pred = time.time() - t0

    knn_path = MULTICLASS_MODELS_DIR / "knn.joblib"
    joblib.dump(knn, knn_path)
    model_paths["kNN"] = str(knn_path)

    metrics = evaluate_multiclass_model("kNN", y_test, y_pred, target_names, t_train, t_pred)
    results.append(metrics)
    logger.info(f"   [kNN] Acc: {metrics['Accuracy']} | Macro F1: {metrics['Macro F1']} | Time: {t_train:.2f}s")

    # 4. Logistic Regression
    logger.info("--> Training [4/5] Multiclass Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)

    t0 = time.time()
    lr.fit(X_train, y_train)
    t_train = time.time() - t0

    t0 = time.time()
    y_pred = lr.predict(X_test)
    t_pred = time.time() - t0

    lr_path = MULTICLASS_MODELS_DIR / "logistic_regression.joblib"
    joblib.dump(lr, lr_path)
    model_paths["Logistic Regression"] = str(lr_path)

    metrics = evaluate_multiclass_model("Logistic Regression", y_test, y_pred, target_names, t_train, t_pred)
    results.append(metrics)
    logger.info(f"   [Logistic Regression] Acc: {metrics['Accuracy']} | Macro F1: {metrics['Macro F1']} | Time: {t_train:.2f}s")

    # 5. SVM (LinearSVC - Primal Optimization dual=False)
    logger.info("--> Training [5/5] Multiclass Support Vector Machine (LinearSVC)...")
    logger.info("   Note: Using LinearSVC(dual=False, max_iter=1000) for fast primal optimization when N > d.")
    svm = LinearSVC(dual=False, max_iter=1000, random_state=RANDOM_SEED)

    t0 = time.time()
    svm.fit(X_train, y_train)
    t_train = time.time() - t0

    t0 = time.time()
    y_pred = svm.predict(X_test)
    t_pred = time.time() - t0

    svm_path = MULTICLASS_MODELS_DIR / "svm.joblib"
    joblib.dump(svm, svm_path)
    model_paths["SVM"] = str(svm_path)

    metrics = evaluate_multiclass_model("SVM (LinearSVC)", y_test, y_pred, target_names, t_train, t_pred)
    results.append(metrics)
    logger.info(f"   [SVM (LinearSVC)] Acc: {metrics['Accuracy']} | Macro F1: {metrics['Macro F1']} | Time: {t_train:.2f}s")

    return results, model_paths
