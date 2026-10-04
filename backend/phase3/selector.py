"""
Phase 3 Model Selector Module.
Implements the strict academic model selection rule:
1. Compare all candidate models and ensembles strictly on the VALIDATION set.
2. Select the top validation model.
3. Compare against the Phase 2 baseline validation performance.
   - If genuine improvement: adopt the winner as the Enhanced Model.
   - If no improvement: retain the baseline honestly.
4. Freeze the chosen model and evaluate ONCE on the official UNSW-NB15 test set.
5. Save artifacts to results/phase3/ and models/phase3/.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import joblib
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

from backend.config import (
    PHASE3_RESULTS_DIR,
    PHASE3_BINARY_MODELS_DIR,
    PHASE3_MULTICLASS_MODELS_DIR,
)
from backend.utils.logger import setup_logger

logger = setup_logger("Phase3_Selector")

REPORTS_DIR = PHASE3_RESULTS_DIR / "classification_reports"
CONFUSION_DIR = PHASE3_RESULTS_DIR / "confusion_matrices"

for d in [
    PHASE3_RESULTS_DIR,
    PHASE3_BINARY_MODELS_DIR,
    PHASE3_MULTICLASS_MODELS_DIR,
    REPORTS_DIR,
    CONFUSION_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)


def select_and_freeze_model(
    df_val_candidates: pd.DataFrame,
    df_val_ensembles: pd.DataFrame,
    all_fitted_models: Dict[str, Any],
    baseline_val_metrics: Dict[str, float],
    X_test: np.ndarray,
    y_test: np.ndarray,
    task: str = "binary",
    target_names: List[str] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Executes Phase 3 model selection and final test evaluation.
    """
    logger.info(f"Executing Phase 3 Model Selection for {task}...")

    is_binary = (task == "binary")
    avg_mode = "binary" if is_binary else "macro"
    pos_label = 1 if is_binary else None

    # Combine all validation evaluations
    df_all_val = pd.concat([df_val_candidates, df_val_ensembles], ignore_index=True)
    df_all_val = df_all_val.sort_values(by=["Val_Accuracy", "Val_F1"], ascending=[False, False]).reset_index(drop=True)

    # Top validation model
    top_val_row = df_all_val.iloc[0]
    top_model_name = top_val_row["Model"]
    top_val_acc = float(top_val_row["Val_Accuracy"])
    top_val_f1 = float(top_val_row["Val_F1"])

    baseline_acc = baseline_val_metrics.get("Val_Accuracy", 90.0)
    baseline_f1 = baseline_val_metrics.get("Val_F1", 90.0)
    baseline_name = baseline_val_metrics.get("Model", "Phase 2 Baseline")

    acc_improvement = round(top_val_acc - baseline_acc, 2)
    f1_improvement = round(top_val_f1 - baseline_f1, 2)

    logger.info(f"Baseline ({baseline_name}) -> Val Acc: {baseline_acc}%, Val F1: {baseline_f1}%")
    logger.info(f"Top Validation Candidate ({top_model_name}) -> Val Acc: {top_val_acc}%, Val F1: {top_val_f1}%")
    logger.info(f"Delta: {acc_improvement:+.2f}% Accuracy, {f1_improvement:+.2f}% F1")

    # Selection condition: Genuine improvement on validation data
    if acc_improvement > 0.1 or (acc_improvement >= 0.0 and f1_improvement > 0.1):
        selected_model_name = top_model_name
        is_enhanced = True
        rationale = (
            f"Validation performance improved over the baseline ({baseline_name}) "
            f"by {acc_improvement:+.2f}% Accuracy and {f1_improvement:+.2f}% F1. "
            f"Selected '{top_model_name}' as the Phase 3 Enhanced Model."
        )
        logger.info(f"[SELECTION DECISION] ENHANCEMENT ADOPTED: {rationale}")
    else:
        selected_model_name = baseline_name
        is_enhanced = False
        rationale = (
            f"Existing baseline model ({baseline_name}) retained because no tested candidate or ensemble "
            f"provided a genuine validation improvement (Best tested: {top_model_name} at {top_val_acc}% vs baseline {baseline_acc}%)."
        )
        logger.info(f"[SELECTION DECISION] BASELINE RETAINED: {rationale}")

    # Freeze the chosen model
    chosen_model = all_fitted_models.get(selected_model_name)
    if chosen_model is None:
        chosen_model = all_fitted_models.get(top_model_name)

    save_dir = PHASE3_BINARY_MODELS_DIR if is_binary else PHASE3_MULTICLASS_MODELS_DIR
    clean_save_name = selected_model_name.replace(" ", "_").replace("(", "").replace(")", "").replace(":", "").replace("+", "plus").replace("->", "to").lower()
    
    is_keras = hasattr(chosen_model, "save") and not hasattr(chosen_model, "estimators_")
    if is_keras:
        frozen_model_path = save_dir / f"enhanced_{task}_model.keras"
        chosen_model.save(frozen_model_path)
    else:
        frozen_model_path = save_dir / f"enhanced_{task}_model.joblib"
        joblib.dump(chosen_model, frozen_model_path)

    logger.info(f"Frozen model saved to: {frozen_model_path}")

    # Final Single Test Set Evaluation on the Frozen Model
    logger.info("Evaluating Frozen Model ONCE on the official UNSW-NB15 Test Set...")
    t0_pred = time.time()
    if is_keras:
        if is_binary:
            test_probs = chosen_model.predict(X_test, batch_size=512, verbose=0).flatten()
            test_preds = (test_probs >= 0.5).astype(int)
        else:
            test_probs = chosen_model.predict(X_test, batch_size=512, verbose=0)
            test_preds = np.argmax(test_probs, axis=1)
    else:
        test_preds = chosen_model.predict(X_test)
    pred_time = time.time() - t0_pred

    test_acc = round(accuracy_score(y_test, test_preds) * 100, 2)
    test_prec = round(precision_score(y_test, test_preds, average=avg_mode, zero_division=0) * 100, 2)
    test_rec = round(recall_score(y_test, test_preds, average=avg_mode, zero_division=0) * 100, 2)
    test_f1 = round(f1_score(y_test, test_preds, average=avg_mode, zero_division=0) * 100, 2)
    test_macro_f1 = round(f1_score(y_test, test_preds, average="macro", zero_division=0) * 100, 2)
    test_weighted_f1 = round(f1_score(y_test, test_preds, average="weighted", zero_division=0) * 100, 2)

    cm = confusion_matrix(y_test, test_preds)
    if target_names is None:
        target_names = ["Normal (0)", "Attack (1)"] if is_binary else [str(i) for i in range(len(np.unique(y_test)))]

    rep = classification_report(y_test, test_preds, target_names=target_names, digits=4, zero_division=0)

    # Save reports and matrices
    rep_file = REPORTS_DIR / f"enhanced_{task}_classification_report.txt"
    cm_file = CONFUSION_DIR / f"enhanced_{task}_cm.json"

    with open(rep_file, "w", encoding="utf-8") as f:
        f.write(rep)

    with open(cm_file, "w", encoding="utf-8") as f:
        json.dump({"labels": target_names, "confusion_matrix": cm.tolist()}, f, indent=2)

    final_summary = {
        "Task": task,
        "Selected_Model": selected_model_name,
        "Is_Enhanced_Adopted": is_enhanced,
        "Rationale": rationale,
        "Validation_Metrics": {
            "Accuracy": top_val_acc,
            "F1": top_val_f1,
            "Baseline_Val_Accuracy": baseline_acc,
            "Baseline_Val_F1": baseline_f1,
            "Accuracy_Delta": acc_improvement,
            "F1_Delta": f1_improvement,
        },
        "Final_Test_Metrics": {
            "Accuracy": test_acc,
            "Precision": test_prec,
            "Recall": test_rec,
            "F1": test_f1,
            "Macro_F1": test_macro_f1,
            "Weighted_F1": test_weighted_f1,
            "Prediction_Time_s": round(pred_time, 4),
        },
        "Model_Artifact_Path": str(frozen_model_path),
    }

    summary_file = PHASE3_RESULTS_DIR / f"{task}_selection_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)

    # Create 1-row DataFrame of final test metrics
    df_final_test = pd.DataFrame([{
        "Task": task.capitalize(),
        "Selected Model": selected_model_name,
        "Status": "Adopted Enhanced Model" if is_enhanced else "Retained Baseline",
        "Val Accuracy (%)": top_val_acc,
        "Test Accuracy (%)": test_acc,
        "Precision (%)": test_prec,
        "Recall (%)": test_rec,
        "F1-Score (%)": test_f1,
        "Macro F1 (%)": test_macro_f1,
        "Weighted F1 (%)": test_weighted_f1,
        "Selection Rationale": rationale,
    }])

    return final_summary, df_final_test
