"""
Phase 2 Comparator Module.
Compares Phase 1 (42 Features) against Phase 2 (19 Features) performance for the 5 common baseline models.
Calculates Accuracy delta, F1 delta, Macro F1 delta, and Training Time ratio/delta.
"""

from pathlib import Path
from typing import Dict, Any
import pandas as pd

from backend.config import RESULTS_DIR
from backend.utils.logger import setup_logger

logger = setup_logger("Comparator")

PHASE2_RESULTS_DIR = RESULTS_DIR / "phase2"

def generate_42_vs_19_comparison() -> pd.DataFrame:
    """
    Loads Phase 1 and Phase 2 result CSV files, aligns the 5 common models,
    computes deltas, and saves comparison_42_vs_19.csv.
    """
    logger.info("Generating 42 vs 19 feature set comparison table...")

    phase1_bin_path = RESULTS_DIR / "binary_results.csv"
    phase1_multi_path = RESULTS_DIR / "multiclass_results.csv"

    phase2_bin_path = PHASE2_RESULTS_DIR / "binary_results.csv"
    phase2_multi_path = PHASE2_RESULTS_DIR / "multiclass_results.csv"

    df_p1_bin = pd.read_csv(phase1_bin_path)
    df_p1_multi = pd.read_csv(phase1_multi_path)

    df_p2_bin = pd.read_csv(phase2_bin_path)
    df_p2_multi = pd.read_csv(phase2_multi_path)

    common_models = ["Decision Tree", "ANN", "kNN", "Logistic Regression", "SVM (LinearSVC)"]

    rows = []

    # Process Binary comparisons
    for m in common_models:
        r1 = df_p1_bin[df_p1_bin["Model"] == m]
        r2 = df_p2_bin[df_p2_bin["Model"] == m]

        if not r1.empty and not r2.empty:
            p1_acc = float(r1["Accuracy"].values[0])
            p2_acc = float(r2["Accuracy"].values[0])
            acc_diff = round(p2_acc - p1_acc, 4)

            p1_f1 = float(r1["F1 (Attack)"].values[0])
            p2_f1 = float(r2["F1 (Attack)"].values[0])
            f1_diff = round(p2_f1 - p1_f1, 4)

            p1_time = float(r1["Training Time (s)"].values[0])
            p2_time = float(r2["Training Time (s)"].values[0])
            time_diff = round(p2_time - p1_time, 4)

            rows.append({
                "Task": "Binary",
                "Model": m,
                "42_Feat_Accuracy": p1_acc,
                "19_Feat_Accuracy": p2_acc,
                "Accuracy_Delta": acc_diff,
                "42_Feat_F1": p1_f1,
                "19_Feat_F1": p2_f1,
                "F1_Delta": f1_diff,
                "42_Feat_Train_Time_s": p1_time,
                "19_Feat_Train_Time_s": p2_time,
                "Train_Time_Delta_s": time_diff,
            })

    # Process Multiclass comparisons
    for m in common_models:
        r1 = df_p1_multi[df_p1_multi["Model"] == m]
        r2 = df_p2_multi[df_p2_multi["Model"] == m]

        if not r1.empty and not r2.empty:
            p1_acc = float(r1["Accuracy"].values[0])
            p2_acc = float(r2["Accuracy"].values[0])
            acc_diff = round(p2_acc - p1_acc, 4)

            p1_mf1 = float(r1["Macro F1"].values[0])
            p2_mf1 = float(r2["Macro F1"].values[0])
            mf1_diff = round(p2_mf1 - p1_mf1, 4)

            p1_time = float(r1["Training Time (s)"].values[0])
            p2_time = float(r2["Training Time (s)"].values[0])
            time_diff = round(p2_time - p1_time, 4)

            rows.append({
                "Task": "Multiclass",
                "Model": m,
                "42_Feat_Accuracy": p1_acc,
                "19_Feat_Accuracy": p2_acc,
                "Accuracy_Delta": acc_diff,
                "42_Feat_F1": p1_mf1,
                "19_Feat_F1": p2_mf1,
                "F1_Delta": mf1_diff,
                "42_Feat_Train_Time_s": p1_time,
                "19_Feat_Train_Time_s": p2_time,
                "Train_Time_Delta_s": time_diff,
            })

    df_comp = pd.DataFrame(rows)
    comp_path = PHASE2_RESULTS_DIR / "comparison_42_vs_19.csv"
    df_comp.to_csv(comp_path, index=False)

    logger.info(f"Saved 42 vs 19 comparison table to: {comp_path}")
    return df_comp
