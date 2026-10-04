"""
Phase 2 Comparator Module.
Compares Phase 1 (42 Features) against Phase 2 (19 Features) performance
for the baseline models (DT, ANN, kNN, LR, SVM) matching Sydney M. Kasongo & Yanxia Sun (2020).
"""

from pathlib import Path
from typing import Dict, Any
import pandas as pd

from backend.config import RESULTS_DIR, PHASE1_RESULTS_DIR, PHASE2_RESULTS_DIR
from backend.utils.logger import setup_logger

logger = setup_logger("Comparator")


def generate_42_vs_19_comparison() -> pd.DataFrame:
    """
    Loads Phase 1 and Phase 2 result CSV files, aligns the models,
    computes test accuracy and F1 deltas, and saves comparison_42_vs_19.csv.
    """
    logger.info("Generating 42 vs 19 feature set comparison table...")

    # Phase 1 paths
    phase1_bin_path = PHASE1_RESULTS_DIR / "binary_results.csv"
    if not phase1_bin_path.exists():
        phase1_bin_path = RESULTS_DIR / "binary_results.csv"

    phase1_multi_path = PHASE1_RESULTS_DIR / "multiclass_results.csv"
    if not phase1_multi_path.exists():
        phase1_multi_path = RESULTS_DIR / "multiclass_results.csv"

    # Phase 2 paths
    phase2_bin_path = PHASE2_RESULTS_DIR / "binary_results.csv"
    phase2_multi_path = PHASE2_RESULTS_DIR / "multiclass_results.csv"

    df_p1_bin = pd.read_csv(phase1_bin_path)
    df_p1_multi = pd.read_csv(phase1_multi_path)

    df_p2_bin = pd.read_csv(phase2_bin_path)
    df_p2_multi = pd.read_csv(phase2_multi_path)

    # Standard model acronyms
    models = ["DT", "ANN", "kNN", "LR", "SVM"]
    name_map = {
        "Decision Tree": "DT",
        "DT": "DT",
        "ANN": "ANN",
        "ANN (Paper-Faithful)": "ANN",
        "kNN": "kNN",
        "Logistic Regression": "LR",
        "LR": "LR",
        "SVM": "SVM",
        "SVM (LinearSVC)": "SVM",
        "SVM (RBF Kernel)": "SVM",
    }

    def _standardize_df(df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        model_col = "ML method" if "ML method" in df_copy.columns else "Model"
        df_copy["Model_Key"] = df_copy[model_col].map(lambda x: name_map.get(str(x), str(x)))
        return df_copy

    p1_bin = _standardize_df(df_p1_bin)
    p2_bin = _standardize_df(df_p2_bin)
    p1_multi = _standardize_df(df_p1_multi)
    p2_multi = _standardize_df(df_p2_multi)

    rows = []

    def _get_metric(row, candidates, default=0.0):
        for c in candidates:
            if c in row:
                val = row[c].values[0]
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return None
        return default

    # Binary comparison
    for m in models:
        r1 = p1_bin[p1_bin["Model_Key"] == m]
        r2 = p2_bin[p2_bin["Model_Key"] == m]

        if not r1.empty and not r2.empty:
            acc1 = _get_metric(r1, ["Test AC (%)", "Accuracy"])
            acc2 = _get_metric(r2, ["Test AC (%)", "Accuracy"])
            f1_1 = _get_metric(r1, ["F1-Score (%)", "F1 (Attack)", "F1"])
            f1_2 = _get_metric(r2, ["F1-Score (%)", "F1 (Attack)", "F1"])

            if acc1 is not None and acc2 is not None:
                acc_diff = round(acc2 - acc1, 2)
            else:
                acc_diff = "N/A"

            if f1_1 is not None and f1_2 is not None:
                f1_diff = round(f1_2 - f1_1, 2)
            else:
                f1_diff = "N/A"

            rows.append({
                "Task": "Binary",
                "Model": m,
                "42_Feat_Test_AC": acc1 if acc1 is not None else "N/A",
                "19_Feat_Test_AC": acc2 if acc2 is not None else "N/A",
                "Accuracy_Delta": acc_diff,
                "42_Feat_F1": f1_1 if f1_1 is not None else "N/A",
                "19_Feat_F1": f1_2 if f1_2 is not None else "N/A",
                "F1_Delta": f1_diff,
            })

    # Multiclass comparison
    for m in models:
        r1 = p1_multi[p1_multi["Model_Key"] == m]
        r2 = p2_multi[p2_multi["Model_Key"] == m]

        if not r1.empty and not r2.empty:
            acc1 = _get_metric(r1, ["Test AC (%)", "Accuracy"])
            acc2 = _get_metric(r2, ["Test AC (%)", "Accuracy"])
            f1_1 = _get_metric(r1, ["F1-Score (%)", "Macro F1 (%)", "Macro F1"])
            f1_2 = _get_metric(r2, ["F1-Score (%)", "Macro F1 (%)", "Macro F1"])

            if acc1 is not None and acc2 is not None:
                acc_diff = round(acc2 - acc1, 2)
            else:
                acc_diff = "N/A"

            if f1_1 is not None and f1_2 is not None:
                f1_diff = round(f1_2 - f1_1, 2)
            else:
                f1_diff = "N/A"

            rows.append({
                "Task": "Multiclass",
                "Model": m,
                "42_Feat_Test_AC": acc1 if acc1 is not None else "N/A",
                "19_Feat_Test_AC": acc2 if acc2 is not None else "N/A",
                "Accuracy_Delta": acc_diff,
                "42_Feat_F1": f1_1 if f1_1 is not None else "N/A",
                "19_Feat_F1": f1_2 if f1_2 is not None else "N/A",
                "F1_Delta": f1_diff,
            })

    df_comp = pd.DataFrame(rows)
    comp_path = PHASE2_RESULTS_DIR / "comparison_42_vs_19.csv"
    df_comp.to_csv(comp_path, index=False)
    logger.info(f"Saved 42 vs 19 comparison table to: {comp_path}")

    return df_comp


def generate_paper_reproduction_comparison_table() -> pd.DataFrame:
    """
    Generates the official Paper Comparison Table matching the user's required schema:
    Model | Features | Task | Paper Accuracy | Our Accuracy | Difference | Status

    Matching Criteria:
    - Difference <= 0.50: "Very close"
    - Difference <= 1.50: "Close"
    - Difference <= 3.00: "Moderate difference"
    - Difference > 3.00:  "Needs investigation"
    """
    from backend.paper_config import PAPER_TARGETS, categorize_difference, PAPER_INCONSISTENCY_NOTE

    logger.info("Generating official Paper vs Our Reproduction comparison table...")

    # Load results
    p1_bin = pd.read_csv(PHASE1_RESULTS_DIR / "binary_results.csv")
    p2_bin = pd.read_csv(PHASE2_RESULTS_DIR / "binary_results.csv")
    p1_multi = pd.read_csv(PHASE1_RESULTS_DIR / "multiclass_results.csv")
    p2_multi = pd.read_csv(PHASE2_RESULTS_DIR / "multiclass_results.csv")

    name_map = {
        "Decision Tree": "DT", "DT": "DT",
        "ANN": "ANN", "kNN": "kNN",
        "Logistic Regression": "LR", "LR": "LR",
        "SVM": "SVM", "SVM (RBF Kernel)": "SVM",
    }

    def _get_acc(df, model_key):
        df_copy = df.copy()
        col = "ML method" if "ML method" in df_copy.columns else "Model"
        df_copy["Key"] = df_copy[col].map(lambda x: name_map.get(str(x), str(x)))
        row = df_copy[df_copy["Key"] == model_key]
        if not row.empty:
            val = row["Test AC (%)"].values[0] if "Test AC (%)" in row.columns else row["Accuracy"].values[0]
            try:
                return float(val)
            except (ValueError, TypeError):
                return None
        return None

    rows = []
    models = ["DT", "ANN", "kNN", "LR", "SVM"]

    groups = [
        ("42", "Binary", p1_bin, PAPER_TARGETS["phase1_binary"]),
        ("19", "Binary", p2_bin, PAPER_TARGETS["phase2_binary"]),
        ("42", "Multiclass", p1_multi, PAPER_TARGETS["phase1_multiclass"]),
        ("19", "Multiclass", p2_multi, PAPER_TARGETS["phase2_multiclass"]),
    ]

    for feats, task, df_actual, target_dict in groups:
        for m in models:
            paper_val = target_dict[m]["Test_AC"]
            our_val = _get_acc(df_actual, m)
            if our_val is not None:
                diff = round(our_val - paper_val, 2)
                status = categorize_difference(diff)
            else:
                diff = None
                status = "Not completed / impractical under current hardware" if m == "SVM" else "Missing"

            rows.append({
                "Model": m,
                "Features": feats,
                "Task": task,
                "Paper Accuracy": paper_val,
                "Our Accuracy": our_val if our_val is not None else "Impractical*",
                "Difference": diff if diff is not None else "N/A",
                "Status": status,
            })

    df_comp = pd.DataFrame(rows)
    out_path = PHASE2_RESULTS_DIR / "paper_vs_our_comparison.csv"
    df_comp.to_csv(out_path, index=False)
    df_comp.to_csv(RESULTS_DIR / "paper_vs_our_comparison.csv", index=False)
    logger.info(f"Saved Paper vs Our comparison table to: {out_path}")

    return df_comp

