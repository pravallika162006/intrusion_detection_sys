"""
FastAPI Routes for the Dedicated Academic PERFORMANCES Dashboard.
Dynamically serves evaluated results for:
- Phase 1: 42-feature ML experiments & paper comparison
- Phase 2: XGBoost Feature Selection, 19-feature ML experiments, and 42 vs 19 delta
- Phase 3: Project Enhancement (Candidates, Ensembles, Selection Decision, Final Test)
- Paper vs Implementation Reproducibility Benchmarks
- Confusion Matrices and Classification Reports
All metrics loaded directly from verified result files (no hardcoding).
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
from fastapi import APIRouter, HTTPException

from backend.config import (
    RESULTS_DIR,
    PHASE1_RESULTS_DIR,
    PHASE2_RESULTS_DIR,
    PHASE3_RESULTS_DIR,
    PAPER_19_IMPORTANCE_SCORES,
    SELECTED_19_FEATURES,
)
from backend.utils.logger import setup_logger

logger = setup_logger("PerformancesRoutes")
router = APIRouter(prefix="/api/performances", tags=["Performances Dashboard"])

# Reference Paper Results (Sydney M. Kasongo & Yanxia Sun, 2020)
PAPER_REFERENCE_RESULTS = {
    "phase1_binary": {
        "ANN": {"Tr_AC": 94.49, "Val_AC": 94.21, "Test_AC": 86.71, "Precision": 81.54, "Recall": 98.06, "F1": 89.04},
        "LR":  {"Tr_AC": 93.22, "Val_AC": 92.87, "Test_AC": 79.59, "Precision": 73.32, "Recall": 98.94, "F1": 84.22},
        "kNN": {"Tr_AC": 96.76, "Val_AC": 93.60, "Test_AC": 83.18, "Precision": 79.15, "Recall": 94.30, "F1": 86.06},
        "SVM": {"Tr_AC": 70.98, "Val_AC": 70.63, "Test_AC": 62.42, "Precision": 60.91, "Recall": 88.58, "F1": 71.18},
        "DT":  {"Tr_AC": 93.65, "Val_AC": 93.37, "Test_AC": 88.13, "Precision": 83.91, "Recall": 96.47, "F1": 90.00},
    },
    "phase1_multiclass": {
        "ANN": {"Tr_AC": 79.91, "Val_AC": 79.61, "Test_AC": 75.62, "Precision": 79.92, "Recall": 75.61, "F1": 76.58},
        "LR":  {"Tr_AC": 75.51, "Val_AC": 73.93, "Test_AC": 65.53, "Precision": 76.91, "Recall": 65.54, "F1": 66.62},
        "kNN": {"Tr_AC": 81.75, "Val_AC": 76.83, "Test_AC": 70.09, "Precision": 75.79, "Recall": 70.21, "F1": 72.03},
        "SVM": {"Tr_AC": 53.43, "Val_AC": 52.67, "Test_AC": 61.09, "Precision": 47.47, "Recall": 62.00, "F1": 53.77},
        "DT":  {"Tr_AC": 77.69, "Val_AC": 77.38, "Test_AC": 66.03, "Precision": 79.82, "Recall": 66.04, "F1": 51.12},
    },
    "phase2_binary": {
        "ANN": {"Tr_AC": 93.75, "Val_AC": 93.66, "Test_AC": 84.39, "Precision": 78.56, "Recall": 98.53, "F1": 87.42},
        "LR":  {"Tr_AC": 89.21, "Val_AC": 89.25, "Test_AC": 77.64, "Precision": 73.18, "Recall": 93.74, "F1": 82.20},
        "kNN": {"Tr_AC": 95.86, "Val_AC": 94.73, "Test_AC": 84.46, "Precision": 80.31, "Recall": 95.09, "F1": 87.08},
        "SVM": {"Tr_AC": 75.42, "Val_AC": 75.51, "Test_AC": 60.89, "Precision": 58.89, "Recall": 95.88, "F1": 72.97},
        "DT":  {"Tr_AC": 94.12, "Val_AC": 93.81, "Test_AC": 90.85, "Precision": 80.33, "Recall": 98.38, "F1": 88.45},
    },
    "phase2_multiclass": {
        "ANN": {"Tr_AC": 79.46, "Val_AC": 78.91, "Test_AC": 77.51, "Precision": 79.50, "Recall": 77.53, "F1": 77.28},
        "LR":  {"Tr_AC": 72.53, "Val_AC": 71.81, "Test_AC": 65.29, "Precision": 70.88, "Recall": 65.29, "F1": 65.96},
        "kNN": {"Tr_AC": 82.66, "Val_AC": 79.87, "Test_AC": 72.30, "Precision": 77.24, "Recall": 72.30, "F1": 73.81},
        "SVM": {"Tr_AC": 53.60, "Val_AC": 52.97, "Test_AC": 61.53, "Precision": 53.95, "Recall": 61.52, "F1": 51.31},
        "DT":  {"Tr_AC": 78.75, "Val_AC": 78.43, "Test_AC": 67.57, "Precision": 79.66, "Recall": 67.56, "F1": 69.26},
    }
}


def _safe_read_csv(filepath: Path) -> List[Dict[str, Any]]:
    """Safely loads CSV into list of dicts."""
    if not filepath.exists():
        return []
    try:
        df = pd.read_csv(filepath)
        # Clean NaN/Inf
        df = df.fillna("N/A")
        return df.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Error reading CSV {filepath}: {e}")
        return []


def _safe_read_json(filepath: Path) -> Dict[str, Any]:
    """Safely loads JSON file."""
    if not filepath.exists():
        return {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading JSON {filepath}: {e}")
        return {}


@router.get("/summary")
def get_performances_summary():
    """
    Returns full summary of all three phases, including 42-feature, 19-feature,
    and Phase 3 enhancement results.
    """
    # Phase 1
    p1_bin = _safe_read_csv(PHASE1_RESULTS_DIR / "binary_results.csv")
    if not p1_bin:
        p1_bin = _safe_read_csv(RESULTS_DIR / "binary_results.csv")

    p1_multi = _safe_read_csv(PHASE1_RESULTS_DIR / "multiclass_results.csv")
    if not p1_multi:
        p1_multi = _safe_read_csv(RESULTS_DIR / "multiclass_results.csv")

    # Phase 2
    p2_bin = _safe_read_csv(PHASE2_RESULTS_DIR / "binary_results.csv")
    p2_multi = _safe_read_csv(PHASE2_RESULTS_DIR / "multiclass_results.csv")
    comp_42_19 = _safe_read_csv(PHASE2_RESULTS_DIR / "comparison_42_vs_19.csv")

    # Phase 3
    p3_summary_bin = _safe_read_json(PHASE3_RESULTS_DIR / "binary_selection_summary.json")
    p3_summary_multi = _safe_read_json(PHASE3_RESULTS_DIR / "multiclass_selection_summary.json")
    p3_final_test = _safe_read_csv(PHASE3_RESULTS_DIR / "final_selected_model_test_results.csv")

    return {
        "status": "SUCCESS",
        "phase1": {
            "title": "Phase 1: Full 42-Feature Baseline Reproduction",
            "feature_count": 42,
            "binary_results": p1_bin,
            "multiclass_results": p1_multi,
            "paper_reference": {
                "binary": PAPER_REFERENCE_RESULTS["phase1_binary"],
                "multiclass": PAPER_REFERENCE_RESULTS["phase1_multiclass"],
            }
        },
        "phase2": {
            "title": "Phase 2: XGBoost Feature Selection & 19-Feature Reproduction",
            "feature_count": 19,
            "selected_features": SELECTED_19_FEATURES,
            "binary_results": p2_bin,
            "multiclass_results": p2_multi,
            "comparison_42_vs_19": comp_42_19,
            "paper_reference": {
                "binary": PAPER_REFERENCE_RESULTS["phase2_binary"],
                "multiclass": PAPER_REFERENCE_RESULTS["phase2_multiclass"],
            }
        },
        "phase3": {
            "title": "Phase 3: Project Enhancement (Candidate ML Models & Ensembles)",
            "binary_summary": p3_summary_bin,
            "multiclass_summary": p3_summary_multi,
            "final_test_results": p3_final_test,
        },
        "meta": {
            "dataset": "UNSW-NB15",
            "paper": "Performance Analysis of Intrusion Detection Systems Using a Feature Selection Method on the UNSW-NB15 Dataset (Kasongo & Sun, 2020)",
            "note": "Results may differ from the reference paper because of implementation details, preprocessing, hardware, randomness, and model tuning."
        }
    }


@router.get("/paper-comparison")
def get_paper_vs_our_comparison():
    """
    Returns side-by-side comparison between Reference Paper and Our Implementation.
    """
    p1_bin = {r.get("ML method", r.get("Model")): r for r in _safe_read_csv(PHASE1_RESULTS_DIR / "binary_results.csv")}
    p1_multi = {r.get("ML method", r.get("Model")): r for r in _safe_read_csv(PHASE1_RESULTS_DIR / "multiclass_results.csv")}
    p2_bin = {r.get("ML method", r.get("Model")): r for r in _safe_read_csv(PHASE2_RESULTS_DIR / "binary_results.csv")}
    p2_multi = {r.get("ML method", r.get("Model")): r for r in _safe_read_csv(PHASE2_RESULTS_DIR / "multiclass_results.csv")}

    def _build_comparison(section_key, our_dict):
        paper_dict = PAPER_REFERENCE_RESULTS[section_key]
        rows = []
        name_map = {"Decision Tree": "DT", "Logistic Regression": "LR"}

        for model_key, p_metrics in paper_dict.items():
            our_row = our_dict.get(model_key)
            if not our_row:
                for k, v in our_dict.items():
                    if name_map.get(k, k) == model_key:
                        our_row = v
                        break

            our_acc = None
            if our_row:
                val = our_row.get("Test AC (%)", our_row.get("Accuracy"))
                try:
                    our_acc = float(val) if val not in ["N/A", None] else None
                except (ValueError, TypeError):
                    our_acc = None

            paper_acc = p_metrics["Test_AC"]
            diff = round(our_acc - paper_acc, 2) if our_acc is not None else "N/A"

            rows.append({
                "Model": model_key,
                "Paper_Test_AC": paper_acc,
                "Our_Test_AC": our_acc if our_acc is not None else "Not completed / impractical",
                "Difference": diff,
                "Paper_F1": p_metrics["F1"],
                "Our_F1": our_row.get("F1-Score (%)", our_row.get("F1 (Attack)", "N/A")) if our_row else "N/A",
            })
        return rows

    return {
        "phase1_binary": _build_comparison("phase1_binary", p1_bin),
        "phase1_multiclass": _build_comparison("phase1_multiclass", p1_multi),
        "phase2_binary": _build_comparison("phase2_binary", p2_bin),
        "phase2_multiclass": _build_comparison("phase2_multiclass", p2_multi),
        "explanation": "Results may differ from the reference paper because of implementation details, preprocessing, hardware, randomness, and model tuning."
    }


@router.get("/feature-importance")
def get_feature_importance():
    """Returns the XGBoost feature ranking and scores (Paper Table 3)."""
    ranking_csv = PHASE2_RESULTS_DIR / "xgboost_feature_ranking.csv"
    if not ranking_csv.exists():
        ranking_csv = RESULTS_DIR / "phase2" / "xgboost_feature_ranking.csv"
    
    records = _safe_read_csv(ranking_csv)
    return {
        "status": "SUCCESS",
        "features": records,
        "selected_19": SELECTED_19_FEATURES,
        "paper_scores": PAPER_19_IMPORTANCE_SCORES,
    }


@router.get("/comparison-42-19")
def get_42_vs_19_comparison():
    """Returns the 42 vs 19 feature set delta comparison table."""
    comp_csv = PHASE2_RESULTS_DIR / "comparison_42_vs_19.csv"
    return {
        "status": "SUCCESS",
        "comparison": _safe_read_csv(comp_csv),
    }


@router.get("/phase3-selection")
def get_phase3_selection_data():
    """Returns Phase 3 validation candidates, ensemble tests, selection decision, and final test results."""
    bin_cand = _safe_read_csv(PHASE3_RESULTS_DIR / "binary_candidates_val.csv")
    bin_ens = _safe_read_csv(PHASE3_RESULTS_DIR / "binary_ensembles_val.csv")
    multi_cand = _safe_read_csv(PHASE3_RESULTS_DIR / "multiclass_candidates_val.csv")
    multi_ens = _safe_read_csv(PHASE3_RESULTS_DIR / "multiclass_ensembles_val.csv")

    summary_bin = _safe_read_json(PHASE3_RESULTS_DIR / "binary_selection_summary.json")
    summary_multi = _safe_read_json(PHASE3_RESULTS_DIR / "multiclass_selection_summary.json")
    final_test = _safe_read_csv(PHASE3_RESULTS_DIR / "final_selected_model_test_results.csv")

    return {
        "binary": {
            "candidates": bin_cand,
            "ensembles": bin_ens,
            "summary": summary_bin,
        },
        "multiclass": {
            "candidates": multi_cand,
            "ensembles": multi_ens,
            "summary": summary_multi,
        },
        "final_test_results": final_test,
    }


@router.get("/confusion-matrix/{phase}/{task}/{model_name}")
def get_confusion_matrix(phase: str, task: str, model_name: str):
    """Returns saved JSON confusion matrix for specified phase, task, and model."""
    clean_model = model_name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus").replace("-", "_").lower()

    if phase == "phase1":
        cm_dir = PHASE1_RESULTS_DIR / "confusion_matrices"
    elif phase == "phase2":
        cm_dir = PHASE2_RESULTS_DIR / "confusion_matrices"
    elif phase == "phase3":
        cm_dir = PHASE3_RESULTS_DIR / "confusion_matrices"
    else:
        raise HTTPException(status_code=400, detail="Invalid phase specified. Must be phase1, phase2, or phase3.")

    if phase == "phase3":
        cm_path = cm_dir / f"enhanced_{task}_cm.json"
        if cm_path.exists():
            with open(cm_path, "r", encoding="utf-8") as f:
                return json.load(f)

    # Alias mapping
    alias_map = {
        "dt": ["dt", "decision_tree"],
        "decision_tree": ["dt", "decision_tree"],
        "ann": ["ann"],
        "knn": ["knn", "k_nearest_neighbors"],
        "lr": ["lr", "logistic_regression"],
        "logistic_regression": ["lr", "logistic_regression"],
        "enhanced": ["enhanced"],
    }
    possible_keys = alias_map.get(clean_model, [clean_model])

    # Search in phase directory
    if cm_dir.exists():
        files = list(cm_dir.glob("*.json"))
        for f in files:
            f_lower = f.name.lower()
            if task.lower() in f_lower:
                for k in possible_keys:
                    if f"{task.lower()}_{k}_cm.json" == f_lower or f"{k}_cm.json" in f_lower:
                        with open(f, "r", encoding="utf-8") as in_f:
                            return json.load(in_f)

    # Search in root confusion matrices directory
    root_cm_dir = RESULTS_DIR / "confusion_matrices"
    if root_cm_dir.exists():
        files = list(root_cm_dir.glob("*.json"))
        for f in files:
            f_lower = f.name.lower()
            if task.lower() in f_lower:
                for k in possible_keys:
                    if f"{task.lower()}_{k}_cm.json" == f_lower:
                        with open(f, "r", encoding="utf-8") as in_f:
                            return json.load(in_f)

    raise HTTPException(status_code=404, detail=f"Confusion matrix for {phase} {task} {model_name} not found.")
