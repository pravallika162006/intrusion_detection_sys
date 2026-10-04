"""
XGBoost Feature Selection Module for Phase 2.
Faithfully reproduces Sydney M. Kasongo & Yanxia Sun (2020) Table 3:
Uses the EXACT 19 features selected by XGBoost in the reference paper:
1. sttl (0.803374)
2. ct_srv_dst (0.039387)
3. sbytes (0.037377)
4. smean (0.019878)
5. proto (0.018848)
6. ct_state_ttl (0.016783)
7. sloss (0.012008)
8. synack (0.010125)
9. ct_dst_src_ltm (0.007203)
10. dmean (0.007134)
11. ct_srv_src (0.006745)
12. service (0.006305)
13. ct_dst_sport_ltm (0.003717)
14. dbytes (0.002706)
15. dloss (0.001793)
16. state (0.001548)
17. tcprtt (0.001224)
18. ct_src_dport_ltm (0.000526)
19. rate (0.000503)

Also trains an XGBClassifier on the training partition to calculate and log
empirical feature importance for comparison.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import xgboost as xgb

from backend.config import (
    ALL_INPUT_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    SELECTED_19_FEATURES,
    PAPER_19_IMPORTANCE_SCORES,
    RANDOM_SEED,
    PHASE2_RESULTS_DIR,
)
from backend.utils.logger import setup_logger, Timer

logger = setup_logger("FeatureSelector")

PHASE2_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_paper_exact_19_features() -> List[str]:
    """Returns the exact 19 features in rank order from Table 3 of the paper."""
    return list(SELECTED_19_FEATURES)


def build_paper_ranking_dataframe() -> pd.DataFrame:
    """Builds DataFrame of Paper Table 3 feature importance ranking."""
    rows = []
    for rank, feat in enumerate(SELECTED_19_FEATURES, 1):
        score = PAPER_19_IMPORTANCE_SCORES.get(feat, 0.0)
        feat_type = "Categorical" if feat in CATEGORICAL_FEATURES else "Numerical"
        rows.append({
            "Rank": rank,
            "Feature": feat,
            "Paper_Importance_Score": score,
            "Type": feat_type,
            "Selected_Top19": True,
        })
    return pd.DataFrame(rows)


def run_xgboost_feature_selection(
    X_train_proc: np.ndarray,
    y_train: np.ndarray,
    prep_info: Dict[str, Any],
    top_k: int = 19,
) -> Tuple[List[str], pd.DataFrame, Dict[str, Any]]:
    """
    1. Loads the paper's exact 19 features and importance scores from Table 3.
    2. Runs an XGBoost classifier on the training partition (TRAIN-1) to measure empirical importances.
    3. Saves official paper ranking and empirical comparison to CSV and JSON artifacts.
    4. Returns (exact_19_features, df_ranking, summary).
    """
    logger.info("==================================================")
    logger.info(" PHASE 2: XGBOOST FEATURE IMPORTANCE SELECTION ")
    logger.info("==================================================")

    # 1. Exact Paper 19 features
    exact_19 = get_paper_exact_19_features()
    logger.info(f"Target Feature Set: Exact Paper Table 3 (19 features): {exact_19}")

    # 2. Train empirical XGBClassifier strictly on training partition
    logger.info("Training empirical XGBClassifier on X_train_proc (TRAIN-1 only)...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        eval_metric="logloss",
    )

    with Timer() as timer:
        xgb_model.fit(X_train_proc, y_train)

    logger.info(f"Empirical XGBoost trained in {timer.interval:.2f} seconds.")

    # 3. Map encoded column importances back to raw 42 features
    raw_importances = xgb_model.feature_importances_
    processed_feature_names = prep_info["processed_feature_names"]

    measured_scores: Dict[str, float] = {f: 0.0 for f in ALL_INPUT_FEATURES}

    for col_name, score in zip(processed_feature_names, raw_importances):
        matched = False
        for cat_feat in CATEGORICAL_FEATURES:
            if col_name.startswith(cat_feat + "_"):
                measured_scores[cat_feat] += float(score)
                matched = True
                break
        if not matched and col_name in NUMERICAL_FEATURES:
            measured_scores[col_name] += float(score)

    total_score = sum(measured_scores.values())
    if total_score > 0:
        for f in measured_scores:
            measured_scores[f] = measured_scores[f] / total_score

    # 4. Construct comprehensive comparison DataFrame
    ranking_data = []
    for rank, feat_name in enumerate(exact_19, 1):
        paper_score = PAPER_19_IMPORTANCE_SCORES.get(feat_name, 0.0)
        measured_score = round(measured_scores.get(feat_name, 0.0), 6)
        ranking_data.append({
            "Rank": rank,
            "Feature": feat_name,
            "Paper_Importance_Score": paper_score,
            "Our_Measured_Score": measured_score,
            "Type": "Categorical" if feat_name in CATEGORICAL_FEATURES else "Numerical",
            "In_Paper_19": True,
        })

    # Add remaining features from 42 not in top 19
    remaining = [f for f in ALL_INPUT_FEATURES if f not in exact_19]
    remaining_sorted = sorted(remaining, key=lambda f: measured_scores.get(f, 0.0), reverse=True)
    for idx, feat_name in enumerate(remaining_sorted, 20):
        ranking_data.append({
            "Rank": idx,
            "Feature": feat_name,
            "Paper_Importance_Score": 0.0,
            "Our_Measured_Score": round(measured_scores.get(feat_name, 0.0), 6),
            "Type": "Categorical" if feat_name in CATEGORICAL_FEATURES else "Numerical",
            "In_Paper_19": False,
        })

    df_ranking = pd.DataFrame(ranking_data)

    # 5. Save artifacts to results/phase2/
    importance_csv = PHASE2_RESULTS_DIR / "xgboost_feature_importance.csv"
    ranking_csv = PHASE2_RESULTS_DIR / "xgboost_feature_ranking.csv"
    selected_json = PHASE2_RESULTS_DIR / "selected_features_19.json"

    df_ranking.to_csv(importance_csv, index=False)
    df_ranking.to_csv(ranking_csv, index=False)

    summary = {
        "description": "Sydney M. Kasongo & Yanxia Sun (2020) Table 3 exact 19 selected features",
        "feature_count": len(exact_19),
        "selected_19_features": exact_19,
        "paper_importance_scores": PAPER_19_IMPORTANCE_SCORES,
        "top_feature": "sttl (0.803374)",
        "lowest_selected_feature": "rate (0.000503)",
        "categorical_count": sum(1 for f in exact_19 if f in CATEGORICAL_FEATURES),
        "numerical_count": sum(1 for f in exact_19 if f not in CATEGORICAL_FEATURES),
    }

    with open(selected_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved feature ranking to: {ranking_csv}")
    logger.info(f"Saved selected 19 features JSON to: {selected_json}")

    return exact_19, df_ranking, summary
