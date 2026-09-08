"""
XGBoost Feature Selection Module for Phase 2.
Fits XGBoost on training data only to rank all 42 original features and select the top 19 features.
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
    RANDOM_SEED,
    RESULTS_DIR,
)
from backend.utils.logger import setup_logger, Timer

logger = setup_logger("FeatureSelector")

# Planned/Paper 19 Features for Comparison
PLANNED_19_FEATURES = [
    "sttl", "ct_srv_dst", "sbytes", "smean", "proto", "ct_state_ttl", 
    "sloss", "synack", "ct_dst_src_ltm", "dmean", "ct_srv_src", "service", 
    "ct_dst_sport_ltm", "dbytes", "dloss", "state", "tcprtt", "ct_src_dport_ltm", "rate"
]

PHASE2_RESULTS_DIR = RESULTS_DIR / "phase2"
PHASE2_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def run_xgboost_feature_selection(
    X_train_proc: np.ndarray,
    y_train: np.ndarray,
    prep_info: Dict[str, Any],
    top_k: int = 19,
) -> Tuple[List[str], pd.DataFrame, Dict[str, Any]]:
    """
    Fits XGBoost Classifier strictly on training data (X_train_proc, y_train).
    Maps gain/importance scores back to the 42 original input features, ranks all 42,
    selects top 19 features, and compares with PLANNED_19_FEATURES.

    Returns:
        (selected_19_features, df_ranking, comparison_summary)
    """
    logger.info("==================================================")
    logger.info(" STARTING XGBOOST FEATURE IMPORTANCE SELECTION ")
    logger.info("==================================================")

    logger.info("Training XGBClassifier on X_train_proc (fitted strictly on training data)...")
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

    logger.info(f"XGBoost feature selection model trained in {timer.interval:.2f} seconds.")

    # Extract raw column importances (194 columns)
    raw_importances = xgb_model.feature_importances_
    processed_feature_names = prep_info["processed_feature_names"]

    # Map processed columns back to 42 original features
    feature_scores: Dict[str, float] = {f: 0.0 for f in ALL_INPUT_FEATURES}

    for col_name, score in zip(processed_feature_names, raw_importances):
        # Determine which original feature this encoded column belongs to
        matched = False
        for cat_feat in CATEGORICAL_FEATURES:
            if col_name.startswith(cat_feat + "_"):
                feature_scores[cat_feat] += float(score)
                matched = True
                break
        if not matched and col_name in NUMERICAL_FEATURES:
            feature_scores[col_name] += float(score)

    # Normalize scores so they sum to 1.0
    total_score = sum(feature_scores.values())
    if total_score > 0:
        for f in feature_scores:
            feature_scores[f] = feature_scores[f] / total_score

    # Rank all 42 features descending
    ranking = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)

    ranking_data = []
    for rank, (feat_name, score) in enumerate(ranking, 1):
        ranking_data.append({
            "Rank": rank,
            "Feature": feat_name,
            "Importance_Score": round(score, 6),
            "Type": "Categorical" if feat_name in CATEGORICAL_FEATURES else "Numerical",
            "In_Planned_19": feat_name in PLANNED_19_FEATURES,
        })

    df_ranking = pd.DataFrame(ranking_data)

    # Select top K features
    selected_19 = [r["Feature"] for r in ranking_data[:top_k]]

    # Compare XGBoost-selected top 19 against Planned 19
    selected_set = set(selected_19)
    planned_set = set(PLANNED_19_FEATURES)

    common_features = sorted(list(selected_set.intersection(planned_set)))
    selected_not_in_planned = sorted(list(selected_set - planned_set))
    planned_not_in_selected = sorted(list(planned_set - selected_set))

    comparison_summary = {
        "top_k_count": top_k,
        "selected_features_19": selected_19,
        "planned_features_19": PLANNED_19_FEATURES,
        "common_features_count": len(common_features),
        "common_features": common_features,
        "selected_not_in_planned": selected_not_in_planned,
        "planned_not_in_selected": planned_not_in_selected,
        "overlap_percentage": round(len(common_features) / top_k * 100, 2),
    }

    # Save artifacts under results/phase2/
    importance_csv = PHASE2_RESULTS_DIR / "xgboost_feature_importance.csv"
    ranking_csv = PHASE2_RESULTS_DIR / "xgboost_feature_ranking.csv"
    selected_json = PHASE2_RESULTS_DIR / "selected_features_19.json"

    df_ranking.to_csv(importance_csv, index=False)
    df_ranking.to_csv(ranking_csv, index=False)

    with open(selected_json, "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, indent=2)

    logger.info("XGBoost Feature Selection Complete:")
    logger.info(f"  - Selected Top {top_k} Features: {selected_19}")
    logger.info(f"  - Overlap with Planned 19: {len(common_features)}/19 ({comparison_summary['overlap_percentage']}%)")
    logger.info(f"  - Saved ranking to: {ranking_csv}")
    logger.info(f"  - Saved selected features JSON to: {selected_json}")

    return selected_19, df_ranking, comparison_summary
