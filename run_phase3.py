"""
Main execution entry point for Phase 3: Project Enhancement (Alternative Models & Ensembles).
Evaluates candidate classifiers and model combinations strictly on the VALIDATION set,
applies the selection rule, freezes the selected model, and evaluates ONCE on TEST.
"""

import sys
import json
import pandas as pd
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.config import PHASE2_RESULTS_DIR, PHASE3_RESULTS_DIR, SELECTED_19_FEATURES
from backend.dataset.splitter import get_train_val_test_splits
from backend.phase2.pipeline_19 import prepare_19_preprocessed_data
from backend.phase3.candidates import train_and_eval_candidates
from backend.phase3.ensembles import build_and_eval_ensembles
from backend.phase3.selector import select_and_freeze_model
from backend.utils.logger import setup_logger, Timer

logger = setup_logger("Phase3_Runner")


def main():
    logger.info("=========================================================================")
    logger.info(" PHASE 3: PROJECT ENHANCEMENT — ALTERNATIVE MODELS & ENSEMBLE EVALUATION ")
    logger.info("=========================================================================")

    with Timer() as total_timer:
        # Step 1: Load and Split Data
        logger.info("\n--- STEP 1: LOADING & STRATIFYING DATASET (75% TRAIN-1 / 25% VAL / TEST) ---")
        df_train1, df_val, df_test = get_train_val_test_splits()

        # Step 2: Preprocess using the 19-feature pipeline (fitted on TRAIN-1 only)
        logger.info("\n--- STEP 2: PREPROCESSING 19-FEATURE REPRESENTATION ---")
        (
            X19_train_proc,
            X19_val_proc,
            X19_test_proc,
            y_bin_train,
            y_bin_val,
            y_bin_test,
            y_multi_train,
            y_multi_val,
            y_multi_test,
            prep19_info,
        ) = prepare_19_preprocessed_data(df_train1, df_val, df_test, SELECTED_19_FEATURES)

        target_names = [str(cls) for cls in prep19_info["classes"]]

        # Read Phase 2 Baseline Results to establish the hurdle
        phase2_bin_csv = PHASE2_RESULTS_DIR / "binary_results.csv"
        phase2_multi_csv = PHASE2_RESULTS_DIR / "multiclass_results.csv"

        baseline_bin_val = {"Model": "Phase 2 Baseline (DT)", "Val_Accuracy": 93.81, "Val_F1": 90.0}
        baseline_multi_val = {"Model": "Phase 2 Baseline (DT)", "Val_Accuracy": 78.43, "Val_F1": 69.26}

        if phase2_bin_csv.exists():
            try:
                df_p2_b = pd.read_csv(phase2_bin_csv)
                best_row = df_p2_b.sort_values(by="Val. AC (%)", ascending=False).iloc[0]
                baseline_bin_val = {
                    "Model": f"Phase 2 Best ({best_row['ML method']})",
                    "Val_Accuracy": float(best_row["Val. AC (%)"]),
                    "Val_F1": float(best_row["F1-Score (%)"]),
                }
            except Exception as e:
                logger.warning(f"Could not parse Phase 2 binary baseline: {e}")

        if phase2_multi_csv.exists():
            try:
                df_p2_m = pd.read_csv(phase2_multi_csv)
                best_row = df_p2_m.sort_values(by="Val. AC (%)", ascending=False).iloc[0]
                baseline_multi_val = {
                    "Model": f"Phase 2 Best ({best_row['ML method']})",
                    "Val_Accuracy": float(best_row["Val. AC (%)"]),
                    "Val_F1": float(best_row["F1-Score (%)"]),
                }
            except Exception as e:
                logger.warning(f"Could not parse Phase 2 multiclass baseline: {e}")

        # =========================================================
        # TASK A: BINARY CLASSIFICATION ENHANCEMENT
        # =========================================================
        logger.info("\n=========================================================")
        logger.info(" PHASE 3: BINARY CLASSIFICATION CANDIDATES & ENSEMBLES ")
        logger.info("=========================================================")

        # 1. Candidate Models on VAL
        df_cand_val_b, fitted_cand_b = train_and_eval_candidates(
            X19_train_proc, y_bin_train, X19_val_proc, y_bin_val, task="binary"
        )
        df_cand_val_b.to_csv(PHASE3_RESULTS_DIR / "binary_candidates_val.csv", index=False)

        # 2. Ensemble Combinations on VAL
        df_ens_val_b, fitted_ens_b = build_and_eval_ensembles(
            X19_train_proc, y_bin_train, X19_val_proc, y_bin_val, task="binary"
        )
        df_ens_val_b.to_csv(PHASE3_RESULTS_DIR / "binary_ensembles_val.csv", index=False)

        # Merge fitted models dictionary
        all_models_b = {**fitted_cand_b, **fitted_ens_b}

        # 3. Model Selector & Single Test Evaluation
        summary_b, df_final_b = select_and_freeze_model(
            df_cand_val_b, df_ens_val_b, all_models_b, baseline_bin_val,
            X19_test_proc, y_bin_test, task="binary"
        )

        # =========================================================
        # TASK B: MULTICLASS CLASSIFICATION ENHANCEMENT
        # =========================================================
        logger.info("\n=========================================================")
        logger.info(" PHASE 3: MULTICLASS CLASSIFICATION CANDIDATES & ENSEMBLES ")
        logger.info("=========================================================")

        # 1. Candidate Models on VAL
        df_cand_val_m, fitted_cand_m = train_and_eval_candidates(
            X19_train_proc, y_multi_train, X19_val_proc, y_multi_val,
            task="multiclass", target_names=target_names
        )
        df_cand_val_m.to_csv(PHASE3_RESULTS_DIR / "multiclass_candidates_val.csv", index=False)

        # 2. Ensemble Combinations on VAL
        df_ens_val_m, fitted_ens_m = build_and_eval_ensembles(
            X19_train_proc, y_multi_train, X19_val_proc, y_multi_val,
            task="multiclass", target_names=target_names
        )
        df_ens_val_m.to_csv(PHASE3_RESULTS_DIR / "multiclass_ensembles_val.csv", index=False)

        # Merge fitted models dictionary
        all_models_m = {**fitted_cand_m, **fitted_ens_m}

        # 3. Model Selector & Single Test Evaluation
        summary_m, df_final_m = select_and_freeze_model(
            df_cand_val_m, df_ens_val_m, all_models_m, baseline_multi_val,
            X19_test_proc, y_multi_test, task="multiclass", target_names=target_names
        )

        # Save Combined Final Selected Model Test Table
        df_final_all = pd.concat([df_final_b, df_final_m], ignore_index=True)
        final_test_csv = PHASE3_RESULTS_DIR / "final_selected_model_test_results.csv"
        df_final_all.to_csv(final_test_csv, index=False)
        logger.info(f"Saved Phase 3 Final Selected Model Test Results to: {final_test_csv}")

    logger.info("=========================================================================")
    logger.info(f" PHASE 3 PIPELINE COMPLETED SUCCESSFULLY IN {total_timer.interval:.2f} SECONDS ")
    logger.info("=========================================================================")

    # Print summary tables to console
    print("\n" + "="*80)
    print(" PHASE 3 BINARY VALIDATION EVALUATION (CANDIDATES & ENSEMBLES)")
    print("="*80)
    df_val_all_b = pd.concat([df_cand_val_b, df_ens_val_b]).sort_values(by="Val_Accuracy", ascending=False)
    print(df_val_all_b.to_string(index=False))

    print("\n" + "="*80)
    print(" PHASE 3 MULTICLASS VALIDATION EVALUATION (CANDIDATES & ENSEMBLES)")
    print("="*80)
    df_val_all_m = pd.concat([df_cand_val_m, df_ens_val_m]).sort_values(by="Val_Accuracy", ascending=False)
    print(df_val_all_m.to_string(index=False))

    print("\n" + "="*80)
    print(" PHASE 3 FINAL FROZEN MODEL TEST PERFORMANCE")
    print("="*80)
    print(df_final_all.to_string(index=False))
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
