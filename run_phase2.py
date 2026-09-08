"""
Main execution entry point for Phase 2: XGBoost Feature Selection & 19-Feature Model Evaluation.
Executes feature ranking, top-19 selection, 19-feature preprocessing, 14 model experiments,
42 vs 19 comparison, and report generation.
"""

import sys
import pandas as pd
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.config import RESULTS_DIR
from backend.dataset.loader import load_data
from backend.preprocessing.pipeline import prepare_preprocessed_data
from backend.phase2.feature_selector import run_xgboost_feature_selection
from backend.phase2.visualizer import generate_feature_importance_plots
from backend.phase2.pipeline_19 import prepare_19_preprocessed_data
from backend.phase2.trainers_19 import (
    train_phase2_binary_models,
    train_phase2_multiclass_models,
)
from backend.phase2.comparator import generate_42_vs_19_comparison
from backend.utils.logger import setup_logger, Timer

logger = setup_logger("Phase2_Runner")

PHASE2_RESULTS_DIR = RESULTS_DIR / "phase2"

def main():
    logger.info("=========================================================================")
    logger.info(" PHASE 2: XGBOOST FEATURE SELECTION & 19-FEATURE MODEL EVALUATION ")
    logger.info("=========================================================================")

    with Timer() as total_timer:
        # Step 1: Load and Validate Data
        logger.info("\n--- STEP 1: LOADING & VALIDATING DATASET ---")
        df_train, df_test = load_data()

        # Step 2: Preprocess 42 Features for XGBoost Selection
        logger.info("\n--- STEP 2: PREPROCESSING 42 FEATURES FOR FEATURE SELECTION ---")
        (
            X_train_proc,
            X_test_proc,
            y_bin_train,
            y_bin_test,
            y_multi_train,
            y_multi_test,
            prep_info,
        ) = prepare_preprocessed_data(df_train, df_test)

        # Step 3: Run XGBoost Feature Selection on Training Data Only
        logger.info("\n--- STEP 3: XGBOOST FEATURE IMPORTANCE & RANKING (42 FEATURES) ---")
        selected_19_features, df_ranking, comp_summary = run_xgboost_feature_selection(
            X_train_proc, y_bin_train, prep_info, top_k=19
        )

        # Step 4: Generate Visualizations
        logger.info("\n--- STEP 4: GENERATING FEATURE IMPORTANCE VISUALIZATIONS ---")
        generate_feature_importance_plots(df_ranking, top_k=19)

        # Step 5: Preprocess 19 Selected Features
        logger.info("\n--- STEP 5: PREPROCESSING 19-FEATURE SUBSET ---")
        (
            X19_train_proc,
            X19_test_proc,
            y_bin_train,
            y_bin_test,
            y_multi_train,
            y_multi_test,
            prep19_info,
        ) = prepare_19_preprocessed_data(df_train, df_test, selected_19_features)

        # Step 6: Train Phase 2 Binary Models (7 Approaches)
        logger.info("\n--- STEP 6: BINARY CLASSIFICATION (19 FEATURES) ---")
        binary_results, binary_paths = train_phase2_binary_models(
            X19_train_proc, X19_test_proc, y_bin_train, y_bin_test
        )

        # Step 7: Train Phase 2 Multiclass Models (7 Approaches)
        logger.info("\n--- STEP 7: MULTICLASS CLASSIFICATION (19 FEATURES) ---")
        target_names = [str(cls) for cls in prep19_info["classes"]]
        multiclass_results, multiclass_paths = train_phase2_multiclass_models(
            X19_train_proc, X19_test_proc, y_multi_train, y_multi_test, target_names
        )

        # Step 8: Save Results Tables
        logger.info("\n--- STEP 8: SAVING PHASE 2 RESULTS ---")
        df_binary = pd.DataFrame(binary_results)
        binary_csv_path = PHASE2_RESULTS_DIR / "binary_results.csv"
        df_binary.to_csv(binary_csv_path, index=False)
        logger.info(f"Saved Phase 2 binary results to: {binary_csv_path}")

        df_multi = pd.DataFrame(multiclass_results)
        multi_csv_path = PHASE2_RESULTS_DIR / "multiclass_results.csv"
        df_multi.to_csv(multi_csv_path, index=False)
        logger.info(f"Saved Phase 2 multiclass results to: {multi_csv_path}")

        # Step 9: 42 vs 19 Comparison
        logger.info("\n--- STEP 9: GENERATING 42 VS 19 DELTA COMPARISON ---")
        df_comp = generate_42_vs_19_comparison()

    logger.info("=========================================================================")
    logger.info(f" PHASE 2 PIPELINE COMPLETED SUCCESSFULLY IN {total_timer.interval:.2f} SECONDS ")
    logger.info("=========================================================================")

    # Print summary tables to console
    print("\n" + "="*80)
    print(" PHASE 2 BINARY CLASSIFICATION RESULTS SUMMARY (19 FEATURES)")
    print("="*80)
    print(df_binary.to_string(index=False))

    print("\n" + "="*80)
    print(" PHASE 2 MULTICLASS CLASSIFICATION RESULTS SUMMARY (19 FEATURES)")
    print("="*80)
    print(df_multi.to_string(index=False))

    print("\n" + "="*80)
    print(" 42 VS 19 FEATURE SET DELTA COMPARISON")
    print("="*80)
    print(df_comp.to_string(index=False))
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
