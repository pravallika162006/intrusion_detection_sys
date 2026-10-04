"""
Main execution entry point for Phase 1: UNSW-NB15 Baseline Model Development.
Faithfully reproduces Sydney M. Kasongo & Yanxia Sun (2020):
- 75% TRAIN-1 (131,506) / 25% VAL (43,835) / TEST (82,332) stratified split
- Min-Max scaling fitted strictly on TRAIN-1
- 5 baseline models across Binary and Multiclass
- Results saved to results/phase1/ and results/
"""

import sys
import pandas as pd
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.config import RESULTS_DIR, PHASE1_RESULTS_DIR
from backend.dataset.splitter import get_train_val_test_splits
from backend.preprocessing.pipeline import prepare_preprocessed_data
from backend.models.trainers import (
    train_and_eval_binary_models,
    train_and_eval_multiclass_models,
)
from backend.utils.logger import setup_logger, Timer

logger = setup_logger("Phase1_Runner")


def main():
    logger.info("=========================================================================")
    logger.info(" PHASE 1: UNSW-NB15 DATASET PREPARATION & 42-FEATURE BASELINE REPRODUCTION ")
    logger.info("=========================================================================")

    with Timer() as total_timer:
        # Step 1: Load and Split Data into 75% TRAIN-1, 25% VAL, and TEST
        logger.info("\n--- STEP 1: LOADING & STRATIFYING DATASET (75% TRAIN-1 / 25% VAL / TEST) ---")
        df_train1, df_val, df_test = get_train_val_test_splits()

        # Step 2: Preprocess Data without Leakage (fitted strictly on TRAIN-1)
        logger.info("\n--- STEP 2: PREPROCESSING DATASET (MIN-MAX SCALING ON TRAIN-1 ONLY) ---")
        (
            X_train_proc,
            X_val_proc,
            X_test_proc,
            y_bin_train,
            y_bin_val,
            y_bin_test,
            y_multi_train,
            y_multi_val,
            y_multi_test,
            prep_info,
        ) = prepare_preprocessed_data(df_train1, df_val, df_test)

        logger.info(f"Feature set = 42 original UNSW-NB15 features (encoded into {prep_info['processed_feature_count']} input columns)")

        # Step 3: Binary Classification Models
        logger.info("\n--- STEP 3: BINARY CLASSIFICATION MODELS (42 FEATURES) ---")
        binary_results, binary_paths = train_and_eval_binary_models(
            X_train_proc, X_val_proc, X_test_proc,
            y_bin_train, y_bin_val, y_bin_test,
        )

        # Step 4: Multiclass Classification Models
        logger.info("\n--- STEP 4: MULTICLASS CLASSIFICATION MODELS (42 FEATURES) ---")
        target_names = [str(cls) for cls in prep_info["classes"]]
        multiclass_results, multiclass_paths = train_and_eval_multiclass_models(
            X_train_proc, X_val_proc, X_test_proc,
            y_multi_train, y_multi_val, y_multi_test,
            target_names,
        )

        # Step 5: Save Comparison Tables
        logger.info("\n--- STEP 5: SAVING PHASE 1 RESULTS & COMPARISON TABLES ---")
        
        # Binary results DataFrame
        df_binary = pd.DataFrame(binary_results)
        df_binary_clean = df_binary.drop(columns=["Confusion Matrix"], errors="ignore")
        
        for p in [PHASE1_RESULTS_DIR / "binary_results.csv", RESULTS_DIR / "binary_results.csv"]:
            df_binary_clean.to_csv(p, index=False)
            logger.info(f"Saved binary classification results to: {p}")

        # Multiclass results DataFrame
        df_multi = pd.DataFrame(multiclass_results)
        df_multi_clean = df_multi.drop(columns=["Confusion Matrix"], errors="ignore")
        
        for p in [PHASE1_RESULTS_DIR / "multiclass_results.csv", RESULTS_DIR / "multiclass_results.csv"]:
            df_multi_clean.to_csv(p, index=False)
            logger.info(f"Saved multiclass classification results to: {p}")

    logger.info("=========================================================================")
    logger.info(f" PHASE 1 PIPELINE COMPLETED SUCCESSFULLY IN {total_timer.interval:.2f} SECONDS ")
    logger.info("=========================================================================")

    # Print summary tables to console
    print("\n" + "="*80)
    print(" PHASE 1 BINARY CLASSIFICATION RESULTS SUMMARY (42 FEATURES)")
    print("="*80)
    print(df_binary_clean.to_string(index=False))

    print("\n" + "="*80)
    print(" PHASE 1 MULTICLASS CLASSIFICATION RESULTS SUMMARY (42 FEATURES)")
    print("="*80)
    print(df_multi_clean.to_string(index=False))
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
