"""
Main execution entry point for Phase 1: UNSW-NB15 Baseline Model Development.
Executes data loading, validation, preprocessing, model training, evaluation, artifact saving, and report generation.
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
from backend.models.trainers import (
    train_and_eval_binary_models,
    train_and_eval_multiclass_models,
)
from backend.utils.logger import setup_logger, Timer

logger = setup_logger("Phase1_Runner")

def main():
    logger.info("=========================================================================")
    logger.info(" PHASE 1: UNSW-NB15 DATASET PREPARATION & BASELINE MODEL DEVELOPMENT ")
    logger.info("=========================================================================")

    with Timer() as total_timer:
        # Step 1: Load and Validate Data
        logger.info("\n--- STEP 1: LOADING & VALIDATING DATASET ---")
        df_train, df_test = load_data()

        # Step 2: Preprocess Data without Leakage
        logger.info("\n--- STEP 2: PREPROCESSING DATASET ---")
        (
            X_train_proc,
            X_test_proc,
            y_bin_train,
            y_bin_test,
            y_multi_train,
            y_multi_test,
            prep_info,
        ) = prepare_preprocessed_data(df_train, df_test)

        logger.info(f"Feature set = 42 original UNSW-NB15 features (encoded into {prep_info['processed_feature_count']} input columns)")

        # Step 3: Binary Classification Models
        logger.info("\n--- STEP 3: BINARY CLASSIFICATION MODELS ---")
        binary_results, binary_paths = train_and_eval_binary_models(
            X_train_proc, X_test_proc, y_bin_train, y_bin_test
        )

        # Step 4: Multiclass Classification Models
        logger.info("\n--- STEP 4: MULTICLASS CLASSIFICATION MODELS ---")
        target_names = [str(cls) for cls in prep_info["classes"]]
        multiclass_results, multiclass_paths = train_and_eval_multiclass_models(
            X_train_proc, X_test_proc, y_multi_train, y_multi_test, target_names
        )

        # Step 5: Save Comparison Tables
        logger.info("\n--- STEP 5: SAVING RESULTS & COMPARISON TABLES ---")
        
        # Binary results DataFrame
        df_binary = pd.DataFrame(binary_results)
        # Drop Confusion Matrix array column for clean CSV output
        df_binary_csv = df_binary.drop(columns=["Confusion Matrix"])
        binary_csv_path = RESULTS_DIR / "binary_results.csv"
        df_binary_csv.to_csv(binary_csv_path, index=False)
        logger.info(f"Saved binary classification comparison to: {binary_csv_path}")

        # Multiclass results DataFrame
        df_multi = pd.DataFrame(multiclass_results)
        df_multi_csv = df_multi.drop(columns=["Confusion Matrix"])
        multi_csv_path = RESULTS_DIR / "multiclass_results.csv"
        df_multi_csv.to_csv(multi_csv_path, index=False)
        logger.info(f"Saved multiclass classification comparison to: {multi_csv_path}")

    logger.info("=========================================================================")
    logger.info(f" PHASE 1 PIPELINE COMPLETED SUCCESSFULLY IN {total_timer.interval:.2f} SECONDS ")
    logger.info("=========================================================================")

    # Print summary tables to console
    print("\n" + "="*80)
    print(" BINARY CLASSIFICATION RESULTS SUMMARY")
    print("="*80)
    print(df_binary_csv.to_string(index=False))

    print("\n" + "="*80)
    print(" MULTICLASS CLASSIFICATION RESULTS SUMMARY")
    print("="*80)
    print(df_multi_csv.to_string(index=False))
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
