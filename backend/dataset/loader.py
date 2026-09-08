"""
Dataset loading and validation module for Phase 1.
"""

from typing import Tuple
import pandas as pd

from backend.config import (
    TRAIN_CSV_PATH,
    TEST_CSV_PATH,
    EXPECTED_TRAIN_ROWS,
    EXPECTED_TEST_ROWS,
    EXPECTED_TOTAL_COLS,
    ALL_INPUT_FEATURES,
    BINARY_TARGET,
    MULTICLASS_TARGET,
    META_COLS,
)
from backend.utils.logger import setup_logger, Timer

logger = setup_logger("DataLoader")

def validate_dataset(df_train: pd.DataFrame, df_test: pd.DataFrame) -> None:
    """
    Performs programmatic validation on loaded UNSW-NB15 dataset splits.
    Raises ValueError if any assertion fails.
    """
    logger.info("Validating training and testing datasets...")

    # Row count checks
    if len(df_train) != EXPECTED_TRAIN_ROWS:
        raise ValueError(f"Train row count mismatch: Expected {EXPECTED_TRAIN_ROWS}, got {len(df_train)}")
    if len(df_test) != EXPECTED_TEST_ROWS:
        raise ValueError(f"Test row count mismatch: Expected {EXPECTED_TEST_ROWS}, got {len(df_test)}")

    # Column count checks
    if len(df_train.columns) != EXPECTED_TOTAL_COLS:
        raise ValueError(f"Train column count mismatch: Expected {EXPECTED_TOTAL_COLS}, got {len(df_train.columns)}")
    if len(df_test.columns) != EXPECTED_TOTAL_COLS:
        raise ValueError(f"Test column count mismatch: Expected {EXPECTED_TOTAL_COLS}, got {len(df_test.columns)}")

    # Column name equality check between train and test
    if list(df_train.columns) != list(df_test.columns):
        raise ValueError("Train and test column names/sequence are not identical!")

    # Check existence of required target and meta columns
    if BINARY_TARGET not in df_train.columns:
        raise ValueError(f"Binary target '{BINARY_TARGET}' missing from dataset!")
    if MULTICLASS_TARGET not in df_train.columns:
        raise ValueError(f"Multiclass target '{MULTICLASS_TARGET}' missing from dataset!")
    for meta_col in META_COLS:
        if meta_col not in df_train.columns:
            raise ValueError(f"Metadata column '{meta_col}' missing from dataset!")

    # Check existence of all 42 input features
    missing_features = [f for f in ALL_INPUT_FEATURES if f not in df_train.columns]
    if missing_features:
        raise ValueError(f"The following required input features are missing: {missing_features}")

    # Verify input feature count excluding meta and targets
    input_cols = [c for c in df_train.columns if c not in META_COLS + [BINARY_TARGET, MULTICLASS_TARGET]]
    if len(input_cols) != 42:
        raise ValueError(f"Expected exactly 42 input features, but found {len(input_cols)}")

    logger.info("Validation PASSED! Dataset matches all Phase 1 schema requirements.")

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads UNSW-NB15 training and testing CSV files and executes validation.
    Returns (df_train, df_test).
    """
    logger.info(f"Loading training data from: {TRAIN_CSV_PATH}")
    logger.info(f"Loading testing data from:  {TEST_CSV_PATH}")

    with Timer() as timer:
        df_train = pd.read_csv(TRAIN_CSV_PATH)
        df_test = pd.read_csv(TEST_CSV_PATH)

    logger.info(f"Loaded datasets in {timer.interval:.2f} seconds.")
    logger.info(f"  - Training shape: {df_train.shape}")
    logger.info(f"  - Testing shape:  {df_test.shape}")

    validate_dataset(df_train, df_test)

    return df_train, df_test
