"""
Dataset splitting module for UNSW-NB15 reproduction.
Implements the paper-mandated 75% Training (TRAIN-1) and 25% Validation (VAL) split
from the official UNSW-NB15 training dataset (175,341 rows), keeping the official
UNSW-NB15 test set (82,332 rows) strictly separate for final evaluation.
"""

from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from backend.config import (
    MULTICLASS_TARGET,
    BINARY_TARGET,
    RANDOM_SEED,
    EXPECTED_TRAIN1_ROWS,
    EXPECTED_VAL_ROWS,
    EXPECTED_TEST_ROWS,
)
from backend.dataset.loader import load_data
from backend.utils.logger import setup_logger

logger = setup_logger("DataSplitter")


def split_training_dataset(df_train: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the UNSW-NB15 training dataset (175,341 records) into:
    - UNSW-NB15-TRAIN-1: 75% (131,506 records)
    - UNSW-NB15-VAL:     25% (43,835 records)
    
    Stratification is performed on attack_cat to preserve identical class
    distribution matching Table 2 of the reference paper (Kasongo & Sun, 2020).
    """
    logger.info("Splitting training dataset into 75% TRAIN-1 and 25% VAL (stratified)...")
    
    df_train1, df_val = train_test_split(
        df_train,
        test_size=EXPECTED_VAL_ROWS,
        random_state=RANDOM_SEED,
        stratify=df_train[MULTICLASS_TARGET],
    )
    
    # Reset indices for clean downstream indexing
    df_train1 = df_train1.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

    logger.info(f"Split completed:")
    logger.info(f"  - TRAIN-1 (75%): {df_train1.shape[0]} rows (expected: {EXPECTED_TRAIN1_ROWS})")
    logger.info(f"  - VAL     (25%): {df_val.shape[0]} rows (expected: {EXPECTED_VAL_ROWS})")

    # Assert exact alignment with Table 2
    if len(df_train1) != EXPECTED_TRAIN1_ROWS:
        logger.warning(f"TRAIN-1 count mismatch: got {len(df_train1)}, expected {EXPECTED_TRAIN1_ROWS}")
    if len(df_val) != EXPECTED_VAL_ROWS:
        logger.warning(f"VAL count mismatch: got {len(df_val)}, expected {EXPECTED_VAL_ROWS}")

    return df_train1, df_val


def get_train_val_test_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience loader that loads the raw datasets and returns the three strictly
    separated partitions: (df_train1, df_val, df_test).
    """
    df_train_full, df_test = load_data()
    df_train1, df_val = split_training_dataset(df_train_full)

    logger.info(f"Final 3-partition setup:")
    logger.info(f"  - df_train1: {df_train1.shape}")
    logger.info(f"  - df_val:    {df_val.shape}")
    logger.info(f"  - df_test:   {df_test.shape}")

    return df_train1, df_val, df_test
