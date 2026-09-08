"""
Configuration settings for Phase 1: UNSW-NB15 Baseline Model Development.
Contains dataset locations, feature schema registries, target columns, and output directory paths.
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent

# Portable Dataset Directory configuration
# Order of preference:
# 1. Environment variable UNSW_DATASET_DIR (if set)
# 2. IDS/data directory in project root (standard recommendation)
# 3. Fallback path for backward compatibility if data/ does not exist yet
ENV_DATASET_DIR = os.getenv("UNSW_DATASET_DIR")
if ENV_DATASET_DIR:
    DATASET_DIR = Path(ENV_DATASET_DIR)
elif (BASE_DIR / "data").exists():
    DATASET_DIR = BASE_DIR / "data"
else:
    DATASET_DIR = Path(r"C:\Users\mvspr\Downloads\unswnb")

# File paths
TRAIN_CSV_PATH = DATASET_DIR / "UNSW_NB15_training-set.csv"
TEST_CSV_PATH = DATASET_DIR / "UNSW_NB15_testing-set.csv"

# Output directories
MODELS_DIR = BASE_DIR / "models"
BINARY_MODELS_DIR = MODELS_DIR / "binary"
MULTICLASS_MODELS_DIR = MODELS_DIR / "multiclass"
PREPROCESSING_DIR = BASE_DIR / "preprocessing_artifacts"
RESULTS_DIR = BASE_DIR / "results"
REPORTS_DIR = RESULTS_DIR / "classification_reports"
CONFUSION_DIR = RESULTS_DIR / "confusion_matrices"

# Create directories if they do not exist
for directory in [
    MODELS_DIR,
    BINARY_MODELS_DIR,
    MULTICLASS_MODELS_DIR,
    PREPROCESSING_DIR,
    RESULTS_DIR,
    REPORTS_DIR,
    CONFUSION_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset validation expectations
EXPECTED_TRAIN_ROWS = 175341
EXPECTED_TEST_ROWS = 82332
EXPECTED_TOTAL_COLS = 45

# Column Definitions
META_COLS = ["id"]
BINARY_TARGET = "label"
MULTICLASS_TARGET = "attack_cat"

# The 3 Categorical Features
CATEGORICAL_FEATURES = ["proto", "service", "state"]

# The 39 Numerical Features
NUMERICAL_FEATURES = [
    "dur",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate",
    "sttl",
    "dttl",
    "sload",
    "dload",
    "sloss",
    "dloss",
    "sinpkt",
    "dinpkt",
    "sjit",
    "djit",
    "swin",
    "stcpb",
    "dtcpb",
    "dwin",
    "tcprtt",
    "synack",
    "ackdat",
    "smean",
    "dmean",
    "trans_depth",
    "response_body_len",
    "ct_srv_src",
    "ct_state_ttl",
    "ct_dst_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_flw_http_mthd",
    "ct_src_ltm",
    "ct_srv_dst",
    "is_sm_ips_ports",
]

# Combined 42 Input Features
ALL_INPUT_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

# Reproducibility
RANDOM_SEED = 42
