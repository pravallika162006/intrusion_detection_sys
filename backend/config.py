"""
Configuration settings for Phase 1: UNSW-NB15 Baseline Model Development.
Contains dataset locations, feature schema registries, target columns, and output directory paths.
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent

# Portable Dataset Directory configuration
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

PHASE1_MODELS_DIR = MODELS_DIR / "phase1"
PHASE1_BINARY_MODELS_DIR = PHASE1_MODELS_DIR / "binary"
PHASE1_MULTICLASS_MODELS_DIR = PHASE1_MODELS_DIR / "multiclass"

PHASE2_MODELS_DIR = MODELS_DIR / "phase2"
PHASE2_BINARY_MODELS_DIR = PHASE2_MODELS_DIR / "binary"
PHASE2_MULTICLASS_MODELS_DIR = PHASE2_MODELS_DIR / "multiclass"

PHASE3_MODELS_DIR = MODELS_DIR / "phase3"
PHASE3_BINARY_MODELS_DIR = PHASE3_MODELS_DIR / "binary"
PHASE3_MULTICLASS_MODELS_DIR = PHASE3_MODELS_DIR / "multiclass"

PREPROCESSING_DIR = BASE_DIR / "preprocessing_artifacts"
PREPROCESSING_PHASE1_DIR = PREPROCESSING_DIR / "phase1"
PREPROCESSING_PHASE2_DIR = PREPROCESSING_DIR / "phase2"
PREPROCESSING_PHASE3_DIR = PREPROCESSING_DIR / "phase3"

RESULTS_DIR = BASE_DIR / "results"
PHASE1_RESULTS_DIR = RESULTS_DIR / "phase1"
PHASE2_RESULTS_DIR = RESULTS_DIR / "phase2"
PHASE3_RESULTS_DIR = RESULTS_DIR / "phase3"

REPORTS_DIR = RESULTS_DIR / "classification_reports"
CONFUSION_DIR = RESULTS_DIR / "confusion_matrices"
UPLOADS_DIR = BASE_DIR / "uploads"

# Create directories if they do not exist
for directory in [
    MODELS_DIR,
    BINARY_MODELS_DIR,
    MULTICLASS_MODELS_DIR,
    PHASE1_MODELS_DIR,
    PHASE1_BINARY_MODELS_DIR,
    PHASE1_MULTICLASS_MODELS_DIR,
    PHASE2_MODELS_DIR,
    PHASE2_BINARY_MODELS_DIR,
    PHASE2_MULTICLASS_MODELS_DIR,
    PHASE3_MODELS_DIR,
    PHASE3_BINARY_MODELS_DIR,
    PHASE3_MULTICLASS_MODELS_DIR,
    PREPROCESSING_DIR,
    PREPROCESSING_PHASE1_DIR,
    PREPROCESSING_PHASE2_DIR,
    PREPROCESSING_PHASE3_DIR,
    RESULTS_DIR,
    PHASE1_RESULTS_DIR,
    PHASE2_RESULTS_DIR,
    PHASE3_RESULTS_DIR,
    REPORTS_DIR,
    CONFUSION_DIR,
    UPLOADS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset validation expectations (UNSW-NB15 & Paper Table 2)
EXPECTED_TRAIN_ROWS = 175341
EXPECTED_TRAIN1_ROWS = 131506  # 75% of training set
EXPECTED_VAL_ROWS = 43835      # 25% of training set
EXPECTED_TEST_ROWS = 82332     # 100% of test set
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

# Exact Paper Table 3 19 Selected Features (Kasongo & Sun, 2020)
SELECTED_19_FEATURES = [
    "sttl",
    "ct_srv_dst",
    "sbytes",
    "smean",
    "proto",
    "ct_state_ttl",
    "sloss",
    "synack",
    "ct_dst_src_ltm",
    "dmean",
    "ct_srv_src",
    "service",
    "ct_dst_sport_ltm",
    "dbytes",
    "dloss",
    "state",
    "tcprtt",
    "ct_src_dport_ltm",
    "rate",
]

# Paper Table 3 Reported Importance Scores
PAPER_19_IMPORTANCE_SCORES = {
    "sttl": 0.803374,
    "ct_srv_dst": 0.039387,
    "sbytes": 0.037377,
    "smean": 0.019878,
    "proto": 0.018848,
    "ct_state_ttl": 0.016783,
    "sloss": 0.012008,
    "synack": 0.010125,
    "ct_dst_src_ltm": 0.007203,
    "dmean": 0.007134,
    "ct_srv_src": 0.006745,
    "service": 0.006305,
    "ct_dst_sport_ltm": 0.003717,
    "dbytes": 0.002706,
    "dloss": 0.001793,
    "state": 0.001548,
    "tcprtt": 0.001224,
    "ct_src_dport_ltm": 0.000526,
    "rate": 0.000503,
}

# Reproducibility
RANDOM_SEED = 42
LR_RANDOM_STATE = 10  # Explicit in paper


