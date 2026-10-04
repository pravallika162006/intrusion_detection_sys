"""
Centralized Reproducibility Configuration for Kasongo & Sun (2020) Paper Reproduction.
Reference Paper:
  "Performance Analysis of Intrusion Detection Systems Using a Feature Selection Method on the UNSW-NB15 Dataset"
  Authors: Sydney M. Kasongo and Yanxia Sun (Journal of Big Data, 2020).
"""

from typing import Dict, Any, List

PAPER_REPRODUCTION = True

# Dataset Partitioning (Table 2 in paper)
TRAIN_SPLIT = 0.75
VALIDATION_SPLIT = 0.25

EXPECTED_TRAIN1_ROWS = 131506  # 75% of UNSW-NB15-TRAIN (175,341)
EXPECTED_VAL_ROWS = 43835      # 25% of UNSW-NB15-TRAIN
EXPECTED_TEST_ROWS = 82332     # 100% of UNSW-NB15-TEST (held-out independent test set)

# Normalization & Preprocessing
SCALER = "minmax"  # Equation 5 in paper: F_norm = (F - F_min) / (F_max - F_min)
SCALER_RANGE = (0, 1)

# Categorical features in UNSW-NB15
CATEGORICAL_FEATURES = ["proto", "service", "state"]

# Exact 19 features from Paper Table 3 (Sydney M. Kasongo & Yanxia Sun, 2020)
SELECTED_19_FEATURES = [
    "sttl", "ct_srv_dst", "sbytes", "smean", "proto",
    "ct_state_ttl", "sloss", "synack", "ct_dst_src_ltm", "dmean",
    "ct_srv_src", "service", "ct_dst_sport_ltm", "dbytes", "dloss",
    "state", "tcprtt", "ct_src_dport_ltm", "rate"
]

# Random Seeds
RANDOM_SEED = 42
LR_RANDOM_STATE = 10  # Explicitly specified on p. 16 of paper: "random state was set at 10"

# Hyperparameter Exploration Grids (Section: "Experiments and Results", pp. 15-16)
MODEL_HYPERPARAMETERS = {
    "ANN": {
        "solver": "adam",
        "hidden_layer_sizes": [5, 10, 15, 30, 50, 100, 150],
        "learning_rate": "adaptive",
        "initial_lr": 0.01,
        "max_iter": 30,
        "batch_size": 256,
    },
    "LR": {
        "random_state": 10,
        "max_iter": 1000,
        "solver": "lbfgs",
    },
    "kNN": {
        "n_neighbors": [3, 5, 7, 9, 11],
        "metric": "euclidean",
    },
    "SVM": {
        "kernel": "rbf",
        "C": 1.12,
        "gamma": "scale",
    },
    "DT": {
        "max_depth": [2, 5, 7, 8, 9],
        "criterion": ["gini", "entropy"],
    },
}

# Reference Paper Targets (Tables 4, 5, 6, 7)
PAPER_TARGETS = {
    "phase1_binary": {  # Table 4 (42 features, binary)
        "ANN": {"Test_AC": 86.71, "Precision": 81.54, "Recall": 98.06, "F1": 89.04, "Tr_AC": 94.49, "Val_AC": 94.21},
        "LR":  {"Test_AC": 79.59, "Precision": 73.32, "Recall": 98.94, "F1": 84.22, "Tr_AC": 93.22, "Val_AC": 92.87},
        "kNN": {"Test_AC": 83.18, "Precision": 79.15, "Recall": 94.30, "F1": 86.06, "Tr_AC": 96.76, "Val_AC": 93.60},
        "SVM": {"Test_AC": 62.42, "Precision": 60.91, "Recall": 88.58, "F1": 71.18, "Tr_AC": 70.98, "Val_AC": 70.63},
        "DT":  {"Test_AC": 88.13, "Precision": 83.91, "Recall": 96.47, "F1": 90.00, "Tr_AC": 93.65, "Val_AC": 93.37},
    },
    "phase2_binary": {  # Table 5 (19 features, binary)
        "ANN": {"Test_AC": 84.39, "Precision": 78.56, "Recall": 98.53, "F1": 87.42, "Tr_AC": 93.75, "Val_AC": 93.66},
        "LR":  {"Test_AC": 77.64, "Precision": 73.18, "Recall": 93.74, "F1": 82.20, "Tr_AC": 89.21, "Val_AC": 89.25},
        "kNN": {"Test_AC": 84.46, "Precision": 80.31, "Recall": 95.09, "F1": 87.08, "Tr_AC": 95.86, "Val_AC": 94.73},
        "SVM": {"Test_AC": 60.89, "Precision": 58.89, "Recall": 95.88, "F1": 72.97, "Tr_AC": 75.42, "Val_AC": 75.51},
        "DT":  {"Test_AC": 90.85, "Precision": 80.33, "Recall": 98.38, "F1": 88.45, "Tr_AC": 94.12, "Val_AC": 93.81},
    },
    "phase1_multiclass": {  # Table 6 (42 features, multiclass)
        "ANN": {"Test_AC": 75.62, "Precision": 79.92, "Recall": 75.61, "F1": 76.58, "Tr_AC": 79.91, "Val_AC": 79.61},
        "LR":  {"Test_AC": 65.53, "Precision": 76.91, "Recall": 65.54, "F1": 66.62, "Tr_AC": 75.51, "Val_AC": 73.93},
        "kNN": {"Test_AC": 70.09, "Precision": 75.79, "Recall": 70.21, "F1": 72.03, "Tr_AC": 81.75, "Val_AC": 76.83},
        "SVM": {"Test_AC": 61.09, "Precision": 47.47, "Recall": 62.00, "F1": 53.77, "Tr_AC": 53.43, "Val_AC": 52.67},
        "DT":  {"Test_AC": 66.03, "Precision": 79.82, "Recall": 66.04, "F1": 51.12, "Tr_AC": 77.69, "Val_AC": 77.38},
    },
    "phase2_multiclass": {  # Table 7 (19 features, multiclass)
        "ANN": {"Test_AC": 77.51, "Precision": 79.50, "Recall": 77.53, "F1": 77.28, "Tr_AC": 79.46, "Val_AC": 78.91},
        "LR":  {"Test_AC": 65.29, "Precision": 70.88, "Recall": 65.29, "F1": 65.96, "Tr_AC": 72.53, "Val_AC": 71.81},
        "kNN": {"Test_AC": 72.30, "Precision": 77.24, "Recall": 72.30, "F1": 73.81, "Tr_AC": 82.66, "Val_AC": 79.87},
        "SVM": {"Test_AC": 61.53, "Precision": 53.95, "Recall": 61.52, "F1": 51.31, "Tr_AC": 53.60, "Val_AC": 52.97},
        "DT":  {"Test_AC": 67.57, "Precision": 79.66, "Recall": 67.56, "F1": 69.26, "Tr_AC": 78.75, "Val_AC": 78.43},
    },
}

# Paper Inconsistency Documentation Note
PAPER_INCONSISTENCY_NOTE = (
    "Note on 19-Feature Binary Decision Tree Inconsistency: "
    "Table 5 of the reference paper (Kasongo & Sun, 2020) reports a test accuracy of 90.85%, "
    "while Section 'Experiments and Results' (p. 16) states: 'whereas it obtained a test score of 85.85% "
    "utilizing 19 features.' For the reproduction benchmark, the official Table 5 value (90.85%) is used."
)

def categorize_difference(diff: float) -> str:
    """
    Categorizes the absolute accuracy difference between our implementation and the reference paper.
    - Difference <= 0.50 percentage points: 'Very close'
    - Difference <= 1.50 percentage points: 'Close'
    - Difference <= 3.00 percentage points: 'Moderate difference'
    - Difference > 3.00 percentage points:  'Needs investigation'
    """
    if diff is None:
        return "N/A"
    abs_d = abs(diff)
    if abs_d <= 0.50:
        return "Very close"
    elif abs_d <= 1.50:
        return "Close"
    elif abs_d <= 3.00:
        return "Moderate difference"
    else:
        return "Needs investigation"
