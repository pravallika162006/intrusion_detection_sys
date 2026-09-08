# Project Setup & Execution Guide (Windows)

This document provides step-by-step instructions for setting up and running the **Intrusion Detection System (IDS)** project on a Windows machine.

---

## 1. Prerequisites

- **Operating System**: Windows 10 / 11 (64-bit)
- **Python**: Version 3.10 or 3.11 recommended (64-bit)
- **Git**: Installed and configured

---

## 2. Step-by-Step Installation

### Step 1: Clone the Repository
Open PowerShell or Command Prompt and clone the repository:
```cmd
git clone <YOUR_REPOSITORY_URL>
cd IDS
```

### Step 2: Create a Virtual Environment
Create an isolated Python virtual environment:
```cmd
python -m venv venv
```

Activate the virtual environment:
```cmd
.\venv\Scripts\activate
```
*(You should see `(venv)` prefixed in your terminal command prompt).*

### Step 3: Install Dependencies
Upgrade pip and install all required packages:
```cmd
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Dataset Configuration

### Step 1: Download Official UNSW-NB15 CSV Dataset
Download the official UNSW-NB15 dataset files:
- `UNSW_NB15_training-set.csv` (175,341 rows)
- `UNSW_NB15_testing-set.csv` (82,332 rows)

### Step 2: Place Dataset Files
Create a folder named `data` inside the project root (`IDS/data/`) and place the two CSV files there:

```
IDS/
├── data/
│   ├── UNSW_NB15_training-set.csv
│   └── UNSW_NB15_testing-set.csv
```

*Note: By default, `backend/config.py` automatically checks `IDS/data/`. Alternatively, you can specify a custom path by setting the environment variable `UNSW_DATASET_DIR="C:\Path\To\Your\Dataset"` or editing `DATASET_DIR` in `backend/config.py`.*

---

## 4. Execution & Model Training Expectations

### Execution Commands

#### Run Phase 1 (Baseline Models — 42 Features)
Trains and evaluates 5 baseline models (Decision Tree, ANN, kNN, Logistic Regression, LinearSVC) across Binary and Multiclass targets:
```cmd
python run_phase1.py
```
*Outputs will be saved in `results/` and `models/`.*

#### Run Phase 2 (XGBoost Feature Selection & 19-Feature Models)
Runs XGBoost feature ranking, selects top 19 features, creates a 19-feature preprocessing pipeline, trains 14 model experiments (7 Binary, 7 Multiclass), generates feature importance visualizations, and computes 42 vs 19 performance deltas:
```cmd
python run_phase2.py
```
*Outputs will be saved in `results/phase2/` and `models/phase2/`.*

### Execution Time & Resource Expectations
Execution times vary significantly depending on hardware and model complexity:
- **Fast Models** (Decision Tree, Logistic Regression, XGBoost feature selection): Complete in 5–30 seconds.
- **Neural Networks (ANN)**: Takes ~1 minute per target (using Keras with early stopping).
- **k-Nearest Neighbors (kNN)**: Training is fast (~0.05s), but evaluation on 82,332 test samples requires non-parametric nearest neighbor searches and may take **1–2 minutes per run**.
- **Multiclass Support Vector Machine (LinearSVC)**: On 42 features, multiclass optimization on 175,341 training rows can take **~10 minutes**. On the reduced 19-feature subset in Phase 2, this speeds up significantly to **~2 minutes**.

---

## 5. Directory Structure Overview

```
IDS/
├── backend/                        # Backend source code modules
│   ├── config.py                   # Global configuration & DATASET_DIR path
│   ├── dataset/                    # Data loader & schema validator
│   ├── preprocessing/              # 42-feature ColumnTransformer pipeline
│   ├── models/                     # ANN builder and Phase 1 trainers
│   ├── evaluation/                 # Metrics calculator
│   ├── utils/                      # Logger and timer helpers
│   └── phase2/                     # Phase 2 feature selection & trainers
├── data/                           # Local dataset folder (ignored by git)
│   ├── UNSW_NB15_training-set.csv
│   └── UNSW_NB15_testing-set.csv
├── results/                        # Phase 1 & Phase 2 results (CSVs, reports, plots)
│   ├── binary_results.csv
│   ├── multiclass_results.csv
│   └── phase2/
│       ├── binary_results.csv
│       ├── multiclass_results.csv
│       ├── comparison_42_vs_19.csv
│       ├── xgboost_feature_importance.csv
│       ├── xgboost_feature_ranking.csv
│       ├── selected_features_19.json
│       ├── classification_reports/
│       ├── confusion_matrices/
│       └── plots/
├── run_phase1.py                   # Main runner for Phase 1
├── run_phase2.py                   # Main runner for Phase 2
├── PHASE_1_README.md               # Detailed Phase 1 documentation
├── PHASE_2_README.md               # Detailed Phase 2 documentation
├── SETUP.md                        # Windows setup & run instructions
├── .gitignore                      # Git exclusion rules
└── requirements.txt                # Python package dependencies
```
