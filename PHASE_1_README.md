# Phase 1 — Dataset Preparation and Baseline Model Development

## 1. Objective
Phase 1 establishes a comprehensive baseline for the Intrusion Detection System project (*"An Efficient Intrusion Detection System Using Machine Learning and XGBoost-Based Feature Selection"*). It implements data validation, leakage-free preprocessing pipelines, and trains 5 baseline machine learning models across both Binary (Normal vs. Attack) and Multiclass (10 attack/normal categories) classification tasks using all 42 original UNSW-NB15 input features.

---

## 2. Dataset Used
- **Dataset Location**: `C:\Users\mvspr\Downloads\unswnb`
- **Training File**: `UNSW_NB15_training-set.csv` (175,341 rows × 45 columns)
- **Testing File**: `UNSW_NB15_testing-set.csv` (82,332 rows × 45 columns)
- **Total Records**: 257,673 rows

---

## 3. The 42 Original Input Features
The dataset contains 45 columns total: `id` (metadata identifier), 2 target columns (`label` and `attack_cat`), and **42 native input features**:

- **Categorical (3 Features)**: `proto`, `service`, `state`
- **Numerical (39 Features)**: `dur`, `spkts`, `dpkts`, `sbytes`, `dbytes`, `rate`, `sttl`, `dttl`, `sload`, `dload`, `sloss`, `dloss`, `sinpkt`, `dinpkt`, `sjit`, `djit`, `swin`, `stcpb`, `dtcpb`, `dwin`, `tcprtt`, `synack`, `ackdat`, `smean`, `dmean`, `trans_depth`, `response_body_len`, `ct_srv_src`, `ct_state_ttl`, `ct_dst_ltm`, `ct_src_dport_ltm`, `ct_dst_sport_ltm`, `ct_dst_src_ltm`, `is_ftp_login`, `ct_ftp_cmd`, `ct_flw_http_mthd`, `ct_src_ltm`, `ct_srv_dst`, `is_sm_ips_ports`

*Exclusions*: `id`, `label`, and `attack_cat` were strictly excluded from input feature matrices.

---

## 4. Preprocessing Pipeline & Data Leakage Prevention
Preprocessing is encapsulated in an `sklearn.compose.ColumnTransformer`:
- **Categorical Encoding**: `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`
- **Numerical Scaling**: `StandardScaler()`

> [!IMPORTANT]
> **Data Leakage Guarantee**: The `ColumnTransformer` and `LabelEncoder` are fitted **ONLY on the training set** (`X_train`). The testing set (`X_test`) is strictly transformed using the fitted transformers. `handle_unknown='ignore'` gracefully handles categories in testing (`ACC`, `CLO` in `state`) not present in training.

---

## 5. Classification Tasks
1. **Binary Classification**:
   - Target: `label` (0 = Normal, 1 = Attack)
2. **Multiclass Classification**:
   - Target: `attack_cat` (10 classes: Normal, Generic, Exploits, Fuzzers, DoS, Reconnaissance, Analysis, Backdoor, Shellcode, Worms)

---

## 6. Baseline ML Models Trained
1. **Decision Tree**: `DecisionTreeClassifier(max_depth=20, random_state=42)`
2. **Artificial Neural Network (ANN)**: Keras Sequential model (Dense 128 -> ReLU -> Dropout(0.2) -> Dense 64 -> ReLU -> Dropout(0.2) -> Output layer), trained with EarlyStopping.
3. **k-Nearest Neighbors (kNN)**: `KNeighborsClassifier(n_neighbors=5, n_jobs=-1)`
4. **Logistic Regression**: `LogisticRegression(max_iter=1000, random_state=42)`
5. **Support Vector Machine (SVM)**: `LinearSVC(dual=False, max_iter=1000, random_state=42)`

---

## 7. SVM Implementation Used
- **Implementation**: `sklearn.svm.LinearSVC(dual=False, max_iter=1000, random_state=42)`
- **Rationale**: Standard kernel `SVC` (RBF/poly) scales quadratically $O(N^2 \sim N^3)$ and is computationally infeasible on 175,341 training samples. `LinearSVC` with primal optimization (`dual=False`) offers linear time complexity $O(N \cdot d)$ on large sample sizes where $N > d$.

---

## 8. Evaluation Metrics
For all models, evaluation computes:
- Accuracy
- Precision (Attack & Macro/Weighted)
- Recall (Attack & Macro/Weighted)
- F1-Score (Attack & Macro/Weighted)
- Training & Prediction Execution Times (seconds)
- Confusion Matrix & Per-Class Classification Reports

---

## 9. Output Files & Directory Structure

```
c:\Users\mvspr\OneDrive\Desktop\IDS\
├── backend/                          # Modular Python source packages
│   ├── config.py
│   ├── dataset/loader.py
│   ├── preprocessing/pipeline.py
│   ├── models/ann_builder.py & trainers.py
│   ├── evaluation/metrics.py
│   └── utils/logger.py
├── models/                           # Trained model artifacts
│   ├── binary/                       # decision_tree.joblib, ann.keras, knn.joblib, logistic_regression.joblib, svm.joblib
│   └── multiclass/                   # decision_tree.joblib, ann.keras, knn.joblib, logistic_regression.joblib, svm.joblib
├── preprocessing_artifacts/          # Preprocessor & label encoder
│   ├── preprocessor.joblib
│   └── label_encoder.joblib
├── results/                          # Evaluation outputs
│   ├── binary_results.csv
│   ├── multiclass_results.csv
│   ├── classification_reports/       # Individual text classification reports per model
│   └── confusion_matrices/           # JSON confusion matrices per model
├── run_phase1.py                     # Main execution runner script
└── PHASE_1_README.md
```

---

## 10. How to Run Phase 1
To re-run the complete Phase 1 pipeline:
```bash
python run_phase1.py
```
