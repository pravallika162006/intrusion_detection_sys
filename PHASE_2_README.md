# Phase 2 — XGBoost Feature Selection and 19-Feature Model Evaluation

## 1. Objective
Phase 2 implements XGBoost-based feature selection to rank all 42 original UNSW-NB15 input features, select the top 19 features, build a dedicated 19-feature preprocessing pipeline, train and evaluate 14 baseline & hybrid model experiments (7 Binary, 7 Multiclass), and perform a direct performance delta comparison between 42-feature and 19-feature representations.

---

## 2. XGBoost Feature Selection Methodology
- **Training Restriction**: XGBoost feature importance calculation was performed **ONLY on the training set** (`X_train`, `y_train`) using `xgboost.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)`.
- **Feature Aggregation**: Feature importance scores from one-hot encoded dummy columns were mapped back to their 42 parent raw features.
- **Ranking**: All 42 original features were ranked from 1 to 42 based on normalized gain/importance scores.

---

## 3. All 42 Feature Importance Ranking & Selected Top 19

XGBoost was used for feature selection. The selected 19 original features were then used by the downstream classifiers.

### Top 19 Selected Features (in Rank Order):
1. `sttl` (Score: 0.637336)
2. `proto` (Score: 0.051366)
3. `ct_srv_dst` (Score: 0.047136)
4. `service` (Score: 0.030254)
5. `sbytes` (Score: 0.028651)
6. `smean` (Score: 0.025092)
7. `ct_dst_sport_ltm` (Score: 0.023571)
8. `dpkts` (Score: 0.016175)
9. `state` (Score: 0.015914)
10. `sloss` (Score: 0.012573)
11. `synack` (Score: 0.011140)
12. `ct_dst_src_ltm` (Score: 0.009525)
13. `dmean` (Score: 0.009126)
14. `dbytes` (Score: 0.008247)
15. `trans_depth` (Score: 0.008136)
16. `ct_state_ttl` (Score: 0.007621)
17. `ct_srv_src` (Score: 0.006969)
18. `dloss` (Score: 0.005667)
19. `tcprtt` (Score: 0.005196)

### Comparison with Planned / Literature 19 Features:
- **Common Features (17/19 - 89.47% Overlap)**: `sttl`, `proto`, `ct_srv_dst`, `service`, `sbytes`, `smean`, `ct_dst_sport_ltm`, `state`, `sloss`, `synack`, `ct_dst_src_ltm`, `dmean`, `dbytes`, `ct_state_ttl`, `ct_srv_src`, `dloss`, `tcprtt`.
- **Selected by XGBoost (Not in Planned 19)**: `dpkts`, `trans_depth`.
- **In Planned 19 (Not in XGBoost Top 19)**: `ct_src_dport_ltm`, `rate`.

---

## 4. 19-Feature Preprocessing Pipeline
- **Categorical Features (3)**: `proto`, `service`, `state`
- **Numerical Features (16)**: `sttl`, `ct_srv_dst`, `sbytes`, `smean`, `ct_dst_sport_ltm`, `dpkts`, `sloss`, `synack`, `ct_dst_src_ltm`, `dmean`, `dbytes`, `trans_depth`, `ct_state_ttl`, `ct_srv_src`, `dloss`, `tcprtt`
- **Transformers**: `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` + `StandardScaler()`
- **Leakage Prevention**: Fitted **ONLY on training set** (`X19_train`). Testing set (`X19_test`) is transformed using the fitted preprocessor.

---

## 5. Model Approaches & Hybrid Methodology
XGBoost was used for feature selection. The selected 19 original features were then used by the downstream classifiers.

For both Binary (`label`) and Multiclass (`attack_cat`) tasks, 7 models were evaluated:
1. **Decision Tree**: Standard Decision Tree trained on the selected 19 features.
2. **ANN**: Keras Sequential ANN trained on the selected 19 features with EarlyStopping.
3. **kNN**: 5-Nearest Neighbors trained on the selected 19 features.
4. **Logistic Regression**: Logistic Regression trained on the selected 19 features.
5. **SVM**: `LinearSVC(dual=False, max_iter=1000, random_state=42)` trained on the selected 19 features.
6. **XGBoost + Decision Tree**: Explicitly designated as **"XGBoost feature selection + Decision Tree"** (Sequential 2-stage approach: Stage 1 = XGBoost Top-19 feature selection -> Stage 2 = Decision Tree classifier trained on the 19 selected features).
7. **XGBoost + kNN**: Explicitly designated as **"XGBoost feature selection + kNN"** (Sequential 2-stage approach: Stage 1 = XGBoost Top-19 feature selection -> Stage 2 = kNN classifier trained on the 19 selected features).

*Note: Neither `XGBoost + Decision Tree` nor `XGBoost + kNN` are voting, stacking, or ensemble models. They are sequential feature selection + classifier pipelines.*

---

## 6. Phase 1 vs Phase 2 Delta Key Findings
- **Feature Reduction**: Reduced feature dimensionality by **54.76%** (from 42 features to 19 features).
- **Binary Decision Tree**: Accuracy increased from 86.85% (42 features) to **87.33%** (19 features); F1 increased from 88.91% to **89.32%** (+0.48% Acc, +0.41% F1).
- **Binary kNN**: Accuracy increased from 84.51% (42 features) to **86.64%** (19 features); F1 increased from 87.27% to **88.84%** (+2.13% Acc, +1.57% F1).
- **Multiclass Decision Tree**: Accuracy increased from 74.76% to **76.10%**; Macro F1 increased from 50.05% to **50.99%**.
- **Multiclass ANN**: Accuracy increased from 75.68% to **76.31%**; Macro F1 increased from 43.65% to **44.44%**.
- **Training Time Savings**: Multiclass LinearSVC training time plummeted from **649.24s (42 features)** down to **120.16s (19 features)** — a **529-second (81.5%) training speedup**.

---

## 7. Artifacts & Directory Structure

```
c:\Users\mvspr\OneDrive\Desktop\IDS\
├── backend/
│   ├── phase2/
│   │   ├── feature_selector.py
│   │   ├── visualizer.py
│   │   ├── pipeline_19.py
│   │   ├── trainers_19.py
│   │   └── comparator.py
├── models/
│   └── phase2/
│       ├── binary/                 # 7 binary trained model joblib/keras files
│       └── multiclass/             # 7 multiclass trained model joblib/keras files
├── preprocessing_artifacts/
│   └── phase2/                     # 19-feature preprocessor & label encoder
├── results/
│   ├── binary_results.csv          # Untouched Phase 1 binary results
│   ├── multiclass_results.csv      # Untouched Phase 1 multiclass results
│   └── phase2/
│       ├── binary_results.csv      # 7 Phase-2 binary results
│       ├── multiclass_results.csv  # 7 Phase-2 multiclass results
│       ├── comparison_42_vs_19.csv # 42 vs 19 delta comparison table
│       ├── xgboost_feature_importance.csv
│       ├── xgboost_feature_ranking.csv
│       ├── selected_features_19.json
│       ├── classification_reports/
│       ├── confusion_matrices/
│       └── plots/                  # XGBoost feature importance bar charts
├── run_phase2.py                   # Main Phase 2 execution runner
└── PHASE_2_README.md
```

---

## 8. How to Run Phase 2
To re-run the complete Phase 2 pipeline:
```bash
python run_phase2.py
```
