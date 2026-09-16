# An Efficient Intrusion Detection System Using Machine Learning and XGBoost-Based Feature Selection

A production-grade, dual-dashboard Intrusion Detection System (IDS) combining machine learning baseline research on the **UNSW-NB15** dataset with **XGBoost-based feature selection**, pre-trained model inference, real-time **passive Wi-Fi network packet capture**, WebSocket streaming, and an **AI Security Agent**.

---

## 1. Project Overview
Network Intrusion Detection Systems (NIDS) are critical defense mechanisms against malicious cyber traffic. Modern high-speed networks generate vast quantities of packet data with high-dimensional feature spaces, creating significant computational bottlenecks for real-time detection. 

This project implements a complete end-to-end framework across three core phases:
* **Phase 1**: Baseline research on the full 42-feature UNSW-NB15 dataset using standard machine learning models.
* **Phase 2**: Dimensionality reduction using XGBoost feature selection to isolate the top **19 original features**, achieving superior classification performance with reduced overhead.
* **Phase 3**: Deployment of a web-based application supporting dual operation modes: **Dataset Analysis & Prediction** and **Real-Time Live Network Monitoring** with an integrated AI Security Agent.

---

## 2. Problem Statement
High-dimensional network traffic features increase computational latency and memory consumption in real-time intrusion detection systems. Furthermore, raw network packet streams lack ground-truth labels, requiring robust flow construction, feature derivation, and compatible preprocessing pipelines to feed pre-trained machine learning models without feature schema mismatches.

---

## 3. Objectives
1. **Develop Baseline Models**: Train and evaluate binary and multiclass classifiers on the full 42-feature UNSW-NB15 dataset (Phase 1).
2. **Optimize Feature Selection**: Apply XGBoost importance scoring on training data to select 19 optimal features (Phase 2).
3. **Build Dataset Dashboard**: Provide batch CSV/Parquet dataset analysis supporting both 42-feature and 19-feature modes with evaluation metrics for labelled data and prediction summaries for unlabelled data.
4. **Implement Live Network Pipeline**: Construct a real-time flow engine capturing passive network packets (Scapy/Npcap), deriving the 19 features, and running pre-trained model inference.
5. **Integrate AI Security Agent**: Provide non-destructive contextual threat explanations, severity ratings, investigation guidance, and security recommendations for detected attacks.

---

## 4. System Architecture

```
                                      +------------------------------------+
                                      |          FastAPI Server            |
                                      |    (backend/app/main.py)           |
                                      +-----------------+------------------+
                                                        |
                         +------------------------------+------------------------------+
                         |                                                             |
         +---------------+---------------+                             +---------------+---------------+
         |    Mode 1: Dataset Dashboard  |                             |   Mode 2: Live Network Dashboard  |
         +---------------+---------------+                             +---------------+---------------+
                         |                                                             |
          - File Upload (CSV/Parquet)                                    - Interface Discovery (Scapy)
          - Schema Validation (42 vs 19)                                 - Async Sniffer (Ethernet/Wi-Fi)
          - Phase 1 / Phase 2 Model Load                                 - Flow State Manager & Expiration
          - Batch Feature Preprocessing                                  - Live 19-Feature Calculator
          - Evaluation (Metrics / CM / ROC)                              - Phase 2 Pipeline Transformation
          - Prediction-Only Distribution                                 - Trained Phase 2 Model Inference
                         |                                               - WebSocket Real-Time Stream
                         v                                                             |
         +---------------+---------------+                                             v
         |   Dataset Analysis Output UI  |                             +---------------+---------------+
         +-------------------------------+                             |   Live Traffic & Alert Stream |
                                                                       +---------------+---------------+
                                                                                       |
                                                                                       v
                                                                       +---------------+---------------+
                                                                       |       AI Security Agent       |
                                                                       |  (backend/ai_agent/agent.py)  |
                                                                       +---------------+---------------+
                                                                        - Contextual Threat Analysis
                                                                        - Attack Explanation & Severity
                                                                        - Actionable Security Guidance
```

---

## 5. Phase 1 — UNSW-NB15 Baseline Model Development

### Dataset Details
* **Source**: Official UNSW-NB15 Training and Testing sets.
* **Training Records**: 175,341 rows
* **Testing Records**: 82,332 rows
* **Feature Schema**: 42 native input features (3 Categorical: `proto`, `service`, `state`; 39 Numerical). Excluded metadata: `id`, `label`, `attack_cat`.

### Preprocessing Pipeline
* `MinMaxScaler(feature_range=(0, 1))` fitted exclusively on numerical training features (Paper-aligned scaling).
* `OneHotEncoder(handle_unknown='ignore')` fitted on categorical features.
* `LabelEncoder` fitted on multiclass target labels.

### Frozen Phase 1 Baseline Results

| Task | Model | Accuracy | F1-Score | Notes |
|---|---|---|---|---|
| **Binary** | **Decision Tree** | **86.85%** | **88.91%** | Baseline tree classifier |
| Binary | ANN | 85.63% | 88.22% | Keras EarlyStopping |
| Binary | kNN (k=5) | 84.51% | 87.27% | Euclidean distance |
| Binary | LinearSVC | 81.22% | 83.10% | Practical SVM baseline |
| Binary | Logistic Regression | 80.97% | 84.91% | L2 penalty |
| **Multiclass** | **ANN** | **75.68%** | **75.20%** | Multiclass deep neural network |
| Multiclass | Decision Tree | 74.76% | 74.30% | Multiclass tree |
| Multiclass | kNN | 70.90% | 70.10% | Multiclass kNN |
| Multiclass | Logistic Regression | 69.81% | 68.50% | Multiclass logistic regression |
| Multiclass | LinearSVC | 68.41% | 67.20% | Multiclass LinearSVC |

*Note: RBF Kernel SVM from the reference paper was computationally impractical on available hardware; LinearSVC was retained as the practical baseline.*

---

## 6. Phase 2 — XGBoost Feature Selection & Reduced Models

### Selection Methodology
XGBoost feature importance scoring was conducted strictly on training data (`df_train`) to rank all 42 input features. The top **19 original features** were selected, capturing 89.47% overlap with planned theoretical rankings.

### Audited 19 Selected Features (`results/phase2/selected_features_19.json`)
1. `sttl`: Source to destination Time to Live
2. `proto`: Transaction protocol (tcp, udp, icmp, etc.)
3. `ct_srv_dst`: No. of connections containing same service & dst IP in last 100 flows
4. `service`: Application service (http, ssl, dns, ftp, smtp, ssh, etc.)
5. `sbytes`: Source to destination transaction bytes
6. `smean`: Mean packet size transmitted by source (`sbytes / spkts`)
7. `ct_dst_sport_ltm`: No. of connections of same dst IP & src port in last 100 flows
8. `state`: Connection state (CON, FIN, INT, REQ, RST, ACC)
9. `dpkts`: Destination to source packet count
10. `sloss`: Source packets retransmitted/dropped
11. `synack`: TCP SYN to SYN-ACK duration (seconds)
12. `ct_dst_src_ltm`: No. of connections of same dst IP & src IP in last 100 flows
13. `dmean`: Mean packet size transmitted by destination (`dbytes / dpkts`)
14. `dbytes`: Destination to source transaction bytes
15. `trans_depth`: HTTP pipelined request depth
16. `ct_state_ttl`: No. of connections with same state & TTL in last 100 flows
17. `ct_srv_src`: No. of connections of same service & src IP in last 100 flows
18. `dloss`: Destination packets retransmitted/dropped
19. `tcprtt`: TCP Round Trip Time (`synack` + `ackdat`)

---

## 7. Final Project Models

### Binary Classifier
* **Architecture**: **XGBoost-Based Feature Selection + Decision Tree Classifier** using the 19 selected features.
* **Clarification**: XGBoost is utilized strictly for **Feature Selection**. Decision Tree performs the final binary classification. This is **not** a true ensemble model.
* **Official Phase 2 Binary Result**: **87.33% Accuracy** | **89.32% Attack F1-Score**.

### Multiclass Classifier
* **Architecture**: **Artificial Neural Network (ANN)** trained on the 19 selected features.
* **Official Phase 2 Multiclass Result**: **76.31% Accuracy** | **44.44% Macro F1-Score**.

### Model Artifacts Location
* Binary Model: `models/phase2/binary/xgboost_dt.joblib`
* Multiclass Model: `models/phase2/multiclass/ann.keras`
* Preprocessor: `preprocessing_artifacts/phase2/preprocessor_19.joblib` (Transforms 19 features -> **171 dense/sparse features**)
* Label Encoder: `preprocessing_artifacts/phase2/label_encoder_19.joblib`

---

## 8. Phase 3 — Dual Application Dashboards & Real-Time Monitoring

The application exposes two independent operational dashboards:

### Dashboard 1 — Dataset Analysis & Prediction
* **Input Sources**: Built-in UNSW-NB15 testing dataset or user-uploaded CSV / Parquet files.
* **Schema Validation**: Checks for required features and highlights missing columns without crashing on invalid files.
* **Feature & Model Selection**: Toggle between 42-feature (Phase 1) and 19-feature (Phase 2) modes across all trained models.
* **Labelled Evaluation Mode**: Computes Accuracy, Precision, Recall, F1, Macro F1, Weighted F1, Confusion Matrix, and Classification Report when ground-truth labels exist.
* **Unlabelled Prediction Mode**: Generates record-level predictions, Normal vs. Attack totals, attack percentage, confidence distribution, and category breakdowns without calculating fake accuracy metrics.

### Dashboard 2 — Live Network Monitoring
* **Interface Discovery**: Scapy adapter discovery listing active Wi-Fi, Ethernet, and Npcap loopback interfaces.
* **Real-Time Metrics**: Live packet count, active flow count, normal traffic total, and attack alert counter.
* **Controlled TEST MODE**: Generates synthetic Scapy packets locally to safely test flow creation, 19-feature extraction, preprocessor transformation, model inference, WebSocket streaming, and AI recommendations without generating real malicious traffic.
* **WebSocket Streaming**: Real-time event feed updating the UI dynamically without page refreshes.
* **AI Security Agent**: Contextual threat analysis drawer.

---

## 9. Live Monitoring Pipeline Architecture

```
Network Interface
       ↓
Packet Capture (Scapy / Npcap)
       ↓
Flow Construction (5-Tuple Key: src_ip, dst_ip, src_port, dst_port, proto)
       ↓
19 Feature Extraction (header fields, TCP state, rolling stats, timing)
       ↓
Phase 2 Preprocessing (preprocessor_19.joblib -> 171 features)
       ↓
Trained ML Model (xgboost_dt.joblib / ann.keras)
       ↓
Prediction (Normal / Attack & Category)
       ↓
Alert Generation
       ↓
AI Security Agent (Recommendation-only advice)
       ↓
Live Dashboard (WebSocket UI update)
```

---

## 10. Important Live Traffic Limitation Notice
* **Ground-Truth Label Absence**: Real live network traffic does **not** contain ground-truth labels.
* **Prediction Distribution vs. Accuracy**: Live Normal/Attack counts and percentages represent model **prediction distributions** on observed network flows. They must **NOT** be interpreted as live accuracy or true positive rates (TPR).

---

## 11. Frozen Benchmark Results

### Official Phase 2 Final Results (19 Selected Features)
* **Binary Classification (XGBoost + Decision Tree)**:
  * **Accuracy**: **87.33%**
  * **Attack F1-Score**: **89.32%**
* **Multiclass Classification (ANN)**:
  * **Accuracy**: **76.31%**
  * **Macro F1-Score**: **44.44%**

---

## 12. Installation & Prerequisites

### Prerequisites
* **Python**: `3.12+`
* **Packet Capture Driver (Windows)**: **Npcap** (installed with WinPcap API compatibility mode enabled).

### Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/pravallika162006/intrusion_detection_sys.git
   cd intrusion_detection_sys
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 13. How to Run the Backend Server
Start the FastAPI server using Uvicorn:
```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```
* Backend API Documentation: `http://localhost:8000/docs`
* API Health Check: `http://localhost:8000/api/health`

---

## 14. How to Run the Frontend Web Application
The single-page web application is automatically served by FastAPI at the root URL:
* Open Google Chrome and navigate to: **`http://localhost:8000/`**

---

## 15. Windows / Npcap Requirements for Live Monitoring
To capture live packets on Windows interfaces:
1. Ensure **Npcap** is installed.
2. If running packet capture on raw network adapters, launch the terminal with **Administrator privileges** if required by Windows security policies.

---

## 16. Controlled TEST MODE Instructions
1. Open `http://localhost:8000/` in Chrome.
2. Click the **Live Monitoring Dashboard** tab.
3. Check the **Controlled TEST MODE** checkbox.
4. Click **Start Monitoring**.
5. Observe synthetic packet flows, real-time counters, prediction table rows, and click **AI Advice** to open the Security Agent drawer.

---

## 17. Project Structure

```
C:\projects\IDS\
├── backend/
│   ├── ai_agent/
│   │   ├── security_agent.py        # AI Security Agent & fallback rule database
│   │   └── prompt_templates.py      # Structured threat analysis prompts
│   ├── app/
│   │   ├── main.py                  # FastAPI server entry point
│   │   ├── routes_dataset.py        # Dataset Dashboard REST APIs
│   │   ├── routes_live.py           # Live Dashboard REST APIs & WebSocket
│   │   ├── schemas.py               # Pydantic data validation schemas
│   │   └── websocket_manager.py     # Real-time WebSocket connection manager
│   ├── live/
│   │   ├── feature_extractor_19.py  # Real-time calculator for 19 Phase 2 features
│   │   ├── flow_key.py              # Bidirectional 5-tuple flow identification
│   │   ├── flow_table.py            # Active flow state table & idle timeout cleanup
│   │   ├── interface_manager.py     # Network adapter discovery (Scapy / netifaces)
│   │   ├── live_detector.py         # Phase 2 model loader and inference engine
│   │   ├── packet_capturer.py       # Scapy AsyncSniffer & Controlled TEST MODE
│   │   └── rolling_stats.py         # Rolling historical statistics for ct_* features
│   ├── phase2/                      # Phase 2 feature selection & training modules
│   ├── preprocessing/               # Phase 1 preprocessing pipeline
│   └── config.py                    # Global paths, constants & feature registries
├── frontend/
│   ├── css/styles.css               # Dashboard stylesheet
│   ├── js/app.js                    # Single-page UI state controller & WebSocket handler
│   └── index.html                   # Main dual-dashboard HTML container
├── models/                          # Saved binary & multiclass model artifacts
├── preprocessing_artifacts/         # Saved fitted preprocessors & label encoders
├── results/                         # Frozen reports, confusion matrices, and plots
├── requirements.txt                 # Project Python dependencies
├── PHASE_1_README.md
├── PHASE_2_README.md
└── SETUP.md
```

---

## 18. Limitations
1. **Live Data Provenance**: Features requiring historical connection counts (`ct_*`) are calculated over a rolling in-memory window of active/recent flows rather than multi-gigabyte historical database logs.
2. **Unlabelled Live Accuracy**: Accuracy and F1 metrics cannot be computed on unlabelled live traffic without ground-truth labels.
3. **No Automatic Destructive Remediation**: The AI Security Agent operates in recommendation-only mode to prevent unintended system disruption.

---

## 19. Future Scope
* **Hardware Acceleration**: Integration with DPDK or eBPF for multi-gigabit packet processing.
* **Distributed Agent Nodes**: Deploying lightweight edge probes with centralized flow aggregation.
* **Automated Playbook Integration**: User-configurable firewall rules (iptables / Windows Firewall) after analyst confirmation.

---

## 20. License & Acknowledgments
* **Dataset Credit**: UNSW-NB15 Dataset created by the Cyber Range Lab of Australian Centre for Cyber Security (ACCS).
* **Project Framework**: Developed for final-year Machine Learning & Cyber Security research.
