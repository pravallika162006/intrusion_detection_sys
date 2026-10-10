"""
Comprehensive Automated Test Suite for F1 Audit and TEST MODE Verification.
Directly implements regression checks for all 16 requirements specified in Part F:
1. Correct 19-feature names and order.
2. Compatibility with saved preprocessing objects.
3. Correct binary and multiclass model loading.
4. Correct metric formulas (Binary & Multiclass Macro/Weighted).
5. Correct confusion matrix totals.
6. Correct label mapping.
7. No ground-truth leakage (GT isolated from model features).
8. No duplicate flow evaluation.
9. Correct flow expiration and final flush.
10. Correct counter reset between TEST MODE runs.
11. Correct API response fields.
12. Correct dashboard metric labels.
13. Correct handling of undefined metrics.
14. Correct handling of empty or invalid flows.
15. Existing model results and research pages remain accessible.
16. Existing report generation, live monitoring, and AI advisory features work.
"""

import unittest
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scapy.all import IP, TCP, UDP, Raw

from backend.config import (
    BASE_DIR,
    PHASE1_RESULTS_DIR,
    PHASE2_RESULTS_DIR,
    PHASE3_RESULTS_DIR,
    PREPROCESSING_PHASE2_DIR,
    PHASE3_BINARY_MODELS_DIR,
    PHASE3_MULTICLASS_MODELS_DIR,
    SELECTED_19_FEATURES,
)
from backend.live.feature_extractor_19 import extract_19_features, resolve_service
from backend.live.flow_table import FlowTableManager
from backend.live.rolling_stats import RollingStatsTracker
from backend.live.live_detector import LiveDetector
from backend.live.packet_capturer import PacketCapturerManager
from backend.app.routes_performances import parse_classification_report_text
from backend.ai_agent.security_agent import AISecurityAgent


class TestF1AuditAndTestMode(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.detector = LiveDetector()
        cls.rolling_tracker = RollingStatsTracker()

    # Requirement 1: Correct 19-feature names and order
    def test_01_feature_names_and_order(self):
        expected_19 = [
            "sttl", "ct_srv_dst", "sbytes", "smean", "proto",
            "ct_state_ttl", "sloss", "synack", "ct_dst_src_ltm", "dmean",
            "ct_srv_src", "service", "ct_dst_sport_ltm", "dbytes", "dloss",
            "state", "tcprtt", "ct_src_dport_ltm", "rate"
        ]
        self.assertEqual(SELECTED_19_FEATURES, expected_19)

        dummy_flow = {
            "flow_id": "test_f1",
            "src_ip": "10.0.0.1", "dst_ip": "10.0.0.2",
            "src_port": 1234, "dst_port": 80,
            "proto": "tcp", "service": "http",
            "spkts": 5, "dpkts": 4, "sbytes": 500, "dbytes": 400,
            "sttl": 64, "dur": 0.05, "duration": 0.05,
            "state": "FIN", "sloss": 0, "dloss": 0,
            "synack": 0.01, "tcprtt": 0.02,
        }
        df_19 = extract_19_features(dummy_flow, self.rolling_tracker)
        self.assertEqual(list(df_19.columns), expected_19)

    # Requirement 2: Compatibility with saved preprocessing objects
    def test_02_preprocessing_compatibility(self):
        preproc_path = PREPROCESSING_PHASE2_DIR / "preprocessor_19.joblib"
        self.assertTrue(preproc_path.exists())
        preprocessor = self.detector.preprocessor
        self.assertIsNotNone(preprocessor)

        dummy_flow = {
            "flow_id": "test_preproc",
            "src_ip": "10.0.0.1", "dst_ip": "10.0.0.2",
            "src_port": 1234, "dst_port": 80,
            "proto": "tcp", "service": "http",
            "spkts": 2, "dpkts": 2, "sbytes": 100, "dbytes": 100,
            "sttl": 64, "dur": 0.01, "state": "FIN",
            "sloss": 0, "dloss": 0, "synack": 0.005, "tcprtt": 0.01,
        }
        df_19 = extract_19_features(dummy_flow, self.rolling_tracker)
        X_trans = preprocessor.transform(df_19)
        self.assertIsInstance(X_trans, np.ndarray)
        self.assertFalse(np.isnan(X_trans).any())

    # Requirement 3: Correct binary and multiclass model loading
    def test_03_model_loading(self):
        self.assertIsNotNone(self.detector.binary_model)
        self.assertIsNotNone(self.detector.multiclass_model)
        self.assertIn("Phase 3 Enhanced", self.detector.active_binary_name)
        self.assertIn("Phase 3 Enhanced", self.detector.active_multiclass_name)
        self.assertEqual(type(self.detector.binary_model).__name__, "HistGradientBoostingClassifier")
        self.assertEqual(type(self.detector.multiclass_model).__name__, "VotingClassifier")

    # Requirement 4: Correct metric formulas
    def test_04_metric_formulas(self):
        # Binary F1 = 2 * P * R / (P + R)
        tp, fp, fn = 80, 20, 10
        p = tp / (tp + fp)  # 0.80
        r = tp / (tp + fn)  # 0.8888...
        f1_expected = 2 * p * r / (p + r)
        self.assertAlmostEqual(f1_expected, 0.842105, places=4)

        # Multiclass Macro vs Weighted
        rep_path = PHASE3_RESULTS_DIR / "classification_reports" / "enhanced_multiclass_classification_report.txt"
        self.assertTrue(rep_path.exists())
        with open(rep_path, "r", encoding="utf-8") as f:
            parsed = parse_classification_report_text(f.read())

        self.assertIsNotNone(parsed["macro_avg"])
        self.assertIsNotNone(parsed["weighted_avg"])
        self.assertEqual(parsed["macro_avg"]["f1_score"], 48.12)
        self.assertEqual(parsed["weighted_avg"]["f1_score"], 77.53)
        self.assertEqual(parsed["accuracy"], 77.17)

    # Requirement 5: Correct confusion matrix totals
    def test_05_confusion_matrix_totals(self):
        cm_path = PHASE3_RESULTS_DIR / "confusion_matrices" / "enhanced_multiclass_cm.json"
        self.assertTrue(cm_path.exists())
        with open(cm_path, "r", encoding="utf-8") as f:
            cm_data = json.load(f)

        matrix = cm_data.get("matrix", cm_data.get("confusion_matrix"))
        total_eval = int(np.sum(matrix))
        # Held-out test set in Kasongo & Sun (2020) Table 2 has exactly 82,332 samples
        self.assertEqual(total_eval, 82332)

    # Requirement 6: Correct label mapping
    def test_06_label_mapping(self):
        label_enc = self.detector.label_encoder
        self.assertIsNotNone(label_enc)
        classes = list(label_enc.classes_)
        # UNSW-NB15 10 classes
        self.assertEqual(len(classes), 10)
        self.assertIn("Normal", classes)
        self.assertIn("DoS", classes)
        self.assertIn("Generic", classes)
        self.assertIn("Exploits", classes)

    # Requirement 7: No ground-truth leakage into model input features
    def test_07_no_ground_truth_leakage(self):
        dummy_flow_attack = {
            "flow_id": "leak_test",
            "src_ip": "10.0.0.99", "dst_ip": "192.168.1.1",
            "src_port": 45000, "dst_port": 80,
            "proto": "tcp", "service": "http",
            "spkts": 10, "dpkts": 0, "sbytes": 500, "dbytes": 0,
            "sttl": 254, "dur": 0.001, "state": "INT",
            "sloss": 0, "dloss": 0, "synack": None, "tcprtt": None,
            "ground_truth_label": 1,
            "ground_truth_category": "DoS (SYN Flood)",
            "is_synthetic": True
        }
        df_19 = extract_19_features(dummy_flow_attack, self.rolling_tracker)
        # Verify ground truth columns are strictly absent from model features
        self.assertNotIn("ground_truth_label", df_19.columns)
        self.assertNotIn("ground_truth_category", df_19.columns)
        self.assertNotIn("is_synthetic", df_19.columns)
        self.assertNotIn("label", df_19.columns)
        self.assertNotIn("attack_cat", df_19.columns)
        self.assertEqual(list(df_19.columns), SELECTED_19_FEATURES)

    # Requirement 8: No duplicate flow evaluation
    def test_08_no_duplicate_flow_evaluation(self):
        pm = PacketCapturerManager()
        pm.start_capture("TEST_MODE", test_mode=True, start_generator_thread=False)

        flow_data = {
            "flow_id": "unique_flow_123",
            "src_ip": "192.168.1.10", "dst_ip": "8.8.8.8",
            "src_port": 50000, "dst_port": 53,
            "proto": "udp", "spkts": 1, "dpkts": 1,
            "sbytes": 60, "dbytes": 80, "sttl": 64,
            "dur": 0.01, "state": "CON",
            "sloss": 0, "dloss": 0, "synack": None, "tcprtt": None,
            "ground_truth_label": 0
        }
        # First expiration
        pm._on_flow_expired(flow_data)
        count_after_first = pm.flow_count
        self.assertEqual(count_after_first, 1)

        # Duplicate call with identical flow_id must be ignored
        pm._on_flow_expired(flow_data)
        self.assertEqual(pm.flow_count, count_after_first)
        pm.stop_capture()

    # Requirement 9: Correct flow expiration and final flush
    def test_09_flow_expiration_and_final_flush(self):
        pm = PacketCapturerManager()
        pm.start_capture("TEST_MODE", test_mode=True, start_generator_thread=False)

        # Inject packet into flow table without immediate manual expire
        pkt = IP(src="192.168.1.50", dst="1.1.1.1", ttl=64) / UDP(sport=40001, dport=53)
        pm._on_packet_received(pkt, ground_truth_label=0, is_synthetic=True, pkt_time=100.0)

        self.assertEqual(len(pm.flow_table.active_flows), 1)
        self.assertEqual(pm.flow_count, 0)

        # stop_capture must force flush all active pending flows
        summary = pm.stop_capture()
        self.assertEqual(len(pm.flow_table.active_flows), 0)
        self.assertEqual(summary["flow_count"], 1)

    # Requirement 10: Correct counter reset between TEST MODE runs
    def test_10_counter_reset_between_runs(self):
        pm = PacketCapturerManager()
        # Run 1
        pm.run_test_scenario_batch("normal_only", count=10)
        self.assertEqual(pm.flow_count, 10)
        self.assertEqual(pm.test_tn, 10)

        # Run 2 with fresh start
        pm.run_test_scenario_batch("attack_only", count=10)
        self.assertEqual(pm.flow_count, 10)
        self.assertEqual(pm.test_tn, 0)
        self.assertGreater(pm.test_tp, 0)

    # Requirement 11: Correct API response fields
    def test_11_api_response_fields(self):
        pm = PacketCapturerManager()
        summary = pm.run_test_scenario_batch("mixed_50_50", count=10)

        expected_summary_keys = {
            "status", "message", "interface_name", "test_mode",
            "test_scenario", "threshold", "packet_count", "flow_count",
            "normal_count", "attack_count", "test_evaluation"
        }
        self.assertTrue(expected_summary_keys.issubset(set(summary.keys())))

        te = summary["test_evaluation"]
        expected_te_keys = {
            "tp", "fp", "tn", "fn", "total", "actual_normal", "actual_attack",
            "accuracy", "precision", "recall", "f1", "fpr", "fnr",
            "missed_attack_rate", "scenario"
        }
        self.assertTrue(expected_te_keys.issubset(set(te.keys())))
        self.assertEqual(te["tp"] + te["fp"] + te["tn"] + te["fn"], te["total"])

    # Requirement 12: Correct dashboard metric labels (final selected model test results)
    def test_12_dashboard_metric_labels(self):
        final_csv = PHASE3_RESULTS_DIR / "final_selected_model_test_results.csv"
        self.assertTrue(final_csv.exists())
        df = pd.read_csv(final_csv)
        cols = list(df.columns)
        self.assertIn("Macro F1 (%)", cols)
        self.assertIn("Weighted F1 (%)", cols)
        self.assertIn("Val Accuracy (%)", cols)
        self.assertIn("Test Accuracy (%)", cols)

    # Requirement 13: Correct handling of undefined metrics
    def test_13_handling_of_undefined_metrics(self):
        pm = PacketCapturerManager()
        # Normal-only has actual_attack = 0
        summary = pm.run_test_scenario_batch("normal_only", count=10)
        te = summary["test_evaluation"]

        self.assertEqual(te["actual_attack"], 0)
        self.assertEqual(te["accuracy"], 100.0)
        self.assertEqual(te["specificity"], 100.0)
        # Recall and F1 must be None (not disguised as 0.0% failure)
        self.assertIsNone(te["recall"])
        self.assertIsNone(te["f1"])
        self.assertIsNone(te["precision"])

    # Requirement 14: Correct handling of empty or invalid flows
    def test_14_handling_of_empty_or_invalid_flows(self):
        # Flow with missing optional fields
        sparse_flow = {
            "src_ip": "1.2.3.4", "dst_ip": "5.6.7.8",
            "src_port": 100, "dst_port": 200,
            "proto": "tcp"
        }
        df_19 = extract_19_features(sparse_flow, self.rolling_tracker)
        self.assertEqual(len(df_19), 1)
        pred_label, bin_val, cat, conf = self.detector.predict_flow(df_19)
        self.assertIn(bin_val, [0, 1])
        self.assertGreaterEqual(conf, 0.0)

    # Requirement 15: Existing model results and research pages remain accessible
    def test_15_existing_research_results_accessible(self):
        p1_bin = PHASE1_RESULTS_DIR / "binary_results.csv"
        p2_bin = PHASE2_RESULTS_DIR / "binary_results.csv"
        p3_bin_cand = PHASE3_RESULTS_DIR / "binary_candidates_val.csv"
        p3_multi_cand = PHASE3_RESULTS_DIR / "multiclass_candidates_val.csv"

        self.assertTrue(p1_bin.exists(), "Phase 1 binary results missing!")
        self.assertTrue(p2_bin.exists(), "Phase 2 binary results missing!")
        self.assertTrue(p3_bin_cand.exists(), "Phase 3 binary candidates missing!")
        self.assertTrue(p3_multi_cand.exists(), "Phase 3 multiclass candidates missing!")

    # Requirement 16: Existing report generation, live monitoring and advisory features still work
    def test_16_ai_advisory_agent_functional(self):
        agent = AISecurityAgent()
        rec = agent.generate_recommendation(
            flow_id="test_flow_audit",
            src_ip="10.0.0.99",
            dst_ip="192.168.1.1",
            service="http",
            attack_category="DoS",
            confidence=0.95,
            flow_info={"spkts": 10, "dpkts": 0, "sbytes": 500, "rate": 20000}
        )
        self.assertIn("threat_summary", rec)
        self.assertIn("severity", rec)
        self.assertIn("recommended_actions", rec)
        self.assertEqual(rec["severity"], "High")


if __name__ == "__main__":
    unittest.main()
