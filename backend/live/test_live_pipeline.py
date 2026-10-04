import time
import unittest

from scapy.all import IP, TCP, Raw

from backend.config import SELECTED_19_FEATURES
from backend.live.feature_extractor_19 import extract_19_features, resolve_service
from backend.live.flow_key import FlowKey
from backend.live.flow_table import FlowRecord
from backend.live.rolling_stats import RollingStatsTracker


class LivePipelineTests(unittest.TestCase):
    def test_service_categories(self):
        self.assertEqual(resolve_service(80, 50000, b""), "http")
        self.assertEqual(resolve_service(443, 50000, b""), "ssl")
        self.assertEqual(resolve_service(50000, 443, b""), "ssl")
        self.assertEqual(resolve_service(31337, 50000, b""), "-")

    def test_midstream_server_packet_is_oriented_by_service_port(self):
        record = FlowRecord(FlowKey("10.0.0.1", "10.0.0.2", 50000, 443, "tcp"), time.time(), 64)
        record.update(IP(src="10.0.0.2", dst="10.0.0.1") / TCP(sport=443, dport=50000, flags="PA", seq=10) / Raw(load=b"x"), 1.0)
        self.assertEqual(record.forward_port, 50000)
        self.assertEqual(record.dpkts, 1)
        self.assertEqual(record.spkts, 0)
        self.assertEqual(record.direction_source, "port_inference")
        summary = record.to_dict()
        self.assertFalse(summary["synack_available"])
        self.assertFalse(summary["tcprtt_available"])

    def test_syn_reorients_provisional_flow_and_ack_is_not_loss(self):
        record = FlowRecord(FlowKey("10.0.0.2", "10.0.0.1", 443, 50000, "tcp"), time.time(), 64)
        record.update(IP(src="10.0.0.2", dst="10.0.0.1") / TCP(sport=443, dport=50000, flags="PA", seq=10) / Raw(load=b"x"), 1.0)
        record.update(IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=50000, dport=443, flags="S", seq=100), 2.0)
        record.update(IP(src="10.0.0.2", dst="10.0.0.1") / TCP(sport=443, dport=50000, flags="SA", seq=200, ack=101), 2.1)
        record.update(IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=50000, dport=443, flags="A", seq=101, ack=201), 2.2)
        record.update(IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=50000, dport=443, flags="A", seq=101, ack=201), 2.3)
        self.assertEqual(record.forward_port, 50000)
        self.assertEqual(record.sloss, 0)
        self.assertEqual(record.dloss, 0)
        self.assertTrue(record.synack_available)
        self.assertTrue(record.tcprtt_available)

    def test_feature_order_and_ssl_value(self):
        flow = {
            "src_ip": "10.0.0.1", "dst_ip": "10.0.0.2", "src_port": 50000,
            "dst_port": 443, "proto": "tcp", "sttl": 128, "sttl_available": True,
            "spkts": 2, "dpkts": 3, "sbytes": 120, "dbytes": 240,
            "state": "CON", "sloss": 0, "dloss": 0, "synack": None,
            "tcprtt": None, "trans_depth": 0, "first_payload": b"",
        }
        frame = extract_19_features(flow, RollingStatsTracker())
        self.assertEqual(list(frame.columns), SELECTED_19_FEATURES)
        self.assertEqual(frame.iloc[0]["service"], "ssl")
        self.assertEqual(frame.iloc[0]["synack"], 0.0)
        self.assertEqual(frame.iloc[0]["tcprtt"], 0.0)

    def test_ct_state_ttl_tracking_consistency(self):
        tracker = RollingStatsTracker()
        flow1 = {
            "src_ip": "10.0.0.1", "dst_ip": "10.0.0.2", "src_port": 50001,
            "dst_port": 443, "proto": "tcp", "sttl": 64, "sttl_available": True,
            "spkts": 2, "dpkts": 2, "sbytes": 100, "dbytes": 200, "state": "CON", "dur": 0.1
        }
        df1 = extract_19_features(flow1, tracker, align_ttl=True)
        self.assertEqual(df1.iloc[0]["ct_state_ttl"], 0)
        tracker.record_flow(flow1)

        flow2 = {
            "src_ip": "10.0.0.1", "dst_ip": "10.0.0.2", "src_port": 50002,
            "dst_port": 443, "proto": "tcp", "sttl": 64, "sttl_available": True,
            "spkts": 2, "dpkts": 2, "sbytes": 100, "dbytes": 200, "state": "CON", "dur": 0.2
        }
        df2 = extract_19_features(flow2, tracker, align_ttl=True)
        self.assertEqual(df2.iloc[0]["ct_state_ttl"], 1)

    def test_modern_web_port_direction_inference(self):
        record = FlowRecord(FlowKey("1.2.3.4", "10.0.0.1", 8080, 50000, "tcp"), time.time(), 64)
        record.update(IP(src="1.2.3.4", dst="10.0.0.1") / TCP(sport=8080, dport=50000, flags="PA", seq=10) / Raw(load=b"HTTP/1.1"), 1.0)
        self.assertEqual(record.forward_ip, "10.0.0.1")
        self.assertEqual(record.forward_port, 50000)
        self.assertEqual(record.dpkts, 1)

    def test_udp_dns_direction_inference(self):
        from scapy.all import UDP
        record = FlowRecord(FlowKey("8.8.8.8", "10.0.0.1", 53, 53210, "udp"), time.time(), 64)
        record.update(IP(src="8.8.8.8", dst="10.0.0.1") / UDP(sport=53, dport=53210) / Raw(load=b"\x81\x80"), 1.0)
        self.assertEqual(record.forward_ip, "10.0.0.1")
        self.assertEqual(record.forward_port, 53210)
        self.assertEqual(record.dpkts, 1)

    def test_trailing_ack_orphan_prevention(self):
        from backend.live.flow_table import FlowTableManager
        mgr = FlowTableManager(idle_timeout=5.0)
        expired = []
        mgr.flow_expired_callback = lambda f: expired.append(f)

        src, dst = "10.0.0.1", "10.0.0.2"
        sp, dp = 50000, 80

        mgr.process_packet(IP(src=src, dst=dst)/TCP(sport=sp, dport=dp, flags="S", seq=100))
        mgr.process_packet(IP(src=dst, dst=src)/TCP(sport=dp, dport=sp, flags="SA", seq=200, ack=101))
        mgr.process_packet(IP(src=src, dst=dst)/TCP(sport=sp, dport=dp, flags="A", seq=101, ack=201))
        mgr.process_packet(IP(src=src, dst=dst)/TCP(sport=sp, dport=dp, flags="FA", seq=101, ack=201))
        mgr.process_packet(IP(src=dst, dst=src)/TCP(sport=dp, dport=sp, flags="FA", seq=201, ack=102))
        mgr.process_packet(IP(src=src, dst=dst)/TCP(sport=sp, dport=dp, flags="A", seq=102, ack=202))

        mgr.expire_idle_flows(force=True)
        # Trailing duplicate arriving after flow close
        mgr.process_packet(IP(src=src, dst=dst)/TCP(sport=sp, dport=dp, flags="A", seq=102, ack=202))

        self.assertEqual(len(expired), 1)
        self.assertEqual(len(mgr.active_flows), 0)

    def test_zero_duration_rate_safety(self):
        flow_zero = {
            "src_ip": "10.0.0.1", "dst_ip": "10.0.0.2", "src_port": 50000,
            "dst_port": 80, "proto": "tcp", "sttl": 64, "spkts": 2, "dpkts": 0,
            "sbytes": 100, "dbytes": 0, "state": "CON", "dur": 0.0
        }
        df_zero = extract_19_features(flow_zero, RollingStatsTracker())
        self.assertEqual(df_zero.iloc[0]["rate"], 0.0)

        flow_short = {
            "src_ip": "10.0.0.1", "dst_ip": "10.0.0.2", "src_port": 50000,
            "dst_port": 80, "proto": "tcp", "sttl": 64, "spkts": 5, "dpkts": 5,
            "sbytes": 500, "dbytes": 500, "state": "CON", "dur": 0.5
        }
        df_short = extract_19_features(flow_short, RollingStatsTracker())
        self.assertEqual(df_short.iloc[0]["rate"], 20.0)

    def test_ground_truth_isolation_from_model_features(self):
        flow_with_gt = {
            "src_ip": "10.0.0.1", "dst_ip": "10.0.0.2", "src_port": 50000,
            "dst_port": 80, "proto": "tcp", "sttl": 64, "spkts": 2, "dpkts": 0,
            "sbytes": 100, "dbytes": 0, "state": "CON", "dur": 0.1,
            "ground_truth_label": 1, "ground_truth_category": "DoS Attack", "is_synthetic": True
        }
        df = extract_19_features(flow_with_gt, RollingStatsTracker())
        # Model input must contain ONLY the 19 features, zero ground truth leakage
        self.assertEqual(list(df.columns), SELECTED_19_FEATURES)
        self.assertNotIn("ground_truth_label", df.columns)
        self.assertNotIn("ground_truth_category", df.columns)
        self.assertNotIn("is_synthetic", df.columns)


if __name__ == "__main__":
    unittest.main()