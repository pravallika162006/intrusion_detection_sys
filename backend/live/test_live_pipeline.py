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


if __name__ == "__main__":
    unittest.main()