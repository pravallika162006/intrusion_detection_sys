"""
Asynchronous Packet Capturer supporting Live Sniffing and Controlled TEST MODE.
Integrates Scapy, FlowTableManager, RollingStatsTracker, and LiveDetector.
Provides rigorous ground-truth evaluation isolated from model inputs.
"""

import time
import random
import threading
from typing import Optional, Callable, Dict, Any, List, Set
from scapy.all import IP, TCP, UDP, Raw, AsyncSniffer

from backend.live.flow_table import FlowTableManager
from backend.live.rolling_stats import RollingStatsTracker
from backend.live.feature_extractor_19 import extract_19_features, resolve_service
from backend.live.live_detector import LiveDetector
from backend.utils.logger import setup_logger

logger = setup_logger("PacketCapturer")


class PacketCapturerManager:
    """
    Manages live packet capture threads, flow state, and live predictions.
    Supports both real Live Network Capture and Controlled TEST MODE.
    """

    def __init__(self):
        self.interface_name: Optional[str] = None
        self.is_running = False
        self.stop_requested = False
        self.test_mode = False
        self.test_scenario = "mixed_75_25"

        self.flow_table = FlowTableManager(idle_timeout=8.0)
        self.rolling_tracker = RollingStatsTracker(max_history=100)
        self.detector = LiveDetector()

        self.sniffer: Optional[AsyncSniffer] = None
        self.test_thread: Optional[threading.Thread] = None
        self.cleanup_thread: Optional[threading.Thread] = None

        self.packet_count = 0
        self.flow_count = 0
        self.normal_count = 0
        self.attack_count = 0

        # Controlled TEST MODE Ground-Truth Evaluation Metrics (strictly isolated from model inputs)
        self.test_tp = 0  # GT=Attack (1), Pred=Attack (1)
        self.test_fp = 0  # GT=Normal (0), Pred=Attack (1)
        self.test_tn = 0  # GT=Normal (0), Pred=Normal (0)
        self.test_fn = 0  # GT=Attack (1), Pred=Normal (0)
        self.evaluated_flow_ids: Set[str] = set()
        self.test_records: List[Dict[str, Any]] = []
        self.decision_threshold: float = 0.50

        self.event_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        self.flow_table.flow_expired_callback = self._on_flow_expired

    def set_threshold(self, threshold: float):
        """Dynamically adjusts the operating decision threshold."""
        self.decision_threshold = max(0.10, min(0.99, float(threshold)))
        logger.info(f"Operating decision threshold updated to {self.decision_threshold:.2f}")

    def get_test_evaluation_summary(self) -> Optional[Dict[str, Any]]:
        """Calculates current evaluation metrics from completed flows with ground truth."""
        total_eval = self.test_tp + self.test_fp + self.test_tn + self.test_fn
        if total_eval == 0 and not self.test_mode:
            return None

        actual_normal = self.test_tn + self.test_fp
        actual_attack = self.test_tp + self.test_fn

        # Strict metric handling with explicit conventions for undefined scenarios
        # Precision: Undefined if no flows are predicted positive (TP + FP == 0)
        prec_val = (self.test_tp / (self.test_tp + self.test_fp)) if (self.test_tp + self.test_fp) > 0 else (0.0 if actual_attack > 0 else None)
        # Recall: Undefined if there are zero attack instances in ground truth (actual_attack == 0, e.g. Normal-Only)
        rec_val = (self.test_tp / actual_attack) if actual_attack > 0 else None
        # F1: Undefined if precision or recall is undefined, or their sum is zero
        f1_val = (2 * prec_val * rec_val / (prec_val + rec_val)) if (prec_val is not None and rec_val is not None and (prec_val + rec_val) > 0) else None
        # Specificity / True Negative Rate (Benign detection rate)
        tnr_val = (self.test_tn / actual_normal) if actual_normal > 0 else None
        fpr_val = (self.test_fp / actual_normal) if actual_normal > 0 else 0.0
        fnr_val = (self.test_fn / actual_attack) if actual_attack > 0 else 0.0
        acc_val = ((self.test_tp + self.test_tn) / total_eval) if total_eval > 0 else 0.0

        return {
            "tp": self.test_tp,
            "fp": self.test_fp,
            "tn": self.test_tn,
            "fn": self.test_fn,
            "total": total_eval,
            "actual_normal": actual_normal,
            "actual_attack": actual_attack,
            "accuracy": round(acc_val * 100, 2),
            "precision": round(prec_val * 100, 2) if prec_val is not None else None,
            "recall": round(rec_val * 100, 2) if rec_val is not None else None,
            "f1": round(f1_val * 100, 2) if f1_val is not None else None,
            "specificity": round(tnr_val * 100, 2) if tnr_val is not None else None,
            "fpr": round(fpr_val * 100, 2),
            "fnr": round(fnr_val * 100, 2),
            "missed_attack_rate": round(fnr_val * 100, 2),
            "scenario": getattr(self, "test_scenario", "mixed_75_25"),
        }

    def start_capture(
        self,
        interface_name: str,
        test_mode: bool = False,
        test_scenario: str = "mixed_75_25",
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        threshold: float = 0.50,
        start_generator_thread: bool = True,
    ):
        """Starts packet capture on specified interface or launches Controlled TEST MODE."""
        if self.is_running:
            self.stop_capture()

        self.interface_name = interface_name
        self.test_mode = test_mode
        self.test_scenario = test_scenario
        self.event_callback = event_callback
        self.decision_threshold = float(threshold)
        self.is_running = True
        self.stop_requested = False

        # Session Reset
        self.packet_count = 0
        self.flow_count = 0
        self.normal_count = 0
        self.attack_count = 0

        self.test_tp = 0
        self.test_fp = 0
        self.test_tn = 0
        self.test_fn = 0
        self.evaluated_flow_ids.clear()
        self.test_records = []

        self.flow_table.clear()
        self.rolling_tracker.clear()

        # Start periodic flow expiration cleanup thread only for live streaming mode
        if start_generator_thread:
            self.cleanup_thread = threading.Thread(target=self._expiration_loop, daemon=True)
            self.cleanup_thread.start()

        if test_mode:
            logger.info(f"Starting Controlled TEST MODE generator [scenario={test_scenario}]...")
            if start_generator_thread:
                self.test_thread = threading.Thread(target=self._run_test_mode_loop, daemon=True)
                self.test_thread.start()
        else:
            logger.info(f"Starting Live Sniffer on interface '{interface_name}'...")
            try:
                self.sniffer = AsyncSniffer(
                    iface=interface_name if interface_name != "default" else None,
                    prn=self._on_packet_received,
                    store=False,
                )
                self.sniffer.start()
            except Exception as e:
                self.is_running = False
                logger.error(f"Failed to start packet capture on interface {interface_name}: {e}")
                raise RuntimeError(
                    f"Capture failed on interface '{interface_name}'. Ensure Npcap is installed and run with administrator permissions if required. Error: {str(e)}"
                )

    def stop_capture(self) -> Dict[str, Any]:
        """Stops packet capture, flushes all pending flows, and returns final session summary."""
        self.stop_requested = True
        if self.sniffer:
            try:
                self.sniffer.stop()
            except Exception:
                pass
            self.sniffer = None

        if self.test_thread and self.test_thread.is_alive():
            self.test_thread.join(timeout=0.5)

        # Flush and evaluate ALL pending active flows before shutting down
        try:
            self.flow_table.expire_idle_flows(force=True)
        except Exception as e:
            logger.error(f"Error expiring pending flows on stop: {e}")

        self.is_running = False

        test_eval = self.get_test_evaluation_summary()
        summary = {
            "status": "STOPPED",
            "message": "Network monitoring stopped.",
            "interface_name": self.interface_name,
            "test_mode": self.test_mode,
            "test_scenario": getattr(self, "test_scenario", "mixed_75_25"),
            "threshold": self.decision_threshold,
            "packet_count": self.packet_count,
            "flow_count": self.flow_count,
            "normal_count": self.normal_count,
            "attack_count": self.attack_count,
            "test_evaluation": test_eval,
        }
        logger.info(f"Packet capture stopped. Total flows evaluated: {self.flow_count}")
        return summary

    def _on_packet_received(
        self,
        packet,
        ground_truth_label: Optional[int] = None,
        ground_truth_category: Optional[str] = None,
        is_synthetic: bool = False,
        pkt_time: Optional[float] = None,
    ):
        if not self.is_running and not is_synthetic:
            return
        self.packet_count += 1
        try:
            self.flow_table.process_packet(
                packet,
                ground_truth_label=ground_truth_label,
                ground_truth_category=ground_truth_category,
                is_synthetic=is_synthetic,
                pkt_time=pkt_time,
            )
        except Exception as e:
            logger.debug(f"Error processing packet: {e}")

    def _on_flow_expired(self, flow_data: Dict[str, Any]):
        """Called when a flow completes or expires in the flow table."""
        flow_id = flow_data.get("flow_id")
        if not flow_id:
            flow_id = f"flow_{self.flow_count + 1}_{int(time.time()*1000)}"
            flow_data["flow_id"] = flow_id

        # Deduplication: Ensure each flow is evaluated exactly once
        if flow_id in self.evaluated_flow_ids:
            return
        self.evaluated_flow_ids.add(flow_id)

        self.flow_count += 1

        flow_data["service"] = resolve_service(
            flow_data.get("dst_port", 0),
            flow_data.get("src_port", 0),
            flow_data.get("first_payload", b""),
            flow_data.get("proto", "tcp"),
        )

        # Extract 19 features & query rolling tracker (GROUND TRUTH IS NEVER IN df_19)
        df_19 = extract_19_features(flow_data, self.rolling_tracker, align_ttl=True)

        # Run Live Preprocessor & Model Inference independently
        pred_label, bin_val, attack_cat, confidence = self.detector.predict_flow(
            df_19, threshold=self.decision_threshold
        )

        if bin_val == 1:
            self.attack_count += 1
        else:
            self.normal_count += 1

        # Update rolling stats tracker with completed flow summary
        self.rolling_tracker.record_flow(flow_data)

        # Ground truth evaluation tracking (isolated from model features)
        gt_label = flow_data.get("ground_truth_label")
        gt_category = flow_data.get("ground_truth_category")

        if gt_label is not None:
            if gt_label == 1:
                if bin_val == 1:
                    self.test_tp += 1
                else:
                    self.test_fn += 1
            elif gt_label == 0:
                if bin_val == 1:
                    self.test_fp += 1
                else:
                    self.test_tn += 1

            self.test_records.append({
                "flow_id": flow_id,
                "gt": gt_label,
                "gt_cat": gt_category,
                "pred": bin_val,
                "confidence": confidence,
                "threshold": self.decision_threshold,
                "state": df_19.iloc[0]["state"],
                "proto": df_19.iloc[0]["proto"],
                "sttl": int(df_19.iloc[0]["sttl"]),
                "ct_state_ttl": int(df_19.iloc[0]["ct_state_ttl"]),
                "rate": float(df_19.iloc[0]["rate"]),
                "sbytes": int(df_19.iloc[0]["sbytes"]),
                "dbytes": int(df_19.iloc[0]["dbytes"]),
                "service": df_19.iloc[0]["service"],
            })

        test_eval = self.get_test_evaluation_summary()

        # Determine severity and operational status
        if bin_val == 1:
            if confidence >= 0.85:
                severity = "High" if attack_cat in ["DoS", "Exploits", "Backdoor", "Shellcode"] else "Medium"
                status = "High Confidence Intrusion"
            elif confidence >= 0.65:
                severity = "Medium"
                status = "Potential Anomaly"
            else:
                severity = "Low"
                status = "Suspicious (Review Needed)"
        else:
            severity = "Informational"
            status = "Normal Traffic"

        flow_stats_dict = {
            "spkts": flow_data["spkts"],
            "dpkts": flow_data["dpkts"],
            "sbytes": flow_data["sbytes"],
            "dbytes": flow_data["dbytes"],
            "duration": flow_data.get("dur", flow_data.get("duration", 0.0)),
            "sttl": flow_data["sttl"],
        }

        # Construct Event Payload for WebSocket / UI
        flow_event = {
            "flow_id": flow_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "flow_key": f"{flow_data['src_ip']}:{flow_data['src_port']} -> {flow_data['dst_ip']}:{flow_data['dst_port']} ({flow_data['proto']})",
            "src_ip": flow_data["src_ip"],
            "dst_ip": flow_data["dst_ip"],
            "src_port": flow_data["src_port"],
            "dst_port": flow_data["dst_port"],
            "proto": flow_data["proto"],
            "service": flow_data["service"],
            "prediction": pred_label,
            "binary_label": bin_val,
            "binary_class": bin_val,
            "attack_category": attack_cat,
            "confidence": confidence,
            "severity": severity,
            "status": status,
            "flow_stats": flow_stats_dict,
            "flow_info": flow_stats_dict,
            "system_totals": {
                "packet_count": self.packet_count,
                "flow_count": self.flow_count,
                "normal_count": self.normal_count,
                "attack_count": self.attack_count,
            },
            "is_test_mode": self.test_mode,
            "is_synthetic": flow_data.get("is_synthetic", self.test_mode),
            "ground_truth_label": gt_label,
            "ground_truth_category": gt_category,
            "test_evaluation": test_eval,
            "decision_threshold": self.decision_threshold,
        }

        if self.event_callback:
            self.event_callback(flow_event)

    def _expiration_loop(self):
        """Background thread expiring idle flows periodically."""
        while self.is_running and not self.stop_requested:
            time.sleep(1.5)
            try:
                self.flow_table.expire_idle_flows()
            except Exception as e:
                logger.error(f"Error during flow expiration loop: {e}")

    def _generate_synthetic_flow(self, is_attack: bool, base_time: Optional[float] = None):
        """
        Generates realistic benign or attack traffic with known ground truth.
        Uses exact protocol dynamics matching the UNSW-NB15 benchmark reference frame.
        """
        t = base_time if base_time is not None else time.time()
        sp = 30000 + (self.flow_count % 30000)

        if is_attack:
            atk_type = random.choice(["dos", "exploit", "generic", "fuzzers", "recon"])
            if atk_type == "dos":
                gt_cat = "DoS (SYN Flood)"
                flood_port = random.choice([8080, 8888, 9000, 1337, 31337])
                num_pkts = random.randint(6, 10)
                burst_delta = random.uniform(0.00003, 0.00006)
                for j in range(num_pkts):
                    self._on_packet_received(
                        IP(src="10.0.0.99", dst="192.168.1.1", ttl=254)/TCP(sport=sp, dport=flood_port, flags="S", seq=5000+j),
                        ground_truth_label=1, ground_truth_category=gt_cat, is_synthetic=True, pkt_time=t + j * burst_delta
                    )
            elif atk_type == "exploit":
                gt_cat = "Web Exploit (SQL Injection)"
                self._on_packet_received(IP(src="10.0.0.99", dst="192.168.1.1", ttl=254)/TCP(sport=sp, dport=80, flags="S", seq=1000), ground_truth_label=1, ground_truth_category=gt_cat, is_synthetic=True, pkt_time=t)
                self._on_packet_received(IP(src="192.168.1.1", dst="10.0.0.99", ttl=64)/TCP(sport=80, dport=sp, flags="SA", seq=2000, ack=1001), ground_truth_label=1, ground_truth_category=gt_cat, is_synthetic=True, pkt_time=t+0.01)
                self._on_packet_received(IP(src="10.0.0.99", dst="192.168.1.1", ttl=254)/TCP(sport=sp, dport=80, flags="A", seq=1001, ack=2001), ground_truth_label=1, ground_truth_category=gt_cat, is_synthetic=True, pkt_time=t+0.02)
                p_load = b"GET /admin?id=1' UNION SELECT username,password FROM users-- HTTP/1.1\r\n\r\n"
                self._on_packet_received(IP(src="10.0.0.99", dst="192.168.1.1", ttl=254)/TCP(sport=sp, dport=80, flags="PA", seq=1001, ack=2001)/Raw(p_load), ground_truth_label=1, ground_truth_category=gt_cat, is_synthetic=True, pkt_time=t+0.03)
                r_load = b"HTTP/1.1 500 Internal Server Error\r\n\r\nSQL Exception"
                self._on_packet_received(IP(src="192.168.1.1", dst="10.0.0.99", ttl=64)/TCP(sport=80, dport=sp, flags="PA", seq=2001, ack=1001+len(p_load))/Raw(r_load), ground_truth_label=1, ground_truth_category=gt_cat, is_synthetic=True, pkt_time=t+0.04)
                self._on_packet_received(IP(src="10.0.0.99", dst="192.168.1.1", ttl=254)/TCP(sport=sp, dport=80, flags="FA", seq=1001+len(p_load), ack=2001+len(r_load)), ground_truth_label=1, ground_truth_category=gt_cat, is_synthetic=True, pkt_time=t+0.05)
                self._on_packet_received(IP(src="192.168.1.1", dst="10.0.0.99", ttl=64)/TCP(sport=80, dport=sp, flags="FA", seq=2001+len(r_load), ack=1002+len(p_load)), ground_truth_label=1, ground_truth_category=gt_cat, is_synthetic=True, pkt_time=t+0.06)
                self._on_packet_received(IP(src="10.0.0.99", dst="192.168.1.1", ttl=254)/TCP(sport=sp, dport=80, flags="A", seq=1002+len(p_load), ack=2002+len(r_load)), ground_truth_label=1, ground_truth_category=gt_cat, is_synthetic=True, pkt_time=t+0.07)
            elif atk_type == "generic":
                gt_cat = "Generic (UDP Flood)"
                for j in range(8):
                    self._on_packet_received(
                        IP(src="10.0.0.99", dst="8.8.8.8", ttl=254)/UDP(sport=sp, dport=53)/Raw(b"\xaa\xbb\x01\x00\x00\x01probe.domain"),
                        ground_truth_label=1, ground_truth_category=gt_cat, is_synthetic=True, pkt_time=t + j*0.0005
                    )
            elif atk_type == "fuzzers":
                gt_cat = "Fuzzers (Malformed TCP)"
                self._on_packet_received(IP(src="10.0.0.99", dst="192.168.1.1", ttl=254)/TCP(sport=sp, dport=80, flags="S", seq=100), ground_truth_label=1, ground_truth_category=gt_cat, is_synthetic=True, pkt_time=t)
                self._on_packet_received(IP(src="192.168.1.1", dst="10.0.0.99", ttl=64)/TCP(sport=80, dport=sp, flags="SA", seq=200, ack=101), ground_truth_label=1, ground_truth_category=gt_cat, is_synthetic=True, pkt_time=t+0.01)
                self._on_packet_received(IP(src="10.0.0.99", dst="192.168.1.1", ttl=254)/TCP(sport=sp, dport=80, flags="A", seq=101, ack=201), ground_truth_label=1, ground_truth_category=gt_cat, is_synthetic=True, pkt_time=t+0.02)
                f_load = b"POST /api/upload HTTP/1.1\r\nContent-Length: 999999\r\n\r\n" + b"\xff" * 200
                self._on_packet_received(IP(src="10.0.0.99", dst="192.168.1.1", ttl=254)/TCP(sport=sp, dport=80, flags="PA", seq=101, ack=201)/Raw(f_load), ground_truth_label=1, ground_truth_category=gt_cat, is_synthetic=True, pkt_time=t+0.03)
                self._on_packet_received(IP(src="10.0.0.99", dst="192.168.1.1", ttl=254)/TCP(sport=sp, dport=80, flags="FA", seq=101+len(f_load), ack=201), ground_truth_label=1, ground_truth_category=gt_cat, is_synthetic=True, pkt_time=t+0.04)
                self._on_packet_received(IP(src="192.168.1.1", dst="10.0.0.99", ttl=64)/TCP(sport=80, dport=sp, flags="FA", seq=201, ack=102+len(f_load)), ground_truth_label=1, ground_truth_category=gt_cat, is_synthetic=True, pkt_time=t+0.05)
                self._on_packet_received(IP(src="10.0.0.99", dst="192.168.1.1", ttl=254)/TCP(sport=sp, dport=80, flags="A", seq=102+len(f_load), ack=202), ground_truth_label=1, ground_truth_category=gt_cat, is_synthetic=True, pkt_time=t+0.06)
            else:  # recon
                gt_cat = "Port Scan (Reconnaissance)"
                dp = random.choice([22, 23, 3306, 8080])
                self._on_packet_received(IP(src="10.0.0.99", dst="192.168.1.1", ttl=254)/TCP(sport=sp, dport=dp, flags="S", seq=999), ground_truth_label=1, ground_truth_category=gt_cat, is_synthetic=True, pkt_time=t)
                self._on_packet_received(IP(src="192.168.1.1", dst="10.0.0.99", ttl=64)/TCP(sport=dp, dport=sp, flags="SA", seq=1999, ack=1000), ground_truth_label=1, ground_truth_category=gt_cat, is_synthetic=True, pkt_time=t+0.01)
                self._on_packet_received(IP(src="10.0.0.99", dst="192.168.1.1", ttl=254)/TCP(sport=sp, dport=dp, flags="FA", seq=1000, ack=2000), ground_truth_label=1, ground_truth_category=gt_cat, is_synthetic=True, pkt_time=t+0.02)
                self._on_packet_received(IP(src="192.168.1.1", dst="10.0.0.99", ttl=64)/TCP(sport=dp, dport=sp, flags="FA", seq=2000, ack=1001), ground_truth_label=1, ground_truth_category=gt_cat, is_synthetic=True, pkt_time=t+0.03)
                self._on_packet_received(IP(src="10.0.0.99", dst="192.168.1.1", ttl=254)/TCP(sport=sp, dport=dp, flags="A", seq=1001, ack=2001), ground_truth_label=1, ground_truth_category=gt_cat, is_synthetic=True, pkt_time=t+0.04)
        else:
            ben_type = random.choice(["http", "https", "dns"])
            if ben_type == "dns":
                gt_cat = "Benign DNS"
                self._on_packet_received(IP(src="192.168.1.15", dst="8.8.8.8", ttl=64)/UDP(sport=sp, dport=53)/Raw(b"\x00\x01query.test"), ground_truth_label=0, ground_truth_category=gt_cat, is_synthetic=True, pkt_time=t)
                self._on_packet_received(IP(src="8.8.8.8", dst="192.168.1.15", ttl=64)/UDP(sport=53, dport=sp)/Raw(b"\x00\x01\x81\x80query.test.answer"), ground_truth_label=0, ground_truth_category=gt_cat, is_synthetic=True, pkt_time=t+0.01)
            elif ben_type == "https":
                gt_cat = "Benign HTTPS (TLS)"
                self._on_packet_received(IP(src="192.168.1.15", dst="142.250.190.46", ttl=64)/TCP(sport=sp, dport=443, flags="S", seq=100), ground_truth_label=0, ground_truth_category=gt_cat, is_synthetic=True, pkt_time=t)
                self._on_packet_received(IP(src="142.250.190.46", dst="192.168.1.15", ttl=64)/TCP(sport=443, dport=sp, flags="SA", seq=200, ack=101), ground_truth_label=0, ground_truth_category=gt_cat, is_synthetic=True, pkt_time=t+0.01)
                self._on_packet_received(IP(src="192.168.1.15", dst="142.250.190.46", ttl=64)/TCP(sport=sp, dport=443, flags="A", seq=101, ack=201), ground_truth_label=0, ground_truth_category=gt_cat, is_synthetic=True, pkt_time=t+0.02)
                t_load = b"\x16\x03\x01\x00\xa5\x01\x00\x00\xa1\x03\x03TLS_Client_Hello"
                self._on_packet_received(IP(src="192.168.1.15", dst="142.250.190.46", ttl=64)/TCP(sport=sp, dport=443, flags="PA", seq=101, ack=201)/Raw(t_load), ground_truth_label=0, ground_truth_category=gt_cat, is_synthetic=True, pkt_time=t+0.03)
                s_load = b"\x16\x03\x03\x00\x5a\x02\x00\x00\x56\x03\x03TLS_Server_Hello"
                self._on_packet_received(IP(src="142.250.190.46", dst="192.168.1.15", ttl=64)/TCP(sport=443, dport=sp, flags="PA", seq=201, ack=101+len(t_load))/Raw(s_load), ground_truth_label=0, ground_truth_category=gt_cat, is_synthetic=True, pkt_time=t+0.04)
                self._on_packet_received(IP(src="192.168.1.15", dst="142.250.190.46", ttl=64)/TCP(sport=sp, dport=443, flags="FA", seq=101+len(t_load), ack=201+len(s_load)), ground_truth_label=0, ground_truth_category=gt_cat, is_synthetic=True, pkt_time=t+0.05)
                self._on_packet_received(IP(src="142.250.190.46", dst="192.168.1.15", ttl=64)/TCP(sport=443, dport=sp, flags="FA", seq=201+len(s_load), ack=102+len(t_load)), ground_truth_label=0, ground_truth_category=gt_cat, is_synthetic=True, pkt_time=t+0.06)
                self._on_packet_received(IP(src="192.168.1.15", dst="142.250.190.46", ttl=64)/TCP(sport=sp, dport=443, flags="A", seq=102+len(t_load), ack=202+len(s_load)), ground_truth_label=0, ground_truth_category=gt_cat, is_synthetic=True, pkt_time=t+0.07)
            else:
                gt_cat = "Benign Web (HTTP)"
                self._on_packet_received(IP(src="192.168.1.15", dst="93.184.216.34", ttl=64)/TCP(sport=sp, dport=80, flags="S", seq=100), ground_truth_label=0, ground_truth_category=gt_cat, is_synthetic=True, pkt_time=t)
                self._on_packet_received(IP(src="93.184.216.34", dst="192.168.1.15", ttl=64)/TCP(sport=80, dport=sp, flags="SA", seq=200, ack=101), ground_truth_label=0, ground_truth_category=gt_cat, is_synthetic=True, pkt_time=t+0.01)
                self._on_packet_received(IP(src="192.168.1.15", dst="93.184.216.34", ttl=64)/TCP(sport=sp, dport=80, flags="A", seq=101, ack=201), ground_truth_label=0, ground_truth_category=gt_cat, is_synthetic=True, pkt_time=t+0.02)
                h_req = b"GET /index.html HTTP/1.1\r\nHost: example.com\r\n\r\n"
                self._on_packet_received(IP(src="192.168.1.15", dst="93.184.216.34", ttl=64)/TCP(sport=sp, dport=80, flags="PA", seq=101, ack=201)/Raw(h_req), ground_truth_label=0, ground_truth_category=gt_cat, is_synthetic=True, pkt_time=t+0.03)
                h_resp = b"HTTP/1.1 200 OK\r\nContent-Length: 12\r\n\r\nHello World!"
                self._on_packet_received(IP(src="93.184.216.34", dst="192.168.1.15", ttl=64)/TCP(sport=80, dport=sp, flags="PA", seq=201, ack=101+len(h_req))/Raw(h_resp), ground_truth_label=0, ground_truth_category=gt_cat, is_synthetic=True, pkt_time=t+0.04)
                self._on_packet_received(IP(src="192.168.1.15", dst="93.184.216.34", ttl=64)/TCP(sport=sp, dport=80, flags="FA", seq=101+len(h_req), ack=201+len(h_resp)), ground_truth_label=0, ground_truth_category=gt_cat, is_synthetic=True, pkt_time=t+0.05)
                self._on_packet_received(IP(src="93.184.216.34", dst="192.168.1.15", ttl=64)/TCP(sport=80, dport=sp, flags="FA", seq=201+len(h_resp), ack=102+len(h_req)), ground_truth_label=0, ground_truth_category=gt_cat, is_synthetic=True, pkt_time=t+0.06)
                self._on_packet_received(IP(src="192.168.1.15", dst="93.184.216.34", ttl=64)/TCP(sport=sp, dport=80, flags="A", seq=102+len(h_req), ack=202+len(h_resp)), ground_truth_label=0, ground_truth_category=gt_cat, is_synthetic=True, pkt_time=t+0.07)

        # Immediately finalize the flow
        self.flow_table.expire_idle_flows(force=True)

    def _run_test_mode_loop(self):
        """
        Continuous streaming worker for UI demonstration.
        Smooth pacing (0.35 - 0.6s) enables visual tracking in the web dashboard.
        """
        flow_counter = 0
        scenario = getattr(self, "test_scenario", "mixed_75_25")
        logger.info(f"TEST MODE live generator running scenario: {scenario}")

        while self.is_running and not self.stop_requested:
            flow_counter += 1

            if scenario == "normal_only":
                is_attack_sim = False
            elif scenario == "attack_only":
                is_attack_sim = True
            elif scenario == "mixed_50_50":
                is_attack_sim = (flow_counter % 2 == 0)
            else:  # mixed_75_25
                is_attack_sim = (flow_counter % 4 == 0)

            self._generate_synthetic_flow(is_attack_sim, base_time=time.time())

            # Interactive pacing for UI dashboard display
            sleep_duration = random.uniform(0.35, 0.60)
            sleep_end = time.time() + sleep_duration
            while time.time() < sleep_end:
                if self.stop_requested or not self.is_running:
                    break
                time.sleep(0.05)

    def run_test_scenario_batch(self, scenario: str, count: int = 100) -> Dict[str, Any]:
        """
        Executes a rapid, repeatable validation batch of flows with simulated timestamps.
        Guarantees exact flow counts and authoritative metric calculation.
        """
        logger.info(f"Executing batch test scenario: {scenario} (count={count})")
        self.start_capture(
            interface_name="TEST_MODE",
            test_mode=True,
            test_scenario=scenario,
            threshold=self.decision_threshold,
            start_generator_thread=False,
        )

        base_time = 10000.0
        for i in range(1, count + 1):
            if scenario == "normal_only":
                is_attack = False
            elif scenario == "attack_only":
                is_attack = True
            elif scenario == "mixed_50_50":
                is_attack = (i % 2 == 0)
            else:  # mixed_75_25
                is_attack = (i % 4 == 0)

            flow_time = base_time + i * 2.0
            self._generate_synthetic_flow(is_attack, base_time=flow_time)

        # Finalize and return authoritative session summary
        summary = self.stop_capture()
        return summary


def resolve_service_simple(dst_port: int) -> str:
    return resolve_service(dst_port, 0, b"")
