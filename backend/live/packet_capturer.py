"""
Asynchronous Packet Capturer supporting Live Sniffing and Controlled TEST MODE.
Integrates Scapy, FlowTableManager, RollingStatsTracker, and LiveDetector.
"""

import time
import random
import threading
from typing import Optional, Callable, Dict, Any
from scapy.all import sniff, IP, TCP, UDP, ICMP, Ether, Raw, AsyncSniffer, conf

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
        self.test_mode = False

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
        self.test_tp = 0  # GT=Attack, Pred=Attack
        self.test_fp = 0  # GT=Normal, Pred=Attack
        self.test_tn = 0  # GT=Normal, Pred=Normal
        self.test_fn = 0  # GT=Attack, Pred=Normal
        self.test_records = []
        self.decision_threshold: float = 0.80

        self.event_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        self.flow_table.flow_expired_callback = self._on_flow_expired

    def set_threshold(self, threshold: float):
        """Dynamically adjusts the live operating decision threshold."""
        self.decision_threshold = max(0.10, min(0.99, float(threshold)))
        logger.info(f"Operating decision threshold updated to {self.decision_threshold:.2f}")

    def start_capture(
        self,
        interface_name: str,
        test_mode: bool = False,
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        threshold: float = 0.80,
    ):
        """Starts packet capture on specified interface or launches Controlled TEST MODE."""
        if self.is_running:
            self.stop_capture()

        self.interface_name = interface_name
        self.test_mode = test_mode
        self.event_callback = event_callback
        self.decision_threshold = float(threshold)
        self.is_running = True

        self.packet_count = 0
        self.flow_count = 0
        self.normal_count = 0
        self.attack_count = 0

        self.test_tp = 0
        self.test_fp = 0
        self.test_tn = 0
        self.test_fn = 0
        self.test_records = []

        self.flow_table.clear()
        self.rolling_tracker.clear()

        # Start periodic flow expiration cleanup thread
        self.cleanup_thread = threading.Thread(target=self._expiration_loop, daemon=True)
        self.cleanup_thread.start()

        if test_mode:
            logger.info("Starting Controlled TEST MODE packet generator...")
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

    def stop_capture(self):
        """Stops packet capture and clears threads."""
        self.is_running = False
        if self.sniffer:
            try:
                self.sniffer.stop()
            except Exception:
                pass
            self.sniffer = None
        logger.info("Packet capture stopped successfully.")

    def _on_packet_received(
        self,
        packet,
        ground_truth_label: Optional[int] = None,
        ground_truth_category: Optional[str] = None,
        is_synthetic: bool = False,
    ):
        if not self.is_running:
            return
        self.packet_count += 1
        try:
            self.flow_table.process_packet(
                packet,
                ground_truth_label=ground_truth_label,
                ground_truth_category=ground_truth_category,
                is_synthetic=is_synthetic,
            )
        except Exception as e:
            logger.debug(f"Error processing packet: {e}")

    def _on_flow_expired(self, flow_data: Dict[str, Any]):
        """Called when a flow completes or expires in the flow table."""
        if not self.is_running:
            return

        self.flow_count += 1

        flow_data["service"] = resolve_service(
            flow_data.get("dst_port", 0),
            flow_data.get("src_port", 0),
            flow_data.get("first_payload", b""),
            flow_data.get("proto", "tcp"),
        )

        # Extract 19 features & query rolling tracker (with benchmark reference alignment)
        df_19 = extract_19_features(flow_data, self.rolling_tracker, align_ttl=True)

        # Run Live Preprocessor & Model Inference (Ground truth is NEVER in df_19)
        pred_label, bin_val, attack_cat, confidence = self.detector.predict_flow(
            df_19, threshold=self.decision_threshold
        )

        if bin_val == 1:
            self.attack_count += 1
        else:
            self.normal_count += 1

        # Update rolling stats tracker with flow summary
        self.rolling_tracker.record_flow(flow_data)

        # Ground truth evaluation tracking (TEST MODE only)
        gt_label = flow_data.get("ground_truth_label")
        gt_category = flow_data.get("ground_truth_category")
        test_eval = None

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

            total_eval = self.test_tp + self.test_fp + self.test_tn + self.test_fn
            if total_eval > 0:
                test_eval = {
                    "tp": self.test_tp,
                    "fp": self.test_fp,
                    "tn": self.test_tn,
                    "fn": self.test_fn,
                    "total": total_eval,
                    "accuracy": round((self.test_tp + self.test_tn) / total_eval * 100, 2),
                    "fpr": round(self.test_fp / max(1, self.test_fp + self.test_tn) * 100, 2),
                    "fnr": round(self.test_fn / max(1, self.test_tp + self.test_fn) * 100, 2),
                    "precision": round(self.test_tp / max(1, self.test_tp + self.test_fp) * 100, 2),
                    "recall": round(self.test_tp / max(1, self.test_tp + self.test_fn) * 100, 2),
                    "f1": round(
                        2 * self.test_tp / max(1, 2 * self.test_tp + self.test_fp + self.test_fn) * 100, 2
                    ),
                }

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
            "flow_id": f"flow_{self.flow_count}_{int(time.time()*1000)}",
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
            "flow_info": flow_stats_dict,  # Dual compatibility for frontend
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
        """Background thread expiring idle flows every 2 seconds."""
        while self.is_running:
            time.sleep(2.0)
            try:
                self.flow_table.expire_idle_flows()
            except Exception as e:
                logger.error(f"Error during flow expiration loop: {e}")

    def _run_test_mode_loop(self):
        """
        Generates realistic synthetic network traffic for Controlled TEST MODE:
        - Simulates regular benign HTTP, HTTPS, and DNS connections.
        - Periodically injects synthetic attack flows (DoS, Reconnaissance, Exploits)
          to safely demonstrate alert generation and AI Security Agent inspection.
        """
        sample_benign_srcs = ["192.168.1.10", "192.168.1.15", "192.168.1.42", "192.168.1.105"]
        sample_servers = ["192.168.1.1", "93.184.216.34", "142.250.190.46", "8.8.8.8"]
        attack_srcs = ["10.0.0.99", "172.16.0.66", "198.51.100.25"]

        flow_counter = 0

        while self.is_running:
            time.sleep(random.uniform(0.6, 1.2))
            flow_counter += 1

            # Every 4th flow, generate a simulated attack flow (25% attack, 75% benign)
            is_attack_sim = (flow_counter % 4 == 0)

            if is_attack_sim:
                # Alternate between DoS flood, Port Scan, and Web Exploit
                attack_type = random.choice(["dos", "scan", "exploit"])
                attacker_ip = random.choice(attack_srcs)
                target_ip = random.choice(sample_servers[:2])
                attack_sport = random.randint(40000, 65000)

                if attack_type == "dos":
                    # Simulated SYN Flood DoS: high packet volume, raw socket TTL=254, state=INT
                    gt_cat = "DoS (SYN Flood)"
                    for seq_offset in range(5):
                        syn_pkt = IP(src=attacker_ip, dst=target_ip, ttl=254) / TCP(
                            sport=attack_sport, dport=80, flags="S", seq=5000 + seq_offset
                        )
                        self._on_packet_received(
                            syn_pkt, ground_truth_label=1, ground_truth_category=gt_cat, is_synthetic=True
                        )
                elif attack_type == "scan":
                    # Simulated Port Scan: single SYN probe to sensitive service, TTL=254, state=REQ
                    gt_cat = "Port Scan (Reconnaissance)"
                    probe_port = random.choice([22, 23, 3306, 3389])
                    scan_pkt = IP(src=attacker_ip, dst=target_ip, ttl=254) / TCP(
                        sport=attack_sport, dport=probe_port, flags="S", seq=9999
                    )
                    self._on_packet_received(
                        scan_pkt, ground_truth_label=1, ground_truth_category=gt_cat, is_synthetic=True
                    )
                else:
                    # Simulated Web Exploit: SQLi/command injection payload, TTL=254
                    gt_cat = "Web Exploit (SQL Injection)"
                    syn_pkt = IP(src=attacker_ip, dst=target_ip, ttl=254) / TCP(
                        sport=attack_sport, dport=80, flags="S", seq=1000
                    )
                    self._on_packet_received(
                        syn_pkt, ground_truth_label=1, ground_truth_category=gt_cat, is_synthetic=True
                    )
                    synack_pkt = IP(src=target_ip, dst=attacker_ip, ttl=64) / TCP(
                        sport=80, dport=attack_sport, flags="SA", seq=2000, ack=1001
                    )
                    self._on_packet_received(
                        synack_pkt, ground_truth_label=1, ground_truth_category=gt_cat, is_synthetic=True
                    )
                    payload = b"GET /admin?id=1' UNION SELECT username,password FROM users-- HTTP/1.1\r\n\r\n"
                    data_pkt = IP(src=attacker_ip, dst=target_ip, ttl=254) / TCP(
                        sport=attack_sport, dport=80, flags="PA", seq=1001, ack=2001
                    ) / Raw(load=payload)
                    self._on_packet_received(
                        data_pkt, ground_truth_label=1, ground_truth_category=gt_cat, is_synthetic=True
                    )
            else:
                # Simulated Benign traffic: normal HTTP/HTTPS/DNS client
                src_ip = random.choice(sample_benign_srcs)
                dst_ip = random.choice(sample_servers)
                dst_port = random.choice([80, 443, 53, 8080])
                src_port = random.randint(49152, 65535)

                if dst_port == 53:
                    # Benign UDP DNS query and response
                    gt_cat = "Benign DNS"
                    dns_query = IP(src=src_ip, dst=dst_ip, ttl=64) / UDP(sport=src_port, dport=53) / Raw(load=b"\x00\x01test.local")
                    self._on_packet_received(
                        dns_query, ground_truth_label=0, ground_truth_category=gt_cat, is_synthetic=True
                    )
                    dns_resp = IP(src=dst_ip, dst=src_ip, ttl=64) / UDP(sport=53, dport=src_port) / Raw(load=b"\x00\x01\x81\x80test.local")
                    self._on_packet_received(
                        dns_resp, ground_truth_label=0, ground_truth_category=gt_cat, is_synthetic=True
                    )
                else:
                    # Benign TCP connection with full handshake, data exchange, and clean teardown
                    gt_cat = "Benign Web (HTTP/HTTPS)"
                    syn_pkt = IP(src=src_ip, dst=dst_ip, ttl=64) / TCP(
                        sport=src_port, dport=dst_port, flags="S", seq=1000
                    )
                    self._on_packet_received(
                        syn_pkt, ground_truth_label=0, ground_truth_category=gt_cat, is_synthetic=True
                    )

                    synack_pkt = IP(src=dst_ip, dst=src_ip, ttl=64) / TCP(
                        sport=dst_port, dport=src_port, flags="SA", seq=2000, ack=1001
                    )
                    self._on_packet_received(
                        synack_pkt, ground_truth_label=0, ground_truth_category=gt_cat, is_synthetic=True
                    )

                    client_ack = IP(src=src_ip, dst=dst_ip, ttl=64) / TCP(
                        sport=src_port, dport=dst_port, flags="A", seq=1001, ack=2001
                    )
                    self._on_packet_received(
                        client_ack, ground_truth_label=0, ground_truth_category=gt_cat, is_synthetic=True
                    )

                    payload = b"GET /index.html HTTP/1.1\r\nHost: example.com\r\n\r\n" if dst_port in [80, 8080] else b"\x16\x03\x01\x00\xa5TLS_Client_Hello"
                    data_pkt = IP(src=src_ip, dst=dst_ip, ttl=64) / TCP(
                        sport=src_port, dport=dst_port, flags="PA", seq=1001, ack=2001
                    ) / Raw(load=payload)
                    self._on_packet_received(
                        data_pkt, ground_truth_label=0, ground_truth_category=gt_cat, is_synthetic=True
                    )

                    resp_payload = b"HTTP/1.1 200 OK\r\nContent-Length: 12\r\n\r\nHello World!"
                    resp_pkt = IP(src=dst_ip, dst=src_ip, ttl=64) / TCP(
                        sport=dst_port, dport=src_port, flags="PA", seq=2001, ack=1001 + len(payload)
                    ) / Raw(load=resp_payload)
                    self._on_packet_received(
                        resp_pkt, ground_truth_label=0, ground_truth_category=gt_cat, is_synthetic=True
                    )

                    fin_pkt = IP(src=src_ip, dst=dst_ip, ttl=64) / TCP(
                        sport=src_port, dport=dst_port, flags="FA", seq=1001 + len(payload), ack=2001 + len(resp_payload)
                    )
                    self._on_packet_received(
                        fin_pkt, ground_truth_label=0, ground_truth_category=gt_cat, is_synthetic=True
                    )

                    fin_ack_pkt = IP(src=dst_ip, dst=src_ip, ttl=64) / TCP(
                        sport=dst_port, dport=src_port, flags="FA", seq=2001 + len(resp_payload), ack=1002 + len(payload)
                    )
                    self._on_packet_received(
                        fin_ack_pkt, ground_truth_label=0, ground_truth_category=gt_cat, is_synthetic=True
                    )

                    # Trailing ACK from client (absorbed without orphan flow)
                    trailing_ack = IP(src=src_ip, dst=dst_ip, ttl=64) / TCP(
                        sport=src_port, dport=dst_port, flags="A", seq=1002 + len(payload), ack=2002 + len(resp_payload)
                    )
                    self._on_packet_received(
                        trailing_ack, ground_truth_label=0, ground_truth_category=gt_cat, is_synthetic=True
                    )


def resolve_service_simple(dst_port: int) -> str:
    return resolve_service(dst_port, 0, b"")
