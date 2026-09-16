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

        self.event_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        self.flow_table.flow_expired_callback = self._on_flow_expired

    def start_capture(
        self,
        interface_name: str,
        test_mode: bool = False,
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """Starts packet capture on specified interface or launches Controlled TEST MODE."""
        if self.is_running:
            self.stop_capture()

        self.interface_name = interface_name
        self.test_mode = test_mode
        self.event_callback = event_callback
        self.is_running = True

        self.packet_count = 0
        self.flow_count = 0
        self.normal_count = 0
        self.attack_count = 0

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

    def _on_packet_received(self, packet):
        if not self.is_running:
            return
        self.packet_count += 1
        try:
            self.flow_table.process_packet(packet)
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

        # Extract 19 features & query rolling tracker
        df_19 = extract_19_features(flow_data, self.rolling_tracker)

        # Run Live Preprocessor & Model Inference
        pred_label, bin_val, attack_cat, confidence = self.detector.predict_flow(df_19)

        if bin_val == 1:
            self.attack_count += 1
        else:
            self.normal_count += 1

        # Update rolling stats tracker with flow summary
        self.rolling_tracker.record_flow(flow_data)

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
            "attack_category": attack_cat,
            "confidence": confidence,
            "flow_stats": {
                "spkts": flow_data["spkts"],
                "dpkts": flow_data["dpkts"],
                "sbytes": flow_data["sbytes"],
                "dbytes": flow_data["dbytes"],
                "duration": flow_data["duration"],
                "sttl": flow_data["sttl"],
            },
            "system_totals": {
                "packet_count": self.packet_count,
                "flow_count": self.flow_count,
                "normal_count": self.normal_count,
                "attack_count": self.attack_count,
            },
            "is_test_mode": self.test_mode,
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
        """Generates synthetic local Scapy packets for Controlled TEST MODE."""
        sample_srcs = ["192.168.1.10", "192.168.1.15", "10.0.0.5", "172.16.0.22"]
        sample_dsts = ["192.168.1.1", "10.0.0.1", "172.16.0.1", "192.168.1.100"]
        sample_ports = [80, 443, 22, 53, 21, 8080, 3306]

        while self.is_running:
            time.sleep(random.uniform(0.3, 1.0))
            src_ip = random.choice(sample_srcs)
            dst_ip = random.choice(sample_dsts)
            dst_port = random.choice(sample_ports)
            src_port = random.randint(49152, 65535)

            # Generate SYN packet
            syn_pkt = IP(src=src_ip, dst=dst_ip, ttl=random.choice([64, 128]))/TCP(sport=src_port, dport=dst_port, flags="S", seq=1000)
            self._on_packet_received(syn_pkt)

            # Generate SYN-ACK response packet
            synack_pkt = IP(src=dst_ip, dst=src_ip, ttl=random.choice([64, 128]))/TCP(sport=dst_port, dport=src_port, flags="SA", seq=2000, ack=1001)
            self._on_packet_received(synack_pkt)

            # Generate Data packet
            if dst_port in [80, 8080]:
                payload = b"GET /index.html HTTP/1.1\r\nHost: example.com\r\n\r\n"
            else:
                payload = b"Test Payload Data"

            data_pkt = IP(src=src_ip, dst=dst_ip, ttl=64)/TCP(sport=src_port, dport=dst_port, flags="PA", seq=1001, ack=2001)/Raw(load=payload)
            self._on_packet_received(data_pkt)

            # Generate FIN packet
            fin_pkt = IP(src=src_ip, dst=dst_ip, ttl=64)/TCP(sport=src_port, dport=dst_port, flags="FA", seq=1001 + len(payload), ack=2001)
            self._on_packet_received(fin_pkt)


def resolve_service_simple(dst_port: int) -> str:
    return resolve_service(dst_port, 0, b"")
