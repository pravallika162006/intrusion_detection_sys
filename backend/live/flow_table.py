"""
Flow Table Manager for active live network flows.
Maintains active flow state, processes incoming packets, updates flow stats,
and handles flow expiration.
"""

import time
import threading
from typing import Dict, List, Optional, Callable, Any
from scapy.all import Packet, IP, IPv6, TCP, UDP

from backend.live.flow_key import FlowKey
from backend.utils.logger import setup_logger

logger = setup_logger("FlowTable")

class FlowRecord:
    """Represents a single active network flow state."""

    COMMON_SERVICE_PORTS = {
        80, 443, 8080, 8443, 8000, 8888, 3000, 5000,  # Web & common alternate web ports
        53,                                           # DNS
        21, 20,                                       # FTP
        22, 23,                                       # SSH, Telnet
        25, 465, 587,                                 # SMTP
        110, 995, 143, 993,                           # POP3, IMAP
        3306, 5432, 27017, 6379,                      # Databases
        161, 162,                                     # SNMP
        67, 68,                                       # DHCP
        123,                                          # NTP
        1812, 1813,                                   # RADIUS
    }

    def __init__(self, key: FlowKey, first_packet_time: float, sttl: int):
        self.key = key
        self.forward_ip = key.src_ip
        self.forward_port = key.src_port
        self.reverse_ip = key.dst_ip
        self.reverse_port = key.dst_port
        self.start_time = first_packet_time
        self.last_time = first_packet_time
        self.sttl = sttl
        self.sttl_available = False
        self.direction_known = False
        self.direction_source = "unknown"
        
        self.spkts = 0
        self.dpkts = 0
        self.sbytes = 0
        self.dbytes = 0

        self.sloss = 0
        self.dloss = 0

        self.syn_time: Optional[float] = None
        self.synack_time: Optional[float] = None
        self.ack_time: Optional[float] = None
        self.syn_seq: Optional[int] = None
        self.synack = 0.0
        self.tcprtt = 0.0
        self.synack_available = False
        self.tcprtt_available = False

        self.state = "CON"
        self.trans_depth = 0
        self.first_payload = b""

        # Teardown tracking & trailing ACK absorption
        self.fin_packets = 0
        self.closing = False
        self.close_time: Optional[float] = None

        # Ground truth metadata for evaluation (isolated from feature inputs)
        self.ground_truth_label: Optional[int] = None
        self.ground_truth_category: Optional[str] = None
        self.is_synthetic: bool = False

        # TCP Seq tracking for loss
        self.payload_ranges = {"forward": [], "reverse": []}

    def _set_forward_direction(
        self,
        forward_ip: str,
        forward_port: int,
        reverse_ip: str,
        reverse_port: int,
        source: str,
    ):
        """Orient the record and preserve counters if an earlier guess was reversed."""
        already_forward = self.forward_ip == forward_ip and self.forward_port == forward_port
        if not already_forward:
            self.spkts, self.dpkts = self.dpkts, self.spkts
            self.sbytes, self.dbytes = self.dbytes, self.sbytes
            self.sloss, self.dloss = self.dloss, self.sloss
            self.payload_ranges["forward"], self.payload_ranges["reverse"] = (
                self.payload_ranges["reverse"],
                self.payload_ranges["forward"],
            )

            self.forward_ip = forward_ip
            self.forward_port = forward_port
            self.reverse_ip = reverse_ip
            self.reverse_port = reverse_port

        self.direction_known = source == "syn" or source == "port_inference"
        self.direction_source = source

    @classmethod
    def _infer_direction(cls, src_port: int, dst_port: int) -> Optional[str]:
        """
        Infers client direction by inspecting common service ports and ephemeral port boundaries.
        Returns:
            'source' if src_port is the client (initiator),
            'destination' if dst_port is the client (so packet came from server),
            None if indeterminate.
        """
        src_is_svc = src_port in cls.COMMON_SERVICE_PORTS
        dst_is_svc = dst_port in cls.COMMON_SERVICE_PORTS

        # 1. Exact match against known service ports
        if dst_is_svc and not src_is_svc:
            return "source"
        if src_is_svc and not dst_is_svc:
            return "destination"

        # 2. Ephemeral vs System/Registered port boundary (32768+ standard ephemeral range)
        if src_port >= 32768 and 0 < dst_port < 32768:
            return "source"
        if 0 < src_port < 32768 and dst_port >= 32768:
            return "destination"

        # 3. Traditional privileged port boundary (1024)
        if src_port > 1023 and 0 < dst_port <= 1023:
            return "source"
        if 0 < src_port <= 1023 and dst_port > 1023:
            return "destination"

        return None

    def _track_payload(self, direction: str, seq: int, payload_length: int):
        """Count overlapping payload ranges as retransmitted segments, not packet loss."""
        if payload_length <= 0:
            return False

        start = int(seq)
        end = start + payload_length
        ranges = self.payload_ranges[direction]
        retransmitted = any(start < old_end and end > old_start for old_start, old_end in ranges)
        ranges.append((start, end))
        ranges.sort()

        merged = []
        for old_start, old_end in ranges:
            if merged and old_start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], old_end))
            else:
                merged.append((old_start, old_end))
        self.payload_ranges[direction] = merged
        return retransmitted

    def update(self, packet: Packet, pkt_time: float):
        self.last_time = pkt_time
        ip_layer = packet[IP] if IP in packet else packet[IPv6]
        packet_src_ip = ip_layer.src
        packet_dst_ip = ip_layer.dst
        packet_src_port = packet.sport if (TCP in packet or UDP in packet) else 0
        packet_dst_port = packet.dport if (TCP in packet or UDP in packet) else 0

        if TCP in packet:
            flags = int(packet[TCP].flags)
            is_syn = bool(flags & 0x02) and not bool(flags & 0x10)
            if is_syn:
                self._set_forward_direction(
                    packet_src_ip,
                    packet_src_port,
                    packet_dst_ip,
                    packet_dst_port,
                    "syn",
                )
            elif not self.direction_known:
                inferred = self._infer_direction(packet_src_port, packet_dst_port)
                if inferred == "source":
                    self._set_forward_direction(
                        packet_src_ip,
                        packet_src_port,
                        packet_dst_ip,
                        packet_dst_port,
                        "port_inference",
                    )
                elif inferred == "destination":
                    self._set_forward_direction(
                        packet_dst_ip,
                        packet_dst_port,
                        packet_src_ip,
                        packet_src_port,
                        "port_inference",
                    )
        elif UDP in packet:
            if not self.direction_known:
                inferred = self._infer_direction(packet_src_port, packet_dst_port)
                if inferred == "source":
                    self._set_forward_direction(
                        packet_src_ip,
                        packet_src_port,
                        packet_dst_ip,
                        packet_dst_port,
                        "port_inference",
                    )
                elif inferred == "destination":
                    self._set_forward_direction(
                        packet_dst_ip,
                        packet_dst_port,
                        packet_src_ip,
                        packet_src_port,
                        "port_inference",
                    )

        is_fwd = packet_src_ip == self.forward_ip and packet_src_port == self.forward_port

        payload_bytes = bytes(packet[TCP].payload) if TCP in packet else (bytes(packet[UDP].payload) if UDP in packet else b"")
        packet_bytes = len(ip_layer)

        if is_fwd:
            self.sttl = int(getattr(ip_layer, "ttl", getattr(ip_layer, "hlim", 0)))
            self.sttl_available = True

        if not self.first_payload and payload_bytes:
            self.first_payload = payload_bytes

        if is_fwd:
            self.spkts += 1
            self.sbytes += packet_bytes
        else:
            self.dpkts += 1
            self.dbytes += packet_bytes

        # HTTP Trans Depth Check
        if payload_bytes:
            payload_str = payload_bytes[:100].decode("utf-8", errors="ignore").upper()
            if any(kw in payload_str for kw in ["GET ", "POST ", "HTTP/1."]):
                self.trans_depth += 1

        # TCP State Machine & Loss Analysis
        if TCP in packet:
            flags = packet[TCP].flags
            seq = packet[TCP].seq
            direction = "forward" if is_fwd else "reverse"
            if self._track_payload(direction, seq, len(payload_bytes)):
                if is_fwd:
                    self.sloss += 1
                else:
                    self.dloss += 1

            # Handshake timing
            if flags.S and not flags.A:  # SYN
                self.syn_time = pkt_time
                self.syn_seq = seq
                self.state = "REQ"
            elif flags.S and flags.A:    # SYN-ACK
                self.synack_time = pkt_time
                if self.syn_time:
                    self.synack = round(self.synack_time - self.syn_time, 6)
                    self.synack_available = True
                self.state = "CON"  # In UNSW-NB15, active connection state is 'CON' (not 'ACC')
            elif flags.A and self.synack_time and not self.ack_time:  # ACK after SYN-ACK
                self.ack_time = pkt_time
                self.tcprtt = round(self.ack_time - self.syn_time, 6)
                self.tcprtt_available = True
                self.state = "CON"

            if flags.F:  # FIN
                self.state = "FIN"
                self.fin_packets = getattr(self, "fin_packets", 0) + 1
                if self.fin_packets >= 2:
                    self.closing = True
                    self.close_time = pkt_time
            elif flags.R:  # RST
                self.state = "RST"
                self.closing = True
                self.close_time = pkt_time
        elif UDP in packet:
            self.state = "CON" if (self.spkts > 0 and self.dpkts > 0) else "INT"

    def to_dict(self) -> Dict[str, Any]:
        dur = round(self.last_time - self.start_time, 4)
        return {
            "src_ip": self.forward_ip,
            "dst_ip": self.reverse_ip,
            "src_port": self.forward_port,
            "dst_port": self.reverse_port,
            "proto": self.key.proto,
            "spkts": self.spkts,
            "dpkts": self.dpkts,
            "sbytes": self.sbytes,
            "dbytes": self.dbytes,
            "sttl": self.sttl if self.sttl_available else 0,
            "sttl_available": self.sttl_available,
            "direction_known": self.direction_known,
            "direction_source": self.direction_source,
            "state": self.state,
            "sloss": self.sloss,
            "dloss": self.dloss,
            "synack": self.synack if self.synack_available else None,
            "tcprtt": self.tcprtt if self.tcprtt_available else None,
            "synack_available": self.synack_available,
            "tcprtt_available": self.tcprtt_available,
            "trans_depth": self.trans_depth,
            "first_payload": self.first_payload,
            "dur": dur,
            "duration": dur,
            "fin_packets": self.fin_packets,
            "closing": self.closing,
            "ground_truth_label": self.ground_truth_label,
            "ground_truth_category": self.ground_truth_category,
            "is_synthetic": self.is_synthetic,
        }


class FlowTableManager:
    """
    Thread-safe Flow Table managing active flows, timeouts, and callbacks.
    """

    def __init__(self, idle_timeout: float = 10.0, max_flows: int = 5000):
        self.idle_timeout = idle_timeout
        self.max_flows = max_flows
        self.active_flows: Dict[FlowKey, FlowRecord] = {}
        self.recently_closed: Dict[FlowKey, float] = {}  # Tombstone cache to absorb trailing ACKs
        self.lock = threading.Lock()
        self.flow_expired_callback: Optional[Callable[[Dict[str, Any]], None]] = None

    def process_packet(
        self,
        packet: Packet,
        ground_truth_label: Optional[int] = None,
        ground_truth_category: Optional[str] = None,
        is_synthetic: bool = False,
    ):
        if not (IP in packet or IPv6 in packet):
            return

        ip_layer = packet[IP] if IP in packet else packet[IPv6]
        src_ip = ip_layer.src
        dst_ip = ip_layer.dst
        sttl = getattr(ip_layer, "ttl", getattr(ip_layer, "hlim", 64))

        src_port = 0
        dst_port = 0
        proto = "other"

        if TCP in packet:
            src_port = packet[TCP].sport
            dst_port = packet[TCP].dport
            proto = "tcp"
        elif UDP in packet:
            src_port = packet[UDP].sport
            dst_port = packet[UDP].dport
            proto = "udp"
        else:
            proto = str(ip_layer.proto) if hasattr(ip_layer, "proto") else "other"

        key = FlowKey(src_ip, dst_ip, src_port, dst_port, proto)
        now = time.time()

        with self.lock:
            # Clean expired tombstone entries older than 3 seconds
            if self.recently_closed:
                self.recently_closed = {k: exp for k, exp in self.recently_closed.items() if exp > now}

            if key not in self.active_flows:
                # Absorb trailing ACKs or teardown duplicates for recently closed flows
                if key in self.recently_closed and TCP in packet:
                    tcp_flags = int(packet[TCP].flags)
                    # If packet is an ACK or FIN without SYN, absorb it without spawning an orphan flow
                    if not (tcp_flags & 0x02):
                        return

                if len(self.active_flows) >= self.max_flows:
                    self._purge_oldest()

                record = FlowRecord(key, now, sttl)
                self.active_flows[key] = record
            else:
                record = self.active_flows[key]

            # Attach ground-truth metadata if provided (isolated from model features)
            if ground_truth_label is not None and record.ground_truth_label is None:
                record.ground_truth_label = ground_truth_label
            if ground_truth_category is not None and record.ground_truth_category is None:
                record.ground_truth_category = ground_truth_category
            if is_synthetic:
                record.is_synthetic = True

            record.update(packet, now)

    def expire_idle_flows(self, force: bool = False):
        """
        Scans active flows and expires flows exceeding idle_timeout (or 1.0s grace period for closing FIN/RST flows).
        If force=True, expires all closing flows immediately.
        """
        now = time.time()
        expired = []

        with self.lock:
            for key, record in list(self.active_flows.items()):
                # Closed FIN/RST flows use 1.0s grace period to capture trailing ACKs
                is_closing = getattr(record, "closing", False) or record.state in ("FIN", "RST")
                if force:
                    timeout = 0.0
                elif is_closing:
                    timeout = 1.0
                else:
                    timeout = self.idle_timeout

                if (now - record.last_time) >= timeout:
                    expired.append(record.to_dict())
                    del self.active_flows[key]
                    self.recently_closed[key] = now + 3.0

        if self.flow_expired_callback:
            for flow_dict in expired:
                self.flow_expired_callback(flow_dict)

    def _purge_oldest(self):
        if not self.active_flows:
            return
        oldest_key = min(self.active_flows.keys(), key=lambda k: self.active_flows[k].last_time)
        del self.active_flows[oldest_key]

    def clear(self):
        with self.lock:
            self.active_flows.clear()
            self.recently_closed.clear()
