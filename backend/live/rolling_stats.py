"""
Rolling statistics tracker for historical connection features (ct_* features).
Maintains a rolling window of recent connection records (default 100) in memory.
"""

from collections import deque
from typing import Dict, Any, List
import threading

class RollingStatsTracker:
    """
    Tracks historical connection counts over the last N (default 100) flows for:
    - ct_srv_dst: same service & dst IP
    - ct_dst_sport_ltm: same dst IP & src port
    - ct_dst_src_ltm: same dst IP & src IP
    - ct_state_ttl: same state & sttl
    - ct_srv_src: same service & src IP
    """

    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.history: deque = deque(maxlen=max_history)
        self.lock = threading.Lock()

    def record_flow(self, flow_summary: Dict[str, Any]):
        """Records a completed flow summary dictionary into the rolling window."""
        service = flow_summary.get("service")
        if not service:
            from backend.live.feature_extractor_19 import resolve_service

            service = resolve_service(
                flow_summary.get("dst_port", 0),
                flow_summary.get("src_port", 0),
                flow_summary.get("first_payload", b""),
                flow_summary.get("proto", "tcp"),
            )
        with self.lock:
            self.history.append({
                "src_ip": flow_summary.get("src_ip"),
                "dst_ip": flow_summary.get("dst_ip"),
                "src_port": flow_summary.get("src_port"),
                "dst_port": flow_summary.get("dst_port"),
                "service": service,
                "state": flow_summary.get("state", "CON"),
                "sttl": flow_summary.get("sttl", 64),
            })

    def get_ct_stats(
        self,
        src_ip: str,
        dst_ip: str,
        src_port: int,
        dst_port: int,
        service: str,
        state: str,
        sttl: int,
    ) -> Dict[str, int]:
        """Calculates ct_* features against the current rolling window."""
        with self.lock:
            snapshot = list(self.history)

        ct_srv_dst = 0
        ct_dst_sport_ltm = 0
        ct_dst_src_ltm = 0
        ct_state_ttl = 0
        ct_srv_src = 0

        for item in snapshot:
            if item["service"] == service and item["dst_ip"] == dst_ip:
                ct_srv_dst += 1
            if item["dst_ip"] == dst_ip and item["src_port"] == src_port:
                ct_dst_sport_ltm += 1
            if item["dst_ip"] == dst_ip and item["src_ip"] == src_ip:
                ct_dst_src_ltm += 1
            if item["state"] == state and item["sttl"] == sttl:
                ct_state_ttl += 1
            if item["service"] == service and item["src_ip"] == src_ip:
                ct_srv_src += 1

        # Self-count minimum of 1 for active flow
        return {
            "ct_srv_dst": max(1, ct_srv_dst + 1),
            "ct_dst_sport_ltm": max(1, ct_dst_sport_ltm + 1),
            "ct_dst_src_ltm": max(1, ct_dst_src_ltm + 1),
            "ct_state_ttl": max(1, ct_state_ttl + 1),
            "ct_srv_src": max(1, ct_srv_src + 1),
        }

    def clear(self):
        with self.lock:
            self.history.clear()
