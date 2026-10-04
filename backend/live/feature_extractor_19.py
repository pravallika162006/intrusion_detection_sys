"""
Feature Extractor for the Sydney M. Kasongo & Yanxia Sun (2020) Table 3 exact 19 features.

Live feature provenance:
  sttl: observed from the forward IP packet header TTL.
  ct_srv_dst: historical count of connections with identical service & dst IP in rolling window.
  sbytes: total IP packet payload + header bytes in forward direction.
  smean: forward byte total divided by forward packet count.
  proto: observed IP/transport layer protocol string (tcp, udp, etc.).
  ct_state_ttl: historical count of connections with same state and sttl in rolling window.
  sloss: approximated count of retransmitted/overlapping TCP packets from source.
  synack: TCP handshake SYN-ACK timing difference (or 0.0 if midstream/non-TCP).
  ct_dst_src_ltm: historical count of connections with same destination and source IP in window.
  dmean: reverse byte total divided by reverse packet count.
  ct_srv_src: historical count of connections with same service and source IP in window.
  service: application layer protocol resolved from port numbers and payload inspection.
  ct_dst_sport_ltm: historical count of connections with same destination IP and source port.
  dbytes: total IP bytes received from destination.
  dloss: approximated count of retransmitted/overlapping TCP packets from destination.
  state: state string derived from TCP flags (FIN, CON, RST) or UDP flow state.
  tcprtt: TCP full round trip time (SYN to ACK).
  ct_src_dport_ltm: historical count of connections with same source IP and destination port.
  rate: packet throughput rate = (spkts + dpkts) / max(0.001, dur).
"""

from typing import Dict, Any
import pandas as pd
from backend.config import SELECTED_19_FEATURES
from backend.live.rolling_stats import RollingStatsTracker


def resolve_service(dst_port: int, src_port: int, payload: bytes, proto: str = "tcp") -> str:
    """Resolves application service based on port numbers and payload inspection."""
    port_map = {
        80: "http",
        8080: "http",
        443: "ssl",
        53: "dns",
        21: "ftp",
        20: "ftp-data",
        25: "smtp",
        22: "ssh",
        110: "pop3",
        67: "dhcp",
        68: "dhcp",
        161: "snmp",
        1812: "radius",
    }
    
    if dst_port in port_map:
        return port_map[dst_port]
    if src_port in port_map:
        return port_map[src_port]

    if proto.lower() == "tcp" and payload[:3] in (b"\x16\x03\x00", b"\x16\x03\x01", b"\x16\x03\x02", b"\x16\x03\x03"):
        return "ssl"

    # Payload signature inspection
    if payload:
        payload_str = payload[:50].decode("utf-8", errors="ignore").upper()
        if any(kw in payload_str for kw in ["GET ", "POST ", "HTTP/"]):
            return "http"
        if "SSH-" in payload_str:
            return "ssh"
        if "220 " in payload_str or "HELO" in payload_str:
            return "smtp"

    return "-"


def align_sttl_for_benchmark(raw_ttl: int, align_mode: bool = True) -> int:
    """
    Aligns observed OS Time-To-Live (TTL) to the UNSW-NB15 benchmark reference frame.

    In the UNSW-NB15 benchmark:
      - Legitimate testbed hosts were configured with initial TTL = 32 (reaching the monitor as 31).
      - Attack injection tools were configured with raw sockets / initial TTL = 255 (reaching monitor as 254).
      - Zero records exist in the benchmark dataset with TTL between 65 and 128 (0.0% of data).

    In real networks:
      - Windows client hosts initialize TTL = 128.
      - Linux / Android / macOS hosts initialize TTL = 64.
      - Raw sockets and network scanners initialize TTL = 255.

    Without alignment, Windows packets (TTL=128) and Linux LAN packets (TTL=64) fall
    above the trained tree threshold (sttl > 61.0), triggering false-positive alerts
    purely due to operating system default differences rather than malicious activity.
    """
    if not align_mode or raw_ttl <= 0:
        return max(0, int(raw_ttl))

    if raw_ttl <= 32:
        # Already matches benchmark reference frame (e.g. UNSW-NB15 normal range)
        return int(raw_ttl)
    elif raw_ttl <= 64:
        # Linux / Android / macOS (initial TTL 64)
        hops = max(0, 64 - raw_ttl)
        return max(1, 31 - hops)
    elif raw_ttl <= 128:
        # Windows client OS (initial TTL 128)
        hops = max(0, 128 - raw_ttl)
        return max(1, 31 - hops)
    else:
        # Raw socket / attack tools / routers (initial TTL 255)
        hops = max(0, 255 - raw_ttl)
        return max(62, 254 - hops)


def extract_19_features(
    flow_data: Dict[str, Any],
    rolling_tracker: RollingStatsTracker,
    align_ttl: bool = True,
) -> pd.DataFrame:
    """
    Extracts the exact 19 features matching Paper Table 3 from a flow summary dictionary,
    queries rolling_tracker for ct_* features, and returns a 1-row DataFrame.
    """
    src_ip = flow_data.get("src_ip", "0.0.0.0")
    dst_ip = flow_data.get("dst_ip", "0.0.0.0")
    src_port = flow_data.get("src_port", 0)
    dst_port = flow_data.get("dst_port", 0)
    proto = str(flow_data.get("proto", "tcp")).lower()
    
    spkts = flow_data.get("spkts", 1)
    dpkts = flow_data.get("dpkts", 0)
    sbytes = flow_data.get("sbytes", 0)
    dbytes = flow_data.get("dbytes", 0)
    dur = float(flow_data.get("dur", flow_data.get("duration", 0.0)))

    raw_sttl = int(flow_data.get("sttl", 64))
    sttl = align_sttl_for_benchmark(raw_sttl, align_mode=align_ttl)
    flow_data["aligned_sttl"] = int(sttl)  # Consistent TTL across extractor & rolling stats
    state = flow_data.get("state", "CON")
    payload = flow_data.get("first_payload", b"")

    service = flow_data.get("service") or resolve_service(dst_port, src_port, payload, proto)

    smean = float(sbytes / spkts) if spkts > 0 else 0.0
    dmean = float(dbytes / dpkts) if dpkts > 0 else 0.0

    sloss = flow_data.get("sloss", 0)
    dloss = flow_data.get("dloss", 0)
    synack_value = flow_data.get("synack")
    tcprtt_value = flow_data.get("tcprtt")
    synack = float(synack_value) if synack_value is not None else 0.0
    tcprtt = float(tcprtt_value) if tcprtt_value is not None else 0.0

    # Rate: packet throughput rate (packets per second)
    # In UNSW-NB15 benchmark, zero-duration flows evaluate to rate = 0.0
    if dur <= 0.0:
        rate = 0.0
    else:
        rate = min(1000000.0, float((spkts + dpkts) / dur))

    # Query rolling stats tracker for ct_* features
    ct_stats = rolling_tracker.get_ct_stats(
        src_ip=src_ip,
        dst_ip=dst_ip,
        src_port=src_port,
        dst_port=dst_port,
        service=service,
        state=state,
        sttl=sttl,
    )

    feature_dict = {
        "sttl": int(sttl),
        "ct_srv_dst": int(ct_stats["ct_srv_dst"]),
        "sbytes": int(sbytes),
        "smean": float(smean),
        "proto": str(proto),
        "ct_state_ttl": int(ct_stats["ct_state_ttl"]),
        "sloss": int(sloss),
        "synack": float(synack),
        "ct_dst_src_ltm": int(ct_stats["ct_dst_src_ltm"]),
        "dmean": float(dmean),
        "ct_srv_src": int(ct_stats["ct_srv_src"]),
        "service": str(service),
        "ct_dst_sport_ltm": int(ct_stats["ct_dst_sport_ltm"]),
        "dbytes": int(dbytes),
        "dloss": int(dloss),
        "state": str(state),
        "tcprtt": float(tcprtt),
        "ct_src_dport_ltm": int(ct_stats.get("ct_src_dport_ltm", 1)),
        "rate": float(rate),
    }

    # Construct DataFrame in exact SELECTED_19_FEATURES ordering
    df_19 = pd.DataFrame([feature_dict])[SELECTED_19_FEATURES]
    return df_19
