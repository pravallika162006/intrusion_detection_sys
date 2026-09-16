"""
Feature Extractor for the Phase 2 selected 19 live features.

Live provenance:
  sttl: observed from the forward IP packet; 0 means unavailable for an
      incomplete/midstream flow.
  proto: directly observed from the IP/TCP/UDP packet.
  service: derived from ports and, secondarily, payload signatures.
  sbytes/dbytes: derived network-layer IP packet bytes by direction.
  smean/dmean: derived current-flow byte totals divided by packet counts.
  dpkts: directly observed packet count in the reverse direction.
  state: derived from observed TCP flags or UDP packet directions.
  sloss/dloss: approximation counting overlapping retransmitted TCP payloads.
  synack/tcprtt: TCP timing derived from a complete observed handshake;
             zero means unavailable because the selected schema is numeric.
  trans_depth: derived count of visible plaintext HTTP transactions.
  ct_*: rolling historical approximation over completed live flows, limited
      to the in-memory window and interface visibility.

The preprocessor requires numeric values, so unavailable timing and TTL values
use the same zero sentinel present in the UNSW-NB15 training representation and
are explicitly marked as unavailable in the flow summary.
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


def extract_19_features(
    flow_data: Dict[str, Any],
    rolling_tracker: RollingStatsTracker
) -> pd.DataFrame:
    """
    Extracts the 19 Phase 2 features from a flow summary dictionary,
    queries rolling_tracker for historical ct_* features,
    and returns a 1-row DataFrame matching SELECTED_19_FEATURES ordering.
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

    sttl = flow_data.get("sttl", 0)
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
    trans_depth = flow_data.get("trans_depth", 0)

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
        "proto": str(proto),
        "ct_srv_dst": int(ct_stats["ct_srv_dst"]),
        "sbytes": int(sbytes),
        "service": str(service),
        "smean": float(smean),
        "ct_dst_sport_ltm": int(ct_stats["ct_dst_sport_ltm"]),
        "state": str(state),
        "dpkts": int(dpkts),
        "sloss": int(sloss),
        "synack": float(synack),
        "ct_dst_src_ltm": int(ct_stats["ct_dst_src_ltm"]),
        "dmean": float(dmean),
        "trans_depth": int(trans_depth),
        "ct_state_ttl": int(ct_stats["ct_state_ttl"]),
        "dbytes": int(dbytes),
        "ct_srv_src": int(ct_stats["ct_srv_src"]),
        "dloss": int(dloss),
        "tcprtt": float(tcprtt),
    }

    # Construct DataFrame in exact SELECTED_19_FEATURES ordering
    df_19 = pd.DataFrame([feature_dict])[SELECTED_19_FEATURES]
    return df_19
