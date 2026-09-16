"""
Bidirectional 5-tuple FlowKey identifier.
"""

from typing import Tuple

class FlowKey:
    """
    Identifies a network flow by 5-tuple: (src_ip, dst_ip, src_port, dst_port, protocol).
    Supports bidirectional equivalence so forward and reverse packets match the same flow.
    """
    def __init__(self, src_ip: str, dst_ip: str, src_port: int, dst_port: int, proto: str):
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.src_port = src_port
        self.dst_port = dst_port
        self.proto = proto.lower()

    @property
    def forward_tuple(self) -> Tuple[str, str, int, int, str]:
        return (self.src_ip, self.dst_ip, self.src_port, self.dst_port, self.proto)

    @property
    def canonical_tuple(self) -> Tuple:
        """Normalized tuple to ensure bidirectional matching."""
        endpoint_a = (self.src_ip, self.src_port)
        endpoint_b = (self.dst_ip, self.dst_port)
        if endpoint_a <= endpoint_b:
            return (self.src_ip, self.dst_ip, self.src_port, self.dst_port, self.proto)
        else:
            return (self.dst_ip, self.src_ip, self.dst_port, self.src_port, self.proto)

    def is_forward(self, packet_src_ip: str, packet_src_port: int) -> bool:
        """Determines if a packet is traveling in the forward direction."""
        return (packet_src_ip == self.src_ip) and (packet_src_port == self.src_port)

    def __hash__(self):
        return hash(self.canonical_tuple)

    def __eq__(self, other):
        if not isinstance(other, FlowKey):
            return False
        return self.canonical_tuple == other.canonical_tuple

    def __str__(self):
        return f"{self.src_ip}:{self.src_port} -> {self.dst_ip}:{self.dst_port} ({self.proto})"
