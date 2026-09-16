"""
Network interface discovery for Windows/Linux environments using Scapy and socket utilities.
"""

import socket
import psutil
from typing import List, Dict, Any
from scapy.all import get_if_list, conf

def get_network_interfaces() -> List[Dict[str, Any]]:
    """
    Discovers available network interfaces on the local machine.
    Returns list of dicts with interface name, description, IP, MAC, and active status.
    """
    interfaces = []
    seen_names = set()

    # Query psutil interface addresses & stats
    stats = psutil.net_if_stats()
    addrs = psutil.net_if_addrs()

    for iface_name, iface_addrs in addrs.items():
        ip_addr = None
        mac_addr = None
        for addr in iface_addrs:
            if addr.family == socket.AF_INET:
                ip_addr = addr.address
            elif addr.family == psutil.AF_LINK or getattr(socket, 'AF_PACKET', -1) == addr.family:
                mac_addr = addr.address

        iface_stat = stats.get(iface_name)
        is_active = iface_stat.isup if iface_stat else False

        # Filter out virtual/loopback if inactive or unneeded, but keep active adapters
        desc = iface_name
        if ip_addr:
            desc = f"{iface_name} ({ip_addr})"

        interfaces.append({
            "name": iface_name,
            "description": desc,
            "ip_address": ip_addr,
            "mac_address": mac_addr,
            "is_active": is_active,
        })
        seen_names.add(iface_name)

    # Cross-reference Scapy interface list
    try:
        scapy_ifaces = get_if_list()
        for s_iface in scapy_ifaces:
            if s_iface not in seen_names:
                interfaces.append({
                    "name": s_iface,
                    "description": f"Scapy Interface: {s_iface}",
                    "ip_address": None,
                    "mac_address": None,
                    "is_active": True,
                })
    except Exception:
        pass

    # Sort active interfaces first
    interfaces.sort(key=lambda x: (not x["is_active"], x["name"]))
    return interfaces
