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
            "ip": ip_addr,
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
                    "ip": None,
                    "ip_address": None,
                    "mac_address": None,
                    "is_active": True,
                })
    except Exception:
        pass

    def _interface_sort_key(iface: Dict[str, Any]):
        name_lower = iface["name"].lower()
        ip = iface.get("ip_address") or ""
        is_loopback = "loopback" in name_lower or ip == "127.0.0.1"
        is_active = bool(iface.get("is_active", False))
        has_real_ip = bool(ip and ip != "127.0.0.1" and not ip.startswith("169.254."))
        is_wifi = any(w in name_lower for w in ["wi-fi", "wifi", "wireless", "wlan"])
        is_eth = any(e in name_lower for e in ["ethernet", "eth", "local area"])

        if is_loopback:
            tier = 5
        elif is_active and has_real_ip and is_wifi:
            tier = 0  # Top priority: active Wi-Fi with valid IP
        elif is_active and has_real_ip and is_eth:
            tier = 1  # Active Ethernet with valid IP
        elif is_active and has_real_ip:
            tier = 2  # Other active interface with valid IP
        elif is_active:
            tier = 3  # Active interface without assigned IPv4
        else:
            tier = 4  # Inactive interface

        return (tier, iface["name"])

    # Sort prioritizing usable active physical interfaces
    interfaces.sort(key=_interface_sort_key)
    return interfaces
