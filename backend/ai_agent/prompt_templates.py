"""
Prompt templates and fallback rule engines for the AI Security Agent.
"""

SYSTEM_PROMPT = """
You are an expert AI Cyber Security Analyst for an Enterprise Intrusion Detection System (IDS).
Analyze the provided network detection event and return a structured JSON assessment.

Your recommendations MUST be recommendation-only. Do NOT execute or propose automatic destructive actions (e.g. blocking IPs, shutting interfaces, deleting files).

Output strict JSON with these keys:
{
  "threat_summary": "Short 1-2 sentence summary of what was detected",
  "explanation": "Detailed technical explanation of the attack category and why it is suspicious",
  "severity": "Low | Medium | High | Critical",
  "recommended_actions": ["Action 1", "Action 2", "Action 3"],
  "investigation_guidance": ["Guidance 1", "Guidance 2"]
}
"""

RULE_FALLBACK_DATABASE = {
    "Reconnaissance": {
        "threat_summary": "Network Reconnaissance / Port Scanning Activity Detected",
        "explanation": "High volume of rapid connection attempts across multiple ports or endpoints indicates active network scanning or service discovery by an external host.",
        "severity": "Medium",
        "recommended_actions": [
            "Verify whether the source host (src_ip) is an authorized vulnerability scanner.",
            "Review firewall logs for unauthorized port scan attempts on sensitive internal ports.",
            "Consider temporary host isolation after human verification."
        ],
        "investigation_guidance": [
            "Inspect web server access logs for automated scanning user-agents.",
            "Check netstat output on target destination host to confirm open listening services."
        ]
    },
    "DoS": {
        "threat_summary": "Denial of Service (DoS) Traffic Pattern Detected",
        "explanation": "Abnormally high packet rate or SYN flooding detected originating from a single source host targeting destination services, potentially overwhelming network capacity.",
        "severity": "High",
        "recommended_actions": [
            "Inspect destination service CPU and bandwidth utilization.",
            "Apply rate limiting at the perimeter router or web application firewall (WAF).",
            "Monitor connection table state for SYN queue exhaustion."
        ],
        "investigation_guidance": [
            "Cross-reference source IP against known malicious threat intelligence lists.",
            "Examine packet dump for forged source headers or repetitive TCP SYN sequences."
        ]
    },
    "Exploits": {
        "threat_summary": "Software Exploit Payload / Vulnerability Target Detected",
        "explanation": "Traffic payload contains signatures or TCP state anomalies corresponding to unpatched software exploitation attempts.",
        "severity": "Critical",
        "recommended_actions": [
            "Immediately inspect target system patch levels for exposed services.",
            "Review application error logs for memory access violations or crash traces.",
            "Prepare incident response team for potential post-exploitation containment."
        ],
        "investigation_guidance": [
            "Capture payload sample and analyze against CVE database.",
            "Check endpoint EDR alerts for process injection or unusual child process spawns."
        ]
    },
    "Fuzzers": {
        "threat_summary": "Protocol Fuzzing / Input Mutation Attack Detected",
        "explanation": "Malformed packet structures or randomized payload lengths detected aiming to trigger application crashes or unhandled exceptions.",
        "severity": "Medium",
        "recommended_actions": [
            "Ensure input validation and sanitization filters are enabled on the target service.",
            "Monitor destination application service uptime for unexpected restarts."
        ],
        "investigation_guidance": [
            "Examine application crash logs or core dumps for buffer overflow indicators."
        ]
    },
    "Generic": {
        "threat_summary": "Generic Anomaly / Cryptographic Collision Attack",
        "explanation": "High collision rate in feature space matching synthetic/generic attack patterns in network traffic.",
        "severity": "Medium",
        "recommended_actions": [
            "Inspect flow byte count and packet ratio for non-standard protocol usage."
        ],
        "investigation_guidance": [
            "Compare flow duration and TCP RTT against baseline network profiles."
        ]
    },
    "Analysis": {
        "threat_summary": "Intrusive Traffic Analysis / Web HTML Probing",
        "explanation": "In-depth HTTP directory traversal, CGI probe, or web spidering behavior detected.",
        "severity": "Low",
        "recommended_actions": [
            "Ensure web server directory listing is disabled.",
            "Review web server 404 error logs for brute-force URI guessing."
        ],
        "investigation_guidance": [
            "Check HTTP transaction depth and response code distribution."
        ]
    },
    "Backdoor": {
        "threat_summary": "Backdoor Connection / Command and Control (C2) Activity",
        "explanation": "Persistent outbound connection to non-standard remote port indicating potential command-and-control beaconing.",
        "severity": "Critical",
        "recommended_actions": [
            "Isolate the affected internal host from the network immediately for forensic review.",
            "Identify running processes bound to the outbound socket."
        ],
        "investigation_guidance": [
            "Check process tree for unauthorized binary execution (e.g. powershell, nc, cmd)."
        ]
    },
    "Shellcode": {
        "threat_summary": "Executable Shellcode Payload Detected",
        "explanation": "Binary payload contains NOP sleds or machine code sequences intended to spawn a remote shell.",
        "severity": "Critical",
        "recommended_actions": [
            "Isolate host and run memory forensic analysis.",
            "Check security event logs for privilege escalation attempts."
        ],
        "investigation_guidance": [
            "Analyze raw packet payload for shellcode instructions."
        ]
    },
    "Worms": {
        "threat_summary": "Self-Propagating Network Worm Traffic Detected",
        "explanation": "Automated scanning coupled with payload execution attempting lateral movement across internal subnets.",
        "severity": "High",
        "recommended_actions": [
            "Segment network subnets to prevent lateral propagation.",
            "Audit SMB / RDP / SSH authentication logs across internal hosts."
        ],
        "investigation_guidance": [
            "Monitor internal subnet traffic for sudden spikes in port 445 / 3389 / 22 attempts."
        ]
    }
}
