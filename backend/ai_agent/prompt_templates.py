"""
Prompt templates and fallback rule engines for the AI Security Agent.
"""

SYSTEM_PROMPT = """
You are an expert AI Cyber Security Analyst for an Enterprise Intrusion Detection System (IDS).
Analyze the provided network detection event and return a structured JSON assessment.

IMPORTANT CONSTRAINTS:
1. Your recommendations MUST be recommendation-only and NON-DESTRUCTIVE. Do NOT propose automatic blocking of IPs, killing processes, or changing firewall rules directly.
2. DO NOT HALLUCINATE. Use ONLY facts explicitly provided in the event input (src_ip, dst_ip, ports, protocol, service, stats). Do NOT invent CVE numbers, malware family names, user accounts, or attacker identities. If information is absent, state "Not available from current network-flow data."
3. Use confidence-aware, non-absolute language (e.g., "The model identified traffic patterns that resemble...", "Potential attack pattern", "Model prediction").
4. If the traffic is Normal, set severity to "Informational" and provide simple reassurance.

Output strict JSON matching this structure:
{
  "threat_summary": "Short 1-2 sentence summary of what was detected",
  "explanation": "Clear explanation of the traffic pattern and why it is classified as this category",
  "why_flagged": "Specific flow characteristics (packet rate, byte ratio, TTL, duration) that matched the model signature",
  "severity": "Informational | Low | Medium | High | Critical",
  "recommended_actions": ["Action 1", "Action 2", "Action 3"],
  "investigation_guidance": ["Guidance 1", "Guidance 2"]
}
"""

RULE_FALLBACK_DATABASE = {
    "Normal": {
        "threat_summary": "Normal Network Traffic Observed",
        "explanation": "The ML model analyzed this network flow and confirmed standard operational parameters with no threat signatures.",
        "why_flagged": "Flow duration, byte count, packet counts, and TTL values match baseline normal network behavior.",
        "severity": "Informational",
        "recommended_actions": [
            "No immediate action required.",
            "Continue routine network telemetry monitoring."
        ],
        "investigation_guidance": [
            "Standard operational traffic.",
            "No deeper investigation required unless unexpected endpoints are involved."
        ]
    },
    "Reconnaissance": {
        "threat_summary": "Potential Network Reconnaissance / Scanning Activity Detected",
        "explanation": "The model detected traffic patterns resembling active network probing or port scanning behavior.",
        "why_flagged": "Observed rapid connection attempts across multiple destination ports or sequential host IPs.",
        "severity": "Medium",
        "recommended_actions": [
            "Review whether the source host is an authorized internal vulnerability scanner.",
            "Inspect recent network log entries for unauthorized port probing on sensitive ports.",
            "Verify perimeter access control policies for exposed services."
        ],
        "investigation_guidance": [
            "Examine web server access logs for automated scanning user-agents.",
            "Check netstat or active connection lists on the target host."
        ]
    },
    "DoS": {
        "threat_summary": "Potential Denial of Service (DoS) Traffic Pattern Detected",
        "explanation": "The model identified high packet volumes or SYN flooding characteristics targeting network endpoints.",
        "why_flagged": "Abnormally high packet rate, elevated SYN-to-FIN packet ratio, or high volume of short-lived connections.",
        "severity": "High",
        "recommended_actions": [
            "Inspect target destination service CPU and network bandwidth utilization.",
            "Review web application firewall (WAF) or perimeter router rate-limiting policies.",
            "Monitor connection state tables for SYN queue saturation."
        ],
        "investigation_guidance": [
            "Cross-reference source IP against internal host inventory and external threat lists.",
            "Examine packet header distributions for repetitive sequence numbers or forged headers."
        ]
    },
    "Exploits": {
        "threat_summary": "Potential Software Exploit Payload / Vulnerability Target Pattern",
        "explanation": "The model detected payload signatures or TCP session characteristics resembling known software exploit attempts.",
        "why_flagged": "Payload byte structures or anomalous window size variations matched known vulnerability exploit signatures.",
        "severity": "Critical",
        "recommended_actions": [
            "Review system patch levels and software versions on the target system.",
            "Check application server error logs for memory access violations or unusual crashes.",
            "Alert security team for post-detection containment and verification."
        ],
        "investigation_guidance": [
            "Capture payload traffic dump for forensic signature analysis against public advisory databases.",
            "Inspect endpoint EDR logs for unusual child process creation."
        ]
    },
    "Fuzzers": {
        "threat_summary": "Potential Protocol Fuzzing / Malformed Input Pattern",
        "explanation": "The model identified traffic resembling randomized or malformed payload structures intended to test application stability.",
        "why_flagged": "Irregular payload lengths, unexpected packet flags, or unparsed byte sequences.",
        "severity": "Medium",
        "recommended_actions": [
            "Verify input validation and sanitization configurations on target applications.",
            "Monitor destination application process stability for unexpected restarts."
        ],
        "investigation_guidance": [
            "Inspect application error logs for unhandled exception traces or core dumps."
        ]
    },
    "Generic": {
        "threat_summary": "Generic Anomaly / Cryptographic Collision Pattern",
        "explanation": "The model flagged traffic matching synthetic or generic intrusion feature space profiles.",
        "why_flagged": "Feature vector distance exceeded normal threshold across packet size and flow duration metrics.",
        "severity": "Medium",
        "recommended_actions": [
            "Review flow byte counts and packet ratios for non-standard protocol usage.",
            "Verify whether source and destination hosts have expected communication history."
        ],
        "investigation_guidance": [
            "Compare flow duration and TCP Round Trip Time (RTT) against baseline traffic profiles."
        ]
    },
    "Analysis": {
        "threat_summary": "Potential Web Directory Probing / Intrusive Traffic Analysis",
        "explanation": "The model detected traffic characteristics resembling URI directory traversal or web structure probing.",
        "why_flagged": "Repeated HTTP request sequences targeting non-existent path resources.",
        "severity": "Low",
        "recommended_actions": [
            "Ensure directory indexing is disabled on target web servers.",
            "Review web server 404 HTTP error logs for automated brute-force URL guessing."
        ],
        "investigation_guidance": [
            "Examine HTTP status code distributions and user-agent strings."
        ]
    },
    "Backdoor": {
        "threat_summary": "Potential Backdoor / Command and Control (C2) Activity",
        "explanation": "The model detected persistent outbound connection patterns associated with remote backdoor beaconing.",
        "why_flagged": "Persistent low-frequency packet transmissions to non-standard remote ports.",
        "severity": "Critical",
        "recommended_actions": [
            "Initiate forensic inspection on the source host for unauthorized running processes.",
            "Identify active network sockets bound to external destination IP addresses."
        ],
        "investigation_guidance": [
            "Inspect process execution tree on source host for unexpected shell or scripting invocations."
        ]
    },
    "Shellcode": {
        "threat_summary": "Potential Executable Shellcode Payload Pattern",
        "explanation": "The model detected binary payload patterns resembling NOP sleds or machine code sequences.",
        "why_flagged": "Raw payload byte entropy and consecutive machine instruction sequences matched shellcode signatures.",
        "severity": "Critical",
        "recommended_actions": [
            "Perform host-based memory inspection on target system.",
            "Check local security event logs for privilege escalation attempts."
        ],
        "investigation_guidance": [
            "Inspect packet payload dumps for shellcode assembly instructions."
        ]
    },
    "Worms": {
        "threat_summary": "Potential Self-Propagating Network Worm Traffic",
        "explanation": "The model identified scanning coupled with payload transmission suggesting automated lateral propagation.",
        "why_flagged": "Rapid scanning across internal IP ranges combined with exploit payload signatures.",
        "severity": "High",
        "recommended_actions": [
            "Verify internal network segmentation between subnets.",
            "Audit SMB, RDP, and SSH authentication logs for unusual lateral connections."
        ],
        "investigation_guidance": [
            "Monitor internal network traffic for sudden spikes targeting ports 445, 3389, or 22."
        ]
    }
}
