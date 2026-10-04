"""
AI Security Agent Engine for IDS Phase 4.
Provides contextual threat explanations, confidence-aware severity assessment,
why-flagged breakdowns, and actionable non-destructive security guidance.
Supports API execution with deterministic rule-based fallback mechanism.
"""

import os
import json
from typing import Dict, Any, Optional
from backend.ai_agent.prompt_templates import RULE_FALLBACK_DATABASE, SYSTEM_PROMPT
from backend.utils.logger import setup_logger

logger = setup_logger("AISecurityAgent")


class AISecurityAgent:
    """
    AI Security Agent evaluating network intrusion alerts and returning
    structured security guidance without hallucination or destructive actions.
    """

    def __init__(self):
        self.api_key = os.getenv("AI_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")

    def generate_recommendation(
        self,
        flow_id: str,
        src_ip: str,
        dst_ip: str,
        service: str,
        attack_category: str,
        confidence: float,
        flow_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generates structured security guidance for a detected flow event.
        """
        category_clean = attack_category.strip() if attack_category else "Generic"

        # Deterministic immediate handling for Normal traffic (saves API costs & avoids unnecessary scary advice)
        if category_clean.lower() == "normal":
            return self._normal_traffic_response(src_ip, dst_ip, service, flow_info)

        # If API key exists, try LLM API generation first
        if self.api_key:
            try:
                res = self._call_ai_api(
                    flow_id, src_ip, dst_ip, service, category_clean, confidence, flow_info
                )
                if res and isinstance(res, dict) and "threat_summary" in res:
                    return res
            except Exception as e:
                logger.warning(f"AI API call failed, switching to rule fallback: {e}")

        # Fallback to Rule Engine
        return self._rule_fallback(src_ip, dst_ip, service, category_clean, confidence, flow_info)

    def _normal_traffic_response(
        self, src_ip: str, dst_ip: str, service: str, flow_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Returns deterministic, non-scary guidance for Normal network traffic."""
        norm_template = RULE_FALLBACK_DATABASE["Normal"]
        spkts = flow_info.get("spkts", "N/A") if flow_info else "N/A"
        dpkts = flow_info.get("dpkts", "N/A") if flow_info else "N/A"
        sbytes = flow_info.get("sbytes", "N/A") if flow_info else "N/A"

        return {
            "threat_summary": f"Normal Network Traffic Observed [{src_ip} -> {dst_ip}]",
            "explanation": f"The ML model analyzed this flow on service [{service}] and confirmed standard protocol operational parameters with no anomaly signatures.",
            "why_flagged": f"Flow characteristics ({spkts} src packets, {dpkts} dst packets, {sbytes} src bytes) matched baseline normal network behavior.",
            "severity": "Informational",
            "recommended_actions": list(norm_template["recommended_actions"]),
            "investigation_guidance": list(norm_template["investigation_guidance"]),
        }

    def _call_ai_api(
        self,
        flow_id: str,
        src_ip: str,
        dst_ip: str,
        service: str,
        attack_category: str,
        confidence: float,
        flow_info: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Calls OpenAI / Gemini API if key is present."""
        import requests
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        prompt_user = (
            f"Event ID: {flow_id}\n"
            f"Source IP: {src_ip}\n"
            f"Destination IP: {dst_ip}\n"
            f"Service: {service}\n"
            f"Detected Attack Category: {attack_category}\n"
            f"Model Confidence: {confidence*100:.1f}%\n"
            f"Flow Statistics: {json.dumps(flow_info)}\n"
        )

        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_user},
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return json.loads(content)
        return None

    def _rule_fallback(
        self,
        src_ip: str,
        dst_ip: str,
        service: str,
        attack_category: str,
        confidence: float,
        flow_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generates deterministic, category-aware, and confidence-calibrated security advice."""
        cat_info = RULE_FALLBACK_DATABASE.get(
            attack_category,
            {
                "threat_summary": f"Potential {attack_category} Traffic Pattern Detected",
                "explanation": f"Network flow characteristics matched feature profiles associated with '{attack_category}'.",
                "why_flagged": f"Anomalous packet ratio or feature vector distance identified for service [{service}].",
                "severity": "Medium",
                "recommended_actions": [
                    f"Review connection logs between {src_ip} and {dst_ip}.",
                    f"Verify whether target service '{service}' is expected to receive this traffic.",
                    "Review recent firewall logs for suspicious connections."
                ],
                "investigation_guidance": [
                    "Examine flow packet/byte ratios and connection state.",
                    "Check target server system logs for abnormal authentication attempts."
                ]
            }
        )

        # Confidence Language Adjustment
        conf_pct = round(confidence * 100, 1)
        if confidence >= 0.85:
            conf_prefix = "The model detected a strong pattern associated with"
            conf_desc = f"high model confidence ({conf_pct}%)"
        elif confidence >= 0.60:
            conf_prefix = "The model detected a pattern that may be associated with"
            conf_desc = f"moderate model confidence ({conf_pct}%)"
        else:
            conf_prefix = "The traffic shows some characteristics associated with"
            conf_desc = f"low model confidence ({conf_pct}%)"

        # Calculate Dynamic Severity based on confidence & category impact
        severity = self._calculate_severity(attack_category, confidence)

        # Dynamic Why Flagged explanation using actual flow stats if available
        spkts = flow_info.get("spkts", "N/A") if flow_info else "N/A"
        dpkts = flow_info.get("dpkts", "N/A") if flow_info else "N/A"
        sbytes = flow_info.get("sbytes", "N/A") if flow_info else "N/A"
        dur = flow_info.get("duration", "N/A") if flow_info else "N/A"

        why_flagged_text = (
            f"{cat_info.get('why_flagged', 'Anomalous flow characteristics observed.')} "
            f"[Flow Stats: {spkts} src pkts, {dpkts} dst pkts, {sbytes} bytes, duration: {dur}s]."
        )

        return {
            "threat_summary": f"{cat_info['threat_summary']} [{src_ip} -> {dst_ip}]",
            "explanation": f"{conf_prefix} {attack_category} with {conf_desc}. {cat_info['explanation']}",
            "why_flagged": why_flagged_text,
            "severity": severity,
            "recommended_actions": list(cat_info["recommended_actions"]),
            "investigation_guidance": list(cat_info["investigation_guidance"]),
        }

    def _calculate_severity(self, category: str, confidence: float) -> str:
        """Computes severity rating based on confidence and potential threat category impact."""
        cat_lower = category.lower()

        if cat_lower == "normal":
            return "Informational"

        critical_cats = ["exploits", "backdoor", "shellcode"]
        high_cats = ["dos", "worms"]
        medium_cats = ["reconnaissance", "fuzzers", "generic"]

        if confidence >= 0.85:
            if cat_lower in critical_cats:
                return "Critical"
            elif cat_lower in high_cats:
                return "High"
            else:
                return "Medium"
        elif confidence >= 0.60:
            if cat_lower in critical_cats:
                return "High"
            elif cat_lower in high_cats or cat_lower in medium_cats:
                return "Medium"
            else:
                return "Low"
        else:
            return "Low"
