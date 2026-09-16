"""
AI Security Agent engine.
Provides contextual threat explanations, severity assessment, and actionable security recommendations.
Supports API call with deterministic rule-based fallback mechanism.
"""

import os
import json
from typing import Dict, Any
from backend.ai_agent.prompt_templates import RULE_FALLBACK_DATABASE, SYSTEM_PROMPT
from backend.utils.logger import setup_logger

logger = setup_logger("AISecurityAgent")


class AISecurityAgent:
    """
    AI Security Agent evaluating network intrusion alerts and returning structured security guidance.
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
        Generates security recommendations for a detected flow event.
        """
        # If API key exists, try API generation first
        if self.api_key:
            try:
                res = self._call_ai_api(
                    flow_id, src_ip, dst_ip, service, attack_category, confidence, flow_info
                )
                if res:
                    return res
            except Exception as e:
                logger.warning(f"AI API call failed, switching to rule fallback: {e}")

        # Fallback to Rule Engine
        return self._rule_fallback(src_ip, dst_ip, service, attack_category, confidence, flow_info)

    def _call_ai_api(
        self,
        flow_id: str,
        src_ip: str,
        dst_ip: str,
        service: str,
        attack_category: str,
        confidence: float,
        flow_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calls OpenAI or Gemini API if requests package and key are present."""
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
            f"Model Confidence: {confidence}\n"
            f"Flow Statistics: {json.dumps(flow_info)}\n"
        )

        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_user},
            ],
            "temperature": 0.3,
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
        """Generates deterministic rule-based security advice."""
        cat_info = RULE_FALLBACK_DATABASE.get(
            attack_category,
            {
                "threat_summary": f"Network Intrusion Detected ({attack_category})",
                "explanation": f"Suspicious traffic pattern matched the '{attack_category}' category with {round(confidence * 100, 1)}% model confidence.",
                "severity": "Medium" if confidence < 0.85 else "High",
                "recommended_actions": [
                    f"Inspect host {src_ip} for unauthorized port activity.",
                    f"Verify whether target service '{service}' on {dst_ip} is protected.",
                    "Review recent firewall logs for suspicious connections."
                ],
                "investigation_guidance": [
                    "Examine flow packet/byte ratios and connection state.",
                    "Check target server system logs for abnormal authentication attempts."
                ]
            }
        )

        return {
            "threat_summary": f"{cat_info['threat_summary']} [{src_ip} -> {dst_ip}]",
            "explanation": f"{cat_info['explanation']} (Model Confidence: {round(confidence*100, 1)}%)",
            "severity": cat_info["severity"],
            "recommended_actions": cat_info["recommended_actions"],
            "investigation_guidance": cat_info["investigation_guidance"],
        }
