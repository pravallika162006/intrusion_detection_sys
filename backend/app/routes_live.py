"""
FastAPI routes for Live Network Monitoring Dashboard.
Supports interface discovery, starting/stopping live capture & Controlled TEST MODE,
status querying, and triggering AI Security Agent recommendations.
"""

import asyncio
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from backend.app.schemas import (
    NetworkInterfaceInfo,
    StartCaptureRequest,
    SetThresholdRequest,
    AIRecommendationRequest,
    AIRecommendationResponse,
)
from backend.live.interface_manager import get_network_interfaces
from backend.live.packet_capturer import PacketCapturerManager
from backend.ai_agent.security_agent import AISecurityAgent
from backend.app.websocket_manager import WebSocketManager
from backend.utils.logger import setup_logger

logger = setup_logger("LiveRoutes")
router = APIRouter(prefix="/api/live", tags=["Live Network Dashboard"])

# Global Singleton Instances for Live Subsystem
ws_manager = WebSocketManager()
capturer = PacketCapturerManager()
ai_agent = AISecurityAgent()
main_event_loop = None


def _ws_event_callback(event_data: Dict[str, Any]):
    """Callback triggered whenever a flow detection event occurs in background worker thread."""
    global main_event_loop
    if main_event_loop is not None and main_event_loop.is_running():
        try:
            asyncio.run_coroutine_threadsafe(ws_manager.broadcast_event(event_data), main_event_loop)
        except Exception as e:
            logger.error(f"Error dispatching WebSocket event across threads: {e}")
    else:
        logger.warning("Main asyncio event loop not captured yet or not running. WebSocket event skipped.")


@router.get("/interfaces", response_model=List[NetworkInterfaceInfo])
def list_interfaces():
    """Returns list of available network interfaces for capture."""
    try:
        return get_network_interfaces()
    except Exception as e:
        logger.error(f"Failed to list network interfaces: {e}")
        raise HTTPException(status_code=500, detail=f"Interface discovery error: {str(e)}")


@router.post("/start")
async def start_monitoring(req: StartCaptureRequest):
    """Starts live packet capture or Controlled TEST MODE."""
    global main_event_loop
    try:
        main_event_loop = asyncio.get_running_loop()
    except Exception:
        pass

    try:
        capturer.start_capture(
            interface_name=req.interface_name,
            test_mode=req.test_mode,
            event_callback=_ws_event_callback,
            threshold=req.threshold,
        )
        mode_str = "Controlled TEST MODE" if req.test_mode else f"Live Capture ({req.interface_name})"
        return {
            "status": "RUNNING",
            "message": f"Network monitoring started in {mode_str}.",
            "interface": req.interface_name,
            "test_mode": req.test_mode,
            "threshold": req.threshold,
        }
    except Exception as e:
        logger.error(f"Failed to start monitoring: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/threshold")
def set_monitoring_threshold(req: SetThresholdRequest):
    """Dynamically updates the operating decision threshold."""
    try:
        capturer.set_threshold(req.threshold)
        return {
            "status": "SUCCESS",
            "threshold": capturer.decision_threshold,
            "message": f"Operating decision threshold set to {capturer.decision_threshold:.2f}",
        }
    except Exception as e:
        logger.error(f"Error setting threshold: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/stop")
def stop_monitoring():
    """Stops live network monitoring and returns final session summary."""
    try:
        summary = {
            "status": "STOPPED",
            "message": "Network monitoring stopped.",
            "interface_name": capturer.interface_name,
            "test_mode": capturer.test_mode,
            "threshold": capturer.decision_threshold,
            "packet_count": capturer.packet_count,
            "flow_count": capturer.flow_count,
            "normal_count": capturer.normal_count,
            "attack_count": capturer.attack_count,
        }
        capturer.stop_capture()
        return summary
    except Exception as e:
        logger.error(f"Error stopping monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
def get_monitoring_status():
    """Returns real-time packet, flow, and detection totals."""
    return {
        "is_running": capturer.is_running,
        "test_mode": capturer.test_mode,
        "interface": capturer.interface_name,
        "threshold": capturer.decision_threshold,
        "packet_count": capturer.packet_count,
        "flow_count": capturer.flow_count,
        "normal_count": capturer.normal_count,
        "attack_count": capturer.attack_count,
    }


@router.post("/ai-recommendation", response_model=AIRecommendationResponse)
def get_ai_recommendation(req: AIRecommendationRequest):
    """Generates AI Security Agent recommendations for a specified detection event."""
    try:
        rec = ai_agent.generate_recommendation(
            flow_id=req.flow_id,
            src_ip=req.src_ip,
            dst_ip=req.dst_ip,
            service=req.service,
            attack_category=req.attack_category,
            confidence=req.confidence,
            flow_info=req.flow_info,
        )
        return AIRecommendationResponse(**rec)
    except Exception as e:
        logger.error(f"Error generating AI recommendation: {e}")
        raise HTTPException(status_code=500, detail=f"AI Agent error: {str(e)}")


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint streaming real-time live flow events."""
    global main_event_loop
    try:
        main_event_loop = asyncio.get_running_loop()
    except Exception:
        pass

    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)

