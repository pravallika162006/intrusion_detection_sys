"""
FastAPI routes for downloading Dataset and Live Monitoring PDF Reports.
"""

from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Response
from backend.reporting.pdf_generator import (
    build_dataset_pdf_report,
    build_live_pdf_report,
    build_performance_pdf_report,
)
from backend.utils.logger import setup_logger

logger = setup_logger("ReportingRoutes")
router = APIRouter(prefix="/api/reports", tags=["PDF Reports"])


@router.post("/dataset-pdf")
def generate_dataset_pdf_report(payload: Dict[str, Any]):
    """
    Generates and returns a PDF report for Dataset Analysis.
    """
    try:
        pdf_bytes = build_dataset_pdf_report(payload)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": "attachment; filename=IDS_Dataset_Analysis_Report.pdf"
            },
        )
    except Exception as e:
        logger.error(f"Error generating dataset PDF report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate dataset PDF: {str(e)}")


@router.post("/live-pdf")
def generate_live_pdf_report(payload: Dict[str, Any]):
    """
    Generates and returns a PDF report for a Live Monitoring Session.
    """
    try:
        pdf_bytes = build_live_pdf_report(payload)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": "attachment; filename=IDS_Live_Monitoring_Session_Report.pdf"
            },
        )
    except Exception as e:
        logger.error(f"Error generating live PDF report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate live PDF: {str(e)}")


@router.post("/performances-pdf")
def generate_performances_pdf_report(payload: Dict[str, Any]):
    """
    Generates and returns an Academic PDF Report for Model Performances (Phases 1, 2, 3).
    """
    try:
        pdf_bytes = build_performance_pdf_report(payload)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": "attachment; filename=IDS_Academic_Performance_Report.pdf"
            },
        )
    except Exception as e:
        logger.error(f"Error generating performance PDF report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate performance PDF: {str(e)}")

