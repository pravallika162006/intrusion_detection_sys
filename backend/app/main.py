"""
Main FastAPI Application Entry Point for IDS Phase 3.
Integrates Dataset Dashboard, Live Monitoring Dashboard, WebSockets, and UI static files.
"""

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.app.routes_dataset import router as dataset_router
from backend.app.routes_live import router as live_router
from backend.config import BASE_DIR
from backend.utils.logger import setup_logger

logger = setup_logger("FastAPIApp")

app = FastAPI(
    title="Intrusion Detection System (IDS) - Phase 3",
    description="Dual-Dashboard IDS supporting Dataset Analysis & Real-Time Live Network Monitoring with AI Security Agent.",
    version="3.0.0",
)

# CORS Configuration for local browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routers
app.include_router(dataset_router)
app.include_router(live_router)

# Mount Frontend Static Directory if present
frontend_dir = BASE_DIR / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/", include_in_schema=False)
    async def read_index():
        index_file = frontend_dir / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return {"message": "IDS Phase 3 API running. Index UI file not found."}


@app.get("/api/health")
def health_check():
    return {
        "status": "HEALTHY",
        "phase": 3,
        "application": "Intrusion Detection System (IDS)",
        "mode": "Dual Dashboard (Dataset & Live Monitoring)",
    }
