from fastapi import APIRouter
from backend.core.connection_manager import manager
from backend.config.settings import settings
from backend.core.model_loader import get_model, is_mock_model
import logging
import datetime

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/health")
async def health_check():
    """
    Production-ready health check endpoint.
    
    Returns:
        JSON with server status, model mode, connection count, version, and timestamp.
    """
    logger.info("Health check endpoint called")
    
    # Get model instance to check status
    model = get_model(settings.MODEL_PATH)
    mock_mode = is_mock_model(model)
    
    # Determine status
    status = "degraded" if mock_mode else "ok"
    model_mode = "mock" if mock_mode else "production"
    
    # Get active connections from manager
    active_connections = len(manager.active_connections)
    
    return {
        "status": status,
        "model_mode": model_mode,
        "active_connections": active_connections,
        "version": settings.APP_VERSION,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }
