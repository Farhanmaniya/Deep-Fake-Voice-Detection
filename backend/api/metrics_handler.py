from fastapi import APIRouter
from backend.core.metrics import metrics

router = APIRouter()

@router.get("/metrics")
async def get_metrics():
    """
    Get current system metrics.
    
    Returns:
        JSON with metrics including:
        - active_connections
        - total_chunks_processed
        - total_errors
        - average_latency_ms
        - uptime_seconds
        - chunks_per_second
    """
    return metrics.get_metrics()
