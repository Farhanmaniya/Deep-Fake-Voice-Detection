import threading
import time
from typing import Dict
from collections import deque

class MetricsCollector:
    """
    Thread-safe singleton metrics collector for monitoring system health.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self.active_connections = 0
        self.total_chunks_processed = 0
        self.total_inference_calls = 0
        self.total_errors = 0
        self.latency_buffer = deque(maxlen=100)  # Keep last 100 latencies
        self.start_time = time.time()
        self._lock = threading.Lock()
    
    def increment_connections(self):
        """Increment active connections count."""
        with self._lock:
            self.active_connections += 1
    
    def decrement_connections(self):
        """Decrement active connections count."""
        with self._lock:
            self.active_connections = max(0, self.active_connections - 1)
    
    def record_chunk(self, latency_ms: float):
        """
        Record a processed chunk.
        
        Args:
            latency_ms: Processing latency in milliseconds
        """
        with self._lock:
            self.total_chunks_processed += 1
            self.latency_buffer.append(latency_ms)
    
    def record_error(self):
        """Record an error."""
        with self._lock:
            self.total_errors += 1
            
    def record_inference(self):
        """Record an inference call."""
        with self._lock:
            self.total_inference_calls += 1
    
    def get_metrics(self) -> Dict:
        """
        Get current metrics snapshot.
        
        Returns:
            Dictionary with current metrics
        """
        with self._lock:
            avg_latency = (
                sum(self.latency_buffer) / len(self.latency_buffer)
                if self.latency_buffer else 0.0
            )
            
            uptime_seconds = time.time() - self.start_time
            
            # Get model status
            from backend.core.model_loader import get_model, is_mock_model
            from backend.config.settings import settings
            model = get_model(settings.MODEL_PATH)
            model_mode = "mock" if is_mock_model(model) else "production"
            
            return {
                "active_connections": self.active_connections,
                "total_chunks_processed": self.total_chunks_processed,
                "total_inference_calls": self.total_inference_calls,
                "total_errors": self.total_errors,
                "average_latency_ms": round(avg_latency, 2),
                "uptime_seconds": round(uptime_seconds, 2),
                "chunks_per_second": round(
                    self.total_chunks_processed / uptime_seconds if uptime_seconds > 0 else 0,
                    2
                ),
                "model_mode": model_mode
            }

# Global singleton instance
metrics = MetricsCollector()
