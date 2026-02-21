import time
from collections import deque
import logging

logger = logging.getLogger(__name__)

class RateLimiter:
    """
    Per-connection rate limiter to prevent flooding.
    """
    
    def __init__(self, max_chunks_per_second: int):
        """
        Initialize rate limiter.
        
        Args:
            max_chunks_per_second: Maximum allowed chunks per second
        """
        self.max_chunks_per_second = max_chunks_per_second
        self.timestamps = deque()
        
        logger.info(f"RateLimiter initialized: max_rate={max_chunks_per_second} chunks/sec")
    
    def check_rate(self) -> tuple[bool, float]:
        """
        Check if current rate is within limits.
        
        Returns:
            (is_allowed, current_rate)
        """
        current_time = time.time()
        
        # Remove timestamps older than 1 second
        while self.timestamps and current_time - self.timestamps[0] > 1.0:
            self.timestamps.popleft()
        
        # Calculate current rate
        current_rate = len(self.timestamps)
        
        # Check if we can accept another chunk
        is_allowed = current_rate < self.max_chunks_per_second
        
        if is_allowed:
            # Add current timestamp
            self.timestamps.append(current_time)
        else:
            logger.warning(
                f"Rate limit exceeded: {current_rate}/{self.max_chunks_per_second} chunks/sec"
            )
        
        return is_allowed, float(current_rate)
