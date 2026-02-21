from collections import deque
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class RiskEngine:
    """
    Rolling risk engine that maintains a sliding window of probabilities
    and computes rolling average risk scores with optional smoothing.
    """
    
    def __init__(
        self, 
        window_size: int, 
        threshold_low: float, 
        threshold_high: float,
        smoothing_factor: float = 0.0
    ):
        """
        Initialize risk engine.
        
        Args:
            window_size: Number of chunks to keep in sliding window
            threshold_low: Threshold for LOW risk level
            threshold_high: Threshold for HIGH risk level
            smoothing_factor: EMA smoothing factor [0, 1]. 0 = no smoothing, 1 = full smoothing
        """
        self.window_size = window_size
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.smoothing_factor = smoothing_factor
        self.probability_buffer = deque(maxlen=window_size)
        self.previous_probability: Optional[float] = None
        
        logger.info(
            f"RiskEngine initialized: window_size={window_size}, "
            f"thresholds=[LOW<{threshold_low}, MEDIUM<{threshold_high}, HIGH≥{threshold_high}], "
            f"smoothing_factor={smoothing_factor}"
        )
    
    def update(self, probability: float) -> Dict:
        """
        Update risk engine with new probability and compute rolling risk.
        
        Args:
            probability: Chunk probability [0, 1]
        
        Returns:
            Dictionary containing:
                - rolling_risk: float (rolling average)
                - risk_level: str (LOW/MEDIUM/HIGH)
                - window_size: int (current buffer size)
                - smoothed_probability: float (after smoothing)
        """
        # Apply exponential moving average smoothing
        if self.smoothing_factor > 0 and self.previous_probability is not None:
            smoothed_prob = (
                (1 - self.smoothing_factor) * probability + 
                self.smoothing_factor * self.previous_probability
            )
        else:
            smoothed_prob = probability
        
        # Update previous for next iteration
        self.previous_probability = smoothed_prob
        
        # Add smoothed probability to buffer
        self.probability_buffer.append(smoothed_prob)
        
        # Compute rolling average
        rolling_risk = sum(self.probability_buffer) / len(self.probability_buffer)
        
        # Categorize risk level
        if rolling_risk < self.threshold_low:
            risk_level = "LOW"
        elif rolling_risk < self.threshold_high:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
        
        logger.info(
            f"Risk updated: raw_prob={probability:.4f}, smoothed_prob={smoothed_prob:.4f}, "
            f"rolling_risk={rolling_risk:.4f}, level={risk_level}, "
            f"buffer_size={len(self.probability_buffer)}/{self.window_size}"
        )
        
        return {
            "rolling_risk": round(rolling_risk, 4),
            "risk_level": risk_level,
            "window_size": len(self.probability_buffer),
            "smoothed_probability": round(smoothed_prob, 4)
        }
    
    def reset(self):
        """Reset the risk engine buffer."""
        self.probability_buffer.clear()
        self.previous_probability = None
        logger.info("Risk engine buffer reset")
