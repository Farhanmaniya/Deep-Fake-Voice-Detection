import numpy as np
import logging

logger = logging.getLogger(__name__)

def detect_voice_activity(audio: np.ndarray, threshold: float) -> tuple[bool, float]:
    """
    Detect voice activity using energy-based thresholding.
    
    Args:
        audio: Audio signal (normalized float32)
        threshold: Energy threshold for voice detection
    
    Returns:
        (is_voice, energy_level)
    """
    # Compute RMS (Root Mean Square) energy
    energy = np.sqrt(np.mean(audio ** 2))
    
    # Check if energy exceeds threshold
    is_voice = energy >= threshold
    
    logger.debug(f"VAD: energy={energy:.6f}, threshold={threshold:.6f}, is_voice={is_voice}")
    
    return is_voice, float(energy)
