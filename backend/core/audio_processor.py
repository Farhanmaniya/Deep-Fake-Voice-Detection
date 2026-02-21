import numpy as np
import logging
from backend.config.settings import settings

logger = logging.getLogger(__name__)

def bytes_to_numpy(data: bytes) -> np.ndarray:
    """
    Convert binary audio data (PCM 16-bit) to numpy float32 array.
    
    Args:
        data: Binary audio data (PCM 16-bit signed integers)
    
    Returns:
        Numpy array of float32 values
    """
    # Convert bytes to int16 array
    audio_int16 = np.frombuffer(data, dtype=np.int16)
    # Convert to float32 in range [-1.0, 1.0]
    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    return audio_float32

def validate_audio(audio: np.ndarray, sample_rate: int) -> tuple[bool, str]:
    """
    Validate audio array against duration and shape constraints.
    
    Args:
        audio: Audio array
        sample_rate: Sample rate in Hz
    
    Returns:
        (is_valid, error_message)
    """
    if len(audio.shape) > 1:
        return False, f"Expected 1D audio array, got shape {audio.shape}"
    
    duration = len(audio) / sample_rate
    
    if duration < settings.MIN_DURATION:
        return False, f"Audio too short: {duration:.2f}s < {settings.MIN_DURATION}s"
    
    if duration > settings.MAX_DURATION:
        return False, f"Audio too long: {duration:.2f}s > {settings.MAX_DURATION}s"
    
    return True, ""

def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Safely normalize audio amplitude to [-1.0, 1.0] range.
    
    Args:
        audio: Audio array
    
    Returns:
        Normalized audio array
    """
    max_val = np.abs(audio).max()
    
    # Avoid division by zero
    if max_val < 1e-8:
        logger.warning("Audio signal is near-silent, skipping normalization")
        return audio
    
    return audio / max_val

def ensure_mono_16khz(audio: np.ndarray, current_sr: int) -> np.ndarray:
    """
    Ensure audio is mono and at 16kHz sample rate.
    
    Args:
        audio: Audio array (can be stereo)
        current_sr: Current sample rate
    
    Returns:
        Mono audio at 16kHz
    """
    # Convert stereo to mono if needed
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    # Resample if needed
    if current_sr != settings.SAMPLE_RATE:
        import librosa
        audio = librosa.resample(audio, orig_sr=current_sr, target_sr=settings.SAMPLE_RATE)
        logger.info(f"Resampled from {current_sr}Hz to {settings.SAMPLE_RATE}Hz")
    
    return audio

def apply_denoising_if_needed(audio: np.ndarray, snr: float, threshold: float = 15.0) -> np.ndarray:
    """
    Apply spectral denoising ONLY if SNR is below threshold.
    """
    if snr < threshold:
        from backend.core.robustness import apply_denoising
        logger.info(f"Applying denoising (SNR {snr:.1f} dB < {threshold} dB)")
        return apply_denoising(audio)
    return audio
