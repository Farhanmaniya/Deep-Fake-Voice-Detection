"""
backend/core/robustness.py
==========================
Signal quality analysis and enhancement module.
Features:
 - SNR Estimation
 - Noise Floor Calculation
 - Spectral Noise Reduction (Denoising)
 - Telephony Simulation (Bandpass)
 - Adversarial Noise Injection (Demo Mode)
"""

import numpy as np
import librosa
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

def estimate_snr(audio: np.ndarray) -> float:
    """
    Estimate Signal-to-Noise Ratio (SNR) in dB.
    Uses a simplified approach: 95th percentile vs 5th percentile (RMS-based).
    """
    if len(audio) == 0:
        return 0.0
    
    # Use windowed RMS to find signal vs noise periods
    frame_length = 1024
    hop_length = 512
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    
    if len(rms) < 2:
        return 0.0

    # Signal is the loudest parts, noise is the quietest parts
    signal_power = np.percentile(rms, 95)
    noise_power = np.percentile(rms, 5)
    
    # If the signal is very steady (std is low), noise_power will be same as signal_power
    # but that usually means it's a clean artificial signal or a constant loud noise.
    # If std is low and signal is loud, it's a clean signal.
    rms_std = np.std(rms)
    rms_mean = np.mean(rms)
    
    if rms_std < 0.15 * rms_mean:
        # Very steady signal. If it's loud (> -40dB), assume high SNR.
        if rms_mean > 0.005:
            return 50.0  # Clean steady signal
        else:
            return 5.0   # Steady quiet noise
            
    if noise_power < 1e-7:
        # Near silent floor
        return 55.0  # High SNR
        
    snr_db = 20 * np.log10(signal_power / (noise_power + 1e-9))
    return float(np.clip(snr_db, 0, 80))

def calculate_noise_floor(audio: np.ndarray) -> float:
    """Calculate the base noise floor in dB."""
    if len(audio) == 0:
        return -100.0
    rms = librosa.feature.rms(y=audio)[0]
    noise_floor = np.min(rms) if len(rms) > 0 else 0.0
    return float(librosa.amplitude_to_db([noise_floor + 1e-9])[0])

def apply_denoising(audio: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    Lightweight spectral noise reduction using STFT thresholding.
    Note: For production, consider using a dedicated library like 'noisereduce'.
    """
    # Simple spectral subtraction approach
    S = librosa.stft(audio)
    mag, phase = librosa.magphase(S)
    
    # Estimate noise from the first few frames (assuming they are silent)
    noise_est = np.mean(mag[:, :min(10, mag.shape[1])], axis=1, keepdims=True)
    
    # Subtract noise
    mag_clean = np.maximum(mag - (noise_est * strength), 0)
    
    # Reconstruct
    S_clean = mag_clean * phase
    audio_clean = librosa.istft(S_clean, length=len(audio))
    
    return audio_clean

def check_clipping(audio: np.ndarray, threshold: float = 0.99) -> float:
    """Returns the percentage of samples that are clipped."""
    if len(audio) == 0:
        return 0.0
    clipped = np.sum(np.abs(audio) >= threshold)
    return float((clipped / len(audio)) * 100)

def build_robustness_analysis(audio: np.ndarray, snr_threshold: float = 15.0) -> Dict:
    """
    Aggregate robustness metrics for the API response.
    """
    snr = estimate_snr(audio)
    noise_floor = calculate_noise_floor(audio)
    clipping = check_clipping(audio)
    
    # Determine overall quality
    is_poor_quality = snr < snr_threshold or clipping > 2.0
    quality_score = np.clip((snr / 40.0) * 100, 0, 100) if not is_poor_quality else np.clip((snr / 40.0) * 80, 0, 70)

    return {
        "snr_db": round(snr, 1),
        "noise_floor_db": round(noise_floor, 1),
        "clipping_percent": round(clipping, 2),
        "quality_score": round(quality_score, 1),
        "is_low_quality": is_poor_quality,
        "warnings": ["Low SNR detected"] if snr < snr_threshold else []
    }

# --- Demo Mode Helpers ---

def inject_noise(audio: np.ndarray, snr_db: float = 15.0) -> np.ndarray:
    """Inject white noise to achieve a target SNR."""
    sig_avg_watts = np.mean(audio**2)
    sig_avg_db = 10 * np.log10(sig_avg_watts + 1e-9)
    noise_avg_db = sig_avg_db - snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    
    noise = np.random.normal(0, np.sqrt(noise_avg_watts), len(audio)).astype(np.float32)
    return audio + noise

def simulate_telephony(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """mimic 8kHz band-limited phone call."""
    # Bandpass filter between 300Hz and 3400Hz
    sos = librosa.filters.butter_bandpass(300, 3400, sr=sr, order=5)
    from scipy.signal import sosfilt
    return sosfilt(sos, audio).astype(np.float32)
