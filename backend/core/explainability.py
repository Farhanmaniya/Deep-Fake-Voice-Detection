import numpy as np
import librosa
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# Mel band index ranges out of 128 total bands (librosa default layout)
BAND_RANGES = {
    "sub_bass":   (0, 10),    # ~0–200 Hz — rumble / GAN noise floor
    "mid":        (10, 50),   # ~200–2000 Hz — core voice formants
    "high":       (50, 100),  # ~2000–6000 Hz — consonants, air
    "ultra_high": (100, 128), # ~6000–8000 Hz — AI artifact zone
}


def compute_band_energies(mel_spec: np.ndarray) -> Dict[str, float]:
    """
    Split the Mel spectrogram into 4 frequency bands and return
    the average energy (dB) of each band over time.

    Args:
        mel_spec: 2D array of shape (n_mels, time_frames), log-Mel scale.

    Returns:
        Dict with keys sub_bass, mid, high, ultra_high — each a float in dB.
    """
    result = {}
    for band_name, (start, end) in BAND_RANGES.items():
        band_slice = mel_spec[start:end, :]          # (band_size, time)
        result[band_name] = float(round(np.mean(band_slice), 4))
    logger.debug(f"Band energies: {result}")
    return result


def compute_spectral_flatness(audio: np.ndarray, sr: int = 16000) -> float:
    """
    Compute mean spectral flatness of the audio signal.

    Spectral flatness (Wiener entropy) measures how noise-like vs
    tone-like a signal is.
      - Values near 1.0 → very flat / machine-like (suspicious)
      - Values near 0.0 → tonal / natural-sounding

    Args:
        audio: 1D numpy array of audio samples (float32, mono).
        sr:    Sample rate (default 16000).

    Returns:
        Float in [0, 1] representing mean spectral flatness.
    """
    flatness = librosa.feature.spectral_flatness(y=audio)
    mean_flatness = float(round(float(np.mean(flatness)), 4))
    logger.debug(f"Spectral flatness: {mean_flatness}")
    return mean_flatness


def compute_temporal_risk(
    mel_spec: np.ndarray,
    chunk_probability: float,
    n_windows: int = 8
) -> List[float]:
    """
    Divide the spectrogram into N equal time windows and estimate a
    per-window risk score using frame-level energy variance.

    High energy variance in a window → high local risk.
    Scores are normalised so they roughly reflect the overall chunk_probability.

    Args:
        mel_spec:          2D log-Mel spectrogram (n_mels, time_frames).
        chunk_probability: Model's overall fake probability for the chunk (0–1).
        n_windows:         Number of temporal windows to produce.

    Returns:
        List of floats (length n_windows), each in [0, 1].
    """
    time_frames = mel_spec.shape[1]
    if time_frames == 0:
        return [0.0] * n_windows

    window_size = max(1, time_frames // n_windows)
    variances = []
    for i in range(n_windows):
        start = i * window_size
        end = min(start + window_size, time_frames)
        window = mel_spec[:, start:end]
        variances.append(float(np.var(window)))

    # Normalise variance to [0, 1]
    max_var = max(variances) if max(variances) > 0 else 1.0
    normalised = [v / max_var for v in variances]

    # Scale by chunk_probability so values reflect the model's confidence
    scaled = [round(v * chunk_probability, 4) for v in normalised]
    logger.debug(f"Temporal risk ({n_windows} windows): {scaled}")
    return scaled


def get_suspicious_band(band_energies: Dict[str, float]) -> str:
    """
    Identify the most suspicious frequency band.

    Heuristic: the band with the highest relative energy compared to
    a reference 'mid' band (GAN artifacts tend to concentrate in
    ultra_high or sub_bass regions).

    Args:
        band_energies: Output of compute_band_energies().

    Returns:
        Name of the most suspicious band (e.g., "ultra_high").
    """
    mid_energy = band_energies.get("mid", -60.0)
    # Compute how much each non-mid band deviates above the mid band
    deviations = {
        band: energy - mid_energy
        for band, energy in band_energies.items()
        if band != "mid"
    }
    suspicious = max(deviations, key=deviations.get)
    logger.debug(f"Suspicious band: {suspicious} (deviation={deviations[suspicious]:.2f} dB)")
    return suspicious


def build_explainability(
    audio: np.ndarray,
    mel_spec: np.ndarray,
    chunk_probability: float,
    sr: int = 16000,
    n_temporal_windows: int = 8,
) -> Dict:
    """
    Convenience wrapper — compute all explainability signals in one call.

    Args:
        audio:             Raw audio array (float32).
        mel_spec:          Log-Mel spectrogram array (n_mels, time_frames).
        chunk_probability: Model fake-probability for the chunk.
        sr:                Sample rate.
        n_temporal_windows: Number of temporal risk windows.

    Returns:
        Dict ready to be embedded in the WebSocket response under
        the "explainability" key.
    """
    band_energies  = compute_band_energies(mel_spec)
    flatness       = compute_spectral_flatness(audio, sr)
    temporal_risk  = compute_temporal_risk(mel_spec, chunk_probability, n_temporal_windows)
    suspicious     = get_suspicious_band(band_energies)

    return {
        "band_energies":    band_energies,
        "spectral_flatness": flatness,
        "suspicious_band":  suspicious,
        "temporal_risk":    temporal_risk,
    }
