"""
backend/core/attribution.py

Feature 2: AI Voice Generator Attribution
==========================================
Uses rule-based spectral fingerprinting to identify which AI voice
generator likely produced an audio chunk.

No ML model is required — each major TTS/voice-cloning tool leaves
measurable spectral signatures that can be detected with fast signal math.

Known Generator Profiles
------------------------
| Generator     | Hallmarks                                                |
|---------------|----------------------------------------------------------|
| ElevenLabs    | Very clean high-freq roll-off, low ZCR, low sub-bass    |
| RVC           | Elevated mid-band resonance, high flatness, moderate ZCR |
| Bark / VALL-E | High temporal energy variance, erratic mel pattern       |
| Generic TTS   | Uniform mel energy, flat centroid, low flatness variance |
| Human         | High centroid variability, natural sub-bass presence     |
"""

import numpy as np
import librosa
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Known generator spectral profiles (heuristic thresholds)
# Each profile is matched by evaluating all rules and scoring.
# ─────────────────────────────────────────────────────────────
_GENERATOR_PROFILES = [
    {
        "name": "ElevenLabs",
        "rules": [
            # Very clean, limited sub-bass; high-freq energy drops sharply
            lambda f: f["sub_bass_ratio"] < 0.12,
            lambda f: f["zero_crossing_rate"] < 0.06,
            lambda f: f["spectral_rolloff_hz"] < 5000,
            lambda f: f["mel_energy_variance"] < 50,
        ],
        "weight": 1.0,
    },
    {
        "name": "RVC (Voice Clone)",
        "rules": [
            # Mid-band resonance artefacts; noisy timbre from vocoder
            lambda f: f["spectral_flatness"] > 0.25,
            lambda f: f["zero_crossing_rate"] > 0.07,
            lambda f: f["spectral_centroid_hz"] > 1500,
            lambda f: f["mel_energy_variance"] > 60,
        ],
        "weight": 1.0,
    },
    {
        "name": "Bark / VALL-E",
        "rules": [
            # Temporal energy is erratic; codec-like artefacts raise ZCR
            lambda f: f["temporal_energy_variance"] > 0.04,
            lambda f: f["zero_crossing_rate"] > 0.08,
            lambda f: f["spectral_rolloff_hz"] > 5500,
        ],
        "weight": 1.0,
    },
    {
        "name": "Generic TTS",
        "rules": [
            # Very flat, steady signal — typical of rule-based / old TTS
            lambda f: f["spectral_flatness"] < 0.15,
            lambda f: f["mel_energy_variance"] < 30,
            lambda f: f["temporal_energy_variance"] < 0.02,
        ],
        "weight": 0.9,
    },
    {
        "name": "Human",
        "rules": [
            # Natural speech has moderate ZCR, richer sub-bass, higher centroid variance
            lambda f: f["sub_bass_ratio"] >= 0.12,
            lambda f: f["spectral_centroid_hz"] > 1200,
            lambda f: 0.03 < f["zero_crossing_rate"] < 0.12,
            lambda f: f["temporal_energy_variance"] > 0.01,
        ],
        "weight": 1.0,
    },
]


def compute_spectral_features(
    audio: np.ndarray,
    mel_spec: np.ndarray,
    band_energies: Dict[str, float],
    sr: int = 16000,
) -> Dict[str, float]:
    """
    Compute spectral features used for generator attribution.

    Args:
        audio:        Raw audio array (float32, mono).
        mel_spec:     Log-Mel spectrogram (n_mels, time_frames).
        band_energies: Output of compute_band_energies() — dict of 4 bands.
        sr:           Sample rate.

    Returns:
        Dict of scalar spectral features.
    """
    # Spectral centroid — "brightness" of the signal
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    spectral_centroid_hz = float(np.mean(centroid))

    # Spectral rolloff — frequency below which 85% of energy sits
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85)
    spectral_rolloff_hz = float(np.mean(rolloff))

    # Zero crossing rate — how oscillatory / noisy the signal is
    zcr = librosa.feature.zero_crossing_rate(audio)
    zero_crossing_rate = float(np.mean(zcr))

    # Mel energy variance across all bands
    mel_energy_variance = float(np.var(mel_spec))

    # Sub-bass ratio — proportion of power in lowest 10 Mel bands
    # Convert log-Mel back to linear power for ratio computation
    linear_mel = librosa.db_to_power(mel_spec)
    total_power = np.sum(linear_mel) + 1e-10
    sub_bass_power = np.sum(linear_mel[:10, :])
    sub_bass_ratio = float(sub_bass_power / total_power)

    # Temporal energy variance — frame-by-frame RMS energy variance
    rms = librosa.feature.rms(y=audio)[0]
    temporal_energy_variance = float(np.var(rms))

    # Spectral flatness — included so the rule table can reference it
    flatness = librosa.feature.spectral_flatness(y=audio)
    spectral_flatness = float(np.mean(flatness))

    features = {
        "spectral_centroid_hz":    round(spectral_centroid_hz, 2),
        "spectral_rolloff_hz":     round(spectral_rolloff_hz, 2),
        "zero_crossing_rate":      round(zero_crossing_rate, 6),
        "mel_energy_variance":     round(mel_energy_variance, 4),
        "sub_bass_ratio":          round(sub_bass_ratio, 6),
        "temporal_energy_variance": round(temporal_energy_variance, 6),
        "spectral_flatness":       round(spectral_flatness, 6),
    }
    logger.debug(f"Spectral features for attribution: {features}")
    return features


def attribute_generator(
    spectral_features: Dict[str, float],
    chunk_probability: float,
) -> Tuple[str, float]:
    """
    Match spectral features against known generator profiles.

    Each profile's rules are evaluated; the proportion that pass
    is the raw score. Scores are normalised to produce a confidence.

    Args:
        spectral_features: Output of compute_spectral_features().
        chunk_probability: Model's fake probability for the chunk.

    Returns:
        Tuple of (suspected_generator: str, confidence: float [0,1]).
    """
    # For real-sounding audio (low fake probability), force Human attribution
    if chunk_probability < 0.35:
        return "Human", round(1.0 - chunk_probability, 4)

    scores: Dict[str, float] = {}
    for profile in _GENERATOR_PROFILES:
        passed = sum(1 for rule in profile["rules"] if _safe_eval(rule, spectral_features))
        ratio = passed / len(profile["rules"])
        scores[profile["name"]] = ratio * profile["weight"]

    best_generator = max(scores, key=scores.get)
    best_score = scores[best_generator]

    # Normalise confidence: scale by fake probability to avoid false positives
    total = sum(scores.values()) or 1.0
    raw_confidence = best_score / total
    # Blend with chunk_probability to dampen confidence on borderline chunks
    confidence = round(raw_confidence * (0.5 + chunk_probability * 0.5), 4)
    confidence = min(1.0, max(0.0, confidence))

    logger.debug(f"Attribution scores: {scores} → {best_generator} @ {confidence:.3f}")
    return best_generator, confidence


def _safe_eval(rule, features: Dict) -> bool:
    """Evaluate a rule safely, returning False on any error."""
    try:
        return bool(rule(features))
    except Exception:
        return False


def build_attribution(
    audio: np.ndarray,
    mel_spec: np.ndarray,
    band_energies: Dict[str, float],
    chunk_probability: float,
    sr: int = 16000,
) -> Dict:
    """
    Convenience wrapper — compute all attribution signals in one call.

    Args:
        audio:             Raw audio array (float32).
        mel_spec:          Log-Mel spectrogram (n_mels, time_frames).
        band_energies:     Output of compute_band_energies().
        chunk_probability: Model's fake probability for the chunk.
        sr:                Sample rate.

    Returns:
        Dict ready to embed in the WebSocket response under "attribution".
    """
    spectral_features = compute_spectral_features(audio, mel_spec, band_energies, sr)
    generator, confidence = attribute_generator(spectral_features, chunk_probability)

    return {
        "suspected_generator":    generator,
        "generator_confidence":   confidence,
        "spectral_centroid_hz":   spectral_features["spectral_centroid_hz"],
        "spectral_rolloff_hz":    spectral_features["spectral_rolloff_hz"],
        "zero_crossing_rate":     spectral_features["zero_crossing_rate"],
        "sub_bass_ratio":         spectral_features["sub_bass_ratio"],
        "note": "Rule-based heuristic — indicative, not definitive",
    }
