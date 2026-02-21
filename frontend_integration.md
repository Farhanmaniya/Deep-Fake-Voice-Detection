# Frontend Integration Guide: Deepfake Voice Detector 🚀

The backend has been upgraded with three powerful features: **Explainability**, **Attribution**, and **Robustness**. 

## WebSocket Connection
- **URL**: `ws://<host>:<port>/ws/audio`
- **Protocol**: Streaming binary audio (PCM 16-bit, 16kHz Mono).

## NEW Response Schema
Every successful chunk processing now returns an enriched JSON object.

### 1. Model Inference (Primary)
- `chunk_probability`: Float [0.0, 1.0]. Probability that the audio is a deepfake.
- `rolling_risk`: A smoothed average risk score across a sliding window.
- `risk_level`: `"LOW"`, `"MEDIUM"`, or `"HIGH"` based on probability.

### 2. Explainability (`explainability`) ✅
Helps explain **WHY** the model flagged the audio.
- `suspicious_band`: The frequency range exhibiting the most abnormal behavior (e.g., `"high"`, `"ultra_high"`).
- `band_energies`: RMS energy across sub-bass, mid, high, and ultra-high.
- `spectral_flatness`: Indicates how "unnatural" or robotic the signal is.

### 3. Attribution (`attribution`) 🕵️‍♂️
Identifies **WHICH** tool likely generated the voice.
- `suspected_generator`: e.g., `"ElevenLabs"`, `"RVC"`, `"Bark"`, or `"Human"`.
- `generator_confidence`: Score [0-1] of the attribution matched against signal footprints.

### 4. Robustness (`robustness`) 🛡️
Shows **SIGNAL QUALITY** and defense status.
- `snr_db`: Signal-to-Noise Ratio.
- `is_low_quality`: Boolean. If `true`, the UI should show a warning (e.g., "Background noise is too high").
- `quality_score`: A percentage (0-100) combining SNR and clipping metrics.

---

## Example JSON Response
```json
{
  "status": "success",
  "chunk_probability": 0.884,
  "rolling_risk": 0.72,
  "risk_level": "HIGH",
  "explainability": {
    "suspicious_band": "high",
    "spectral_flatness": 0.12
  },
  "attribution": {
    "suspected_generator": "ElevenLabs",
    "generator_confidence": 0.82
  },
  "robustness": {
    "snr_db": 12.5,
    "is_low_quality": true,
    "warnings": ["Low SNR detected"]
  },
  "processing_latency_ms": 145.2
}
```

## Best Practices for Frontend
1. **Dynamic Warnings**: If `robustness.is_low_quality` is true, display a "Poor Audio Quality" badge.
2. **Attribution Display**: Only show `attribution` if `chunk_probability > 0.5`. Otherwise, it's likely a human voice.
3. **Risk Level Colors**: 
   - `LOW`: Green
   - `MEDIUM`: Yellow/Orange
   - `HIGH`: Red
