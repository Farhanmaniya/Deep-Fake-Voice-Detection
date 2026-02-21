# Project Presentation Guide: Real-time Deepfake Voice Detection 🛡️🎙️

This document provides a comprehensive overview of the system for use in the Hackathon PowerPoint presentation.

## 1. The Core Vision
The system provides a **low-latency, real-time defense** against audio deepfakes. It doesn't just say "Fake" or "Real"—it explains **why**, identifies the **source**, and handles **real-world noise**.

---

## 2. Technical Stack
- **Backend Framework**: FastAPI (Python)
- **Communication**: WebSockets (Bi-directional streaming)
- **Signal Processing**: Librosa & SciPy
- **AI Framework**: PyTorch
- **Key Architectures**: 
    - **CNN (Convolutional Neural Network)**: For frequency pattern recognition.
    - **LSTM (Long Short-Term Memory)**: For temporal/rhythm analysis.

---

## 3. Five Pillar Features (The "Wow" Factors)

### ✅ Pillar 1: Real-time ML Inference
The system processes audio in small "chunks" (~1 second) to provide near-instant feedback.
- **Model**: Custom `DeepfakeCNN` trained on MFCC (Mel-Frequency Cepstral Coefficients).
- **Latency**: Sub-200ms processing time.

### ✅ Pillar 2: Explainability (XAI)
Transparency for the user. We don't believe in "Black Box" AI.
- **Suspicious Band Detection**: Identifies exactly which frequency range (Sub-bass, Mid, High, Ultra-high) is showing AI artifacts.
- **Spectral Flatness**: Measures the "unnaturalness" of the voice signal.

### ✅ Pillar 3: Attribution (Source Identification)
Identifies the "DNA" of the generator.
- **Profiles**: Matches signal footprints against known generators like **ElevenLabs**, **RVC (Retrieval-based Voice Conversion)**, and **Bark**.
- **Human Baseline**: Can distinguish between a synthesized voice and a natural human profile.

### ✅ Pillar 4: Robustness & Self-Healing
Real-world audio is dirty; our AI is ready for it.
- **SNR Estimation**: Real-time Signal-to-Noise Ratio calculation.
- **Adaptive Denoising**: If the background noise is too high (>15dB), the system automatically triggers a **Spectral Noise Reduction** filter to "clean" the voice before inference.
- **Clipping Detection**: Alerts the user if the audio is distorted/peaking.

### ✅ Pillar 5: Multi-Model Consensus (In Progress)
Two minds are better than one.
- **Mechanism**: The system runs a **Temporal CNN** and a **Temporal LSTM** in parallel.
- **Decision**: The final "Risk Level" is a consensus between both models, significantly reducing False Positives.

---

## 4. Architectural Workflow (For the "System Diagram" Slide)
1. **Input**: Raw PCM 16-bit audio streamed via WebSockets.
2. **Pre-processing**: Normalization -> Resampling -> VAD (Voice Activity Detection).
3. **Robustness Check**: If noisy? -> **Denoise**. If clipping? -> **Warn**.
4. **Feature Extraction**: 
    - Mel Spectrogram (for XAI & Attribution)
    - MFCC (for CNN Inference)
    - Spectral Time-Frames (for LSTM Inference)
5. **Parallel Inference**: Models process features simultaneously.
6. **Output**: Consolidated JSON response with probability, risk levels, and forensic metadata.

---

## 5. Potential Use Cases
- **FinTech Security**: Preventing "Voice Clone" fraud in banking calls.
- **Media Verification**: Newsrooms verifying "leaked" audio clips.
- **Personal Defense**: Real-time protection for individuals against social engineering.
