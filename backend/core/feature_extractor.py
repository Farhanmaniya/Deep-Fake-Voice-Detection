import numpy as np
import librosa
import asyncio
import logging
import time
from backend.config.settings import settings

logger = logging.getLogger(__name__)

def extract_mel_spectrogram(
    audio: np.ndarray,
    sr: int,
    n_mels: int,
    hop_length: int,
    n_fft: int
) -> np.ndarray:
    """
    Compute Log-Mel spectrogram from audio signal.
    
    Args:
        audio: Audio time series
    """
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        hop_length=hop_length,
        n_fft=n_fft
    )
    
    # Convert to log scale (dB)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    return log_mel_spec

def extract_mfcc(
    audio: np.ndarray,
    source_sr: int,
    target_sr: int = 22050,
    n_mfcc: int = 40,
    max_len: int = 174
) -> np.ndarray:
    """
    Extract MFCC features for the real inference model.
    Includes resampling to target_sr and padding/trimming to max_len.
    
    Args:
        audio: Input audio array
        source_sr: Current sample rate of audio
        target_sr: Sample rate expected by model
        n_mfcc: Number of MFCC coefficients
        max_len: Fixed number of time frames
        
    Returns:
        Numpy array of shape (n_mfcc, max_len)
    """
    # Step 1: Resample if necessary
    if source_sr != target_sr:
        audio = librosa.resample(y=audio, orig_sr=source_sr, target_sr=target_sr)
    
    # Step 2: Extract MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=target_sr, n_mfcc=n_mfcc)  # (40, T)
    
    # Step 3: Pad or trim along the time axis
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :max_len]
        
    return mfcc

def prepare_mfcc_tensor(
    mfcc: np.ndarray,
    scaler
) -> np.ndarray:
    """
    Normalize and reshape MFCC for model input.
    
    Args:
        mfcc: Array of shape (40, 174)
        scaler: StandardScaler instance from training
        
    Returns:
        Standardized array reshaped to (1, 1, 40, 174)
    """
    # Flatten for scaler: (40, 174) -> (1, 6960)
    flat = mfcc.reshape(1, -1)
    
    # Transform
    norm = scaler.transform(flat)
    
    # Reshape back to model input: (1, 1, 40, 174)
    final = norm.reshape(1, 1, mfcc.shape[0], mfcc.shape[1])
    
    return final.astype(np.float32)

async def extract_features_async(audio: np.ndarray) -> dict:
    """
    Async wrapper for feature extraction to prevent event loop blocking.
    
    Args:
        audio: Audio array (mono, 16kHz)
    
    Returns:
        Dictionary containing:
            - status: "success" or "error"
            - feature_shape: [n_mels, time_frames]
            - processing_latency_ms: float
            - audio_duration_sec: float
            - message: error details if failed
    """
    start_time = time.time()
    
    try:
        # Run blocking feature extraction in thread pool
        features = await asyncio.to_thread(
            extract_mel_spectrogram,
            audio,
            settings.SAMPLE_RATE,
            settings.N_MELS,
            settings.HOP_LENGTH,
            settings.N_FFT
        )
        
        processing_time = (time.time() - start_time) * 1000  # ms
        audio_duration = len(audio) / settings.SAMPLE_RATE
        
        logger.info(
            f"Feature extraction complete: shape={features.shape}, "
            f"duration={audio_duration:.2f}s, latency={processing_time:.2f}ms"
        )
        
        return {
            "status": "success",
            "feature_shape": list(features.shape),
            "processing_latency_ms": round(processing_time, 2),
            "audio_duration_sec": round(audio_duration, 2)
        }
    
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        logger.error(f"Feature extraction failed: {e}")
        
        return {
            "status": "error",
            "message": str(e),
            "processing_latency_ms": round(processing_time, 2)
        }
