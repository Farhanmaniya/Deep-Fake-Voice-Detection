from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from backend.core.connection_manager import manager
from backend.config.settings import settings
from backend.core.audio_processor import (
    bytes_to_numpy,
    validate_audio,
    normalize_audio,
    ensure_mono_16khz
)
from backend.core.feature_extractor import extract_features_async
from backend.core.model_loader import get_model, get_temporal_model, is_mock_model
from backend.core.inference_engine import run_consensus_inference
from backend.core.risk_engine import RiskEngine
from backend.core.vad import detect_voice_activity
from backend.core.rate_limiter import RateLimiter
from backend.core.metrics import metrics
from backend.core.explainability import build_explainability
from backend.core.attribution import build_attribution
from backend.core.robustness import build_robustness_analysis, inject_noise
from backend.core.feature_extractor import (
    extract_mel_spectrogram,
    extract_mfcc,
    prepare_mfcc_tensor
)
import logging
import numpy as np
import time
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Load models once at module level (singleton)
model = get_model(settings.MODEL_PATH)
temporal_model = get_temporal_model()

@router.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    
    # Track connection in metrics
    metrics.increment_connections()
    
    # Create risk engine instance for this connection
    risk_engine = RiskEngine(
        window_size=settings.RISK_WINDOW_SIZE,
        threshold_low=settings.RISK_THRESHOLD_LOW,
        threshold_high=settings.RISK_THRESHOLD_HIGH,
        smoothing_factor=settings.PREDICTION_SMOOTHING_FACTOR
    )
    
    # Create rate limiter for this connection
    rate_limiter = RateLimiter(settings.MAX_CHUNKS_PER_SECOND)
    
    # Initialize buffers for this connection
    overlap_buffer = np.array([], dtype=np.float32)
    temporal_audio_buffer = np.array([], dtype=np.float32)
    MAX_TEMPORAL_SAMPLES = settings.SAMPLE_RATE * 2 # 2 seconds of history
    
    try:
        while True:
            # Receive binary data
            data = await websocket.receive_bytes()
            
            data_size = len(data)
            logger.info(f"Received {data_size} bytes")
            
            # Start timing
            start_time = time.time()
            
            # Check rate limit
            is_allowed, current_rate = rate_limiter.check_rate()
            if not is_allowed:
                await manager.send_json({
                    "status": "error",
                    "message": f"Rate limit exceeded: {current_rate:.0f}/{settings.MAX_CHUNKS_PER_SECOND} chunks/sec"
                }, websocket)
                metrics.record_error()
                continue

            # Validate size against config
            if data_size > settings.MAX_CHUNK_SIZE:
                 logger.warning(f"Chunk too large: {data_size} > {settings.MAX_CHUNK_SIZE}")
                 await manager.send_json({
                     "status": "error",
                     "message": "Chunk size exceeds limit"
                 }, websocket)
                 continue

            try:
                # Step 1: Convert bytes to numpy array
                audio = bytes_to_numpy(data)
                logger.info(f"Converted to numpy array: {len(audio)} samples")
                
                # Step 2: Ensure mono 16kHz (assuming input is already 16kHz)
                audio = ensure_mono_16khz(audio, settings.SAMPLE_RATE)
                
                # Step 2.5: Apply chunk overlap
                if settings.CHUNK_OVERLAP_SAMPLES > 0:
                    if overlap_buffer.size > 0:
                        audio = np.concatenate([overlap_buffer, audio])
                    
                    # Update overlap buffer for next chunk from the current raw samples
                    if len(audio) >= settings.CHUNK_OVERLAP_SAMPLES:
                        overlap_buffer = audio[-settings.CHUNK_OVERLAP_SAMPLES:].copy()
                    else:
                        overlap_buffer = audio.copy()
                    logger.info(f"Applied overlap: new length = {len(audio)} samples")
                
                # Step 3: Validate audio duration
                is_valid, error_msg = validate_audio(audio, settings.SAMPLE_RATE)
                if not is_valid:
                    logger.warning(f"Audio validation failed: {error_msg}")
                    await manager.send_json({
                        "status": "error",
                        "message": error_msg
                    }, websocket)
                    continue
                
                # Step 4: Normalize audio
                audio = normalize_audio(audio)
                
                # Step 5: Voice Activity Detection
                is_voice, energy = detect_voice_activity(audio, settings.VAD_ENERGY_THRESHOLD)
                
                # Update temporal history even if it's silence (to maintain context rhythm)
                temporal_audio_buffer = np.concatenate([temporal_audio_buffer, audio])
                if len(temporal_audio_buffer) > MAX_TEMPORAL_SAMPLES:
                    temporal_audio_buffer = temporal_audio_buffer[-MAX_TEMPORAL_SAMPLES:]

                if not is_voice:
                    # logger.info(f"Silence detected: energy={energy:.6f} < threshold={settings.VAD_ENERGY_THRESHOLD}")
                    await manager.send_json({
                        "status": "success",
                        "is_silence": true,
                        "message": "Analysing background noise...",
                        "energy_level": round(energy, 6),
                        "chunk_probability": 0.0,
                        "cnn_probability": 0.0,
                        "lstm_probability": 0.0,
                        "rolling_risk": 0.0
                    }, websocket)
                    continue
                
                # Step 5b: Robustness Analysis
                robustness_data = await asyncio.to_thread(
                    build_robustness_analysis, 
                    audio, 
                    settings.ROBUSTNESS_SNR_THRESHOLD
                )
                
                # Step 6: Get features for inference
                # We always need Mel Spectrogram for Explainability/Attribution
                mel_features = await asyncio.to_thread(
                    extract_mel_spectrogram,
                    audio,
                    settings.SAMPLE_RATE,
                    settings.N_MELS,
                    settings.HOP_LENGTH,
                    settings.N_FFT
                )
                
                # Step 6: Get features for inference
                audio_duration = len(audio) / settings.SAMPLE_RATE
                feature_result = {
                    "feature_shape": list(mel_features.shape),
                    "audio_duration_sec": round(audio_duration, 2)
                }
                
                # For the inference model, decide between Mel or MFCC
                from backend.core.model_loader import RealModelContainer
                if isinstance(model, RealModelContainer):
                    # Real model expects MFCC at 22050 Hz
                    mfcc = await asyncio.to_thread(
                        extract_mfcc,
                        audio,
                        settings.SAMPLE_RATE,
                        settings.MODEL_SAMPLE_RATE,
                        settings.MFCC_N_MFCC,
                        settings.MFCC_MAX_LEN
                    )
                    inference_features = await asyncio.to_thread(
                        prepare_mfcc_tensor,
                        mfcc,
                        model.scaler
                    )
                else:
                    # Mock or fallback models use Mel Spectrogram
                    inference_features = mel_features
                
                # Step 6b: Prepare features for Temporal LSTM model
                lstm_features = None
                if temporal_model:
                    # LSTM expects 64 mels, normalized
                    # Use the 2-second history for better temporal context
                    lstm_mel = await asyncio.to_thread(
                        extract_mel_spectrogram,
                        temporal_audio_buffer,
                        settings.SAMPLE_RATE,
                        n_mels=64,
                        hop_length=settings.HOP_LENGTH,
                        n_fft=settings.N_FFT
                    )
                    # Normalize logic must match train_temporal_model.py
                    import librosa
                    lstm_mel_db = librosa.power_to_db(lstm_mel, ref=1.0)
                    lstm_mel_db = np.clip(lstm_mel_db, -80, 0)
                    lstm_features = (lstm_mel_db + 40.0) / 40.0
                    # Pad/Trim to 128 frames
                    max_frames = 128
                    if lstm_features.shape[1] < max_frames:
                        lstm_features = np.pad(lstm_features, ((0, 0), (0, max_frames - lstm_features.shape[1])), mode='constant')
                    else:
                        lstm_features = lstm_features[:, :max_frames]
                    lstm_features = lstm_features.T # (time, mels)
                
                # Step 7: Run Consensus Inference
                inference_result = await run_consensus_inference(
                    inference_features, 
                    lstm_features, 
                    model, 
                    temporal_model
                )
                metrics.record_inference()
                
                # Step 8: Build explainability data from mel spectrogram
                import asyncio as _asyncio
                explainability_data = await _asyncio.to_thread(
                    build_explainability,
                    audio,
                    mel_features,
                    inference_result["chunk_probability"],
                    settings.SAMPLE_RATE,
                )

                # Step 8b: Build attribution — who generated this audio?
                attribution_data = await _asyncio.to_thread(
                    build_attribution,
                    audio,
                    mel_features,
                    explainability_data["band_energies"],
                    inference_result["chunk_probability"],
                    settings.SAMPLE_RATE,
                )
                
                # Step 9: Update risk engine
                risk_result = risk_engine.update(inference_result["chunk_probability"])
                
                # Step 10: Combine results and send response
                total_latency = (time.time() - start_time) * 1000
                
                response = {
                    "status": "success",
                    "feature_shape": feature_result["feature_shape"],
                    "audio_duration_sec": feature_result["audio_duration_sec"],
                    "chunk_probability": inference_result["chunk_probability"],
                    "rolling_risk": risk_result["rolling_risk"],
                    "risk_level": risk_result["risk_level"],
                    "processing_latency_ms": round(total_latency, 2),
                    "energy_level": round(energy, 6),
                    "explainability": explainability_data,
                    "attribution": attribution_data,
                    "robustness": robustness_data,
                    "consensus": {
                        "cnn_score": inference_result.get("cnn_probability", 0.0),
                        "lstm_score": inference_result.get("lstm_probability", 0.0),
                        "is_active": inference_result.get("consensus_active", False)
                    }
                }
                
                # Record metrics
                metrics.record_chunk(total_latency)
                
                logger.info(f"Consensus Scores: CNN={response['consensus']['cnn_score']}, LSTM={response['consensus']['lstm_score']}")
                await manager.send_json(response, websocket)
                
            except Exception as e:
                logger.error(f"Audio processing error: {e}")
                metrics.record_error()
                await manager.send_json({
                    "status": "error",
                    "message": f"Processing failed: {str(e)}"
                }, websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        metrics.decrement_connections()
        logger.info("Client disconnected normally")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)
        metrics.decrement_connections()
        metrics.record_error()
