import torch
import numpy as np
import asyncio
import logging
import time
from typing import Union

logger = logging.getLogger(__name__)

def _run_inference_sync(features: Union[np.ndarray, torch.Tensor], model) -> float:
    """
    Synchronous inference function (runs in thread pool).
    
    Args:
        features: Input features (numpy array or torch tensor)
        model: Loaded model (RealModelContainer, TorchScript, or Mock)
    
    Returns:
        Probability score [0, 1]
    """
    # If features are already a tensor (e.g. prepared MFCC), use as is
    if isinstance(features, torch.Tensor):
        features_tensor = features
    else:
        # Convert numpy to torch tensor and add batch dimension
        features_tensor = torch.from_numpy(features).float().unsqueeze(0)
    
    # Ensure tensor is on the same device as the model
    from backend.core.model_loader import RealModelContainer
    if isinstance(model, RealModelContainer):
        device = next(model.model.parameters()).device
        features_tensor = features_tensor.to(device)
    
    # Run inference
    with torch.no_grad():
        if isinstance(model, RealModelContainer):
            # Output is (batch, 2) logits
            output = model.model(features_tensor)
            # Apply softmax to get probabilities
            probs = torch.softmax(output, dim=1)
            # Class 1 is "Fake"
            probability = probs[0, 1].item()
        else:
            # Fallback for TorchScript or MockModel
            output = model(features_tensor)
            if isinstance(output, torch.Tensor):
                probability = output.item() if output.numel() == 1 else output[0, 1].item() if output.shape[1] > 1 else output[0].item()
            else:
                probability = float(output)
    
    # Ensure probability is in [0, 1] range
    probability = max(0.0, min(1.0, probability))
    
    return probability

async def run_inference(features: np.ndarray, model) -> dict:
    """
    Async wrapper for model inference to prevent event loop blocking.
    
    Args:
        features: Log-Mel spectrogram features (n_mels, time_frames)
        model: Loaded model (TorchScript or Mock)
    
    Returns:
        Dictionary containing:
            - chunk_probability: float [0, 1]
            - inference_latency_ms: float
    """
    start_time = time.time()
    
    try:
        # Run inference in thread pool to avoid blocking
        probability = await asyncio.to_thread(
            _run_inference_sync,
            features,
            model
        )
        
        inference_time = (time.time() - start_time) * 1000  # ms
        
        logger.info(
            f"Inference complete: probability={probability:.4f}, "
            f"latency={inference_time:.2f}ms"
        )
        
        return {
            "chunk_probability": round(probability, 4),
            "inference_latency_ms": round(inference_time, 2)
        }
    
    except Exception as e:
        inference_time = (time.time() - start_time) * 1000
        logger.error(f"Inference failed: {e}")
        
        # Return neutral probability on error
        return {
            "chunk_probability": 0.5,
            "inference_latency_ms": round(inference_time, 2),
            "error": str(e)
        }
