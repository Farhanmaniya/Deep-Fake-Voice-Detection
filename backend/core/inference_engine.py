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

def _run_temporal_inference_sync(features: np.ndarray, model) -> float:
    """Run secondary LSTM inference."""
    if model is None:
        return 0.5 # Neutral fallback
        
    # Features already normalized in training script logic
    # Expected shape: (time, mels) -> (1, time, mels) for batch
    features_tensor = torch.from_numpy(features).float().unsqueeze(0)
    
    with torch.no_grad():
        output = model(features_tensor)
        probability = output.item()
        
    return float(np.clip(probability, 0, 1))

async def run_consensus_inference(
    cnn_features: torch.Tensor, 
    lstm_features: np.ndarray, 
    cnn_model, 
    lstm_model
) -> dict:
    """
    Runs both models and returns a weighted consensus.
    """
    start_time = time.time()
    
    try:
        # Run CNN in parallel with LSTM
        cnn_prob_task = asyncio.to_thread(_run_inference_sync, cnn_features, cnn_model)
        lstm_prob_task = asyncio.to_thread(_run_temporal_inference_sync, lstm_features, lstm_model)
        
        cnn_prob, lstm_prob = await asyncio.gather(cnn_prob_task, lstm_prob_task)
        
        # Calculate Weighted Consensus
        # If LSTM is None, rely 100% on CNN
        if lstm_model is None:
            final_prob = cnn_prob
        else:
            w = settings.CONSENSUS_CNN_WEIGHT
            final_prob = (cnn_prob * w) + (lstm_prob * (1 - w))
            
        latency = (time.time() - start_time) * 1000
        
        return {
            "chunk_probability": round(float(final_prob), 4),
            "cnn_probability": round(float(cnn_prob), 4),
            "lstm_probability": round(float(lstm_prob), 4),
            "inference_latency_ms": round(latency, 2),
            "consensus_active": lstm_model is not None
        }
        
    except Exception as e:
        logger.error(f"Consensus inference failed: {e}")
        return {
            "chunk_probability": 0.5,
            "error": str(e)
        }
