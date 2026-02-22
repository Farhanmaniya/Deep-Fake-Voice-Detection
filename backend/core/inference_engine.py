import torch
import numpy as np
import asyncio
import logging
import time
from typing import Union
import torch.nn as nn
import torch.nn.functional as F
from backend.config.settings import settings

logger = logging.getLogger(__name__)

# New Architecture Components
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, x):
        weights = self.attention(x)
        weights = F.softmax(weights, dim=1)
        context = torch.sum(weights * x, dim=1)
        return context, weights

class TemporalModel(nn.Module):
    def __init__(self, input_size=64, hidden_size=128, num_layers=2):
        super(TemporalModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.2)
        self.attention = Attention(hidden_size * 2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        gru_out, _ = self.gru(x)
        context, _ = self.attention(gru_out)
        return self.fc(context)

def _run_inference_sync(features: Union[np.ndarray, torch.Tensor], model) -> float:
    """
    Synchronous inference function (runs in thread pool).
    
    Args:
        features: Input features (numpy array or torch tensor)
        model: Loaded model (RealModelContainer, TorchScript, or Mock)
    
    Returns:
        Probability score [0, 1]
    """
    # Convert numpy to torch tensor and handle batch dimension
    if not isinstance(features, torch.Tensor):
        features_tensor = torch.from_numpy(features).float()
        # Only add batch dimension if not already present
        if features_tensor.ndim < 4:
            features_tensor = features_tensor.unsqueeze(0)
    else:
        features_tensor = features
    
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
    """Run secondary Bi-GRU + Attention inference."""
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
        print(f"CRITICAL: Consensus inference failed: {e}")
        import traceback
        traceback.print_exc()
        logger.error(f"Consensus inference failed: {e}", exc_info=True)
        return {
            "chunk_probability": 0.5,
            "cnn_probability": 0.5,
            "lstm_probability": 0.5,
            "inference_latency_ms": 0,
            "consensus_active": False,
            "error": str(e)
        }
async def run_inference(features: np.ndarray, model) -> dict:
    """Async wrapper for single model inference (backward compatibility)."""
    start_time = time.time()
    try:
        probability = await asyncio.to_thread(_run_inference_sync, features, model)
        latency = (time.time() - start_time) * 1000
        return {
            "chunk_probability": round(probability, 4),
            "inference_latency_ms": round(latency, 2)
        }
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return {"chunk_probability": 0.5, "error": str(e)}
