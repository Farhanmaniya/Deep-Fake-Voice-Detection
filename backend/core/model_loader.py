import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
from typing import Union, NamedTuple
from sklearn.preprocessing import StandardScaler
from backend.core.deepfake_cnn import DeepfakeCNN
from backend.config.settings import settings

logger = logging.getLogger(__name__)

class RealModelContainer(NamedTuple):
    """Container for the real model and its associated scaler."""
    model: DeepfakeCNN
    scaler: StandardScaler
    n_mfcc: int
    max_len: int
    sample_rate: int

class MockModel:
    """
    Mock ML model for testing when real model is not available.
    Returns random probabilities between 0 and 1.
    """
    def __init__(self):
        logger.warning("⚠️  Using MOCK MODEL - predictions are random!")
        logger.warning("   Place a real TorchScript model at the configured MODEL_PATH")
    
    def forward(self, x):
        """
        Mock forward pass - returns random probability.
        
        Args:
            x: Input tensor (ignored)
        
        Returns:
            Random probability between 0 and 1
        """
        # Return random probability
        return torch.rand(1).item()
    
    def __call__(self, x):
        return self.forward(x)

def load_model(model_path: str) -> Union[RealModelContainer, MockModel]:
    """
    Load PyTorch model checkpoint (.pth) or fall back to mock model.
    
    Args:
        model_path: Path to model checkpoint file (.pth)
    
    Returns:
        Loaded model container or MockModel
    """
    model_file = Path(model_path)
    
    if model_file.exists():
        try:
            logger.info(f"Loading model checkpoint from: {model_path}")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(str(model_file), map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                logger.info("Detected new model checkpoint format")
                
                # Initialize model architecture
                model = DeepfakeCNN().to(device)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()
                
                # Reconstruct scaler
                scaler = StandardScaler()
                scaler.mean_ = checkpoint["scaler_mean"]
                scaler.scale_ = checkpoint["scaler_scale"]
                scaler.n_features_in_ = len(scaler.mean_)
                
                logger.info("✅ Real model and scaler loaded successfully")
                return RealModelContainer(
                    model=model,
                    scaler=scaler,
                    n_mfcc=checkpoint.get("n_mfcc", 40),
                    max_len=checkpoint.get("max_len", 174),
                    sample_rate=checkpoint.get("sample_rate", 22050)
                )
            
            # Fallback for old TorchScript format if it was actually used
            logger.warning("Unknown checkpoint format. Attempting TorchScript load...")
            model = torch.jit.load(str(model_file))
            model.eval()
            logger.info("✅ Loaded as TorchScript model (legacy fallback)")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.warning("Falling back to mock model")
            return MockModel()
    else:
        logger.warning(f"Model file not found: {model_path}")
        logger.info("Using mock model for testing")
        return MockModel()

# Singleton model instances
_model_instance = None
_temporal_model_instance = None

def get_model(model_path: str):
    """Get legacy CNN singleton."""
    global _model_instance
    if _model_instance is None:
        _model_instance = load_model(model_path)
    return _model_instance

def get_temporal_model():
    """Load the LSTM temporal model as a singleton."""
    global _temporal_model_instance
    if _temporal_model_instance is None:
        path = Path(settings.TEMPORAL_MODEL_PATH)
        if not path.exists():
            logger.warning(f"Temporal model not found at {path}. Using None.")
            return None
            
        try:
            from backend.core.deepfake_cnn import DeepfakeCNN # Not needed but good for imports
            # Architecture must match train_temporal_model.py
            # Re-defining here locally to avoid circular imports or missing definitions
            class TemporalLSTM(nn.Module):
                def __init__(self, input_size=64, hidden_size=256, num_layers=3):
                    super().__init__()
                    import torch.nn as nn
                    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
                    self.fc = nn.Sequential(
                        nn.Linear(hidden_size, 128),
                        nn.ReLU(),
                        nn.BatchNorm1d(128),
                        nn.Dropout(0.3),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Linear(64, 1),
                        nn.Sigmoid()
                    )
                def forward(self, x):
                    _, (hn, _) = self.lstm(x)
                    return self.fc(hn[-1])

            model = TemporalLSTM().to(torch.device("cpu")) # Default to CPU
            model.load_state_dict(torch.load(str(path), map_location="cpu", weights_only=True))
            model.eval()
            _temporal_model_instance = model
            logger.info("✅ Temporal LSTM model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load temporal model: {e}")
            _temporal_model_instance = None
            
    return _temporal_model_instance

def is_mock_model(model) -> bool:
    """
    Check if the model is a MockModel instance.
    
    Args:
        model: Model instance to check
        
    Returns:
        True if mock, False otherwise
    """
    return isinstance(model, MockModel)
