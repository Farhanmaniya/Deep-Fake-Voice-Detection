import torch
import numpy as np
import logging
from pathlib import Path
from typing import Union, NamedTuple
from sklearn.preprocessing import StandardScaler
from backend.core.deepfake_cnn import DeepfakeCNN

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

# Singleton model instance
_model_instance = None

def get_model(model_path: str):
    """
    Get singleton model instance.
    
    Args:
        model_path: Path to model file
    
    Returns:
        Model instance (singleton)
    """
    global _model_instance
    if _model_instance is None:
        _model_instance = load_model(model_path)
    return _model_instance

def is_mock_model(model) -> bool:
    """
    Check if the model is a MockModel instance.
    
    Args:
        model: Model instance to check
        
    Returns:
        True if mock, False otherwise
    """
    return isinstance(model, MockModel)
