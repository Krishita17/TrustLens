"""
Model utility functions for TrustLens.

Provides model save/load functionality using joblib,
plus common model evaluation helpers.
"""

import os
import logging
import joblib
from datetime import datetime

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'experiments', 'results'
)


def save_model(model, model_name: str, model_dir: str = None) -> str:
    """
    Save a trained model to disk using joblib.
    
    Args:
        model: Trained model object.
        model_name: Name for the saved model file.
        model_dir: Directory to save in. Uses default if None.
        
    Returns:
        Path to saved model file.
    """
    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR
    
    os.makedirs(model_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{model_name}_{timestamp}.joblib"
    filepath = os.path.join(model_dir, filename)
    
    joblib.dump(model, filepath)
    logger.info(f"Model saved to {filepath}")
    return filepath


def load_model(filepath: str):
    """
    Load a trained model from disk.
    
    Args:
        filepath: Path to model file.
        
    Returns:
        Loaded model object.
        
    Raises:
        FileNotFoundError: If model file doesn't exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    model = joblib.load(filepath)
    logger.info(f"Model loaded from {filepath}")
    return model


def get_latest_model(model_name: str, model_dir: str = None) -> str:
    """
    Find the most recently saved model matching a name pattern.
    
    Args:
        model_name: Model name prefix to match.
        model_dir: Directory to search in.
        
    Returns:
        Path to the most recent matching model file.
        
    Raises:
        FileNotFoundError: If no matching model found.
    """
    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR
    
    import glob
    pattern = os.path.join(model_dir, f"{model_name}_*.joblib")
    files = sorted(glob.glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"No models matching '{model_name}' in {model_dir}")
    
    return files[-1]  # Most recent by timestamp in filename
