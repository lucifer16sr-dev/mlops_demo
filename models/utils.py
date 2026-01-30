import torch
import os
from pathlib import Path
from typing import Optional, Dict, Any

def save_model(model: torch.nn.Module, path: str, metadata: Optional[dict] = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Automatically extract model parameters if they exist
    model_params = {}
    if hasattr(model, 'vocab_size'):
        model_params['vocab_size'] = model.vocab_size
    if hasattr(model, 'embedding_dim'):
        model_params['embedding_dim'] = model.embedding_dim
    if hasattr(model, 'hidden_dim'):
        model_params['hidden_dim'] = model.hidden_dim
    if hasattr(model, 'num_layers'):
        model_params['num_layers'] = model.num_layers
    if hasattr(model, 'num_classes'):
        model_params['num_classes'] = model.num_classes
    if hasattr(model, 'dropout'):
        # Get dropout from the first dropout layer if it exists
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                model_params['dropout'] = module.p
                break
    
    # Merge with provided metadata
    if metadata:
        model_params.update(metadata)
    
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'model_params': model_params  # Save model parameters
    }
    
    torch.save(save_dict, path)
    print(f"Model saved to {path}")


def load_model(model_class: type, path: str, device: str = 'cpu'):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    
    checkpoint = torch.load(path, map_location=device)
    
    # Get model parameters from checkpoint
    model_params = checkpoint.get('model_params', {})
    
    # Create model with saved parameters, or use defaults
    if model_params:
        # Filter out None values and use only valid parameters
        filtered_params = {k: v for k, v in model_params.items() if v is not None}
        model = model_class(**filtered_params)
    else:
        # Fallback to default parameters
        model = model_class()
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {path}")
    return model


def get_model_path(model_name: str, base_dir: str = "models"):
    return os.path.join(base_dir, model_name, "checkpoint.pth")