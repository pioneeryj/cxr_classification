"""Utility functions for CXR classification training."""

import os
import random
import yaml
import numpy as np
import torch


def load_config(config_path):
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Dictionary containing configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
 

def set_seed(seed):
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_output_dir(output_dir):
    """Create output directory if it doesn't exist.

    Args:
        output_dir: Path to output directory

    Returns:
        Path to output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_config(config, output_dir):
    """Save configuration to output directory.

    Args:
        config: Configuration dictionary
        output_dir: Path to output directory
    """
    config_path = os.path.join(output_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def count_parameters(model):
    """Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_lr(optimizer):
    """Get current learning rate from optimizer.

    Args:
        optimizer: PyTorch optimizer

    Returns:
        Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']
