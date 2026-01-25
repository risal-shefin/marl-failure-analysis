"""
Constants and configuration variables for the detection statistics system.
"""
import torch

USE_CUDA = torch.cuda.is_available()
DEVICE = 'gpu' if USE_CUDA else 'cpu'
torch_device = torch.device("cuda" if USE_CUDA else "cpu")
K_SIGMA = 2.0
DEFAULT_INFLUENCE_DECAY_LAMBDA = 0.3