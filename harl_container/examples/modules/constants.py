import torch

USE_CUDA = torch.cuda.is_available()
DEVICE = 'gpu' if USE_CUDA else 'cpu'
torch_device = torch.device("cuda" if USE_CUDA else "cpu")
K_SIGMA = 3.0  # Changed from 1.0 to 3.0 for stricter anomaly detection


# FILEPATH = "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/pettingzoo_mpe/simple_spread_v3-discrete/hatrpo/q_three_agent/seed-00001-2025-09-23-22-29-22/models"
# REWARD=-89.021
FILEPATH = "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/pettingzoo_mpe/simple_spread_v3-discrete/hatrpo/q_five_agent/seed-00001-2025-09-23-22-32-22/models"
REWARD = -565.155 
ATTACK_ID = 2
