"""
Attack functions for adversarial environments.
"""
import torch
from torch.autograd import Variable
from ..constants import torch_device

try:
    import supersuit
except ImportError:
    print("Warning: supersuit not available. Atari preprocessing will not work.")
    supersuit = None


def preprocess_env_atari(env):
    """
    Preprocess Atari environment with standard wrappers.
    
    Args:
        env: Raw Atari environment
        
    Returns:
        Wrapped environment
    """
    if supersuit is None:
        raise ImportError("supersuit is required for Atari preprocessing but not available")
    
    # as per openai baseline's MaxAndSKip wrapper, maxes over the last 2 frames
    # to deal with frame flickering
    env = supersuit.max_observation_v0(env, 2)
    # skip frames for faster processing and less control
    # to be compatible with gym, use frame_skip(env, (2,5))
    env = supersuit.frame_skip_v0(env, 4)
    # downscale observation for faster processing
    env = supersuit.resize_v1(env, 84, 84)
    # allow agent to see everything on the screen despite Atari's flickering screen problem
    env = supersuit.frame_stack_v1(env, 4)
    return env


def fgsm_attack(maddpg, obs, actions, attacked_agent_id, epsilon):
    """
    Fast Gradient Sign Method (FGSM) attack on observations.
    
    Args:
        maddpg: MADDPG agent
        obs: List of observations for all agents
        actions: List of actions for all agents
        attacked_agent_id: ID of agent to attack
        epsilon: Attack strength
        
    Returns:
        Perturbed observation for the attacked agent
    """
    # Convert to tensors with gradient tracking
    torch_obs = [Variable(torch.Tensor([obs[i]]).to(torch_device), requires_grad=True) for i in range(maddpg.nagents)]
    torch_actions = [Variable(torch.Tensor([actions[i]]).to(torch_device), requires_grad=False) for i in range(maddpg.nagents)]
    # Concatenate for critic input
    vf_in = torch.cat((*torch_obs, *torch_actions), dim=1)
    # Loss to maximize (degrade agent performance)
    loss = -(maddpg.agents[attacked_agent_id].critic(vf_in)).mean()  # Negative to maximize via gradient ascent
    # Compute gradient
    grad = torch.autograd.grad(loss, torch_obs[attacked_agent_id], retain_graph=True)[0]
    # FGSM perturbation: move in direction of gradient sign
    perturbation = epsilon * grad.sign()
    # Apply perturbation element-wise
    obs_perturbed = obs[attacked_agent_id] + perturbation.squeeze().cpu().numpy()
    return obs_perturbed