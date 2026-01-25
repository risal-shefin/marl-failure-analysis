"""
Environment setup and initialization functions.
"""
try:
    import pettingzoo.mpe as mpe
    import pettingzoo.sisl as sisl
    import pettingzoo.atari as atari
    from utils.pettingzoo_wrapper import PettingZooWrapper
except ImportError as e:
    print(f"Warning: PettingZoo components not available: {e}")
    mpe = sisl = atari = PettingZooWrapper = None

from ..attacks import preprocess_env_atari


def create_environment(config, maddpg):
    """
    Create and configure the appropriate environment based on config.
    
    Args:
        config: Configuration object with env_id
        maddpg: MADDPG agent for environment compatibility
        
    Returns:
        Configured environment wrapped for PettingZoo
    """
    if mpe is None or sisl is None or atari is None or PettingZooWrapper is None:
        raise ImportError("PettingZoo components are required but not available")
    
    try:
        env_func = getattr(mpe, config.env_id)
        if config.env_id == "simple_spread_v3":
            env = env_func.parallel_env(
                continuous_actions=not maddpg.discrete_action, 
                render_mode='rgb_array', 
                N=maddpg.nagents
            )
        else:
            env = env_func.parallel_env(
                continuous_actions=not maddpg.discrete_action, 
                render_mode='rgb_array'
            )
    except:
        try:
            env_func = getattr(sisl, config.env_id)
            if config.env_id == 'waterworld_v4':
                env = env_func.parallel_env(n_pursuers=5, render_mode='rgb_array')
            else:
                env = env_func.parallel_env(render_mode='rgb_array')
        except:
            env_func = getattr(atari, config.env_id)
            env = env_func.parallel_env(render_mode='rgb_array')
            env = preprocess_env_atari(env)

    env = PettingZooWrapper.wrap_env(env)
    env.reset()
    
    return env