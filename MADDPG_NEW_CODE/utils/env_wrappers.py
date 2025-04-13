import numpy as np
import multiprocessing as mp
from abc import ABC, abstractmethod
from gym.spaces import Discrete

class VecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    """
    closed = False
    viewer = None

    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step_async(self, actions):
        pass

    @abstractmethod
    def step_wait(self):
        pass

    def close_extras(self):
        pass

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        try:
            import cloudpickle
            return cloudpickle.dumps(self.x)
        except ImportError:
            import pickle
            return pickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns):
        """
        env_fns: list of environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(nenvs)])
        self.ps = [mp.Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # If the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs_list = [remote.recv() for remote in self.remotes]
        # Handle different reset formats
        if isinstance(obs_list[0], tuple):
            obs_list = [o[0] for o in obs_list]  # Extract observations
        return np.stack(obs_list)

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:            
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

class DummyVecEnv:
    def __init__(self, env_fns):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        
        # Initialize self.keys for observation handling
        self.keys = None
        if isinstance(env.observation_space, dict):
            # If observation space is a dictionary, get its keys
            self.keys = list(env.observation_space.keys())
        
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        
        # Handle PettingZoo environments
        if hasattr(env, 'env_type') and env.env_type == 'pettingzoo':
            # Set custom attributes for PettingZoo environments
            self.agent_ids = env.agent_ids
            self.n_agents = len(env.agent_ids)
            
        self.actions = []
    
    def step(self, actions):
        """
        Step the environments with the given actions
        """
        self.actions = actions
        return self.step_wait()
        
    def step_wait(self):
        """
        Wait for the step to complete in all environments
        """
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, rews, dones, infos = zip(*results)
        
        # For PettingZoo environments, handle differently
        if hasattr(self.envs[0], 'env_type') and self.envs[0].env_type == 'pettingzoo':
            # Just return the observation as is
            self.actions = []
            return obs[0], rews[0], dones[0], infos[0]
        
        # Properly handle different types of done flags
        all_done = False
        for d in dones:
            # Convert to numpy if it's not already
            if not isinstance(d, np.ndarray):
                d = np.array(d)
            
            # Check if any agent is done
            if d.any():
                all_done = True
                break
        
        if all_done:
            # Reset all environments to get fresh observations
            # Convert tuple to list so we can modify it
            obs_list = list(obs)
            for (i, env) in enumerate(self.envs):
                obs_i = env.reset()
                if isinstance(obs_i, tuple) and len(obs_i) == 2:  # Handle case where reset returns (obs, info)
                    obs_i = obs_i[0]
                # Update the observation in the list
                obs_list[i] = obs_i
            # Use the updated list instead of the original tuple
            obs = obs_list
                
        self.actions = []
        
        # Handle observations based on their type
        if self.keys is not None:
            # For dictionary observations
            obs_dict = {k: np.stack([o[k] for o in obs]) for k in self.keys}
            return obs_dict, np.stack(rews), np.stack(dones), infos
        else:
            # For normal observations
            try:
                return np.stack(obs), np.stack(rews), np.stack(dones), infos
            except:
                # If stacking fails, return as is
                return obs, np.stack(rews), np.stack(dones), infos
    
    def reset(self):
        """
        Reset all environments and return observations
        """
        obs = [env.reset() for env in self.envs]
        
        # Handle the case where reset returns (obs, info)
        for i, o in enumerate(obs):
            if isinstance(o, tuple) and len(o) == 2:
                obs[i] = o[0]  # Extract just the observation
                
        # For PettingZoo environments, return the observation as is
        if hasattr(self.envs[0], 'env_type') and self.envs[0].env_type == 'pettingzoo':
            return obs[0]
        
        # For dictionary observations
        if self.keys is not None:
            obs_dict = {k: np.stack([o[k] for o in obs]) for k in self.keys}
            return obs_dict
        
        # For normal observations
        try:
            return np.stack(obs)
        except:
            # If stacking fails, return as is
            return obs
    
    def close(self):
        """
        Clean up the environments
        """
        for env in self.envs:
            env.close()

def _flatten_obs(obs, keys=None):
    """
    Flatten observations into a single numpy array
    Args:
        obs: list of observations from multiple environments
        keys: keys to use for dictionary observations (None if not dictionary)
    Returns:
        flattened observations
    """
    if keys is not None:
        # For dictionary observations
        assert isinstance(obs, dict)
        return np.stack([obs[k] for k in keys])
    else:
        # For non-dictionary observations
        if isinstance(obs[0], np.ndarray):
            # If observations are numpy arrays
            return np.stack(obs)
        elif isinstance(obs[0], (list, tuple)):
            # If observations are lists or tuples
            return np.array(obs)
        else:
            # Try conversion to numpy array
            try:
                return np.stack([o for o in obs])
            except:
                # If that fails, return as is
                return obs

def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        try:
            cmd, data = remote.recv()
            
            if cmd == 'step':
                result = env.step(data)
                # Handle different return formats
                if len(result) == 3:
                    ob, reward, done = result
                    info = {}
                elif len(result) == 4:
                    ob, reward, done, info = result
                else:
                    ob, reward, terminated, truncated, info = result
                    done = terminated or truncated
                    
                # Reset if done
                if any(done) if hasattr(done, '__iter__') else done:
                    reset_result = env.reset()
                    if isinstance(reset_result, tuple):
                        ob = reset_result[0]  # Extract observation
                    else:
                        ob = reset_result
                        
                remote.send((ob, reward, done, info))
                
            elif cmd == 'reset':
                reset_result = env.reset()
                if isinstance(reset_result, tuple):
                    remote.send(reset_result[0])  # Send observation only
                else:
                    remote.send(reset_result)
                    
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
                
            elif cmd == 'close':
                env.close()
                remote.close()
                break
                
            else:
                raise NotImplementedError(f"Unknown command: {cmd}")
                
        except EOFError:
            break