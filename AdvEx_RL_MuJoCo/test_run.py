from .sac import SAC
from .safety_agent import Safety_Agent
from .AdvEx_RL_config.victim_config import get_victim_args
from .AdvEx_RL_config.adversary_config import get_adv_args
from .AdvEx_RL_config.safety_config import get_safety_args
import argparse
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
import os
from matplotlib import pyplot 
from matplotlib import cm
import pickle
import matplotlib.pyplot as plt
import os
import warnings
import torch.nn.functional as F
warnings.filterwarnings("ignore")

#====================================================================
def torchify(x):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
    return torch.FloatTensor(x).to(device).unsqueeze(0)
#====================================================================

def compute_critic_loss(agent, states, actions, next_states, rewards, cont_flags):
    with torch.no_grad():
        next_state_action, next_state_log_pi, _,_ = agent.policy.sample(next_states)
        qf1_next_target, qf2_next_target = agent.critic_target(next_states, next_state_action)
        
        min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - agent.alpha * next_state_log_pi
        next_q_value = rewards.unsqueeze(axis=-1) + cont_flags.unsqueeze(axis=-1) * agent.gamma * min_qf_next_target
        
        qf1, qf2 = agent.critic(states, actions)

        #==========Critic Loss =====================
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        critic_loss = ((qf1_loss + qf2_loss) / 2.0).item()
    return critic_loss

def compute_policy_loss(agent, state, action):
    _, _, mean, stddev = agent.policy.sample(state)    # agent's actual policy distribution
    action_log_prob = agent.policy.compute_log_prob(action, mean, stddev) # applied action's log probability
    qf1_pi, qf2_pi = agent.critic(state, action)
    min_qf_pi = torch.min(qf1_pi, qf2_pi)
    policy_loss = ((agent.alpha * action_log_prob) - min_qf_pi).mean() 
    return policy_loss.item()


def random_policy_sample_fn(env):
    # Extract properties
    low = env.action_space.low
    high = env.action_space.high
    num_dimensions = env.action_space.shape[0]
    pi = env.action_space.sample()
    log_pi = -num_dimensions * np.log(high - low)   # k * log(1/(r-l))
    return torchify(pi), torchify(log_pi), None, None


# source: https://arxiv.org/pdf/2306.05873
def fo_inrd(agent, state, action, epsilon=0.01):
    # Sample a small random perturbation
    perturbation = torch.randn_like(state) * epsilon
    
    # Perturb the state
    perturbed_state = state + perturbation
    
    # Compute the cost for the original state
    cost_original = compute_policy_loss(agent, state, action)
    
    # Compute the cost for the perturbed state
    cost_perturbed = compute_policy_loss(agent, perturbed_state, action)
    
    # Compute the difference in cost
    delta_cost = cost_perturbed - cost_original
    
    return delta_cost


def run_eval_episode(env, 
                    expert_agent, 
                    safety_agent=None, 
                    use_safety=False,
                    shield_threshold = 0.0, 
                    atk_rate=0.20, 
                    aaa_agent = None,
                    aaa_atk = True,
                    ):
    rec_cnt = 0
    tsk_cnt = 0

    done =False
    epi_reward = 0
    epi_step_count=0
    state = env.reset()
    adv_reward = 0
    safety = 0
    unsafe_cnt = 0
    info_vec = []
    while not done:
        epi_step_count+=1

        action_tsk = expert_agent.select_action(state, eval=True)
        random_action = env.action_space.sample()
        adv_action = aaa_agent.select_action(state)
        #******************************************************************************   
        if np.random.rand()<atk_rate:
            if aaa_atk:
                action_tsk = aaa_agent.select_action(state)
            else:
                action_tsk = env.action_space.sample()
        #******************************************************************************   
        if use_safety:
            shield_val_tsk = safety_agent.get_shield_value(torchify(state), torchify(action_tsk))
            if shield_val_tsk>=shield_threshold:
                action = safety_agent.select_action(state, eval=True)
                rec_cnt+=1
            else:
                action = action_tsk 
                tsk_cnt+=1     
        else:
            action = action_tsk 
            tsk_cnt+=1    

        nxt_state, reward, done, info = env.step(action)
        epi_reward+=reward

        adv_reward +=info['adv_reward']
        if info['adv_reward']>0:
            unsafe_cnt+=1
        state = nxt_state
        done = done or (epi_step_count==env._max_episode_steps)
        info['done'] = done
        info['action_random'] = random_action
        info['action_adv'] = adv_action
        info_vec.append(info)
        if done:
            if adv_reward<0:
                safety=1
            else:
                safety = epi_step_count/env._max_episode_steps
            break
          
    return_data = {
        'ep_safety_ratio': safety,
        'ep_task_policy_count': tsk_cnt,
        'ep_safety_policy_count': rec_cnt,
        'ep_reward': epi_reward,
        'ep_adv_reward': adv_reward,
        'ep_step_count': epi_step_count,
        'ep_info_vec': info_vec
    }
    return return_data
#====================================================================
#====================================================================
   
            
def run(env_name=None, eval_epi_no=100, exp_data_dir=''):
  if not os.path.exists(exp_data_dir):
    os.makedirs(exp_data_dir)
  eval_epi_no = eval_epi_no
#   atk_rate = [0.0, 0.25, 0.5, 0.75]
  atk_rate = [0.0]
  
  if env_name == "maze":
      from .env.maze import MazeNavigation
      env = MazeNavigation()
    #   env.seed(1234)
  elif env_name == 'nav1':
      from .env.navigation1 import Navigation1
      env = Navigation1()
    #   env.seed(31415)
  elif env_name == 'nav2':
      from .env.navigation2 import Navigation2
      env = Navigation2()
    #   env.seed(27156)

  agent_cfg =  get_victim_args(env_name)
  safety_cfg = get_safety_args(env_name)
  adv_cfg = get_adv_args(env_name)
  current_path = os.getcwd()
  
  expert_agent_path = current_path + agent_cfg.saved_model_path
  safety_policy_path = current_path + safety_cfg.saved_model_path
  adv_path = current_path + adv_cfg.saved_model_path

  agent_observation_space = env.observation_space.shape[0]
  agent_action_space = env.action_space.shape[0]
  logdir = ' '
  #====================================================================
  expert_agent = SAC(agent_observation_space,
                   agent_action_space,
                   agent_cfg,
                   logdir,
                   env=env
                  )
  task_algorithm = "SAC"
  expert_agent.load_best_model(expert_agent_path)
  #====================================================================
  adv_agent = SAC(agent_observation_space,
                   agent_action_space,
                   adv_cfg,
                   logdir,
                   env=env
                  )
  adv_agent.load_best_model(adv_path)
  #====================================================================
  safety_agent = Safety_Agent(observation_space = agent_observation_space, 
                                action_space= agent_action_space,
                                args=safety_cfg,
                                logdir=logdir,
                                env = env,
                                adv_agent=adv_agent
                                )
  safety_agent.load_safety_model(safety_policy_path)
  #====================================================================
  
  fig = plt.figure()

  # run episode for different attack rates
  for atk_rate in atk_rate:
    ep_data = run_eval_episode(env, expert_agent, safety_agent, False, 5, atk_rate, adv_agent, False)

    print("Episode Reward: ", ep_data['ep_reward'])
    print("Episode Adv reward: ", ep_data['ep_adv_reward'])

    # Collect episode data
    info_vec = ep_data['ep_info_vec']
    states = torchify([info['state'] for info in info_vec]).squeeze(axis=0)
    next_states = torchify([info['next_state'] for info in info_vec]).squeeze(axis=0)
    actions = torchify([info['action'] for info in info_vec]).squeeze(axis=0)
    actions_random = torchify([info['action_random'] for info in info_vec]).squeeze(axis=0)
    actions_adv = torchify([info['action_adv'] for info in info_vec]).squeeze(axis=0)
    rewards = torchify([info['reward'] for info in info_vec]).squeeze(axis=0)
    cont_flags = torchify([info['done'] for info in info_vec]).squeeze(axis=0)
                        
    critic_loss_li = []
    critic_loss_random = []
    critic_loss_adv = []
    policy_loss_li = []
    policy_loss_random = []
    policy_loss_adv = []
    fo_inrd_li = []
    fo_inrd_adv = []
    fo_inrd_random = []
    for i in range(0, states.shape[0]):
        critic_loss = compute_critic_loss(expert_agent, states[i].unsqueeze(0), actions[i].unsqueeze(0), next_states[i].unsqueeze(0),
                                        rewards[i].unsqueeze(0), cont_flags[i].unsqueeze(0))
        critic_loss_li.append(critic_loss)

        critic_loss_random.append(compute_critic_loss(expert_agent, states[i].unsqueeze(0), actions_random[i].unsqueeze(0), next_states[i].unsqueeze(0),
                                        rewards[i].unsqueeze(0), cont_flags[i].unsqueeze(0)))
        critic_loss_adv.append(compute_critic_loss(expert_agent, states[i].unsqueeze(0), actions_adv[i].unsqueeze(0), next_states[i].unsqueeze(0),
                                        rewards[i].unsqueeze(0), cont_flags[i].unsqueeze(0)))
        
        policy_loss_li.append(compute_policy_loss(expert_agent, states[i].unsqueeze(0), actions[i].unsqueeze(0)))
        policy_loss_random.append(compute_policy_loss(expert_agent, states[i].unsqueeze(0), actions_random[i].unsqueeze(0)))
        policy_loss_adv.append(compute_policy_loss(expert_agent, states[i].unsqueeze(0), actions_adv[i].unsqueeze(0)))

        fo_inrd_li.append(fo_inrd(expert_agent, states[i].unsqueeze(0), actions[i].unsqueeze(0)))
        fo_inrd_adv.append(fo_inrd(expert_agent, states[i].unsqueeze(0), actions_adv[i].unsqueeze(0)))
        fo_inrd_random.append(fo_inrd(expert_agent, states[i].unsqueeze(0), actions_random[i].unsqueeze(0)))

    # Plot Episode Data

    # plt.plot(np.arange(0, states.shape[0], 1), critic_loss_li, label='Task Policy')
    # plt.plot(np.arange(0, states.shape[0], 1), critic_loss_adv, label='Adversary Policy')
    # plt.plot(np.arange(0, states.shape[0], 1), critic_loss_random, label='Random Policy')
    # plt.plot(np.arange(0, states.shape[0], 1), critic_loss_li, label='Attack Rate: {}'.format(atk_rate))
    # plt.plot(np.arange(0, states.shape[0], 1), policy_loss_li, label='Attack Rate: {}'.format(atk_rate))

    # plt.plot(np.arange(0, states.shape[0], 1), policy_loss_li, label='Task Action')
    # plt.plot(np.arange(0, states.shape[0], 1), policy_loss_adv, label='Adversary Action')
    # plt.plot(np.arange(0, states.shape[0], 1), policy_loss_random, label='Random Action')

    # plt.plot(np.arange(0, states.shape[0], 1), fo_inrd_li, label='Attack Rate: {}'.format(atk_rate))
    plt.plot(np.arange(0, states.shape[0], 1), fo_inrd_li, label='Task Action')
    plt.plot(np.arange(0, states.shape[0], 1), fo_inrd_random, label='Random Action')
    plt.plot(np.arange(0, states.shape[0], 1), fo_inrd_adv, label='Adversary Action')

  plt.xlabel('Steps')
#   plt.ylabel('Loss')
  plt.ylabel('delta_loss')
  plt.yscale('log')
  plt.title('FO-INRD Comparison - Same States, Different Actions ({})'.format(env_name))
  plt.legend()
  plt.savefig(os.path.join(exp_data_dir, 'fo_inrd_comparison_task_rnd_adv_{}.png'.format(env_name)), dpi=300, format='png',bbox_inches='tight')
  plt.close(fig)  # Close the figure to free memory
