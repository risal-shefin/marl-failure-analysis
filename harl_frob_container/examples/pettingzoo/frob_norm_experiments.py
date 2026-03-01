"""
Frobenius Norm Experiments for HARL PettingZoo Environments.

Mirrors maddpg-pytorch/scripts/pettingzoo/frob_norm_experiments.py but adapted
for HARL off-policy algorithms (haddpg, maddpg, hatd3, matd3).

Cross-Hessian Frobenius norm definition (HARL):
    H[i][j] = || ∂²Q / (∂a_i ∂a_j) ||_F

where Q = critic(share_obs, cat(a_0, ..., a_{N-1})) is the shared centralized
Q-function.  Because HARL uses a cooperative shared reward, Q is the same for
all agents; the off-diagonal H[i][j] (i ≠ j) captures how a change in agent
j's action direction distorts the gradient signal received by agent i.

Supported algorithms: haddpg, maddpg, hatd3, matd3
  (continuous-action off-policy algorithms with ContinuousQCritic)

Experiments
-----------
episode_gif_frob_norm
    Run a single seeded episode.  Save a per-timestep Frobenius norm log and
    a heatmap PNG (rows = agent pairs, cols = timesteps).

grad_shift_svd_coupling
    Run many seeded episodes.  For each (i, j) pair and every timestep measure
    ||H||_F and the induced gradient shift ||Δg||_2 after perturbing a_j along
    the top right singular vector of H.  Saves CSVs and scatter plots.

Usage examples
--------------
    python frob_norm_experiments.py episode_gif_frob_norm \\
        --algo haddpg \\
        --scenario simple_spread_v3 \\
        --model_dir /path/to/models \\
        --seed 0

    python frob_norm_experiments.py grad_shift_svd_coupling \\
        --algo haddpg \\
        --scenario simple_spread_v3 \\
        --model_dir /path/to/models \\
        --epsilon 0.01 \\
        --total_experiments 100
"""
import argparse
import os
import sys
import random
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from tqdm import tqdm
from torch.autograd import Variable

# Allow invocation from examples/pettingzoo/ or from harl_frob_container root
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..'))

from harl.envs.pettingzoo_mpe.pettingzoo_mpe_env import PettingZooMPEEnv
from harl.algorithms.actors import ALGO_REGISTRY
from harl.algorithms.critics import CRITIC_REGISTRY
from harl.algorithms.critics.centralized_q_critic import CentralizedQCritic
from harl.utils.configs_tools import get_defaults_yaml_args, update_args
from harl.utils.envs_tools import get_shape_from_act_space
from harl.utils.models_tools import find_checkpoint, init_device

# Off-policy: single shared ContinuousQCritic (actions as explicit input)
OFF_POLICY_ALGOS = {"haddpg", "maddpg", "hatd3", "matd3"}
# On-policy: per-agent CentralizedQCritic (stored as central_q_critic_agent{i})
ON_POLICY_ALGOS  = {"happo", "hatrpo", "haa2c", "mappo"}
ALL_ALGOS = OFF_POLICY_ALGOS | ON_POLICY_ALGOS


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_harl_model(algo, model_dir, scenario):
    """
    Instantiate and restore HARL actors and the appropriate Q-function(s) from
    model_dir, returning a unified ``get_Q`` callable.

    Supported algorithm families
    ----------------------------
    Off-policy (haddpg, maddpg, hatd3, matd3)
        One shared ``ContinuousQCritic``.  ``get_Q(i, s, acts)`` is the same
        function for every agent i.

    On-policy with central-Q (happo, hatrpo, haa2c, mappo)
        Per-agent ``CentralizedQCritic`` loaded from ``central_q_critic_agent{i}``.
        ``get_Q(i, s, acts)`` returns ``Q_i(share_obs, all_actions)``.
        Requires that the model_dir was saved with ``enable_central_q: True``.

    Args:
        algo      : algorithm name
        model_dir : directory containing actor/critic .pt checkpoints
        scenario  : PettingZoo MPE scenario (e.g. 'simple_spread_v3')

    Returns:
        actors     : list[actor obj] — one per agent
        get_Q      : callable(agent_i, share_obs_t, torch_actions_list) -> scalar
        env        : PettingZooMPEEnv
        device     : torch.device
        num_agents : int
    """
    assert algo in ALL_ALGOS, (
        f"Algorithm '{algo}' is not supported. "
        f"Choose from: {sorted(ALL_ALGOS)}"
    )

    algo_args, env_args = get_defaults_yaml_args(algo, 'pettingzoo_mpe')
    update_args({'scenario': scenario}, algo_args, env_args)

    device = init_device(algo_args['device'])

    env = PettingZooMPEEnv(dict(env_args))
    env.reset()

    num_agents   = env.n_agents
    obs_spaces   = env.observation_space
    act_spaces   = env.action_space
    share_obs_sp = env.share_observation_space[0]

    actor_args = {**algo_args['model'], **algo_args['algo']}

    # --- Actors (same loading path for both families) ---
    actors = []
    for agent_id in range(num_agents):
        actor = ALGO_REGISTRY[algo](
            actor_args, obs_spaces[agent_id], act_spaces[agent_id], device=device
        )
        sd = torch.load(
            find_checkpoint(model_dir, f'actor_agent{agent_id}'),
            map_location=device,
        )
        actor.actor.load_state_dict(sd)
        actor.actor.eval()
        actors.append(actor)

    # --- Q function ---
    if algo in OFF_POLICY_ALGOS:
        # Single shared ContinuousQCritic: Q(share_obs, cat(all_actions))
        critic_args = {**algo_args['train'], **algo_args['model'], **algo_args['algo']}
        critic = CRITIC_REGISTRY[algo](
            critic_args, share_obs_sp, act_spaces, num_agents, 'EP', device=device
        )
        sd = torch.load(find_checkpoint(model_dir, 'critic_agent'), map_location=device)
        critic.critic.load_state_dict(sd)
        critic.critic.eval()

        def get_Q(agent_i, share_obs_t, torch_actions):
            actions_cat = torch.cat(torch_actions, dim=-1)
            return critic.critic(share_obs_t, actions_cat).mean()

    else:  # ON_POLICY_ALGOS — per-agent CentralizedQCritic
        total_act_dim = sum(
            get_shape_from_act_space(act_spaces[i]) for i in range(num_agents)
        )
        critic_args = {**algo_args['model'], **algo_args['algo']}
        central_q_critics = []
        for agent_id in range(num_agents):
            cq = CentralizedQCritic(
                {**algo_args['train'], **algo_args['model'], **algo_args['algo']},
                share_obs_sp,
                total_act_dim,
                device=device,
            )
            sd = torch.load(
                find_checkpoint(model_dir, f'central_q_critic_agent{agent_id}'),
                map_location=device,
            )
            cq.critic.load_state_dict(sd)
            cq.critic.eval()
            central_q_critics.append(cq)

        # Dummy RNN inputs for non-recurrent VNet (not used in forward, just passed)
        _rnn_zeros = np.zeros((1, 1, 1), dtype=np.float32)
        _mask_ones = np.ones((1, 1), dtype=np.float32)

        def get_Q(agent_i, share_obs_t, torch_actions):
            actions_cat = torch.cat(torch_actions, dim=-1)  # (1, total_act_dim)
            # CentralizedQCritic expects combined [share_obs, all_actions] as input
            combined = torch.cat([share_obs_t, actions_cat], dim=-1)
            q_values, _ = central_q_critics[agent_i].critic(combined, _rnn_zeros, _mask_ones)
            return q_values.mean()

    return actors, get_Q, env, device, num_agents


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def _make_logdir(base_name, scenario, algo, nagents, seed):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join(
        os.getcwd(), 'runs', 'frob_norm_experiments', base_name,
        f"{scenario}_{algo}_nagents{nagents}_seed{seed}_{timestamp}"
    )
    os.makedirs(logdir, exist_ok=True)
    return logdir


def _make_multiseed_logdir(base_name, scenario, algo, nagents, n_seeds):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join(
        os.getcwd(), 'runs', 'frob_norm_experiments', base_name,
        f"{scenario}_{algo}_nagents{nagents}_seeds{n_seeds}_{timestamp}"
    )
    os.makedirs(logdir, exist_ok=True)
    return logdir


def _actor_step(actor, obs_t):
    """
    Get a single deterministic action from an actor, handling both
    off-policy (DeterministicPolicy) and on-policy (StochasticPolicy) actors.

    Off-policy actors expose ``get_actions(obs, add_noise)``.
    On-policy actors expose ``act(obs, rnn_states, masks, deterministic)``.
    """
    if hasattr(actor, 'target_actor'):       # off-policy
        return actor.get_actions(obs_t, add_noise=False)
    else:                                    # on-policy stochastic
        _rnn = np.zeros((1, 1, 1), dtype=np.float32)
        _msk = np.ones((1, 1),     dtype=np.float32)
        actions, _ = actor.act(obs_t, _rnn, _msk, deterministic=True)
        return actions


def _get_actions(actors, obs_list, device):
    """
    Get deterministic actions from all actors.

    Args:
        actors   : list of HARL actor objects
        obs_list : np.ndarray (n_agents, obs_dim)
        device   : torch.device

    Returns:
        np.ndarray (n_agents, action_dim)
    """
    actions = []
    for i, actor in enumerate(actors):
        obs_t = torch.FloatTensor(obs_list[i]).unsqueeze(0).to(device)
        with torch.no_grad():
            a = _actor_step(actor, obs_t)
        actions.append(a.cpu().numpy().squeeze(0))
    return np.array(actions)


# ---------------------------------------------------------------------------
# Cross-Hessian computations
# ---------------------------------------------------------------------------

def _compute_pairwise_frob_norms(actors, get_Q, obs_list, s_obs, device):
    """
    Compute cross-Hessian Frobenius norms for all (i, j) agent pairs.

        H[i][j] = || ∂²Q_i / (∂a_i ∂a_j) ||_F

    For off-policy algorithms Q_i is the same shared critic for all i.
    For on-policy algorithms Q_i is agent i's per-agent CentralizedQCritic.

    Args:
        actors   : list of actor objects
        get_Q    : callable(agent_i, share_obs_t, torch_actions_list) -> scalar
        obs_list : np.ndarray (n_agents, obs_dim)
        s_obs    : np.ndarray (state_dim,) — global/shared state
        device   : torch.device

    Returns:
        N×N list of floats
    """
    N = len(actors)

    raw_actions = []
    for i, actor in enumerate(actors):
        obs_t = torch.FloatTensor(obs_list[i]).unsqueeze(0).to(device)
        with torch.no_grad():
            a = _actor_step(actor, obs_t)
        raw_actions.append(a.detach())

    torch_actions = [Variable(raw_actions[i].clone(), requires_grad=True) for i in range(N)]
    share_obs_t   = torch.FloatTensor(s_obs).unsqueeze(0).to(device)

    results = [[0.0] * N for _ in range(N)]

    for i in range(N):
        Q_i    = get_Q(i, share_obs_t, torch_actions)
        grad_i = torch.autograd.grad(
            Q_i, torch_actions[i], create_graph=True, retain_graph=True
        )[0]  # (1, action_dim_i)

        for j in range(N):
            rows = []
            for k in range(grad_i.shape[1]):
                row = torch.autograd.grad(
                    grad_i[0, k],
                    torch_actions[j],
                    retain_graph=True,
                    allow_unused=True,
                    create_graph=False,
                )[0]
                rows.append(row.flatten())
            H = torch.stack(rows)                            # (dim_i, dim_j)
            results[i][j] = H.norm(p='fro').item()

    return results


def _compute_pairwise_svd_gradient_shift(actors, get_Q, obs_list, s_obs, device, epsilon=0.01):
    """
    Cross-Hessian Frobenius norm and SVD-directed gradient shift for every (i, j).

    Steps per pair:
      1. Compute H = ∂²Q_i/(∂a_i ∂a_j)  →  Frobenius norm ||H||_F
      2. Top right singular vector v_max of H (direction in a_j space that most
         rotates ∇_{a_i} Q_i)
      3. Perturb  a_j' = a_j + ε · v_max
      4. Measure  ||Δg||_2 = ||∇_{a_i} Q_i(a_j') – ∇_{a_i} Q_i(a_j)||_2

    Args:
        actors, get_Q, obs_list, s_obs, device: same semantics as above
        epsilon: perturbation magnitude along v_max

    Returns:
        dict mapping (agent_i, agent_j) -> {'frob_norm': float, 'delta_g_norm': float}
    """
    N = len(actors)

    raw_actions = []
    for i, actor in enumerate(actors):
        obs_t = torch.FloatTensor(obs_list[i]).unsqueeze(0).to(device)
        with torch.no_grad():
            a = _actor_step(actor, obs_t)
        raw_actions.append(a.detach())

    torch_actions = [Variable(raw_actions[i].clone(), requires_grad=True) for i in range(N)]
    share_obs_t   = torch.FloatTensor(s_obs).unsqueeze(0).to(device)

    results = {}

    for i in range(N):
        Q_i    = get_Q(i, share_obs_t, torch_actions)
        grad_i = torch.autograd.grad(
            Q_i, torch_actions[i], create_graph=True, retain_graph=True
        )[0]  # (1, dim_i)

        for j in range(N):
            # Build cross-Hessian H (dim_i × dim_j)
            rows = []
            for k in range(grad_i.shape[1]):
                row = torch.autograd.grad(
                    grad_i[0, k],
                    torch_actions[j],
                    retain_graph=True,
                    allow_unused=True,
                    create_graph=False,
                )[0]
                rows.append(row.flatten())
            H = torch.stack(rows)                            # (dim_i, dim_j)
            frob_norm = H.norm(p='fro').item()

            # Top right singular vector of H
            _, _, Vt = torch.linalg.svd(H.detach(), full_matrices=False)
            v_max    = Vt[0].unsqueeze(0)                   # (1, dim_j)

            # Perturbed action_j
            perturbed_aj = (raw_actions[j] + epsilon * v_max).detach().requires_grad_(True)

            actions_p = [
                raw_actions[idx].clone().detach().requires_grad_(True) if idx != j
                else perturbed_aj
                for idx in range(N)
            ]

            grad_i_p = torch.autograd.grad(
                get_Q(i, share_obs_t, actions_p),
                actions_p[i],
                retain_graph=False,
            )[0]

            delta_g_norm = (grad_i_p - grad_i.detach()).norm(p=2).item()

            results[(i, j)] = {'frob_norm': frob_norm, 'delta_g_norm': delta_g_norm}

    return results


# ---------------------------------------------------------------------------
# GIF saving
# ---------------------------------------------------------------------------

def _save_gif(frames, filepath, fps=10):
    """Save a list of RGB numpy frames as an animated GIF using imageio."""
    import imageio
    if not frames:
        print(f"  Warning: no frames to save for {filepath}")
        return
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    imageio.mimsave(filepath, frames, duration=1.0 / fps)
    print(f"  GIF saved ({len(frames)} frames): {filepath}")


# ---------------------------------------------------------------------------
# Episode runners
# ---------------------------------------------------------------------------

def _run_episode_with_frob_norms(actors, get_Q, env, seed, device):
    """
    Run one episode and collect per-timestep cross-Hessian Frobenius norms
    and RGB frames for GIF rendering.

    Returns:
        frob_norms_history : list of N×N float matrices, one per timestep
        frames             : list of RGB numpy arrays, one per timestep
        episode_length     : int
    """
    _set_seeds(seed)
    env.seed(seed)
    obs, s_obs, _ = env.reset()

    frob_norms_history = []
    frames = [env.render()]  # initial frame

    while True:
        frob_matrix = _compute_pairwise_frob_norms(actors, get_Q, obs, s_obs[0], device)
        frob_norms_history.append(frob_matrix)

        actions = _get_actions(actors, obs, device)
        obs, s_obs, _, dones, _, _ = env.step(actions)
        frames.append(env.render())

        if all(dones):
            break

    return frob_norms_history, frames, len(frob_norms_history)


def _run_svd_coupling_episode(actors, get_Q, env, seed, device, epsilon):
    """
    Run one episode collecting SVD-based coupling data at every timestep.

    Returns:
        list of dicts: seed, timestep, agent_i, agent_j, frob_norm, delta_g_norm
    """
    _set_seeds(seed)
    env.seed(seed)
    obs, s_obs, _ = env.reset()

    records  = []
    timestep = 0

    while True:
        coupling = _compute_pairwise_svd_gradient_shift(
            actors, get_Q, obs, s_obs[0], device, epsilon
        )
        for (agent_i, agent_j), m in coupling.items():
            records.append({
                'seed':         seed,
                'timestep':     timestep,
                'agent_i':      agent_i,
                'agent_j':      agent_j,
                'frob_norm':    m['frob_norm'],
                'delta_g_norm': m['delta_g_norm'],
            })

        actions = _get_actions(actors, obs, device)
        obs, s_obs, _, dones, _, _ = env.step(actions)
        timestep += 1

        if all(dones):
            break

    return records


# ---------------------------------------------------------------------------
# Frob-norm log + heatmap  (identical logic to MADDPG version)
# ---------------------------------------------------------------------------

def _save_frob_norm_log(frob_norms_history, nagents, logdir, seed):
    """Write per-timestep Frobenius norms and summary statistics to a text file."""
    pair_stats = {
        (i, j): {'max': -float('inf'), 'max_t': -1, 'min': float('inf'), 'min_t': -1}
        for i in range(nagents) for j in range(nagents) if i != j
    }
    for t, frob_matrix in enumerate(frob_norms_history):
        for i in range(nagents):
            for j in range(nagents):
                if i == j:
                    continue
                v = frob_matrix[i][j]
                s = pair_stats[(i, j)]
                if v > s['max']:
                    s['max'], s['max_t'] = v, t
                if v < s['min']:
                    s['min'], s['min_t'] = v, t

    overall_max = {'val': -float('inf'), 't': -1, 'pair': None}
    overall_min = {'val':  float('inf'), 't': -1, 'pair': None}
    for (i, j), s in pair_stats.items():
        if s['max'] > overall_max['val']:
            overall_max.update({'val': s['max'], 't': s['max_t'], 'pair': (i, j)})
        if s['min'] < overall_min['val']:
            overall_min.update({'val': s['min'], 't': s['min_t'], 'pair': (i, j)})

    log_path = os.path.join(logdir, f"frob_norms_seed{seed}.txt")
    with open(log_path, 'w') as f:
        f.write(f"Cross-Hessian Frobenius Norms — seed {seed}\n")
        f.write("=" * 60 + "\n\n")

        f.write("--- Per-Timestep Values ---\n\n")
        for t, frob_matrix in enumerate(frob_norms_history):
            f.write(f"Timestep {t:4d}:\n")
            for i in range(nagents):
                for j in range(nagents):
                    if i == j:
                        continue
                    f.write(f"  agent_{j} -> agent_{i}: {frob_matrix[i][j]:.6f}\n")
            f.write("\n")

        f.write("--- Per-Pair Max / Min ---\n\n")
        for i in range(nagents):
            for j in range(nagents):
                if i == j:
                    continue
                s = pair_stats[(i, j)]
                f.write(f"  agent_{j} -> agent_{i}:\n")
                f.write(f"    max: {s['max']:.6f}  at t={s['max_t']}\n")
                f.write(f"    min: {s['min']:.6f}  at t={s['min_t']}\n")
        f.write("\n")

        f.write("--- Overall Max / Min ---\n\n")
        om = overall_max
        f.write(f"  max: {om['val']:.6f}  at t={om['t']}  pair: agent_{om['pair'][1]} -> agent_{om['pair'][0]}\n")
        on = overall_min
        f.write(f"  min: {on['val']:.6f}  at t={on['t']}  pair: agent_{on['pair'][1]} -> agent_{on['pair'][0]}\n")

    print(f"  Frob norm log saved: {log_path}")
    return log_path


def _save_frob_norm_heatmap(frob_norms_history, nagents, logdir, seed, max_cols=60):
    """
    Plot and save a heatmap of per-timestep Frobenius norms.

    Rows  : agent pairs  (agent_j → agent_i, i ≠ j)
    Cols  : timesteps, subsampled to at most *max_cols* evenly-spaced points
    Color : Frobenius norm value
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    T      = len(frob_norms_history)
    pairs  = [(i, j) for i in range(nagents) for j in range(nagents) if i != j]
    n_pairs = len(pairs)

    full_matrix = np.array(
        [[frob_norms_history[t][i][j] for t in range(T)] for i, j in pairs]
    )  # (n_pairs, T)

    col_indices = (
        np.round(np.linspace(0, T - 1, max_cols)).astype(int) if T > max_cols
        else np.arange(T)
    )
    matrix     = full_matrix[:, col_indices]
    col_labels = col_indices.tolist()
    row_labels = [f"agent_{j} → agent_{i}" for i, j in pairs]
    n_cols     = len(col_indices)

    fig, ax = plt.subplots(figsize=(max(10, n_cols * 0.22), max(3, n_pairs * 0.75)))
    sns.heatmap(
        matrix, ax=ax, cmap='viridis',
        xticklabels=col_labels, yticklabels=row_labels,
        cbar_kws={'label': 'Frobenius Norm'},
        linewidths=0.3, linecolor='#444444',
    )
    show_every = max(1, n_cols // 15)
    for idx, tick in enumerate(ax.get_xticklabels()):
        tick.set_visible(idx % show_every == 0)
        tick.set_rotation(45)
        tick.set_ha('right')

    ax.set_xlabel('Timestep', labelpad=8)
    ax.set_ylabel('Agent Pair', labelpad=8)
    ax.set_title('Cross-Hessian Frobenius Norms', pad=12)
    plt.tight_layout()

    heatmap_path = os.path.join(logdir, f"frob_norm_heatmap_seed{seed}.png")
    fig.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Heatmap saved  : {heatmap_path}")
    return heatmap_path


# ---------------------------------------------------------------------------
# SVD coupling CSV + plots  (identical logic to MADDPG version)
# ---------------------------------------------------------------------------

def _save_svd_coupling_csv(df, logdir):
    """Save raw per-timestep coupling data and a per-pair summary CSV."""
    csv_dir = os.path.join(logdir, 'csv_data')
    os.makedirs(csv_dir, exist_ok=True)

    raw_path = os.path.join(csv_dir, 'raw_coupling_data.csv')
    df.to_csv(raw_path, index=False)
    print(f"  Raw data saved      : {raw_path}")

    summary = (
        df.groupby(['agent_j', 'agent_i'])
          .agg(
              mean_frob_norm   = ('frob_norm',    'mean'),
              std_frob_norm    = ('frob_norm',    'std'),
              mean_delta_g_norm= ('delta_g_norm', 'mean'),
              std_delta_g_norm = ('delta_g_norm', 'std'),
              n_samples        = ('frob_norm',    'count'),
          )
          .reset_index()
    )
    summary_path = os.path.join(csv_dir, 'mean_coupling_by_pair.csv')
    summary.to_csv(summary_path, index=False)
    print(f"  Summary CSV saved   : {summary_path}")
    print(f"\n  Per-pair summary:")
    print(summary.to_string(index=False))
    return summary_path


def _save_svd_coupling_plots(df, logdir):
    """
    Scatter plots of ||H||_F vs ||Δg||_2 per agent pair (combined + individual).
    Each subplot shows a linear regression line and Pearson r.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy import stats

    plots_dir = os.path.join(logdir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    pairs  = df[['agent_i', 'agent_j']].drop_duplicates().values
    n_pairs = len(pairs)
    n_cols  = min(3, int(np.ceil(np.sqrt(n_pairs))))
    n_rows  = int(np.ceil(n_pairs / n_cols))

    # --- Combined figure ---
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_pairs == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (agent_i, agent_j) in enumerate(pairs):
        ax        = axes[idx]
        pair_data = df[(df['agent_i'] == agent_i) & (df['agent_j'] == agent_j)].dropna(
            subset=['frob_norm', 'delta_g_norm']
        )
        if len(pair_data) == 0:
            ax.text(0.5, 0.5, f'No data ({agent_i},{agent_j})',
                    ha='center', va='center', transform=ax.transAxes)
            continue

        x = pair_data['frob_norm'].values
        y = pair_data['delta_g_norm'].values
        ax.scatter(x, y, alpha=0.3, s=10, c='steelblue')

        if len(x) > 1:
            try:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                x_fit = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_fit, p(x_fit), 'r--', linewidth=2,
                        label=f'y={z[0]:.3f}x+{z[1]:.3f}')
                r, pval = stats.pearsonr(x, y)
                ax.text(0.05, 0.95, f'r = {r:.3f}\np = {pval:.3e}',
                        transform=ax.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                ax.legend(fontsize=9)
            except Exception:
                pass

        ax.set_xlabel('||H||_F (Frobenius Norm)')
        ax.set_ylabel('||Δg||₂ (Gradient Shift)')
        ax.set_title(f'agent_{agent_j} → agent_{agent_i}')
        ax.grid(True, alpha=0.3)

    for idx in range(n_pairs, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    combined_path = os.path.join(plots_dir, 'coupling_analysis_all_pairs.png')
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Combined scatter plot: {combined_path}")

    # --- Individual plots per pair ---
    for agent_i, agent_j in pairs:
        fig, ax   = plt.subplots(figsize=(8, 6))
        pair_data = df[(df['agent_i'] == agent_i) & (df['agent_j'] == agent_j)].dropna(
            subset=['frob_norm', 'delta_g_norm']
        )
        if len(pair_data) == 0:
            plt.close()
            continue

        x = pair_data['frob_norm'].values
        y = pair_data['delta_g_norm'].values
        ax.scatter(x, y, alpha=0.3, s=20, c='steelblue')

        if len(x) > 1:
            try:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                x_fit = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_fit, p(x_fit), 'r--', linewidth=2,
                        label=f'y={z[0]:.3f}x+{z[1]:.3f}')
                r, pval = stats.pearsonr(x, y)
                ax.text(0.05, 0.95,
                        f'Pearson r = {r:.3f}\np-value = {pval:.3e}',
                        transform=ax.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                        fontsize=12)
                ax.legend(fontsize=11)
            except Exception:
                pass

        ax.set_xlabel('||H||_F (Frobenius Norm)', fontsize=12)
        ax.set_ylabel('||Δg||₂ (Gradient Shift)', fontsize=12)
        ax.set_title(f'SVD Coupling: agent_{agent_j} → agent_{agent_i}', fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        ind_path = os.path.join(plots_dir, f'svd_coupling_pair_{agent_j}_to_{agent_i}.png')
        plt.savefig(ind_path, dpi=150, bbox_inches='tight')
        plt.close()

    print(f"  Individual pair plots: {plots_dir}")
    return plots_dir


# ---------------------------------------------------------------------------
# Experiment 1: single seeded episode — frob norm log + heatmap
# ---------------------------------------------------------------------------

def frob_norm_episode_experiment(config):
    """
    Run a single seeded episode, log the cross-Hessian Frobenius norm for every
    agent pair at every timestep, and save a heatmap PNG.

    Args:
        config: Namespace with fields:
                  algo, scenario, model_dir, seed
    """
    actors, get_Q, env, device, nagents = _load_harl_model(
        config.algo, config.model_dir, config.scenario
    )
    logdir = _make_logdir('episode_frob_norm', config.scenario, config.algo, nagents, config.seed)

    print(f"[frob_norm_episode_experiment]")
    print(f"  algo     : {config.algo}")
    print(f"  env      : {config.scenario}")
    print(f"  agents   : {nagents}")
    print(f"  seed     : {config.seed}")
    print(f"  output   : {logdir}")

    frob_norms_history, frames, episode_length = _run_episode_with_frob_norms(
        actors, get_Q, env, config.seed, device
    )
    print(f"  episode length: {episode_length} timesteps")

    gif_path = os.path.join(logdir, f"episode_seed{config.seed}.gif")
    _save_gif(frames, gif_path, fps=10)

    _save_frob_norm_log(frob_norms_history, nagents, logdir, config.seed)
    _save_frob_norm_heatmap(frob_norms_history, nagents, logdir, config.seed)

    env.close()
    print("Done.")


# ---------------------------------------------------------------------------
# Experiment 2: multi-seed SVD coupling — frob norm vs gradient shift
# ---------------------------------------------------------------------------

def svd_coupling_experiment(config):
    """
    Run a multi-seed SVD-based gradient coupling experiment.

    For every (i, j) pair at every timestep of every seed episode:
      - Computes H = ∂²Q/(∂a_i ∂a_j) and ||H||_F
      - Perturbs a_j along the top right singular vector of H (pure second-order
        direction in action space)
      - Measures gradient shift ||Δg||_2 = ||∇_{a_i} Q(a_j') – ∇_{a_i} Q(a_j)||_2

    Saves:
      csv_data/raw_coupling_data.csv        — every (seed, timestep, pair) row
      csv_data/mean_coupling_by_pair.csv    — mean ± std per pair
      plots/coupling_analysis_all_pairs.png — combined scatter
      plots/svd_coupling_pair_<j>_to_<i>.png

    Args:
        config: Namespace with fields:
                  algo, scenario, model_dir, epsilon, total_experiments
    """
    actors, get_Q, env, device, nagents = _load_harl_model(
        config.algo, config.model_dir, config.scenario
    )
    logdir = _make_multiseed_logdir(
        'grad_shift_svd_coupling', config.scenario, config.algo,
        nagents, config.total_experiments
    )

    print(f"[grad_shift_svd_coupling_experiment]")
    print(f"  algo         : {config.algo}")
    print(f"  env          : {config.scenario}")
    print(f"  agents       : {nagents}")
    print(f"  seeds        : {config.total_experiments}")
    print(f"  epsilon      : {config.epsilon}")
    print(f"  output       : {logdir}")

    all_records = []
    for seed in tqdm(range(config.total_experiments), desc="Seeds"):
        records = _run_svd_coupling_episode(actors, get_Q, env, seed, device, config.epsilon)
        all_records.extend(records)

    env.close()

    n_points = len(all_records)
    print(f"\n  Collected {n_points} data points across {config.total_experiments} seeds")

    df = pd.DataFrame(all_records)
    _save_svd_coupling_csv(df, logdir)
    _save_svd_coupling_plots(df, logdir)

    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser():
    parser = argparse.ArgumentParser(
        description="Frobenius Norm Experiments for HARL PettingZoo environments"
    )
    sub = parser.add_subparsers(dest='experiment', required=True)

    # Shared arguments added to both subcommands
    def _add_common(p):
        p.add_argument('--algo', type=str, default='haddpg',
                       choices=sorted(ALL_ALGOS),
                       help='HARL algorithm (default: haddpg)')
        p.add_argument('--scenario', type=str, default='simple_spread_v3',
                       help='PettingZoo MPE scenario (default: simple_spread_v3)')
        p.add_argument('--model_dir', type=str, required=True,
                       help='Path to directory containing actor/critic checkpoints')

    # --- episode_gif_frob_norm ---
    ep = sub.add_parser('episode_gif_frob_norm',
                        help='Single seeded episode: frob norm log + heatmap')
    _add_common(ep)
    ep.add_argument('--seed', type=int, default=0, help='Random seed (default: 0)')

    # --- grad_shift_svd_coupling ---
    sc = sub.add_parser('grad_shift_svd_coupling',
                        help='Multi-seed SVD coupling: frob norm vs gradient shift')
    _add_common(sc)
    sc.add_argument('--epsilon', type=float, default=0.01,
                    help='Perturbation magnitude along v_max (default: 0.01)')
    sc.add_argument('--total_experiments', type=int, default=100,
                    help='Number of seed episodes (default: 100)')

    return parser


def main():
    config = _build_parser().parse_args()

    if config.experiment == 'episode_gif_frob_norm':
        frob_norm_episode_experiment(config)
    elif config.experiment == 'grad_shift_svd_coupling':
        svd_coupling_experiment(config)


if __name__ == '__main__':
    main()
