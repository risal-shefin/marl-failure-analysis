"""
Frobenius Norm Experiments for PettingZoo Environments.

Houses multiple experiment initiators, each as a separate function called from main.
"""
import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from tqdm import tqdm
from torch.autograd import Variable

from algorithms.maddpg import MADDPG
from modules.constants import DEVICE, torch_device
from modules.environment import create_environment
from modules.metrics import compute_pairwise_frob_norms
from modules.metrics.basic_metrics import compute_pairwise_svd_gradient_shift
from modules.visualization.utils import save_frames_as_gif


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def _make_logdir(base_name, env_id, env_type, nagents, seed):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join(
        os.getcwd(), 'runs', 'frob_norm_experiments', base_name,
        f"{env_id}_{env_type}_nagents{nagents}_seed{seed}_{timestamp}"
    )
    os.makedirs(logdir, exist_ok=True)
    return logdir


def _make_multiseed_logdir(base_name, env_id, env_type, nagents, n_seeds):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join(
        os.getcwd(), 'runs', 'frob_norm_experiments', base_name,
        f"{env_id}_{env_type}_nagents{nagents}_seeds{n_seeds}_{timestamp}"
    )
    os.makedirs(logdir, exist_ok=True)
    return logdir


def _render_frame(env):
    try:
        return env.render(mode='rgb_array')
    except TypeError:
        return env.render()


def _run_episode_with_frob_norms(maddpg, env, seed, collect_frames=True):
    """
    Run one episode and collect per-timestep cross-Hessian Frobenius norms.

    Returns:
        frob_norms_history : list of N×N float matrices, one per timestep
        frames             : list of RGB frames (empty if collect_frames=False)
        episode_length     : int
    """
    _set_seeds(seed)
    with torch.no_grad():
        maddpg.prep_rollouts(device=DEVICE)

    obs = env.reset(seed=seed)
    frob_norms_history = []
    frames = []

    if collect_frames:
        frames.append(_render_frame(env))

    while True:
        torch_obs = [
            Variable(torch.tensor([obs[i]], dtype=torch.float32).to(torch_device),
                     requires_grad=False)
            for i in range(maddpg.nagents)
        ]

        with torch.no_grad():
            torch_agent_actions = maddpg.step(torch_obs, explore=False)
            agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions]

        if maddpg.discrete_action:
            actions = {name: agent_actions[i].argmax()
                       for i, name in enumerate(env.possible_agents)}
        else:
            actions = {name: agent_actions[i].squeeze()
                       for i, name in enumerate(env.possible_agents)}
            # actions['agent_1'] = np.zeros_like(actions['agent_1'])  # freeze an agent to identify it in the video

        # Compute cross-Hessian Frobenius norms for all (i, j) pairs
        frob_matrix = compute_pairwise_frob_norms(
            maddpg, obs, list(actions.values()), env.action_space
        )
        frob_norms_history.append(frob_matrix)

        next_obs, _, dones, _ = env.step(actions)
        obs = next_obs

        if collect_frames:
            frames.append(_render_frame(env))

        if dones.all():
            break

    return frob_norms_history, frames, len(frob_norms_history)


def _save_frob_norm_log(frob_norms_history, nagents, logdir, seed):
    """Write per-timestep Frobenius norms and summary statistics to a text file."""
    # Pre-compute per-pair stats in one pass
    # pair_stats[i][j] = {'max': val, 'max_t': t, 'min': val, 'min_t': t}
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

    # Overall max/min across all pairs
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

        # Per-timestep values
        f.write("--- Per-Timestep Values ---\n\n")
        for t, frob_matrix in enumerate(frob_norms_history):
            f.write(f"Timestep {t:4d}:\n")
            for i in range(nagents):
                for j in range(nagents):
                    if i == j:
                        continue
                    f.write(f"  agent_{j} -> agent_{i}: {frob_matrix[i][j]:.6f}\n")
            f.write("\n")

        # Per-pair max/min summary
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

        # Overall max/min
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

    Rows  : agent pairs  (agent_i → agent_j, i ≠ j)
    Cols  : timesteps, subsampled to at most *max_cols* evenly-spaced points
            so each column represents a well-distributed slice of the episode.
    Color : Frobenius norm value
    """
    import matplotlib
    matplotlib.use('Agg')  # headless rendering on HPC
    import matplotlib.pyplot as plt
    import seaborn as sns

    T = len(frob_norms_history)
    pairs = [(i, j) for i in range(nagents) for j in range(nagents) if i != j]
    n_pairs = len(pairs)

    # Build full data matrix (n_pairs × T)
    full_matrix = np.array(
        [[frob_norms_history[t][i][j] for t in range(T)] for i, j in pairs]
    )  # shape: (n_pairs, T)

    # Subsample columns to at most max_cols, evenly distributed across [0, T-1]
    if T > max_cols:
        col_indices = np.round(np.linspace(0, T - 1, max_cols)).astype(int)
    else:
        col_indices = np.arange(T)

    matrix = full_matrix[:, col_indices]          # (n_pairs, n_cols)
    col_labels = col_indices.tolist()              # actual timestep numbers
    row_labels = [f"agent_{j} → agent_{i}" for i, j in pairs]

    n_cols = len(col_indices)

    # Figure size scales with number of columns / pairs
    fig_w = max(10, n_cols * 0.22)
    fig_h = max(3, n_pairs * 0.75)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        matrix,
        ax=ax,
        cmap='viridis',
        xticklabels=col_labels,
        yticklabels=row_labels,
        cbar_kws={'label': 'Frobenius Norm'},
        linewidths=0.3,
        linecolor='#444444',
    )

    # Show at most 15 x-tick labels to avoid crowding
    show_every = max(1, n_cols // 15)
    for idx, tick in enumerate(ax.get_xticklabels()):
        tick.set_visible(idx % show_every == 0)
        tick.set_rotation(45)
        tick.set_ha('right')

    ax.set_xlabel('Timestep', labelpad=8)
    ax.set_ylabel('Agent Pair', labelpad=8)
    ax.set_title(f'Cross-Hessian Frobenius Norms', pad=12)

    plt.tight_layout()
    heatmap_path = os.path.join(logdir, f"frob_norm_heatmap_seed{seed}.png")
    fig.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Heatmap saved  : {heatmap_path}")
    return heatmap_path


def _run_svd_coupling_episode(maddpg, env, seed, epsilon):
    """
    Run one episode and collect SVD-based gradient coupling data at each timestep.

    For every agent pair (i, j) at every timestep, records the cross-Hessian
    Frobenius norm and the resulting gradient shift after perturbing agent j's
    action along the top right singular vector of H.

    Returns:
        list of dicts with keys:
          seed, timestep, agent_i, agent_j, frob_norm, delta_g_norm
    """
    _set_seeds(seed)
    with torch.no_grad():
        maddpg.prep_rollouts(device=DEVICE)

    obs = env.reset(seed=seed)
    records = []
    timestep = 0

    while True:
        torch_obs = [
            Variable(torch.tensor([obs[i]], dtype=torch.float32).to(torch_device),
                     requires_grad=False)
            for i in range(maddpg.nagents)
        ]
        with torch.no_grad():
            torch_agent_actions = maddpg.step(torch_obs, explore=False)
            agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions]

        actions = {name: agent_actions[i].squeeze()
                   for i, name in enumerate(env.possible_agents)}

        coupling = compute_pairwise_svd_gradient_shift(
            maddpg, obs, list(actions.values()), epsilon
        )

        for (agent_i, agent_j), m in coupling.items():
            records.append({
                'seed': seed,
                'timestep': timestep,
                'agent_i': agent_i,
                'agent_j': agent_j,
                'frob_norm': m['frob_norm'],
                'delta_g_norm': m['delta_g_norm'],
            })

        next_obs, _, dones, _ = env.step(actions)
        obs = next_obs
        timestep += 1

        if dones.all():
            break

    return records


def _save_svd_coupling_csv(df, logdir):
    """
    Save raw per-timestep coupling data and a per-pair summary CSV.

    Summary columns: agent_j, agent_i, mean_frob_norm, std_frob_norm,
                     mean_delta_g_norm, std_delta_g_norm, n_samples
    """
    csv_dir = os.path.join(logdir, 'csv_data')
    os.makedirs(csv_dir, exist_ok=True)

    raw_path = os.path.join(csv_dir, 'raw_coupling_data.csv')
    df.to_csv(raw_path, index=False)
    print(f"  Raw data saved      : {raw_path}")

    summary = (
        df.groupby(['agent_j', 'agent_i'])
          .agg(
              mean_frob_norm=('frob_norm', 'mean'),
              std_frob_norm=('frob_norm', 'std'),
              mean_delta_g_norm=('delta_g_norm', 'mean'),
              std_delta_g_norm=('delta_g_norm', 'std'),
              n_samples=('frob_norm', 'count')
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

    Each subplot shows a linear regression line and Pearson correlation.
    Pair label convention: frob[i][j] means agent_j influences agent_i,
    so the title reads "agent_j → agent_i".
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy import stats

    plots_dir = os.path.join(logdir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    pairs = df[['agent_i', 'agent_j']].drop_duplicates().values
    n_pairs = len(pairs)
    n_cols = min(3, int(np.ceil(np.sqrt(n_pairs))))
    n_rows = int(np.ceil(n_pairs / n_cols))

    # --- Combined figure ---
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_pairs == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (agent_i, agent_j) in enumerate(pairs):
        ax = axes[idx]
        pair_data = df[
            (df['agent_i'] == agent_i) & (df['agent_j'] == agent_j)
        ].dropna(subset=['frob_norm', 'delta_g_norm'])

        if len(pair_data) == 0:
            ax.text(0.5, 0.5, f'No data for ({agent_i}, {agent_j})',
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
        ax.set_ylabel('||\u0394g||\u2082 (Gradient Shift)')
        ax.set_title(f'agent_{agent_j} \u2192 agent_{agent_i}')
        ax.grid(True, alpha=0.3)

    for idx in range(n_pairs, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    combined_path = os.path.join(plots_dir, 'svd_coupling_frob_vs_deltag_all_pairs.png')
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Combined scatter plot: {combined_path}")

    # --- Individual plots per pair ---
    for agent_i, agent_j in pairs:
        fig, ax = plt.subplots(figsize=(8, 6))
        pair_data = df[
            (df['agent_i'] == agent_i) & (df['agent_j'] == agent_j)
        ].dropna(subset=['frob_norm', 'delta_g_norm'])

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
        ax.set_ylabel('||\u0394g||\u2082 (Gradient Shift)', fontsize=12)
        ax.set_title(f'SVD Coupling: agent_{agent_j} \u2192 agent_{agent_i}', fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        ind_path = os.path.join(plots_dir,
                                f'svd_coupling_pair_{agent_j}_to_{agent_i}.png')
        plt.savefig(ind_path, dpi=150, bbox_inches='tight')
        plt.close()

    print(f"  Individual pair plots: {plots_dir}")
    return plots_dir


# ---------------------------------------------------------------------------
# Experiment 1: single seeded episode — GIF + frob norm log
# ---------------------------------------------------------------------------

def frob_norm_episode_experiment(config):
    """
    Run a single seeded episode, save a GIF of the episode, and log the
    cross-Hessian Frobenius norm for every agent pair at every timestep.

    Args:
        config: Namespace with fields:
                  env_id, model_path, seed
    """
    maddpg = MADDPG.init_from_save(config.model_path)
    env_type = 'discrete' if maddpg.discrete_action else 'continuous'

    device_str = 'gpu' if DEVICE == 'gpu' else 'cpu'
    maddpg.prep_training(device=device_str)

    env = create_environment(config, maddpg)
    logdir = _make_logdir('episode_gif_frob_norm', config.env_id, env_type, maddpg.nagents, config.seed)

    print(f"[frob_norm_episode_experiment]")
    print(f"  env      : {config.env_id}  ({env_type})")
    print(f"  agents   : {maddpg.nagents}")
    print(f"  seed     : {config.seed}")
    print(f"  output   : {logdir}")

    frob_norms_history, frames, episode_length = _run_episode_with_frob_norms(
        maddpg, env, config.seed, collect_frames=True
    )

    print(f"  episode length: {episode_length} timesteps")

    # Save GIF
    gif_path = os.path.join(logdir, f"episode_seed{config.seed}.gif")
    save_frames_as_gif(frames, gif_path, fps=10)
    print(f"  GIF saved      : {gif_path}")

    # Save frob norm log
    _save_frob_norm_log(frob_norms_history, maddpg.nagents, logdir, config.seed)

    # Save frob norm heatmap (rows=agent pairs, cols=well-distributed timesteps)
    _save_frob_norm_heatmap(frob_norms_history, maddpg.nagents, logdir, config.seed)

    env.close()
    print("Done.")


# ---------------------------------------------------------------------------
# Experiment 2: multi-seed SVD coupling — frob norm vs gradient shift
# ---------------------------------------------------------------------------

def svd_coupling_experiment(config):
    """
    Run a multi-seed SVD-based gradient coupling experiment and report the
    pointwise relationship between cross-Hessian Frobenius norm and the
    induced gradient shift in agent i caused by an orthogonally-projected
    perturbation of agent j.

    For every (i, j) pair at every timestep of every seed episode:
      - Computes H = ∇_{a_j} ∇_{a_i} Q_i  and its Frobenius norm ||H||_F
      - Perturbs a_j along d_orthogonal = H^T g projected out of ∇_{a_j} Q_i
        so that the perturbation has zero first-order effect on Q_i
      - Measures gradient shift ||Δg||_2 = ||∇_{a_i} Q_i(a_j') - g||_2

    Saves:
      csv_data/raw_coupling_data.csv      — every (seed, timestep, pair) row
      csv_data/mean_coupling_by_pair.csv  — mean frob norm and mean gradient
                                            shift per (agent_j, agent_i) pair
      plots/svd_coupling_frob_vs_deltag_all_pairs.png  — combined scatter
      plots/svd_coupling_pair_<j>_to_<i>.png           — per-pair scatter

    Args:
        config: Namespace with fields:
                  env_id, model_path, epsilon, total_experiments
    """
    maddpg = MADDPG.init_from_save(config.model_path)

    if maddpg.discrete_action:
        print("ERROR: svd_coupling_experiment requires continuous action spaces.")
        print("SVD perturbation on one-hot vectors is not supported.")
        return

    env_type = 'continuous'
    device_str = 'gpu' if DEVICE == 'gpu' else 'cpu'
    maddpg.prep_training(device=device_str)

    env = create_environment(config, maddpg)
    logdir = _make_multiseed_logdir(
        'grad_shift_svd_coupling', config.env_id, env_type,
        maddpg.nagents, config.total_experiments
    )

    print(f"[grad_shift_svd_coupling_experiment]")
    print(f"  env          : {config.env_id}  ({env_type})")
    print(f"  agents       : {maddpg.nagents}")
    print(f"  seeds        : {config.total_experiments}")
    print(f"  epsilon      : {config.epsilon}")
    print(f"  output       : {logdir}")

    all_records = []
    for seed in tqdm(range(config.total_experiments), desc="Seeds"):
        records = _run_svd_coupling_episode(maddpg, env, seed, config.epsilon)
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
        description="Frobenius Norm Experiments for PettingZoo environments"
    )
    sub = parser.add_subparsers(dest='experiment', required=True)

    # --- frob_norm_episode ---
    ep = sub.add_parser('episode_gif_frob_norm',
                        help='Run a seeded episode, save GIF and frob norm log')
    ep.add_argument('env_id', help="PettingZoo environment name (e.g. simple_spread)")
    ep.add_argument('model_path', help="Path to saved MADDPG model")
    ep.add_argument('--seed', type=int, default=0, help="Random seed (default: 0)")

    # --- svd_coupling ---
    sc = sub.add_parser('grad_shift_svd_coupling',
                        help='Multi-seed SVD coupling: frob norm vs gradient shift')
    sc.add_argument('env_id', help="PettingZoo environment name (e.g. simple_spread)")
    sc.add_argument('model_path', help="Path to saved MADDPG model")
    sc.add_argument('--epsilon', type=float, default=0.01,
                    help="Perturbation magnitude for SVD direction (default: 0.01)")
    sc.add_argument('--total_experiments', type=int, default=100,
                    help="Number of seed episodes to run (default: 100)")

    return parser


def main():
    config = _build_parser().parse_args()

    if config.experiment == 'episode_gif_frob_norm':
        frob_norm_episode_experiment(config)
    elif config.experiment == 'grad_shift_svd_coupling':
        svd_coupling_experiment(config)


if __name__ == '__main__':
    main()
