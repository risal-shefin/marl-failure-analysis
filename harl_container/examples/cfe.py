# cfe.py
# Windowed-attack counterfactual explainability for HARL (HAPPO + PettingZoo MPE)
import argparse
import os
import json
import itertools
import random
from math import factorial
from collections import deque, defaultdict
from datetime import datetime
import csv
import re

import numpy as np
import torch
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harl.utils.configs_tools import get_defaults_yaml_args, update_args
from harl.utils.trans_tools import _t2n
from harl.runners import RUNNER_REGISTRY


# ------------------------------- Util helpers -------------------------------

def in_window(t: int, min_w: int, max_w: int) -> bool:
    return (t >= min_w) and (t <= max_w)

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def slice_avail(avail, aid):
    if avail is None:
        return None
    first = avail[0]
    if first is None:
        return None
    return avail[:, aid]

def discrete_n_actions(runner, agent_id):
    if hasattr(runner, "eval_envs") and hasattr(runner.eval_envs.action_space[agent_id], "n"):
        return int(runner.eval_envs.action_space[agent_id].n)
    return int(runner.envs.action_space[agent_id].n)

def available_action_indices(avail_slice, n_act):
    if (avail_slice is None) or (avail_slice[0] is None):
        return list(range(n_act))
    mask = (avail_slice[0] > 0.5)
    idxs = np.nonzero(mask)[0].tolist()
    if len(idxs) == 0:
        return list(range(n_act))
    return idxs

def factorial_ratio(m, n):
    return (factorial(m) * factorial(n - m - 1)) / factorial(n)

# safe caster for extra CLI k/v overrides (no eval)
_INT_RE   = re.compile(r'^[+-]?\d+$')
_FLOAT_RE = re.compile(r'^[+-]?((\d+\.\d*)|(\.\d+)|(\d+))(e[+-]?\d+)?$', re.IGNORECASE)
def safe_cast(arg: str):
    lower = arg.lower()
    if lower == "true":  return True
    if lower == "false": return False
    if lower in ("none","null"): return None
    if _INT_RE.fullmatch(arg):   return int(arg)
    if _FLOAT_RE.fullmatch(arg): return float(arg)
    return arg


# --------------------------- Worst action (sampling) ---------------------------

def pick_worst_action_by_sampling(
    runner,
    agent_id,
    obs,            # (1, obs_dim)
    rnn_state,      # (1, rec_n, hid)
    mask,           # (1, 1)
    avail_slice,    # (1, action_dim) or None
    branch_seed,    # deterministic per (t, agent)
    N=256 #Here, N is the number of times to sample from the agent’s own policy at the same observation to estimate how likely each action is according to that policy.

):
    n_act = discrete_n_actions(runner, agent_id)
    allowed = available_action_indices(avail_slice, n_act)

    torch_state = torch.random.get_rng_state()
    np_state = np.random.get_state()
    torch.manual_seed(branch_seed)
    np.random.seed(branch_seed)

    counts = np.zeros(n_act, dtype=np.int64)
    for _ in range(N):
        a, logp, _ = runner.actor[agent_id].get_actions(
            obs,
            rnn_state,
            mask,
            avail_slice,
            deterministic=False,
        )
        a_idx = int(_t2n(a)[0, 0])
        counts[a_idx] += 1

    torch.random.set_rng_state(torch_state)
    np.random.set_state(np_state)

    counts_allowed = counts[allowed]
    argmin_local = int(np.argmin(counts_allowed))
    return int(allowed[argmin_local])


# --------------------------- Taylor / Frob / 2nd-ord ---------------------------

def compute_taylor_policy(runner, eval_obs, eval_available_actions, eval_rnn_states):
    eval_obs = torch.tensor(eval_obs, dtype=torch.float32, requires_grad=True)
    delta_errors = []
    eval_masks = np.ones(
        (runner.algo_args["eval"]["n_eval_rollout_threads"], runner.num_agents, 1),
        dtype=np.float32,
    )
    for agent_id in range(runner.num_agents):
        cur_obs = eval_obs[:, agent_id]
        avail_slice = (eval_available_actions[:, agent_id]
                       if (eval_available_actions is not None) and (eval_available_actions[0] is not None)
                       else None)
        _, eval_actions_log_prob, _ = runner.actor[agent_id].get_actions(
            cur_obs,
            eval_rnn_states[:, agent_id],
            eval_masks[:, agent_id],
            avail_slice,
            deterministic=True,
        )
        grad_i = torch.autograd.grad(
            outputs=eval_actions_log_prob, inputs=cur_obs, create_graph=True, retain_graph=True
        )[0]
        eta_i = 0.001 * grad_i.sign() / torch.max(grad_i.norm(p=2), torch.tensor(1e-6))
        j_tilde = eval_actions_log_prob + torch.dot(grad_i.flatten(), eta_i.flatten())

        p_obs = cur_obs + eta_i
        _, perturb_log_prob, _ = runner.actor[agent_id].get_actions(
            p_obs,
            eval_rnn_states[:, agent_id],
            eval_masks[:, agent_id],
            avail_slice,
            deterministic=True,
        )
        j_perturbed = perturb_log_prob
        delta_error = abs(j_perturbed - j_tilde).item()
        delta_errors.append(delta_error)

    return delta_errors


def compute_frob_norms(runner, eval_obs, vulnerable_agent_id, eval_rnn_states_critic, eval_masks):
    eval_obs = torch.tensor(eval_obs, dtype=torch.float32)
    agent_obs_tensors = []
    n_agents = runner.num_agents
    for i in range(n_agents):
        agent_obs = eval_obs[0][i].clone().detach()
        agent_obs_tensor = agent_obs.clone().detach().requires_grad_(True)
        agent_obs_tensors.append(agent_obs_tensor)

    concatenated_obs = torch.cat(agent_obs_tensors, dim=0)
    share_obs = concatenated_obs.unsqueeze(0).unsqueeze(0).expand(1, n_agents, -1)

    values, _ = runner.critic.get_values(
        share_obs,
        eval_rnn_states_critic,
        eval_masks,
    )
    values = values.squeeze()

    results = []
    for i in range(runner.num_agents):
        grad_i = torch.autograd.grad(values[i], agent_obs_tensors[i], create_graph=True, retain_graph=True)[0]
        hessian_rows = []
        for k in range(grad_i.shape[0]):
            second_grad = torch.autograd.grad(
                grad_i[k],
                agent_obs_tensors[vulnerable_agent_id],
                retain_graph=True,
                allow_unused=True
            )[0]
            hessian_rows.append(second_grad.flatten())
        H = torch.stack(hessian_rows)
        results.append(H.norm(p='fro').item())
    return results


def compute_2nd_ord_dir_derivatives(runner, eval_obs, vulnerable_agent_id, eval_rnn_states_critic, eval_masks):
    eval_obs = torch.tensor(eval_obs, dtype=torch.float32)
    agent_obs_tensors = []
    n_agents = runner.num_agents
    for i in range(n_agents):
        agent_obs = eval_obs[0][i].clone().detach()
        agent_obs_tensor = agent_obs.clone().detach().requires_grad_(True)
        agent_obs_tensors.append(agent_obs_tensor)

    concatenated_obs = torch.cat(agent_obs_tensors, dim=0)
    share_obs = concatenated_obs.unsqueeze(0).unsqueeze(0).expand(1, n_agents, -1)

    values, _ = runner.critic.get_values(
        share_obs,
        eval_rnn_states_critic,
        eval_masks,
    )
    values = values.squeeze()

    results = []
    for i in range(runner.num_agents):
        grad_i = torch.autograd.grad(values[i], agent_obs_tensors[i], create_graph=True, retain_graph=True)[0]
        v = grad_i / torch.max(grad_i.norm(p=2), torch.tensor(1e-6))
        hvp = torch.autograd.grad(
            outputs=grad_i,
            inputs=agent_obs_tensors[vulnerable_agent_id],
            grad_outputs=v,
            retain_graph=True,
            allow_unused=True
        )[0]
        grad_j = torch.autograd.grad(-values[i], agent_obs_tensors[vulnerable_agent_id], create_graph=True, retain_graph=True)[0]
        u = grad_j / torch.max(grad_j.norm(p=2), torch.tensor(1e-6))
        u = -u
        curvature_val = torch.dot(u.flatten(), hvp.flatten())
        results.append(curvature_val.item())
    return results


# ----------------------------- Plotting helpers -----------------------------

def plot_results(results, results_attacked, atk_agent_id, logdir):
    ensure_dir(logdir)
    n = len(results[0])
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    fig.suptitle(f'Taylor Error (Worst Action Attack | Attacked Agent ID: {atk_agent_id})', fontsize=16, y=0.95)
    axes = [axes] if n == 1 else axes
    for i in range(n):
        ax = axes[i]
        normal_series = [results[t][i] for t in range(len(results))]
        attacked_series = [results_attacked[t][i] for t in range(len(results_attacked))]
        steps = range(len(normal_series))
        ax.plot(steps, normal_series, 'b-', label='Normal', linewidth=2)
        ax.plot(steps, attacked_series, 'r-', label='Attacked', linewidth=2)
        ax.set_xlabel('Step'); ax.set_ylabel('Taylor Delta Error'); ax.set_title(f'Agent {i}')
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(logdir, f'plot_analysis_attacked_{atk_agent_id}.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_frobs(frobs_normal, frobs_atk, attacked_steps, atk_agent_id, logdir):
    n = len(frobs_normal[0])
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    fig.suptitle(f'Frobenius Norms (Worst Action Attack | Attacked Agent ID: {atk_agent_id})', fontsize=16, y=0.95)
    axes = [axes] if n == 1 else axes
    for i in range(n):
        ax = axes[i]
        normal_series = [frobs_normal[t][i] for t in range(len(frobs_normal))]
        attacked_series = [frobs_atk[t][i] for t in range(len(frobs_atk))]
        steps = range(len(normal_series))
        ax.plot(steps, attacked_series, 'r-', label='Under Attack', linewidth=2)
        ax.plot(steps, normal_series, 'g-', label='Normal', linewidth=2)
        if attacked_steps:
            for attack_step in attacked_steps:
                ax.axvline(x=attack_step, color='orange', linestyle=':', alpha=0.5, linewidth=1.5)
            ax.axvline(x=attacked_steps[0], color='orange', linestyle=':', alpha=0.5, linewidth=1.5, label='Attacked Steps')
        ax.set_xlabel('Step'); ax.set_ylabel('Frobenius Norm'); ax.set_title(f'Agent {i}')
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(logdir, f'plot_frobs_attacked_{atk_agent_id}.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_sec_dir_derivatives(s_dir_derv_normal, s_dir_derv_atk, attacked_steps, atk_agent_id, logdir):
    n = len(s_dir_derv_normal[0])
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    fig.suptitle(f'2nd Ord. Dir. Derivatives (Worst Action Attack | Attacked Agent ID: {atk_agent_id})', fontsize=16, y=0.95)
    axes = [axes] if n == 1 else axes
    for i in range(n):
        ax = axes[i]
        normal_series = [s_dir_derv_normal[t][i] for t in range(len(s_dir_derv_normal))]
        attacked_series = [s_dir_derv_atk[t][i] for t in range(len(s_dir_derv_atk))]
        steps = range(len(normal_series))
        ax.plot(steps, attacked_series, 'r-', label='Under Attack', linewidth=2)
        ax.plot(steps, normal_series, 'g-', label='Normal', linewidth=2)
        y_min = min(min(normal_series), min(attacked_series))
        if y_min < 0:
            ax.axhspan(y_min * 1.1, 0, alpha=0.2, color='red')
        if attacked_steps:
            for attack_step in attacked_steps:
                ax.axvline(x=attack_step, color='orange', linestyle=':', alpha=0.5, linewidth=1.5)
            ax.axvline(x=attacked_steps[0], color='orange', linestyle=':', alpha=0.5, linewidth=1.5, label='Attacked Steps')
        ax.set_xlabel('Step'); ax.set_ylabel('2nd Ord. Dir. Derivative'); ax.set_title(f'Agent {i}')
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(logdir, f'plot_sec_dir_derivatives_attacked_{atk_agent_id}.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_totase(series, save_path):
    plt.figure(figsize=(10, 4))
    plt.plot(range(len(series)), series, linewidth=2)
    plt.xlabel("Timestep"); plt.ylabel("tot-ASE"); plt.title("tot-ASE per timestep")
    plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(save_path, dpi=160); plt.close()


def plot_shapley(shapley_matrix, save_path):
    arr = np.array(shapley_matrix)
    T, n_agents = arr.shape
    plt.figure(figsize=(12, 5))
    for aid in range(n_agents):
        plt.plot(range(T), arr[:, aid], linewidth=2, label=f"agent {aid}")
    plt.xlabel("Timestep"); plt.ylabel("Shapley value"); plt.title("Per-timestep Shapley (non-attacked agents)")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(save_path, dpi=160); plt.close()


# ----------------------------- CSV helpers -----------------------------

def save_matrix_to_files(matrix, attacked_steps, attacked_agent_id, total_agents, logdir, filename):
    filepath = os.path.join(logdir, filename)
    header = ["timestep", "is_attacked", "attacked_agent"] + [f"agent_{i}" for i in range(total_agents)]
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for timestep, timestep_data in enumerate(matrix):
            is_attacked = 1 if timestep in attacked_steps else 0
            row = [timestep, is_attacked, attacked_agent_id] + [timestep_data[i] for i in range(total_agents)]
            writer.writerow(row)


# ----------------------------- Baseline episode -----------------------------

def run_baseline_episode_and_record(runner, seed=23, vulnerable_id=0):
    """Deterministic baseline episode — returns joint actions and (Taylor/Frob/2nd) metrics along the way."""
    eval_obs, eval_share_obs, eval_avail = runner.eval_envs.reset(seed=seed)

    n_agents = runner.num_agents
    rnn = np.zeros((1, n_agents, runner.recurrent_n, runner.rnn_hidden_size), dtype=np.float32)
    rnn_c = np.zeros_like(rnn)
    masks = np.ones((1, n_agents, 1), dtype=np.float32)

    actions_list = []
    total_return = 0.0

    taylor_error_list = []
    frob_norms_list = []
    sec_dir_derivatives_list = []

    result_deques = [deque(maxlen=5) for _ in range(n_agents)]
    frob_deques = [deque(maxlen=5) for _ in range(n_agents)]
    sdd_deques = [deque(maxlen=5) for _ in range(n_agents)]

    while True:
        # --- NEW: align critic RNN with current share_obs BEFORE metrics ---
        _, rnn_c = runner.critic.get_values(
            eval_share_obs,  # current joint observations
            rnn_c,
            masks,
        )

        # (Optional) align actor RNNs too — uncomment if you want fully state-aligned Taylor metrics
        # for aid in range(n_agents):
        #     _, rnn_next = runner.actor[aid].act(
        #         eval_obs[:, aid], rnn[:, aid], masks[:, aid],
        #         slice_avail(eval_avail, aid), deterministic=True,
        #     )
        #     rnn[:, aid] = _t2n(rnn_next)

        # metrics BEFORE stepping (at current obs)
        delta_errors = compute_taylor_policy(runner, eval_obs, eval_avail, rnn.copy())
        frobs = compute_frob_norms(runner, eval_obs, vulnerable_id, rnn_c, masks)
        sdds = compute_2nd_ord_dir_derivatives(runner, eval_obs, vulnerable_id, rnn_c, masks)

        for i in range(n_agents):
            result_deques[i].append(delta_errors[i])
            frob_deques[i].append(frobs[i])
            sdd_deques[i].append(sdds[i])

        taylor_error_list.append([np.mean(list(result_deques[j])) for j in range(n_agents)])
        frob_norms_list.append([np.mean(frob_deques[i]) for i in range(n_agents)])
        sec_dir_derivatives_list.append([np.mean(sdd_deques[i]) for i in range(n_agents)])

        # act deterministically for baseline actions
        actions_col = []
        for aid in range(n_agents):
            a, rnn_next = runner.actor[aid].act(
                eval_obs[:, aid], rnn[:, aid], masks[:, aid],
                slice_avail(eval_avail, aid), deterministic=True,
            )
            rnn[:, aid] = _t2n(rnn_next)
            actions_col.append(_t2n(a))
        actions = np.array(actions_col).transpose(1, 0, 2)

        # step
        eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_avail = runner.eval_envs.step(actions)
        actions_list.append(actions.copy())
        total_return += float(eval_rewards.sum())

        if np.all(eval_dones):
            break

        done_env = np.all(eval_dones, axis=1)
        rnn[done_env == True] = 0
        masks[:] = 1.0
        masks[done_env == True] = 0.0

    return actions_list, total_return, len(actions_list), taylor_error_list, frob_norms_list, sec_dir_derivatives_list


# ----------------------------- Attacked episode -----------------------------

def run_attacked_episode_with_metrics(runner, attacked_id, seed=23, min_window=0, max_window=10):
    """Episode where the attacked agent executes the worst action only when t is in [min_window, max_window]."""
    eval_obs, eval_share_obs, eval_avail = runner.eval_envs.reset(seed=seed)

    n_agents = runner.num_agents
    rnn = np.zeros((1, n_agents, runner.recurrent_n, runner.rnn_hidden_size), dtype=np.float32)
    rnn_c = np.zeros_like(rnn)
    masks = np.ones((1, n_agents, 1), dtype=np.float32)

    taylor_error_list, frob_norms_list, sec_dir_derivatives_list = [], [], []
    result_deques = [deque(maxlen=5) for _ in range(n_agents)]
    frob_deques   = [deque(maxlen=5) for _ in range(n_agents)]
    sdd_deques    = [deque(maxlen=5) for _ in range(n_agents)]

    attacked_steps = []
    #Rolling 5-step averages to smooth for plotting.
    t = 0
    while True:
        # --- NEW: align critic RNN with current share_obs BEFORE metrics ---
        _, rnn_c = runner.critic.get_values(
            eval_share_obs,
            rnn_c,
            masks,
        )

        # (Optional) align actor RNNs too — uncomment if desired
        # for aid in range(n_agents):
        #     _, rnn_next = runner.actor[aid].act(
        #         eval_obs[:, aid], rnn[:, aid], masks[:, aid],
        #         slice_avail(eval_avail, aid), deterministic=True,
        #     )
        #     rnn[:, aid] = _t2n(rnn_next)

        # metrics at current obs
        delta_errors = compute_taylor_policy(runner, eval_obs, eval_avail, rnn.copy())
        frobs = compute_frob_norms(runner, eval_obs, attacked_id, rnn_c, masks)
        sdds  = compute_2nd_ord_dir_derivatives(runner, eval_obs, attacked_id, rnn_c, masks)
        for i in range(n_agents):
            result_deques[i].append(delta_errors[i]);  frob_deques[i].append(frobs[i]);  sdd_deques[i].append(sdds[i])
        taylor_error_list.append([np.mean(list(result_deques[j])) for j in range(n_agents)])
        frob_norms_list.append([np.mean(frob_deques[i]) for i in range(n_agents)])
        sec_dir_derivatives_list.append([np.mean(sdd_deques[i]) for i in range(n_agents)])

        # actions: worst only inside window
        actions_col = []
        for aid in range(n_agents):
            if (aid == attacked_id) and in_window(t, min_window, max_window):
                derived_seed = seed #* 100003 + t * 1009 + attacked_id * 101
                worst_idx = pick_worst_action_by_sampling(
                    runner, attacked_id,
                    eval_obs[:, attacked_id], rnn[:, attacked_id], masks[:, attacked_id],
                    slice_avail(eval_avail, attacked_id),
                    branch_seed=derived_seed, N=256,
                )
                a_np = np.array([[worst_idx]], dtype=np.int64)
                actions_col.append(a_np)
                attacked_steps.append(t)
            else:
                a, rnn_next = runner.actor[aid].act(
                    eval_obs[:, aid], rnn[:, aid], masks[:, aid],
                    slice_avail(eval_avail, aid), deterministic=True,
                )
                rnn[:, aid] = _t2n(rnn_next)
                actions_col.append(_t2n(a))
        actions = np.array(actions_col).transpose(1, 0, 2)

        eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_avail = runner.eval_envs.step(actions)
        if np.all(eval_dones):
            break

        done_env = np.all(eval_dones, axis=1)
        rnn[done_env == True] = 0
        masks[:] = 1.0
        masks[done_env == True] = 0.0
        t += 1

    return taylor_error_list, frob_norms_list, sec_dir_derivatives_list, attacked_steps


# ----------------------- Coalition branch: v(S) at time t -----------------------

def branch_rollout_value(
    runner,
    seed,
    baseline_actions,
    branch_t,
    attacked_id,
    coalition_allows_react,
    min_window,
    max_window,
):
    obs, share_obs, avail = runner.eval_envs.reset(seed=seed)

    n_agents = runner.num_agents
    rnn = np.zeros((1, n_agents, runner.recurrent_n, runner.rnn_hidden_size), dtype=np.float32)
    masks = np.ones((1, n_agents, 1), dtype=np.float32)

    # replay baseline to t-1, and forward RNNs to match obs at k+1
    for k in range(branch_t):
        obs, share_obs, rewards, dones, infos, avail = runner.eval_envs.step(baseline_actions[k])
        for aid in range(n_agents):
            _, rnn_next = runner.actor[aid].act(
                obs[:, aid], rnn[:, aid], masks[:, aid], slice_avail(avail, aid), deterministic=True
            )
            rnn[:, aid] = _t2n(rnn_next)
        done_env = np.all(dones, axis=1)
        masks[:] = 1.0; masks[done_env == True] = 0.0

    # at t: if inside window, force worst for attacked; else baseline
    actions_t = baseline_actions[branch_t].copy()
    if in_window(branch_t, min_window, max_window):
        derived_seed = seed # * 100003 + branch_t * 1009 + attacked_id * 101
        worst_a = pick_worst_action_by_sampling(
            runner, attacked_id,
            obs[:, attacked_id], rnn[:, attacked_id], masks[:, attacked_id],
            slice_avail(avail, attacked_id),
            branch_seed=derived_seed, N=256,
        )
        actions_t[0, attacked_id, 0] = worst_a
    obs, share_obs, rewards, dones, infos, avail = runner.eval_envs.step(actions_t)
    ep_return = float(rewards.sum())

    # t+1..end
    k = branch_t + 1
    while True:
        if np.all(dones):
            break
        actions_col = []
        for aid in range(n_agents):
            if aid == attacked_id:
                if in_window(k, min_window, max_window):
                    derived_seed = seed #* 100003 + k * 1009 + attacked_id * 101
                    worst_idx = pick_worst_action_by_sampling(
                        runner, attacked_id,
                        obs[:, attacked_id], rnn[:, attacked_id], masks[:, attacked_id],
                        slice_avail(avail, attacked_id),
                        branch_seed=derived_seed, N=256,
                    )
                    a_np = np.array([[worst_idx]], dtype=np.int64)
                    actions_col.append(a_np)
                else:
                    if k < len(baseline_actions):
                        a_idx = int(baseline_actions[k][0, aid, 0])
                        a_np = np.array([[a_idx]], dtype=np.int64)
                        actions_col.append(a_np)
                    else:
                        a, rnn_next = runner.actor[aid].act(
                            obs[:, aid], rnn[:, aid], masks[:, aid],
                            slice_avail(avail, aid), deterministic=True,
                        )
                        rnn[:, aid] = _t2n(rnn_next)
                        actions_col.append(_t2n(a))
            else:
                if aid in coalition_allows_react:
                    a, rnn_next = runner.actor[aid].act(
                        obs[:, aid], rnn[:, aid], masks[:, aid], slice_avail(avail, aid), deterministic=True,
                    )
                    rnn[:, aid] = _t2n(rnn_next)
                    actions_col.append(_t2n(a))
                else:
                    if k < len(baseline_actions):
                        a_idx = int(baseline_actions[k][0, aid, 0])
                        a_np = np.array([[a_idx]], dtype=np.int64)
                        actions_col.append(a_np)
                    else:
                        a, rnn_next = runner.actor[aid].act(
                            obs[:, aid], rnn[:, aid], masks[:, aid],
                            slice_avail(avail, aid), deterministic=True,
                        )
                        rnn[:, aid] = _t2n(rnn_next)
                        actions_col.append(_t2n(a))

        actions = np.array(actions_col).transpose(1, 0, 2)
        obs, share_obs, rewards, dones, infos, avail = runner.eval_envs.step(actions)
        ep_return += float(rewards.sum())
        k += 1

        done_env = np.all(dones, axis=1)
        masks[:] = 1.0
        masks[done_env == True] = 0.0

    return ep_return


def tot_ase_and_shapley_at_t(runner, seed, baseline_actions, branch_t, attacked_id, min_window, max_window):
    all_non_attacked = [a for a in range(runner.num_agents) if a != attacked_id]
    cache = {}
    def v_of(S_frozen):
        if S_frozen in cache:
            return cache[S_frozen]
        val = branch_rollout_value(
            runner=runner, seed=seed, baseline_actions=baseline_actions,
            branch_t=branch_t, attacked_id=attacked_id,
            coalition_allows_react=set(S_frozen),
            min_window=min_window, max_window=max_window,
        )
        cache[S_frozen] = val
        return val
    v_empty = v_of(frozenset())
    v_all   = v_of(frozenset(all_non_attacked))
    tot_ase = v_all - v_empty

    n = len(all_non_attacked)
    shap = defaultdict(float)
    def w(m,n): return (factorial(m)*factorial(n-m-1))/factorial(n)
    for j in all_non_attacked:
        others = [x for x in all_non_attacked if x != j]
        for m in range(len(others)+1):
            for S in itertools.combinations(others, m):
                vS  = v_of(frozenset(S))
                vSj = v_of(frozenset(set(S)|{j}))
                shap[j] += w(m, n) * (vSj - vS)
    shap_vec = [0.0]*runner.num_agents
    for aid,val in shap.items(): shap_vec[aid] = float(val)
    return float(tot_ase), shap_vec, float(v_all), float(v_empty)


# --------------------------------- Restore ---------------------------------

def restore(runner, reward, episode, filepath):
    for agent_id in range(runner.num_agents):
        policy_actor_state_dict = torch.load(
            os.path.join(filepath, f"actor_agent{agent_id}_reward_{reward}_episode_{episode}.pt"),
            weights_only=False
        )
        runner.actor[agent_id].actor.load_state_dict(policy_actor_state_dict)
    if not runner.algo_args["render"]["use_render"]:
        policy_critic_state_dict = torch.load(
            os.path.join(filepath, f"critic_agent_reward_{reward}_episode_{episode}.pt"),
            weights_only=False
        )
        runner.critic.critic.load_state_dict(policy_critic_state_dict)
        if runner.value_normalizer is not None:
            value_normalizer_state_dict = torch.load(
                os.path.join(filepath, f"value_normalizer_reward_{reward}_episode_{episode}.pt"),
                weights_only=False
            )
            runner.value_normalizer.load_state_dict(value_normalizer_state_dict)


# ----------------------------------- Main -----------------------------------

def main():
    # seeds
    set_all_seeds(23)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--algo", type=str, default="happo",
                        choices=["happo","hatrpo","haa2c","haddpg","hatd3","hasac","had3qn","maddpg","matd3","mappo"])
    parser.add_argument("--env", type=str, default="pettingzoo_mpe",
                        choices=["smac","mamujoco","pettingzoo_mpe","gym","football","dexhands","smacv2","lag"])
    parser.add_argument("--exp_name", type=str, default="cfe_eval")
    parser.add_argument("--load_config", type=str, default="")
    parser.add_argument("--attack_id", type=int, default=0)
    parser.add_argument("--restore_dir", type=str, default="")
    parser.add_argument("--restore_reward", type=str, default="")
    parser.add_argument("--restore_episode", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="results",
                        help="Root folder where a timestamped subdir will be created.")
    parser.add_argument("--min_window", type=int, default=0, help="inclusive attack start step")
    parser.add_argument("--max_window", type=int, default=10, help="inclusive attack end step")

    args, unparsed_args = parser.parse_known_args()

    # build output dir once, based on --save_dir
    base_dir = os.path.abspath(args.save_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_path = os.path.join(base_dir, timestamp)
    ensure_dir(log_path)

    # safe parse extra overrides for yaml configs
    keys   = [k[2:] for k in unparsed_args[0::2]]
    values = [safe_cast(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}

    # configs
    if args.load_config != "":
        with open(args.load_config, encoding="utf-8") as file:
            all_config = json.load(file)
        main_args = all_config["main_args"]
        algo_args = all_config["algo_args"]
        env_args  = all_config["env_args"]
        main_args["exp_name"] = args.exp_name
        main_args["algo"]     = args.algo or main_args["algo"]
        main_args["env"]      = args.env  or main_args["env"]
    else:
        algo_args, env_args = get_defaults_yaml_args(args.algo, args.env)
        main_args = {"algo": args.algo, "env": args.env, "exp_name": args.exp_name}

    algo_args["eval"]["n_eval_rollout_threads"] = 1
    algo_args["eval"]["eval_episodes"] = 1
    update_args(unparsed_dict, algo_args, env_args)

    if main_args["env"] == "dexhands":
        import isaacgym  # noqa: F401
        algo_args["eval"]["use_eval"] = False
        algo_args["train"]["episode_length"] = env_args["hands_episode_length"]

    # build runner
    runner = RUNNER_REGISTRY[main_args["algo"]](main_args, algo_args, env_args)
    if (args.restore_dir != "") and (args.restore_reward != "") and (args.restore_episode != ""):
        restore(runner, args.restore_reward, args.restore_episode, args.restore_dir)
    runner.prep_training()

    attacked_id = args.attack_id
    seed = 23

    # 1) Baseline episode with metrics
    (baseline_actions, base_return, T,
     taylor_normal, frob_normal, sdd_normal) = run_baseline_episode_and_record(
        runner, seed=seed, vulnerable_id=attacked_id
    )
    print(f"[Baseline] steps={T}, team_return={base_return:.3f}")

    # 2) Attacked episode with metrics (worst action only within window)
    (taylor_atk, frob_atk, sdd_atk, attacked_steps) = run_attacked_episode_with_metrics(
        runner, attacked_id, seed=seed, min_window=args.min_window, max_window=args.max_window
    )

    # 3) tot-ASE + Shapley per timestep using baseline actions
    timesteps = list(range(T))
    totASE_series = []
    shapley_series = []
    for t in timesteps:
        tot_ase_t, shap_vec_t, v_all, v_empty = tot_ase_and_shapley_at_t(
            runner=runner, seed=seed, baseline_actions=baseline_actions,
            branch_t=t, attacked_id=attacked_id,
            min_window=args.min_window, max_window=args.max_window
        )
        totASE_series.append(tot_ase_t)
        shapley_series.append(shap_vec_t)
        # (Optional) efficiency check
        # if abs(tot_ase_t - sum(shap_vec_t)) > 1e-6:
        #     print("WARN: efficiency mismatch at t=", t)

    # 4) save matrices to CSVs (normal + attacked)
    save_matrix_to_files(taylor_normal, [], attacked_id, runner.num_agents, log_path,
                         f'happo_taylor_error_normal.csv')
    save_matrix_to_files(taylor_atk, attacked_steps, attacked_id, runner.num_agents, log_path,
                         f'happo_taylor_error_atk_{attacked_id}.csv')

    save_matrix_to_files(frob_normal, [], attacked_id, runner.num_agents, log_path,
                         f'happo_frobenius_norms_normal.csv')
    save_matrix_to_files(frob_atk, attacked_steps, attacked_id, runner.num_agents, log_path,
                         f'happo_frobenius_norms_atk_{attacked_id}.csv')

    save_matrix_to_files(sdd_normal, [], attacked_id, runner.num_agents, log_path,
                         f'happo_sec_dir_derivatives_normal.csv')
    save_matrix_to_files(sdd_atk, attacked_steps, attacked_id, runner.num_agents, log_path,
                         f'happo_sec_dir_derivatives_atk_{attacked_id}.csv')

    # Save tot-ASE and Shapley
    with open(os.path.join(log_path, "totASE_per_t.csv"), "w") as f:
        f.write("timestep,totASE\n")
        for i, val in enumerate(totASE_series):
            f.write(f"{timesteps[i]},{val}\n")

    with open(os.path.join(log_path, "shapley_per_t.csv"), "w") as f:
        header = ["timestep"] + [f"agent_{i}" for i in range(runner.num_agents)]
        f.write(",".join(header) + "\n")
        for i, vec in enumerate(shapley_series):
            row = [str(timesteps[i])] + [f"{x:.6f}" for x in vec]
            f.write(",".join(row) + "\n")

    # 5) plots
    plot_results(taylor_normal, taylor_atk, attacked_id, log_path)
    plot_frobs(frob_normal, frob_atk, attacked_steps, attacked_id, log_path)
    plot_sec_dir_derivatives(sdd_normal, sdd_atk, attacked_steps, attacked_id, log_path)
    plot_totase(totASE_series, os.path.join(log_path, "totASE_plot.png"))
    plot_shapley(shapley_series, os.path.join(log_path, "shapley_plot.png"))

    print(f"All results saved in: {log_path}")
    runner.close()


if __name__ == "__main__":
    main()
