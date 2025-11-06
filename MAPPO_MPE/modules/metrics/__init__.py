"""
Metrics module initialization.

NOTE: All metrics functions have been updated to work with MAPPO.
- Functions that compute gradients with respect to actions or Q-values now use MAPPO's centralized Q network
- Most functions now require a 'state' parameter (global state) in addition to observations
- The 'action_spaces' parameter is kept for API compatibility but is no longer used
- Functions will raise RuntimeError if centralized Q network is not enabled (use_central_q=True)
"""
from .basic_metrics import (
    compute_taylor_delta_policy,
    compute_frob_norms,
    compute_pairwise_frob_norms,
    compute_2nd_ord_dir_derivatives,
    collect_agent_q_values,
    collect_agent_q_value
)
from .influence_metrics import (
    compute_pairwise_action_influences,
    compute_second_order_action_influences,
    compute_pairwise_observation_influences,
    compute_second_order_observation_influences,
    compute_pairwise_action_directional_second_derivatives
)
from .attack_metrics import AttackMetricsComputer

__all__ = [
    'compute_taylor_delta_policy',
    'compute_frob_norms',
    'compute_pairwise_frob_norms',
    'compute_2nd_ord_dir_derivatives',
    'compute_pairwise_action_influences',
    'compute_second_order_action_influences',
    'compute_pairwise_observation_influences',
    'compute_second_order_observation_influences',
    'collect_agent_q_values',
    'compute_pairwise_action_directional_second_derivatives',
    'AttackMetricsComputer',
    'collect_agent_q_value'
]