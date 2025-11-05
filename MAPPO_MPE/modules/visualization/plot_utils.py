"""
Visualization utilities and plotting functions.
This module serves as the main entry point for all plotting functions.
"""

# Import utility functions
from .utils import get_agent_colors

# Import all plotting functions from sub-modules
from .basic_plots import (
    plot_results,
    plot_frobs,
    plot_sec_dir_derivatives
)

from .timeline_plots import (
    plot_fault_timeline,
    plot_contributor_barchart
)

from .influence_plots import (
    plot_action_influences,
    plot_pairwise_action_influences, 
    plot_second_order_action_influences,
    plot_pairwise_second_order_action_influences,
    plot_observation_influences,
    plot_pairwise_observation_influences,
    plot_second_order_observation_influences,
    plot_pairwise_second_order_observation_influences,
    plot_frob_norm_influences
)

from .stacked_plots import (
    plot_fault_timeline_action_influences_stacked,
    plot_normal_scenario_action_influences_stacked,
    plot_normal_scenario_frob_norms_stacked,
    plot_attacked_scenario_frob_norms_stacked
)

from .advanced_timeline_plots import (
    plot_fault_timeline_action_influences,
    plot_fault_timeline_second_order_action_influences,
    plot_fault_timeline_observation_influences,
    plot_fault_timeline_second_order_observation_influences
)


# Make all plotting functions available at package level
__all__ = [
    'get_agent_colors',
    # Basic plots
    'plot_results',
    'plot_frobs', 
    'plot_sec_dir_derivatives',
    # Timeline plots
    'plot_fault_timeline',
    'plot_contributor_barchart',
    # Influence plots
    'plot_action_influences',
    'plot_pairwise_action_influences',
    'plot_second_order_action_influences', 
    'plot_pairwise_second_order_action_influences',
    'plot_observation_influences',
    'plot_pairwise_observation_influences',
    'plot_second_order_observation_influences',
    'plot_pairwise_second_order_observation_influences',
    'plot_frob_norm_influences',
    # Stacked plots
    'plot_fault_timeline_action_influences_stacked',
    'plot_normal_scenario_action_influences_stacked',
    'plot_normal_scenario_frob_norms_stacked',
    'plot_attacked_scenario_frob_norms_stacked',
    # Advanced timeline plots
    'plot_fault_timeline_action_influences',
    'plot_fault_timeline_second_order_action_influences',
    'plot_fault_timeline_observation_influences',
    'plot_fault_timeline_second_order_observation_influences'
]