"""
Experiments module for Assembly-Net.

Contains experiment runners for ablation studies and research questions.
"""

from assembly_net.experiments.ablation_topology_vs_structure import (
    TopologyVsStructureAblation,
    run_topology_ablation,
)
from assembly_net.experiments.assembly_order_randomization import (
    AssemblyOrderRandomization,
    run_order_randomization_experiment,
)
from assembly_net.experiments.experiment_runner import (
    ExperimentRunner,
    ExperimentConfig,
    log_metrics,
)

__all__ = [
    "TopologyVsStructureAblation",
    "run_topology_ablation",
    "AssemblyOrderRandomization",
    "run_order_randomization_experiment",
    "ExperimentRunner",
    "ExperimentConfig",
    "log_metrics",
]
