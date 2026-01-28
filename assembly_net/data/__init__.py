"""
Data module for Assembly-Net.

Contains core data structures for representing assembly graphs and trajectories,
as well as the stochastic coordination network simulator.
"""

from assembly_net.data.core import (
    AssemblyGraph,
    AssemblyTrajectory,
    NetworkState,
    NodeType,
    EdgeType,
    AssemblyEvent,
)
from assembly_net.data.synthetic_assembly_simulator.simulator import (
    CoordinationNetworkSimulator,
    SimulationParameters,
)
from assembly_net.data.dataset import AssemblyDataset, collate_trajectories

__all__ = [
    "AssemblyGraph",
    "AssemblyTrajectory",
    "NetworkState",
    "NodeType",
    "EdgeType",
    "AssemblyEvent",
    "CoordinationNetworkSimulator",
    "SimulationParameters",
    "AssemblyDataset",
    "collate_trajectories",
]
