"""
Assembly-Net: Topology-Aware Learning of Emergent Material Properties
from Coordination-Network Assembly Graphs

This framework enables materials property prediction from assembly processes
rather than static structures, using topological data analysis and temporal
graph neural networks.
"""

__version__ = "0.1.0"
__author__ = "Assembly-Net Contributors"

from assembly_net.data.core import (
    AssemblyGraph,
    AssemblyTrajectory,
    NetworkState,
    NodeType,
    EdgeType,
)
from assembly_net.topology.persistent_homology import (
    PersistentHomologyComputer,
    TopologicalFeatures,
)
from assembly_net.topology.graph_rigidity import RigidityAnalyzer
from assembly_net.models.temporal_gnn import TemporalAssemblyGNN
from assembly_net.models.baseline_static_gnn import StaticGNN

__all__ = [
    # Core data structures
    "AssemblyGraph",
    "AssemblyTrajectory",
    "NetworkState",
    "NodeType",
    "EdgeType",
    # Topology
    "PersistentHomologyComputer",
    "TopologicalFeatures",
    "RigidityAnalyzer",
    # Models
    "TemporalAssemblyGNN",
    "StaticGNN",
]
