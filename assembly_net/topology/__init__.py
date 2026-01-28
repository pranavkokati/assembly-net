"""
Topology module for Assembly-Net.

Provides topological data analysis tools including persistent homology computation,
Betti number extraction, and graph rigidity analysis.
"""

from assembly_net.topology.persistent_homology import (
    PersistentHomologyComputer,
    TopologicalFeatures,
    compute_betti_numbers,
    compute_persistence_diagram,
    persistence_landscape,
)
from assembly_net.topology.graph_rigidity import (
    RigidityAnalyzer,
    RigidityMetrics,
    compute_rigidity_matrix,
    count_floppy_modes,
)
from assembly_net.topology.descriptors import (
    TopologicalDescriptorExtractor,
    extract_loop_statistics,
    compute_percolation_threshold,
)

__all__ = [
    # Persistent homology
    "PersistentHomologyComputer",
    "TopologicalFeatures",
    "compute_betti_numbers",
    "compute_persistence_diagram",
    "persistence_landscape",
    # Rigidity
    "RigidityAnalyzer",
    "RigidityMetrics",
    "compute_rigidity_matrix",
    "count_floppy_modes",
    # Descriptors
    "TopologicalDescriptorExtractor",
    "extract_loop_statistics",
    "compute_percolation_threshold",
]
