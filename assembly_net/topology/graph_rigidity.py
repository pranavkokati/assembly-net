"""
Graph Rigidity Analysis for coordination networks.

This module provides tools for analyzing the mechanical rigidity of
assembly networks using:
- Maxwell counting
- Pebble game algorithms
- Rigidity matrix analysis

Rigidity is a key predictor of mechanical properties in materials.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy import linalg
import networkx as nx

from assembly_net.data.core import AssemblyGraph, AssemblyTrajectory


@dataclass
class RigidityMetrics:
    """
    Container for rigidity analysis results.

    Attributes:
        is_rigid: Whether the network is globally rigid.
        num_floppy_modes: Number of zero-frequency modes.
        rigidity_fraction: Fraction of edges that are rigid.
        rigid_clusters: List of rigid sub-clusters.
        redundant_edges: Edges that are over-constrained.
        mean_coordination: Average coordination number.
        maxwell_count: Maxwell counting result (positive = overconstrained).
    """

    is_rigid: bool = False
    num_floppy_modes: int = 0
    rigidity_fraction: float = 0.0
    rigid_clusters: List[Set[int]] = field(default_factory=list)
    redundant_edges: List[Tuple[int, int]] = field(default_factory=list)
    mean_coordination: float = 0.0
    maxwell_count: int = 0
    isostatic_threshold: float = 0.0
    degrees_of_freedom: int = 0

    def to_vector(self) -> np.ndarray:
        """Convert metrics to a feature vector."""
        return np.array(
            [
                float(self.is_rigid),
                self.num_floppy_modes / 10.0,  # Normalize
                self.rigidity_fraction,
                len(self.rigid_clusters) / 10.0,
                len(self.redundant_edges) / 10.0,
                self.mean_coordination / 6.0,  # Normalize by typical max coordination
                self.maxwell_count / 100.0,
                self.isostatic_threshold,
            ],
            dtype=np.float32,
        )


class RigidityAnalyzer:
    """
    Analyzes mechanical rigidity of assembly networks.

    Implements Maxwell counting and simplified pebble game for
    2D and 3D networks.
    """

    def __init__(self, dimension: int = 3):
        """
        Initialize the rigidity analyzer.

        Args:
            dimension: Spatial dimension (2 or 3).
        """
        self.dimension = dimension
        # Degrees of freedom per node
        self.dof_per_node = dimension  # Translational DOF
        # Rigid body modes
        self.rigid_body_modes = dimension * (dimension + 1) // 2

    def analyze(self, graph: AssemblyGraph) -> RigidityMetrics:
        """
        Perform complete rigidity analysis on a graph.

        Args:
            graph: AssemblyGraph to analyze.

        Returns:
            RigidityMetrics containing all analysis results.
        """
        metrics = RigidityMetrics()

        if graph.num_nodes == 0:
            return metrics

        n_nodes = graph.num_nodes
        n_edges = len(graph.edges)

        # Maxwell counting
        # Total DOF = d * n_nodes
        # Constraints = n_edges (each edge removes 1 DOF)
        # Rigid body modes = d(d+1)/2
        total_dof = self.dof_per_node * n_nodes
        constraints = n_edges
        net_dof = total_dof - constraints - self.rigid_body_modes

        metrics.degrees_of_freedom = max(0, net_dof)
        metrics.maxwell_count = constraints - (total_dof - self.rigid_body_modes)

        # Isostatic threshold: <z> = 2d - 2d/n
        metrics.isostatic_threshold = 2 * self.dimension * (1 - 1 / max(1, n_nodes))

        # Mean coordination
        degrees = graph.get_degree_sequence()
        metrics.mean_coordination = np.mean(degrees) if len(degrees) > 0 else 0.0

        # Global rigidity check
        # Network is rigid if it has enough constraints
        metrics.is_rigid = metrics.maxwell_count >= 0

        # Compute floppy modes
        metrics.num_floppy_modes = self._count_floppy_modes(graph)

        # Rigidity fraction
        if n_nodes > 1:
            max_edges = n_nodes * (n_nodes - 1) // 2
            metrics.rigidity_fraction = n_edges / max_edges
        else:
            metrics.rigidity_fraction = 0.0

        # Find rigid clusters using simplified pebble game
        metrics.rigid_clusters = self._find_rigid_clusters(graph)

        # Find redundant edges
        metrics.redundant_edges = self._find_redundant_edges(graph)

        return metrics

    def _count_floppy_modes(self, graph: AssemblyGraph) -> int:
        """
        Count zero-frequency (floppy) modes using rigidity matrix.

        The rigidity matrix R has rows for each edge constraint and
        columns for each node coordinate. Floppy modes are in the
        null space of R.
        """
        rigidity_matrix = compute_rigidity_matrix(graph, self.dimension)

        if rigidity_matrix.size == 0:
            return self.dimension * graph.num_nodes

        # Compute rank
        rank = np.linalg.matrix_rank(rigidity_matrix, tol=1e-10)

        # Floppy modes = total DOF - rank
        total_dof = self.dimension * graph.num_nodes
        floppy = total_dof - rank

        # Subtract rigid body modes (if network is connected)
        G = graph.to_networkx()
        if nx.is_connected(G):
            floppy = max(0, floppy - self.rigid_body_modes)

        return floppy

    def _find_rigid_clusters(self, graph: AssemblyGraph) -> List[Set[int]]:
        """
        Find rigid clusters in the network.

        Uses a simplified approach based on local rigidity:
        a cluster is rigid if its internal DOF are fully constrained.
        """
        G = graph.to_networkx()
        rigid_clusters = []

        # Start with connected components
        for component in nx.connected_components(G):
            if len(component) < 3:
                continue

            # Check if component is rigid
            subgraph_nodes = list(component)
            edges = [
                (s, t)
                for s, t in graph.edges
                if s in component and t in component
            ]

            n = len(subgraph_nodes)
            m = len(edges)

            # Maxwell count for this component
            dof = self.dimension * n
            constraints = m
            rigid_body = self.rigid_body_modes

            if constraints >= dof - rigid_body:
                rigid_clusters.append(set(subgraph_nodes))
            else:
                # Try to find rigid sub-clusters using k-cores
                subG = G.subgraph(component)
                for k in range(self.dimension + 1, 0, -1):
                    core = nx.k_core(subG, k=k)
                    if core.number_of_nodes() >= 3:
                        core_nodes = set(core.nodes())
                        core_edges = [
                            (s, t)
                            for s, t in edges
                            if s in core_nodes and t in core_nodes
                        ]
                        if len(core_edges) >= self.dimension * len(core_nodes) - rigid_body:
                            rigid_clusters.append(core_nodes)
                            break

        return rigid_clusters

    def _find_redundant_edges(self, graph: AssemblyGraph) -> List[Tuple[int, int]]:
        """
        Find redundant (over-constraining) edges.

        An edge is redundant if removing it doesn't change the rigidity.
        """
        redundant = []

        # Check each edge
        for edge in graph.edges:
            # Create graph without this edge
            test_graph = graph.copy()
            test_graph.remove_edge(edge[0], edge[1])

            # Check if still rigid
            test_metrics = self._quick_rigidity_check(test_graph)
            original_metrics = self._quick_rigidity_check(graph)

            if test_metrics["is_rigid"] == original_metrics["is_rigid"]:
                redundant.append(edge)

        return redundant

    def _quick_rigidity_check(self, graph: AssemblyGraph) -> Dict:
        """Quick Maxwell counting check."""
        n_nodes = graph.num_nodes
        n_edges = len(graph.edges)

        total_dof = self.dof_per_node * n_nodes
        constraints = n_edges

        return {
            "is_rigid": constraints >= total_dof - self.rigid_body_modes,
            "maxwell_count": constraints - (total_dof - self.rigid_body_modes),
        }

    def analyze_trajectory(
        self, trajectory: AssemblyTrajectory
    ) -> List[RigidityMetrics]:
        """Analyze rigidity for each state in a trajectory."""
        return [self.analyze(state.graph) for state in trajectory.states]


def compute_rigidity_matrix(
    graph: AssemblyGraph,
    dimension: int = 3,
    positions: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute the rigidity matrix for a graph.

    The rigidity matrix R has dimensions (n_edges, d * n_nodes).
    For edge (i, j), the row contains:
    - (p_i - p_j) at columns [d*i : d*i + d]
    - (p_j - p_i) at columns [d*j : d*j + d]

    Args:
        graph: AssemblyGraph to analyze.
        dimension: Spatial dimension (2 or 3).
        positions: Node positions. If None, uses positions from features
                   or generates random positions.

    Returns:
        Rigidity matrix of shape (n_edges, d * n_nodes).
    """
    n_nodes = graph.num_nodes
    n_edges = len(graph.edges)

    if n_nodes == 0 or n_edges == 0:
        return np.array([]).reshape(0, 0)

    # Get or generate positions
    if positions is None:
        positions = np.zeros((n_nodes, dimension))
        for i, nf in enumerate(graph.node_features):
            if nf.position is not None and len(nf.position) >= dimension:
                positions[i] = nf.position[:dimension]
            else:
                # Random position for analysis
                positions[i] = np.random.randn(dimension)

    # Build rigidity matrix
    R = np.zeros((n_edges, dimension * n_nodes))

    for e_idx, (i, j) in enumerate(graph.edges):
        # Edge vector
        edge_vec = positions[i] - positions[j]

        # Normalize (for numerical stability)
        norm = np.linalg.norm(edge_vec)
        if norm > 1e-10:
            edge_vec = edge_vec / norm

        # Fill matrix
        R[e_idx, dimension * i : dimension * i + dimension] = edge_vec
        R[e_idx, dimension * j : dimension * j + dimension] = -edge_vec

    return R


def count_floppy_modes(graph: AssemblyGraph, dimension: int = 3) -> int:
    """
    Convenience function to count floppy modes.

    Args:
        graph: AssemblyGraph to analyze.
        dimension: Spatial dimension.

    Returns:
        Number of floppy modes.
    """
    analyzer = RigidityAnalyzer(dimension=dimension)
    metrics = analyzer.analyze(graph)
    return metrics.num_floppy_modes


def rigidity_percolation_threshold(dimension: int = 3) -> float:
    """
    Return the theoretical rigidity percolation threshold.

    For a random network, rigidity percolates when the mean
    coordination exceeds 2d (Maxwell criterion).

    Args:
        dimension: Spatial dimension.

    Returns:
        Critical mean coordination number.
    """
    return 2.0 * dimension


def compute_stress_tensor(
    graph: AssemblyGraph,
    forces: Optional[np.ndarray] = None,
    dimension: int = 3,
) -> np.ndarray:
    """
    Compute the virial stress tensor for the network.

    Args:
        graph: AssemblyGraph to analyze.
        forces: Optional external forces on nodes.
        dimension: Spatial dimension.

    Returns:
        Stress tensor of shape (dimension, dimension).
    """
    n_nodes = graph.num_nodes

    if n_nodes == 0:
        return np.zeros((dimension, dimension))

    # Get positions
    positions = np.zeros((n_nodes, dimension))
    for i, nf in enumerate(graph.node_features):
        if nf.position is not None and len(nf.position) >= dimension:
            positions[i] = nf.position[:dimension]
        else:
            positions[i] = np.random.randn(dimension)

    # Compute stress tensor from edge contributions
    stress = np.zeros((dimension, dimension))

    for (i, j), ef in zip(graph.edges, graph.edge_features):
        r_ij = positions[j] - positions[i]
        r_norm = np.linalg.norm(r_ij)

        if r_norm > 1e-10:
            # Force along bond (proportional to bond strength)
            f_mag = ef.bond_strength
            f_ij = f_mag * r_ij / r_norm

            # Virial contribution
            stress += np.outer(r_ij, f_ij)

    # Normalize by volume (assume unit volume)
    return stress


def compute_bulk_modulus(
    graph: AssemblyGraph,
    dimension: int = 3,
) -> float:
    """
    Estimate bulk modulus from network topology.

    Uses simplified effective medium theory:
    K ~ (z - z_c) * k_bond

    where z is mean coordination, z_c is critical coordination,
    and k_bond is average bond stiffness.

    Args:
        graph: AssemblyGraph to analyze.
        dimension: Spatial dimension.

    Returns:
        Estimated bulk modulus (relative units).
    """
    if graph.num_nodes == 0:
        return 0.0

    # Mean coordination
    degrees = graph.get_degree_sequence()
    z = np.mean(degrees) if len(degrees) > 0 else 0.0

    # Critical coordination
    z_c = 2 * dimension

    # Average bond stiffness
    if graph.edge_features:
        k_bond = np.mean([ef.bond_strength for ef in graph.edge_features])
    else:
        k_bond = 1.0

    # Bulk modulus
    if z > z_c:
        K = (z - z_c) * k_bond
    else:
        K = 0.0  # Below rigidity threshold

    return K


def compute_shear_modulus(
    graph: AssemblyGraph,
    dimension: int = 3,
) -> float:
    """
    Estimate shear modulus from network topology.

    Similar to bulk modulus but with different prefactor.

    Args:
        graph: AssemblyGraph to analyze.
        dimension: Spatial dimension.

    Returns:
        Estimated shear modulus (relative units).
    """
    if graph.num_nodes == 0:
        return 0.0

    # Mean coordination
    degrees = graph.get_degree_sequence()
    z = np.mean(degrees) if len(degrees) > 0 else 0.0

    # Critical coordination
    z_c = 2 * dimension

    # Average bond stiffness
    if graph.edge_features:
        k_bond = np.mean([ef.bond_strength for ef in graph.edge_features])
    else:
        k_bond = 1.0

    # Shear modulus (typically smaller prefactor than bulk)
    if z > z_c:
        G = 0.5 * (z - z_c) * k_bond
    else:
        G = 0.0

    return G
