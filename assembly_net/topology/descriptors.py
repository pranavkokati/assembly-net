"""
Topological Descriptor Extraction for assembly networks.

This module provides comprehensive topological feature extraction
combining persistent homology, rigidity analysis, and network statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import networkx as nx

from assembly_net.data.core import AssemblyGraph, AssemblyTrajectory, NetworkState
from assembly_net.topology.persistent_homology import (
    PersistentHomologyComputer,
    TopologicalFeatures,
    compute_betti_numbers,
)
from assembly_net.topology.graph_rigidity import (
    RigidityAnalyzer,
    RigidityMetrics,
    compute_bulk_modulus,
    compute_shear_modulus,
)


@dataclass
class FullTopologicalDescriptor:
    """
    Complete topological descriptor combining all feature types.

    This is the primary feature representation for topology-aware ML.
    """

    # Persistent homology features
    homology_features: Optional[TopologicalFeatures] = None

    # Rigidity features
    rigidity_metrics: Optional[RigidityMetrics] = None

    # Network statistics
    num_nodes: int = 0
    num_edges: int = 0
    num_components: int = 0
    largest_component_fraction: float = 0.0

    # Loop statistics
    num_triangles: int = 0
    num_squares: int = 0
    cycle_basis_size: int = 0
    loop_density: float = 0.0

    # Percolation
    is_percolated: bool = False
    percolation_strength: float = 0.0

    # Clustering
    clustering_coefficient: float = 0.0
    transitivity: float = 0.0

    # Degree statistics
    mean_degree: float = 0.0
    degree_variance: float = 0.0
    max_degree: int = 0
    degree_assortativity: float = 0.0

    # Centrality measures
    mean_betweenness: float = 0.0
    mean_closeness: float = 0.0

    # Mechanical estimates
    bulk_modulus: float = 0.0
    shear_modulus: float = 0.0

    def to_vector(self) -> np.ndarray:
        """Convert full descriptor to a feature vector."""
        features = []

        # Homology features
        if self.homology_features is not None:
            features.extend(self.homology_features.to_vector().tolist())
        else:
            # Placeholder
            features.extend([0.0] * 30)

        # Rigidity features
        if self.rigidity_metrics is not None:
            features.extend(self.rigidity_metrics.to_vector().tolist())
        else:
            features.extend([0.0] * 8)

        # Network statistics
        features.extend(
            [
                np.log1p(self.num_nodes) / 5,
                np.log1p(self.num_edges) / 5,
                self.num_components / max(1, self.num_nodes),
                self.largest_component_fraction,
            ]
        )

        # Loop statistics
        features.extend(
            [
                np.log1p(self.num_triangles) / 5,
                np.log1p(self.num_squares) / 5,
                np.log1p(self.cycle_basis_size) / 5,
                self.loop_density,
            ]
        )

        # Percolation
        features.extend([float(self.is_percolated), self.percolation_strength])

        # Clustering
        features.extend([self.clustering_coefficient, self.transitivity])

        # Degree statistics
        features.extend(
            [
                self.mean_degree / 10,
                self.degree_variance / 10,
                self.max_degree / 20,
                (self.degree_assortativity + 1) / 2,  # Map [-1, 1] to [0, 1]
            ]
        )

        # Centrality
        features.extend([self.mean_betweenness, self.mean_closeness])

        # Mechanical
        features.extend(
            [np.log1p(self.bulk_modulus), np.log1p(self.shear_modulus)]
        )

        return np.array(features, dtype=np.float32)


class TopologicalDescriptorExtractor:
    """
    Extracts comprehensive topological descriptors from assembly graphs.

    Combines:
    - Persistent homology (Betti numbers, persistence diagrams)
    - Rigidity analysis (Maxwell counting, floppy modes)
    - Network statistics (clustering, centrality, degree distribution)
    - Loop analysis (cycles, triangles, squares)
    """

    def __init__(
        self,
        compute_homology: bool = True,
        compute_rigidity: bool = True,
        homology_max_dim: int = 2,
        rigidity_dimension: int = 3,
    ):
        """
        Initialize the descriptor extractor.

        Args:
            compute_homology: Whether to compute persistent homology.
            compute_rigidity: Whether to compute rigidity metrics.
            homology_max_dim: Maximum homology dimension.
            rigidity_dimension: Spatial dimension for rigidity.
        """
        self.compute_homology = compute_homology
        self.compute_rigidity = compute_rigidity

        if compute_homology:
            self.homology_computer = PersistentHomologyComputer(
                max_dimension=homology_max_dim
            )

        if compute_rigidity:
            self.rigidity_analyzer = RigidityAnalyzer(dimension=rigidity_dimension)

    def extract(self, graph: AssemblyGraph) -> FullTopologicalDescriptor:
        """
        Extract complete topological descriptor from a graph.

        Args:
            graph: AssemblyGraph to analyze.

        Returns:
            FullTopologicalDescriptor containing all features.
        """
        descriptor = FullTopologicalDescriptor()

        # Basic counts
        descriptor.num_nodes = graph.num_nodes
        descriptor.num_edges = len(graph.edges)

        if graph.num_nodes == 0:
            return descriptor

        # Convert to NetworkX for analysis
        G = graph.to_networkx()

        # Persistent homology
        if self.compute_homology:
            descriptor.homology_features = self.homology_computer.compute(graph)

        # Rigidity
        if self.compute_rigidity:
            descriptor.rigidity_metrics = self.rigidity_analyzer.analyze(graph)

        # Component analysis
        components = list(nx.connected_components(G))
        descriptor.num_components = len(components)
        if components:
            largest = max(len(c) for c in components)
            descriptor.largest_component_fraction = largest / graph.num_nodes
        descriptor.is_percolated = descriptor.largest_component_fraction > 0.5
        descriptor.percolation_strength = descriptor.largest_component_fraction

        # Loop statistics
        loop_stats = extract_loop_statistics(graph)
        descriptor.num_triangles = loop_stats["num_triangles"]
        descriptor.num_squares = loop_stats["num_squares"]
        descriptor.cycle_basis_size = loop_stats["cycle_basis_size"]
        descriptor.loop_density = loop_stats["loop_density"]

        # Clustering
        descriptor.clustering_coefficient = nx.average_clustering(G)
        descriptor.transitivity = nx.transitivity(G)

        # Degree statistics
        degrees = graph.get_degree_sequence()
        if len(degrees) > 0:
            descriptor.mean_degree = np.mean(degrees)
            descriptor.degree_variance = np.var(degrees)
            descriptor.max_degree = int(np.max(degrees))

        # Degree assortativity
        if len(graph.edges) > 0:
            try:
                descriptor.degree_assortativity = nx.degree_assortativity_coefficient(G)
            except Exception:
                descriptor.degree_assortativity = 0.0

        # Centrality measures
        if G.number_of_nodes() > 0:
            try:
                betweenness = nx.betweenness_centrality(G)
                descriptor.mean_betweenness = np.mean(list(betweenness.values()))
            except Exception:
                pass

            try:
                # Use only largest connected component for closeness
                if descriptor.is_percolated:
                    largest_cc = max(components, key=len)
                    subG = G.subgraph(largest_cc)
                    closeness = nx.closeness_centrality(subG)
                    descriptor.mean_closeness = np.mean(list(closeness.values()))
            except Exception:
                pass

        # Mechanical estimates
        descriptor.bulk_modulus = compute_bulk_modulus(graph)
        descriptor.shear_modulus = compute_shear_modulus(graph)

        return descriptor

    def extract_trajectory(
        self, trajectory: AssemblyTrajectory
    ) -> List[FullTopologicalDescriptor]:
        """Extract descriptors for each state in a trajectory."""
        return [self.extract(state.graph) for state in trajectory.states]

    def extract_evolution_features(
        self, trajectory: AssemblyTrajectory
    ) -> Dict[str, np.ndarray]:
        """
        Extract time-evolution features from a trajectory.

        Returns dictionary of arrays showing how each feature evolves.
        """
        descriptors = self.extract_trajectory(trajectory)

        # Extract time series for key features
        features = {
            "times": np.array([s.time for s in trajectory.states]),
            "num_edges": np.array([d.num_edges for d in descriptors]),
            "num_components": np.array([d.num_components for d in descriptors]),
            "largest_component_fraction": np.array(
                [d.largest_component_fraction for d in descriptors]
            ),
            "loop_density": np.array([d.loop_density for d in descriptors]),
            "clustering_coefficient": np.array(
                [d.clustering_coefficient for d in descriptors]
            ),
            "mean_degree": np.array([d.mean_degree for d in descriptors]),
            "is_percolated": np.array(
                [float(d.is_percolated) for d in descriptors]
            ),
        }

        # Add rate of change features
        for key in ["num_edges", "loop_density", "mean_degree"]:
            values = features[key]
            rates = np.gradient(values, features["times"])
            features[f"{key}_rate"] = rates

        return features


def extract_loop_statistics(graph: AssemblyGraph) -> Dict[str, Any]:
    """
    Extract detailed loop statistics from a graph.

    Args:
        graph: AssemblyGraph to analyze.

    Returns:
        Dictionary containing loop statistics.
    """
    stats = {
        "num_triangles": 0,
        "num_squares": 0,
        "cycle_basis_size": 0,
        "loop_density": 0.0,
        "largest_loop_size": 0,
        "mean_loop_size": 0.0,
    }

    if graph.num_nodes == 0 or len(graph.edges) == 0:
        return stats

    G = graph.to_networkx()

    # Count triangles
    triangles = nx.triangles(G)
    stats["num_triangles"] = sum(triangles.values()) // 3

    # Count squares (4-cycles)
    # This is expensive, so we estimate for large graphs
    if graph.num_nodes < 100:
        squares = 0
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            for i, n1 in enumerate(neighbors):
                for n2 in neighbors[i + 1 :]:
                    # Check if n1 and n2 share another common neighbor
                    common = set(G.neighbors(n1)) & set(G.neighbors(n2))
                    common.discard(node)
                    squares += len(common)
        stats["num_squares"] = squares // 4  # Each square counted 4 times
    else:
        # Estimate from clustering
        stats["num_squares"] = int(
            stats["num_triangles"] * (1 - nx.average_clustering(G))
        )

    # Cycle basis
    try:
        cycles = nx.cycle_basis(G)
        stats["cycle_basis_size"] = len(cycles)

        if cycles:
            cycle_sizes = [len(c) for c in cycles]
            stats["largest_loop_size"] = max(cycle_sizes)
            stats["mean_loop_size"] = np.mean(cycle_sizes)
    except Exception:
        pass

    # Loop density: cycles per edge
    if len(graph.edges) > 0:
        stats["loop_density"] = stats["cycle_basis_size"] / len(graph.edges)

    return stats


def compute_percolation_threshold(
    trajectory: AssemblyTrajectory,
    threshold_fraction: float = 0.5,
) -> Tuple[Optional[float], float]:
    """
    Compute when percolation occurs in an assembly trajectory.

    Args:
        trajectory: AssemblyTrajectory to analyze.
        threshold_fraction: Fraction of nodes for percolation.

    Returns:
        Tuple of (percolation_time, final_percolation_strength).
        percolation_time is None if percolation never occurs.
    """
    percolation_time = None
    final_strength = 0.0

    for state in trajectory.states:
        if state.largest_component_size is None:
            state.compute_properties()

        strength = (state.largest_component_size or 0) / max(1, state.graph.num_nodes)
        final_strength = strength

        if percolation_time is None and strength >= threshold_fraction:
            percolation_time = state.time

    return percolation_time, final_strength


def compute_loop_formation_rate(
    trajectory: AssemblyTrajectory,
) -> np.ndarray:
    """
    Compute the rate of loop formation over time.

    Args:
        trajectory: AssemblyTrajectory to analyze.

    Returns:
        Array of loop formation rates at each timestep.
    """
    loop_counts = []
    times = []

    for state in trajectory.states:
        G = state.graph.to_networkx()
        try:
            num_cycles = len(nx.cycle_basis(G))
        except Exception:
            num_cycles = 0
        loop_counts.append(num_cycles)
        times.append(state.time)

    loop_counts = np.array(loop_counts)
    times = np.array(times)

    # Compute rate using gradient
    if len(times) > 1:
        rates = np.gradient(loop_counts, times)
    else:
        rates = np.zeros_like(loop_counts)

    return rates


def compute_growth_exponents(
    trajectory: AssemblyTrajectory,
) -> Dict[str, float]:
    """
    Compute growth exponents for network properties.

    Fits power-law growth: property ~ t^alpha

    Args:
        trajectory: AssemblyTrajectory to analyze.

    Returns:
        Dictionary of growth exponents for different properties.
    """
    exponents = {}

    times = np.array([s.time for s in trajectory.states])
    if len(times) < 3 or times[-1] <= times[0]:
        return exponents

    # Normalize times
    t = (times - times[0]) / (times[-1] - times[0])
    t = np.maximum(t, 1e-6)  # Avoid log(0)

    # Properties to analyze
    properties = {
        "num_edges": [len(s.graph.edges) for s in trajectory.states],
        "largest_component": [
            s.largest_component_size or 0 for s in trajectory.states
        ],
        "mean_degree": [
            np.mean(s.graph.get_degree_sequence()) if s.graph.num_nodes > 0 else 0
            for s in trajectory.states
        ],
    }

    for name, values in properties.items():
        values = np.array(values)
        values = np.maximum(values, 1e-6)

        # Linear regression in log-log space
        try:
            log_t = np.log(t + 1e-6)
            log_v = np.log(values + 1e-6)

            # Remove infinities
            mask = np.isfinite(log_t) & np.isfinite(log_v)
            if np.sum(mask) >= 2:
                coeffs = np.polyfit(log_t[mask], log_v[mask], 1)
                exponents[name] = coeffs[0]
        except Exception:
            pass

    return exponents
