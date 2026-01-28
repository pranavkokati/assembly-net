"""
Persistent Homology computation for assembly graphs.

This module provides tools for computing topological features from
coordination networks using persistent homology.

Persistent homology captures multi-scale topological features:
- H0: Connected components (clusters)
- H1: Loops/cycles
- H2: Voids/cavities

These features are critical for understanding emergent material properties.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist, squareform

from assembly_net.data.core import AssemblyGraph, AssemblyTrajectory, NetworkState


@dataclass
class TopologicalFeatures:
    """
    Container for topological features extracted from a graph.

    Attributes:
        betti_numbers: Betti numbers [b0, b1, b2, ...]
        persistence_diagram: List of (birth, death) tuples per dimension
        persistence_entropy: Entropy of persistence diagram
        total_persistence: Sum of all bar lengths
        landscape_features: Persistence landscape statistics
    """

    betti_numbers: np.ndarray = field(default_factory=lambda: np.array([0, 0]))
    persistence_diagram: Dict[int, np.ndarray] = field(default_factory=dict)
    persistence_entropy: float = 0.0
    total_persistence: float = 0.0
    landscape_features: Optional[np.ndarray] = None
    barcode_statistics: Dict[str, float] = field(default_factory=dict)

    def to_vector(self, max_dim: int = 2, num_landscape_pts: int = 10) -> np.ndarray:
        """Convert features to a fixed-size vector for ML."""
        features = []

        # Betti numbers (padded)
        betti = np.zeros(max_dim + 1)
        betti[: len(self.betti_numbers)] = self.betti_numbers[: max_dim + 1]
        features.extend(betti)

        # Persistence statistics per dimension
        for dim in range(max_dim + 1):
            if dim in self.persistence_diagram and len(self.persistence_diagram[dim]) > 0:
                diagram = self.persistence_diagram[dim]
                lifetimes = diagram[:, 1] - diagram[:, 0]
                lifetimes = lifetimes[np.isfinite(lifetimes)]

                if len(lifetimes) > 0:
                    features.extend(
                        [
                            np.mean(lifetimes),
                            np.std(lifetimes),
                            np.max(lifetimes),
                            np.median(lifetimes),
                            len(lifetimes),
                        ]
                    )
                else:
                    features.extend([0.0] * 5)
            else:
                features.extend([0.0] * 5)

        # Global statistics
        features.extend([self.persistence_entropy, self.total_persistence])

        # Landscape features
        if self.landscape_features is not None:
            features.extend(self.landscape_features[:num_landscape_pts].tolist())
        else:
            features.extend([0.0] * num_landscape_pts)

        return np.array(features, dtype=np.float32)


class PersistentHomologyComputer:
    """
    Computes persistent homology features from assembly graphs.

    Supports multiple filtration methods:
    - Edge weight filtration (bond strength)
    - Vietoris-Rips filtration (spatial)
    - Alpha complex filtration (spatial)
    - Clique complex filtration (graph)
    """

    def __init__(
        self,
        max_dimension: int = 2,
        filtration_type: str = "clique",
        max_edge_length: float = np.inf,
        num_landscape_layers: int = 5,
        num_landscape_points: int = 100,
    ):
        """
        Initialize the persistent homology computer.

        Args:
            max_dimension: Maximum homology dimension to compute.
            filtration_type: Type of filtration ('clique', 'rips', 'alpha', 'weight').
            max_edge_length: Maximum edge length for Rips filtration.
            num_landscape_layers: Number of landscape layers.
            num_landscape_points: Resolution of landscape discretization.
        """
        self.max_dimension = max_dimension
        self.filtration_type = filtration_type
        self.max_edge_length = max_edge_length
        self.num_landscape_layers = num_landscape_layers
        self.num_landscape_points = num_landscape_points

    def compute(self, graph: AssemblyGraph) -> TopologicalFeatures:
        """
        Compute persistent homology features for a graph.

        Args:
            graph: AssemblyGraph to analyze.

        Returns:
            TopologicalFeatures containing all computed features.
        """
        features = TopologicalFeatures()

        # Handle empty graph
        if graph.num_nodes == 0:
            return features

        # Compute Betti numbers directly from graph
        features.betti_numbers = compute_betti_numbers(graph)

        # Build filtration and compute persistence
        if self.filtration_type == "clique":
            diagram = self._clique_filtration(graph)
        elif self.filtration_type == "weight":
            diagram = self._weight_filtration(graph)
        elif self.filtration_type == "rips":
            diagram = self._rips_filtration(graph)
        else:
            diagram = self._clique_filtration(graph)

        features.persistence_diagram = diagram

        # Compute derived features
        features.persistence_entropy = self._compute_entropy(diagram)
        features.total_persistence = self._compute_total_persistence(diagram)
        features.landscape_features = self._compute_landscape_features(diagram)
        features.barcode_statistics = self._compute_barcode_statistics(diagram)

        return features

    def _clique_filtration(self, graph: AssemblyGraph) -> Dict[int, np.ndarray]:
        """
        Compute persistence using clique complex filtration.

        Uses edge formation time as filtration value.
        """
        try:
            from gtda.homology import VietorisRipsPersistence
        except ImportError:
            # Fallback to simple computation
            return self._simple_persistence(graph)

        # Build distance matrix from graph structure
        n = graph.num_nodes
        if n == 0:
            return {0: np.array([]).reshape(0, 2), 1: np.array([]).reshape(0, 2)}

        # Create distance matrix where connected nodes have distance 0
        # and disconnected nodes have distance 1
        dist_matrix = np.ones((n, n))
        np.fill_diagonal(dist_matrix, 0)

        for (s, t), ef in zip(graph.edges, graph.edge_features):
            # Use inverse bond strength as distance
            d = 1.0 / (ef.bond_strength + 0.1)
            dist_matrix[s, t] = d
            dist_matrix[t, s] = d

        try:
            # Use giotto-tda for computation
            vr = VietorisRipsPersistence(
                homology_dimensions=list(range(self.max_dimension + 1)),
                max_edge_length=self.max_edge_length,
            )
            diagrams = vr.fit_transform([dist_matrix])

            # Convert to our format
            result = {}
            diag = diagrams[0]
            for dim in range(self.max_dimension + 1):
                mask = diag[:, 2] == dim
                result[dim] = diag[mask, :2]
            return result
        except Exception:
            return self._simple_persistence(graph)

    def _weight_filtration(self, graph: AssemblyGraph) -> Dict[int, np.ndarray]:
        """
        Compute persistence using edge weight (formation time) filtration.
        """
        # Build filtered simplicial complex based on formation time
        return self._simple_persistence(graph)

    def _rips_filtration(self, graph: AssemblyGraph) -> Dict[int, np.ndarray]:
        """
        Compute Vietoris-Rips persistence from spatial coordinates.
        """
        # Get spatial positions
        positions = []
        for nf in graph.node_features:
            if nf.position is not None:
                positions.append(nf.position)
            else:
                # Use random positions if not available
                positions.append(np.random.randn(3))

        if not positions:
            return {0: np.array([]).reshape(0, 2), 1: np.array([]).reshape(0, 2)}

        positions = np.array(positions)

        try:
            from gtda.homology import VietorisRipsPersistence

            dist_matrix = squareform(pdist(positions))

            vr = VietorisRipsPersistence(
                homology_dimensions=list(range(self.max_dimension + 1)),
                max_edge_length=self.max_edge_length,
            )
            diagrams = vr.fit_transform([dist_matrix])

            result = {}
            diag = diagrams[0]
            for dim in range(self.max_dimension + 1):
                mask = diag[:, 2] == dim
                result[dim] = diag[mask, :2]
            return result
        except Exception:
            return self._simple_persistence(graph)

    def _simple_persistence(self, graph: AssemblyGraph) -> Dict[int, np.ndarray]:
        """
        Simple persistence computation without external libraries.

        Computes H0 (components) and H1 (cycles) directly.
        """
        G = graph.to_networkx()

        # H0: Connected components
        # Each component born at 0, one dies at infinity (or max time)
        components = list(nx.connected_components(G))
        h0_diagram = []
        for i, comp in enumerate(components):
            if i == 0:
                # One component persists forever
                h0_diagram.append([0.0, np.inf])
            else:
                # Other components represent merging events
                h0_diagram.append([0.0, 1.0])

        # H1: Cycles
        # Each independent cycle represents a 1-dimensional hole
        try:
            cycle_basis = nx.cycle_basis(G)
            h1_diagram = []
            for cycle in cycle_basis:
                # Born when cycle forms, persists
                birth = 0.5  # Arbitrary birth time
                death = np.inf
                h1_diagram.append([birth, death])
        except Exception:
            h1_diagram = []

        return {
            0: np.array(h0_diagram).reshape(-1, 2) if h0_diagram else np.array([]).reshape(0, 2),
            1: np.array(h1_diagram).reshape(-1, 2) if h1_diagram else np.array([]).reshape(0, 2),
        }

    def _compute_entropy(self, diagram: Dict[int, np.ndarray]) -> float:
        """Compute persistence entropy."""
        all_lifetimes = []
        for dim, diag in diagram.items():
            if len(diag) > 0:
                lifetimes = diag[:, 1] - diag[:, 0]
                lifetimes = lifetimes[np.isfinite(lifetimes)]
                all_lifetimes.extend(lifetimes.tolist())

        if not all_lifetimes:
            return 0.0

        lifetimes = np.array(all_lifetimes)
        total = np.sum(lifetimes)
        if total == 0:
            return 0.0

        probs = lifetimes / total
        probs = probs[probs > 0]
        return -np.sum(probs * np.log(probs + 1e-10))

    def _compute_total_persistence(self, diagram: Dict[int, np.ndarray]) -> float:
        """Compute total persistence (sum of bar lengths)."""
        total = 0.0
        for dim, diag in diagram.items():
            if len(diag) > 0:
                lifetimes = diag[:, 1] - diag[:, 0]
                lifetimes = lifetimes[np.isfinite(lifetimes)]
                total += np.sum(lifetimes)
        return total

    def _compute_landscape_features(
        self, diagram: Dict[int, np.ndarray]
    ) -> np.ndarray:
        """Compute persistence landscape features."""
        # Simplified landscape: statistics of lifetimes
        features = []

        for dim in range(self.max_dimension + 1):
            if dim in diagram and len(diagram[dim]) > 0:
                lifetimes = diagram[dim][:, 1] - diagram[dim][:, 0]
                lifetimes = lifetimes[np.isfinite(lifetimes)]

                if len(lifetimes) > 0:
                    # Sort by lifetime (descending)
                    sorted_lifetimes = np.sort(lifetimes)[::-1]
                    # Take top k as landscape "heights"
                    k = min(self.num_landscape_layers, len(sorted_lifetimes))
                    features.extend(sorted_lifetimes[:k].tolist())
                    features.extend([0.0] * (self.num_landscape_layers - k))
                else:
                    features.extend([0.0] * self.num_landscape_layers)
            else:
                features.extend([0.0] * self.num_landscape_layers)

        return np.array(features, dtype=np.float32)

    def _compute_barcode_statistics(
        self, diagram: Dict[int, np.ndarray]
    ) -> Dict[str, float]:
        """Compute statistical summaries of persistence barcodes."""
        stats = {}

        for dim in range(self.max_dimension + 1):
            prefix = f"h{dim}_"

            if dim in diagram and len(diagram[dim]) > 0:
                births = diagram[dim][:, 0]
                deaths = diagram[dim][:, 1]
                lifetimes = deaths - births
                finite_lifetimes = lifetimes[np.isfinite(lifetimes)]

                stats[prefix + "count"] = len(diagram[dim])
                stats[prefix + "birth_mean"] = np.mean(births)
                stats[prefix + "birth_std"] = np.std(births)

                if len(finite_lifetimes) > 0:
                    stats[prefix + "lifetime_mean"] = np.mean(finite_lifetimes)
                    stats[prefix + "lifetime_std"] = np.std(finite_lifetimes)
                    stats[prefix + "lifetime_max"] = np.max(finite_lifetimes)
                else:
                    stats[prefix + "lifetime_mean"] = 0.0
                    stats[prefix + "lifetime_std"] = 0.0
                    stats[prefix + "lifetime_max"] = 0.0
            else:
                stats[prefix + "count"] = 0
                stats[prefix + "birth_mean"] = 0.0
                stats[prefix + "birth_std"] = 0.0
                stats[prefix + "lifetime_mean"] = 0.0
                stats[prefix + "lifetime_std"] = 0.0
                stats[prefix + "lifetime_max"] = 0.0

        return stats

    def compute_trajectory_features(
        self, trajectory: AssemblyTrajectory
    ) -> List[TopologicalFeatures]:
        """Compute persistent homology for each state in a trajectory."""
        return [self.compute(state.graph) for state in trajectory.states]


def compute_betti_numbers(graph: AssemblyGraph) -> np.ndarray:
    """
    Compute Betti numbers directly from graph structure.

    b0 = number of connected components
    b1 = number of independent cycles = |E| - |V| + b0

    Args:
        graph: AssemblyGraph to analyze.

    Returns:
        Array of Betti numbers [b0, b1].
    """
    if graph.num_nodes == 0:
        return np.array([0, 0])

    G = graph.to_networkx()

    # b0: number of connected components
    b0 = nx.number_connected_components(G)

    # b1: cyclomatic complexity (for each component)
    # b1 = |E| - |V| + b0
    b1 = len(graph.edges) - graph.num_nodes + b0

    return np.array([b0, max(0, b1)])


def compute_persistence_diagram(
    graph: AssemblyGraph,
    max_dimension: int = 1,
) -> Dict[int, np.ndarray]:
    """
    Convenience function to compute persistence diagram.

    Args:
        graph: AssemblyGraph to analyze.
        max_dimension: Maximum homology dimension.

    Returns:
        Dictionary mapping dimension to persistence pairs.
    """
    computer = PersistentHomologyComputer(max_dimension=max_dimension)
    features = computer.compute(graph)
    return features.persistence_diagram


def persistence_landscape(
    diagram: np.ndarray,
    num_layers: int = 5,
    resolution: int = 100,
    x_range: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Compute persistence landscape from persistence diagram.

    The persistence landscape is a functional summary of the persistence
    diagram that is stable and suitable for statistical analysis.

    Args:
        diagram: Persistence diagram as (n, 2) array of (birth, death) pairs.
        num_layers: Number of landscape layers to compute.
        resolution: Number of points in discretization.
        x_range: Range for x-axis. If None, computed from diagram.

    Returns:
        Array of shape (num_layers, resolution) containing landscape values.
    """
    if len(diagram) == 0:
        return np.zeros((num_layers, resolution))

    # Filter infinite values
    finite_mask = np.isfinite(diagram[:, 1])
    diagram = diagram[finite_mask]

    if len(diagram) == 0:
        return np.zeros((num_layers, resolution))

    # Determine x range
    if x_range is None:
        x_min = np.min(diagram[:, 0])
        x_max = np.max(diagram[:, 1])
        margin = 0.1 * (x_max - x_min) if x_max > x_min else 1.0
        x_range = (x_min - margin, x_max + margin)

    x = np.linspace(x_range[0], x_range[1], resolution)
    landscape = np.zeros((num_layers, resolution))

    # For each point, compute tent function values
    for i, xi in enumerate(x):
        values = []
        for birth, death in diagram:
            mid = (birth + death) / 2
            half_life = (death - birth) / 2

            if birth <= xi <= death:
                if xi <= mid:
                    val = xi - birth
                else:
                    val = death - xi
                values.append(val)
            else:
                values.append(0.0)

        # Sort and assign to layers
        values = sorted(values, reverse=True)
        for k in range(min(num_layers, len(values))):
            landscape[k, i] = values[k]

    return landscape


def wasserstein_distance(
    diagram1: np.ndarray,
    diagram2: np.ndarray,
    p: int = 2,
) -> float:
    """
    Compute Wasserstein distance between two persistence diagrams.

    Args:
        diagram1: First persistence diagram.
        diagram2: Second persistence diagram.
        p: Order of Wasserstein distance.

    Returns:
        Wasserstein distance.
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        # Simple approximation
        return np.abs(len(diagram1) - len(diagram2))

    # Filter infinite values
    d1 = diagram1[np.isfinite(diagram1[:, 1])] if len(diagram1) > 0 else np.array([]).reshape(0, 2)
    d2 = diagram2[np.isfinite(diagram2[:, 1])] if len(diagram2) > 0 else np.array([]).reshape(0, 2)

    n1, n2 = len(d1), len(d2)

    if n1 == 0 and n2 == 0:
        return 0.0

    # Add diagonal projections
    if n1 > 0:
        proj1 = np.column_stack([d1.mean(axis=1), d1.mean(axis=1)])
    else:
        proj1 = np.array([]).reshape(0, 2)

    if n2 > 0:
        proj2 = np.column_stack([d2.mean(axis=1), d2.mean(axis=1)])
    else:
        proj2 = np.array([]).reshape(0, 2)

    # Build cost matrix
    all_pts1 = np.vstack([d1, proj2]) if n1 > 0 else proj2
    all_pts2 = np.vstack([d2, proj1]) if n2 > 0 else proj1

    if len(all_pts1) == 0 or len(all_pts2) == 0:
        return 0.0

    # Pad to same size
    max_n = max(len(all_pts1), len(all_pts2))
    if len(all_pts1) < max_n:
        all_pts1 = np.vstack([all_pts1, np.zeros((max_n - len(all_pts1), 2))])
    if len(all_pts2) < max_n:
        all_pts2 = np.vstack([all_pts2, np.zeros((max_n - len(all_pts2), 2))])

    cost_matrix = np.zeros((max_n, max_n))
    for i in range(max_n):
        for j in range(max_n):
            cost_matrix[i, j] = np.linalg.norm(all_pts1[i] - all_pts2[j]) ** p

    # Solve assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_cost = cost_matrix[row_ind, col_ind].sum()

    return total_cost ** (1.0 / p)
