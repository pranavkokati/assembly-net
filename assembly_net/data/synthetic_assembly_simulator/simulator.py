"""
Stochastic Coordination Network Simulator.

This module provides a physics-motivated simulator for generating assembly
trajectories of coordination networks. The simulator models:
- Metal-ligand coordination
- pH-dependent binding
- Concentration effects
- Kinetic rates

The simulator generates ground truth labels based on network topology.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import distance_matrix

from assembly_net.data.core import (
    AssemblyEvent,
    AssemblyGraph,
    AssemblyTrajectory,
    EdgeFeatures,
    EdgeType,
    NetworkState,
    NodeFeatures,
    NodeType,
)


class PropertyClass(Enum):
    """Classification labels for emergent material properties."""

    # Mechanical resilience
    BRITTLE = auto()
    DUCTILE = auto()
    ROBUST = auto()

    # Diffusion behavior
    FAST_DIFFUSION = auto()
    SLOW_DIFFUSION = auto()
    BARRIER = auto()

    # Optical properties
    TRANSPARENT = auto()
    NARROW_ABSORPTION = auto()
    BROADBAND_ABSORPTION = auto()

    # Responsiveness
    STABLE = auto()
    PH_RESPONSIVE = auto()
    ION_RESPONSIVE = auto()


@dataclass
class SimulationParameters:
    """Parameters for the coordination network simulation."""

    # System composition
    num_metal_ions: int = 20
    num_ligands: int = 40
    metal_valency: int = 6  # Coordination number of metal
    ligand_valency: int = 2  # Binding sites on ligand

    # Concentrations (relative)
    metal_concentration: float = 1.0
    ligand_concentration: float = 2.0

    # Environmental parameters
    ph: float = 7.0  # pH value (affects binding)
    ionic_strength: float = 0.1  # Ionic strength (M)
    temperature: float = 298.0  # Temperature (K)

    # Kinetic parameters
    base_formation_rate: float = 1.0  # k_on
    base_dissociation_rate: float = 0.01  # k_off
    ph_sensitivity: float = 0.5  # How much pH affects rates

    # Simulation parameters
    dt: float = 0.1  # Time step
    total_time: float = 100.0  # Total simulation time
    snapshot_interval: float = 1.0  # How often to save states

    # Spatial parameters (optional)
    box_size: float = 10.0  # Simulation box size
    use_spatial: bool = False  # Whether to use spatial coordinates

    # Random seed
    seed: Optional[int] = None

    # Additional edge types to include
    include_hydrogen_bonds: bool = False
    include_pi_stacking: bool = False

    def __post_init__(self):
        if self.seed is not None:
            np.random.seed(self.seed)


@dataclass
class PropertyLabels:
    """Ground truth labels computed from network topology."""

    # Classification labels
    mechanical_class: PropertyClass = PropertyClass.DUCTILE
    diffusion_class: PropertyClass = PropertyClass.FAST_DIFFUSION
    optical_class: PropertyClass = PropertyClass.TRANSPARENT
    responsiveness_class: PropertyClass = PropertyClass.STABLE

    # Regression labels
    mechanical_score: float = 0.0  # 0 = brittle, 1 = robust
    diffusion_coefficient: float = 1.0  # Relative diffusion
    absorption_breadth: float = 0.0  # Optical absorption width
    ph_response_magnitude: float = 0.0  # pH responsiveness

    # Topological signatures used to compute labels
    percolation_achieved: bool = False
    loop_density: float = 0.0
    mean_cluster_size: float = 0.0
    rigidity_fraction: float = 0.0


class CoordinationNetworkSimulator:
    """
    Stochastic simulator for coordination network assembly.

    Uses Gillespie-like kinetic Monte Carlo with spatial awareness.
    """

    def __init__(self, params: SimulationParameters):
        self.params = params
        self.rng = np.random.default_rng(params.seed)

        # Initialize state
        self.graph: Optional[AssemblyGraph] = None
        self.time = 0.0
        self.events: List[AssemblyEvent] = []

        # Spatial positions (if enabled)
        self.positions: Optional[np.ndarray] = None

    def initialize(self) -> AssemblyGraph:
        """Initialize the simulation with starting nodes."""
        self.graph = AssemblyGraph(time=0.0)
        self.time = 0.0
        self.events = []

        # Add metal ions
        for i in range(self.params.num_metal_ions):
            features = NodeFeatures(
                node_type=NodeType.METAL_ION,
                valency=self.params.metal_valency,
                charge=2.0,  # Typical for divalent metals
                mass=56.0,  # Iron-like
            )
            self.graph.add_node(features)

        # Add ligands
        for i in range(self.params.num_ligands):
            features = NodeFeatures(
                node_type=NodeType.LIGAND,
                valency=self.params.ligand_valency,
                charge=-1.0,  # Typical for anionic ligands
                mass=150.0,  # Typical small ligand
            )
            self.graph.add_node(features)

        # Initialize spatial positions if enabled
        if self.params.use_spatial:
            self.positions = self.rng.uniform(
                0, self.params.box_size, (self.graph.num_nodes, 3)
            )

        return self.graph

    def _compute_formation_rate(
        self, node_i: int, node_j: int
    ) -> float:
        """
        Compute the rate of bond formation between two nodes.

        Considers:
        - Available coordination sites
        - pH effects
        - Spatial proximity (if enabled)
        - Concentration effects
        """
        nf_i = self.graph.node_features[node_i]
        nf_j = self.graph.node_features[node_j]

        # Check if both have available sites
        if nf_i.current_coordination >= nf_i.valency:
            return 0.0
        if nf_j.current_coordination >= nf_j.valency:
            return 0.0

        # Base rate
        rate = self.params.base_formation_rate

        # Concentration effect (mass action kinetics)
        if nf_i.node_type == NodeType.METAL_ION:
            rate *= self.params.metal_concentration
        if nf_j.node_type == NodeType.LIGAND:
            rate *= self.params.ligand_concentration

        # pH effect: lower pH reduces metal-ligand binding
        # (protonation of ligand binding sites)
        ph_factor = 1.0 / (
            1.0 + 10 ** (self.params.ph_sensitivity * (7.0 - self.params.ph))
        )
        rate *= ph_factor

        # Spatial effect (if enabled)
        if self.params.use_spatial and self.positions is not None:
            dist = np.linalg.norm(
                self.positions[node_i] - self.positions[node_j]
            )
            # Exponential distance dependence
            rate *= np.exp(-dist / (self.params.box_size / 5))

        # Available sites factor
        avail_i = nf_i.valency - nf_i.current_coordination
        avail_j = nf_j.valency - nf_j.current_coordination
        rate *= (avail_i / nf_i.valency) * (avail_j / nf_j.valency)

        return rate

    def _compute_dissociation_rate(
        self, edge_idx: int
    ) -> float:
        """
        Compute the rate of bond dissociation.

        Considers:
        - Bond strength
        - pH effects
        - Temperature
        """
        ef = self.graph.edge_features[edge_idx]

        # Base rate
        rate = self.params.base_dissociation_rate

        # Bond strength effect
        rate /= ef.bond_strength

        # pH effect: extreme pH increases dissociation
        ph_deviation = abs(self.params.ph - 7.0)
        rate *= 1.0 + 0.1 * ph_deviation * ef.ph_sensitivity

        # Temperature effect (Arrhenius-like)
        rate *= np.exp((self.params.temperature - 298.0) / 100.0)

        return rate

    def _get_possible_reactions(
        self,
    ) -> Tuple[List[Tuple[str, Any]], List[float]]:
        """
        Enumerate all possible reactions and their rates.

        Returns:
            Tuple of (reactions, rates) where reactions are tuples of
            (reaction_type, reaction_data).
        """
        reactions = []
        rates = []

        # Formation reactions: metal-ligand pairs
        metals = [
            i
            for i, nf in enumerate(self.graph.node_features)
            if nf.node_type == NodeType.METAL_ION
        ]
        ligands = [
            i
            for i, nf in enumerate(self.graph.node_features)
            if nf.node_type == NodeType.LIGAND
        ]

        # Check existing edges to avoid duplicates
        existing_edges = set()
        for s, t in self.graph.edges:
            existing_edges.add((min(s, t), max(s, t)))

        for m in metals:
            for l in ligands:
                edge_key = (min(m, l), max(m, l))
                if edge_key not in existing_edges:
                    rate = self._compute_formation_rate(m, l)
                    if rate > 0:
                        reactions.append(("formation", (m, l)))
                        rates.append(rate)

        # Dissociation reactions
        for i, (s, t) in enumerate(self.graph.edges):
            rate = self._compute_dissociation_rate(i)
            if rate > 0:
                reactions.append(("dissociation", i))
                rates.append(rate)

        return reactions, rates

    def _execute_reaction(
        self, reaction_type: str, reaction_data: Any
    ) -> Optional[AssemblyEvent]:
        """Execute a reaction and update the graph."""
        if reaction_type == "formation":
            source, target = reaction_data
            # Determine edge type based on node types
            nf_s = self.graph.node_features[source]
            nf_t = self.graph.node_features[target]

            if (
                nf_s.node_type == NodeType.METAL_ION
                or nf_t.node_type == NodeType.METAL_ION
            ):
                edge_type = EdgeType.COORDINATION
            else:
                edge_type = EdgeType.COVALENT

            # Create edge features
            bond_strength = 1.0 + self.rng.exponential(0.5)
            ph_sens = self.rng.uniform(0.2, 0.8)

            ef = EdgeFeatures(
                edge_type=edge_type,
                bond_strength=bond_strength,
                formation_time=self.time,
                ph_sensitivity=ph_sens,
                formation_rate=self._compute_formation_rate(source, target),
            )

            self.graph.add_edge(source, target, ef)

            return AssemblyEvent(
                time=self.time,
                event_type="formation",
                source_node=source,
                target_node=target,
                edge_features=ef,
            )

        elif reaction_type == "dissociation":
            edge_idx = reaction_data
            source, target = self.graph.edges[edge_idx]
            ef = self.graph.edge_features[edge_idx]

            self.graph.remove_edge(source, target)

            return AssemblyEvent(
                time=self.time,
                event_type="dissociation",
                source_node=source,
                target_node=target,
                edge_features=ef,
            )

        return None

    def step(self) -> Optional[AssemblyEvent]:
        """
        Perform one Gillespie step.

        Returns:
            The assembly event that occurred, or None if no reaction possible.
        """
        reactions, rates = self._get_possible_reactions()

        if not reactions or sum(rates) == 0:
            # No reactions possible, advance time
            self.time += self.params.dt
            return None

        # Gillespie algorithm
        total_rate = sum(rates)

        # Time to next reaction
        tau = self.rng.exponential(1.0 / total_rate)
        self.time += tau

        # Choose reaction
        probs = np.array(rates) / total_rate
        reaction_idx = self.rng.choice(len(reactions), p=probs)
        reaction_type, reaction_data = reactions[reaction_idx]

        # Execute reaction
        event = self._execute_reaction(reaction_type, reaction_data)
        if event:
            self.events.append(event)

        return event

    def run(self) -> AssemblyTrajectory:
        """
        Run the full simulation.

        Returns:
            AssemblyTrajectory containing all states and events.
        """
        self.initialize()

        trajectory = AssemblyTrajectory(metadata={"params": self.params.__dict__})

        # Save initial state
        initial_state = NetworkState(
            graph=self.graph.copy(), time=self.time
        )
        trajectory.add_state(initial_state)

        last_snapshot = 0.0

        while self.time < self.params.total_time:
            event = self.step()

            if event:
                trajectory.add_event(event)

            # Save snapshot at intervals
            if self.time - last_snapshot >= self.params.snapshot_interval:
                state = NetworkState(
                    graph=self.graph.copy(), time=self.time
                )
                trajectory.add_state(state)
                last_snapshot = self.time

        # Compute properties and labels
        trajectory.compute_all_properties()
        labels = self._compute_labels(trajectory)
        trajectory.labels = labels.__dict__

        return trajectory

    def _compute_labels(self, trajectory: AssemblyTrajectory) -> PropertyLabels:
        """
        Compute ground truth labels from the assembly trajectory.

        Uses physics-motivated rules:
        - Percolation -> gel-like behavior
        - High loop density -> mechanical robustness
        - Sparse tree-like -> brittle behavior
        """
        labels = PropertyLabels()

        final_state = trajectory.final_state
        if final_state is None:
            return labels

        # Ensure properties are computed
        if final_state.num_cycles is None:
            final_state.compute_properties()

        # Percolation
        labels.percolation_achieved = final_state.is_percolated or False

        # Loop density: cycles per node
        num_nodes = final_state.graph.num_nodes
        num_cycles = final_state.num_cycles or 0
        labels.loop_density = num_cycles / max(1, num_nodes)

        # Mean cluster size
        labels.mean_cluster_size = (final_state.largest_component_size or 0) / max(
            1, num_nodes
        )

        # Compute rigidity (simplified)
        num_edges = len(final_state.graph.edges)
        # Maxwell counting: rigid if edges >= 2*nodes - 3 (2D) or 3*nodes - 6 (3D)
        rigidity_threshold = 2 * num_nodes - 3
        labels.rigidity_fraction = min(1.0, num_edges / max(1, rigidity_threshold))

        # --- Mechanical classification ---
        if labels.loop_density > 0.3 and labels.percolation_achieved:
            labels.mechanical_class = PropertyClass.ROBUST
            labels.mechanical_score = 0.8 + 0.2 * min(1.0, labels.loop_density)
        elif labels.percolation_achieved:
            labels.mechanical_class = PropertyClass.DUCTILE
            labels.mechanical_score = 0.4 + 0.3 * labels.rigidity_fraction
        else:
            labels.mechanical_class = PropertyClass.BRITTLE
            labels.mechanical_score = 0.2 * labels.mean_cluster_size

        # --- Diffusion classification ---
        # Dense, percolated networks act as barriers
        if labels.percolation_achieved and (final_state.density or 0) > 0.3:
            labels.diffusion_class = PropertyClass.BARRIER
            labels.diffusion_coefficient = 0.1 * (1.0 - (final_state.density or 0))
        elif labels.percolation_achieved:
            labels.diffusion_class = PropertyClass.SLOW_DIFFUSION
            labels.diffusion_coefficient = 0.5 * (1.0 - labels.mean_cluster_size)
        else:
            labels.diffusion_class = PropertyClass.FAST_DIFFUSION
            labels.diffusion_coefficient = 1.0 - 0.3 * labels.mean_cluster_size

        # --- Optical classification ---
        # Based on network topology and metal content
        metal_fraction = sum(
            1
            for nf in final_state.graph.node_features
            if nf.node_type == NodeType.METAL_ION
        ) / max(1, num_nodes)

        coordination_variance = np.var(
            [nf.current_coordination for nf in final_state.graph.node_features]
        )

        if metal_fraction < 0.2:
            labels.optical_class = PropertyClass.TRANSPARENT
            labels.absorption_breadth = 0.1
        elif coordination_variance > 2.0:
            labels.optical_class = PropertyClass.BROADBAND_ABSORPTION
            labels.absorption_breadth = 0.8 + 0.2 * min(1.0, coordination_variance / 4)
        else:
            labels.optical_class = PropertyClass.NARROW_ABSORPTION
            labels.absorption_breadth = 0.3 + 0.2 * metal_fraction

        # --- Responsiveness classification ---
        # Based on pH sensitivity of bonds
        if final_state.graph.edge_features:
            mean_ph_sens = np.mean(
                [ef.ph_sensitivity for ef in final_state.graph.edge_features]
            )
        else:
            mean_ph_sens = 0.0

        labels.ph_response_magnitude = mean_ph_sens

        if mean_ph_sens > 0.6:
            labels.responsiveness_class = PropertyClass.PH_RESPONSIVE
        elif self.params.ionic_strength > 0.5:
            labels.responsiveness_class = PropertyClass.ION_RESPONSIVE
        else:
            labels.responsiveness_class = PropertyClass.STABLE

        return labels


def generate_dataset(
    num_samples: int,
    params_generator: Optional[Callable[[], SimulationParameters]] = None,
    seed: int = 42,
) -> List[AssemblyTrajectory]:
    """
    Generate a dataset of assembly trajectories.

    Args:
        num_samples: Number of trajectories to generate.
        params_generator: Optional function to generate varied parameters.
        seed: Random seed for reproducibility.

    Returns:
        List of AssemblyTrajectory objects.
    """
    rng = np.random.default_rng(seed)
    trajectories = []

    for i in range(num_samples):
        if params_generator is not None:
            params = params_generator()
        else:
            # Default varied parameters
            params = SimulationParameters(
                num_metal_ions=rng.integers(10, 50),
                num_ligands=rng.integers(20, 100),
                metal_valency=rng.choice([4, 6, 8]),
                ligand_valency=rng.choice([1, 2, 3]),
                ph=rng.uniform(4.0, 10.0),
                ionic_strength=rng.uniform(0.01, 1.0),
                base_formation_rate=rng.uniform(0.5, 2.0),
                base_dissociation_rate=rng.uniform(0.001, 0.05),
                total_time=100.0,
                snapshot_interval=1.0,
                seed=int(rng.integers(0, 2**31)),
            )

        simulator = CoordinationNetworkSimulator(params)
        trajectory = simulator.run()
        trajectories.append(trajectory)

    return trajectories


def generate_paired_dataset(
    num_pairs: int,
    seed: int = 42,
) -> List[Tuple[AssemblyTrajectory, AssemblyTrajectory]]:
    """
    Generate pairs of trajectories with same final structure but different
    assembly histories.

    This is crucial for testing whether assembly history matters.

    Args:
        num_pairs: Number of pairs to generate.
        seed: Random seed.

    Returns:
        List of (trajectory_1, trajectory_2) tuples.
    """
    rng = np.random.default_rng(seed)
    pairs = []

    for i in range(num_pairs):
        # Generate base parameters
        base_params = SimulationParameters(
            num_metal_ions=rng.integers(15, 30),
            num_ligands=rng.integers(30, 60),
            ph=7.0,  # Neutral pH for both
            total_time=100.0,
            seed=int(rng.integers(0, 2**31)),
        )

        # Fast assembly (high rates)
        params_fast = copy.deepcopy(base_params)
        params_fast.base_formation_rate = 2.0
        params_fast.base_dissociation_rate = 0.001
        params_fast.seed = base_params.seed

        # Slow assembly (low rates)
        params_slow = copy.deepcopy(base_params)
        params_slow.base_formation_rate = 0.5
        params_slow.base_dissociation_rate = 0.01
        params_slow.total_time = 200.0  # More time for slow assembly
        params_slow.seed = base_params.seed + 1

        # Run simulations
        sim_fast = CoordinationNetworkSimulator(params_fast)
        traj_fast = sim_fast.run()

        sim_slow = CoordinationNetworkSimulator(params_slow)
        traj_slow = sim_slow.run()

        pairs.append((traj_fast, traj_slow))

    return pairs
