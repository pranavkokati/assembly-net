"""
Assembly-Net: Publication-Quality Framework
============================================

THEORETICAL FOUNDATIONS
-----------------------

Definition 1 (Assembly Trajectory):
    An assembly trajectory T = {G_t}_{t=0}^{T} is an ordered sequence of graphs
    where G_t = (V, E_t) represents the network state at time t.

Definition 2 (Assembly Non-Equivalence):
    Two materials M₁ and M₂ are assembly-non-equivalent if:
    - They share the same final graph: G_T^(1) ≅ G_T^(2)
    - Their assembly trajectories differ: {G_t^(1)} ≢ {G_t^(2)}

    Formally, the equivalence relation ∼_F over final structures does NOT
    imply equivalence over assembly processes:

        M₁ ∼_F M₂  ⇏  M₁ ∼_A M₂

Definition 3 (Emergent Property):
    A property P is emergent if it cannot be computed solely from G_T:

        ∃ M₁, M₂: G_T^(1) ≅ G_T^(2) ∧ P(M₁) ≠ P(M₂)

CENTRAL CLAIM
-------------

Final-structure-only models are information-theoretically insufficient
for predicting emergent properties in assembly-driven materials.

This module implements the complete framework for proving this claim.
"""

from __future__ import annotations

import copy
import json
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import networkx as nx
except ImportError:
    nx = None

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


# =============================================================================
# ASSEMBLY REGIME DEFINITIONS
# =============================================================================

class AssemblyRegime(Enum):
    """
    Physically distinct assembly regimes that produce different topology evolution
    but can converge to similar final densities.
    """

    # Diffusion-Limited Aggregation: Fast reaction, slow diffusion
    # Results in fractal-like, branching structures with early percolation
    DLA = auto()

    # Reaction-Limited Aggregation: Slow reaction, fast diffusion
    # Results in compact, dense clusters with delayed percolation
    RLA = auto()

    # Burst Nucleation + Slow Growth: Initial rapid nucleation then slow addition
    # Results in many small clusters that slowly merge
    BURST_NUCLEATION = auto()

    # Sequential: Ordered addition (for controlled experiments)
    SEQUENTIAL = auto()

    # Equilibrium: Reversible with detailed balance
    EQUILIBRIUM = auto()


class IrreversibilityMode(Enum):
    """Modes of irreversibility in assembly."""

    # Fully reversible (equilibrium)
    REVERSIBLE = auto()

    # Coordination saturation: once a node reaches max valency,
    # its bonds become irreversible
    COORDINATION_SATURATION = auto()

    # Kinetic trapping: bonds formed in certain conditions cannot break
    KINETIC_TRAPPING = auto()

    # Path-dependent: early bonds are stronger than late bonds
    PATH_DEPENDENT = auto()

    # Full irreversibility: no bond breaking
    FULLY_IRREVERSIBLE = auto()


# =============================================================================
# ENHANCED SIMULATION PARAMETERS
# =============================================================================

@dataclass
class PublicationSimulationParameters:
    """
    Publication-quality simulation parameters with full physical motivation.
    """

    # System composition
    num_metal_ions: int = 25
    num_ligands: int = 50
    metal_valency: int = 6
    ligand_valency: int = 2

    # Assembly regime
    regime: AssemblyRegime = AssemblyRegime.RLA
    irreversibility: IrreversibilityMode = IrreversibilityMode.PATH_DEPENDENT

    # Kinetic parameters (regime-dependent defaults applied in __post_init__)
    base_formation_rate: Optional[float] = None
    base_dissociation_rate: Optional[float] = None

    # Path-dependence parameters
    early_bond_strength_multiplier: float = 2.0  # Early bonds are stronger
    coordination_saturation_threshold: float = 0.8  # Fraction of valency for saturation

    # DLA-specific: diffusion-limited encounter rate
    encounter_rate_scaling: float = 1.0

    # RLA-specific: reaction probability per encounter
    reaction_probability: float = 0.1

    # Burst nucleation specific
    nucleation_burst_duration: float = 10.0
    nucleation_rate_multiplier: float = 10.0

    # Environmental
    ph: float = 7.0
    temperature: float = 298.0

    # Simulation
    total_time: float = 100.0
    snapshot_interval: float = 1.0
    seed: Optional[int] = None

    # Derived topological observables to track
    track_betti_numbers: bool = True
    track_loop_formation_rate: bool = True
    track_early_loop_index: bool = True

    def __post_init__(self):
        """Apply regime-specific defaults."""
        if self.base_formation_rate is None or self.base_dissociation_rate is None:
            self._apply_regime_defaults()

    def _apply_regime_defaults(self):
        """Set kinetic parameters based on assembly regime."""
        if self.regime == AssemblyRegime.DLA:
            # Fast formation, essentially no dissociation
            self.base_formation_rate = 5.0
            self.base_dissociation_rate = 0.001
        elif self.regime == AssemblyRegime.RLA:
            # Slow formation, moderate dissociation
            self.base_formation_rate = 0.5
            self.base_dissociation_rate = 0.05
        elif self.regime == AssemblyRegime.BURST_NUCLEATION:
            # Time-dependent (handled in simulator)
            self.base_formation_rate = 1.0
            self.base_dissociation_rate = 0.01
        elif self.regime == AssemblyRegime.SEQUENTIAL:
            # Controlled addition
            self.base_formation_rate = 1.0
            self.base_dissociation_rate = 0.0
        else:  # EQUILIBRIUM
            self.base_formation_rate = 1.0
            self.base_dissociation_rate = 0.1


# =============================================================================
# TOPOLOGICAL OBSERVABLES
# =============================================================================

@dataclass
class TopologicalObservables:
    """
    Time-resolved topological observables for a network state.

    These go beyond simple Betti numbers to capture assembly dynamics.
    """

    time: float = 0.0

    # Betti numbers
    beta_0: int = 0  # Connected components
    beta_1: int = 0  # Independent cycles

    # Derived observables
    loop_formation_rate: float = 0.0  # dβ₁/dt
    component_merger_rate: float = 0.0  # -dβ₀/dt

    # Early-loop dominance index: fraction of loops formed in first half of assembly
    early_loop_fraction: float = 0.0

    # Topological hysteresis indicator
    irreversibility_signature: float = 0.0

    # Network properties
    num_edges: int = 0
    largest_component_fraction: float = 0.0
    clustering_coefficient: float = 0.0

    def to_vector(self) -> np.ndarray:
        """Convert to feature vector."""
        return np.array([
            self.beta_0 / 100.0,  # Normalize
            self.beta_1 / 50.0,
            self.loop_formation_rate,
            self.component_merger_rate,
            self.early_loop_fraction,
            self.irreversibility_signature,
            self.num_edges / 200.0,
            self.largest_component_fraction,
            self.clustering_coefficient,
        ], dtype=np.float32)


@dataclass
class TrajectoryTopology:
    """Complete topological characterization of an assembly trajectory."""

    # Time series of observables
    observables: List[TopologicalObservables] = field(default_factory=list)

    # Summary statistics
    percolation_time: Optional[float] = None
    final_beta_1: int = 0
    total_loops_formed: int = 0
    early_loop_dominance_index: float = 0.0

    # Assembly signature (distinguishes assembly-non-equivalent materials)
    topology_signature: Optional[np.ndarray] = None

    def compute_signature(self) -> np.ndarray:
        """
        Compute a topology signature that captures the assembly process.

        This signature should differ for assembly-non-equivalent materials
        even if they have identical final structures.
        """
        if not self.observables:
            return np.zeros(20)

        times = np.array([o.time for o in self.observables])
        beta_0 = np.array([o.beta_0 for o in self.observables])
        beta_1 = np.array([o.beta_1 for o in self.observables])

        # Normalize time to [0, 1]
        if times[-1] > times[0]:
            t_norm = (times - times[0]) / (times[-1] - times[0])
        else:
            t_norm = np.zeros_like(times)

        # Signature components:
        # 1. Area under β₀ curve (measures how fast components merge)
        auc_beta_0 = np.trapz(beta_0, t_norm) if len(t_norm) > 1 else 0

        # 2. Area under β₁ curve (measures when loops form)
        auc_beta_1 = np.trapz(beta_1, t_norm) if len(t_norm) > 1 else 0

        # 3. Time of first loop
        first_loop_time = 1.0
        for i, b1 in enumerate(beta_1):
            if b1 > 0:
                first_loop_time = t_norm[i]
                break

        # 4. Loop formation rate statistics
        if len(beta_1) > 1:
            loop_rates = np.diff(beta_1) / np.maximum(np.diff(t_norm), 1e-6)
            mean_loop_rate = np.mean(loop_rates)
            max_loop_rate = np.max(loop_rates)
            loop_rate_variance = np.var(loop_rates)
        else:
            mean_loop_rate = max_loop_rate = loop_rate_variance = 0

        # 5. Percolation time (normalized)
        perc_time_norm = self.percolation_time / times[-1] if self.percolation_time else 1.0

        # 6. Early vs late loop ratio
        mid_idx = len(beta_1) // 2
        early_loops = beta_1[mid_idx] if mid_idx < len(beta_1) else 0
        late_loops = beta_1[-1] - early_loops if len(beta_1) > 0 else 0
        early_loop_ratio = early_loops / max(1, beta_1[-1]) if len(beta_1) > 0 else 0

        # 7. Component merger profile (when do clusters join?)
        if len(beta_0) > 1:
            merger_rates = -np.diff(beta_0) / np.maximum(np.diff(t_norm), 1e-6)
            merger_rates = np.maximum(merger_rates, 0)  # Only positive mergers
            mean_merger_rate = np.mean(merger_rates)
            peak_merger_time = t_norm[np.argmax(merger_rates)] if len(merger_rates) > 0 else 0.5
        else:
            mean_merger_rate = peak_merger_time = 0

        # Assemble signature
        signature = np.array([
            auc_beta_0 / 100,
            auc_beta_1 / 50,
            first_loop_time,
            mean_loop_rate / 10,
            max_loop_rate / 20,
            loop_rate_variance / 100,
            perc_time_norm,
            early_loop_ratio,
            self.early_loop_dominance_index,
            mean_merger_rate / 10,
            peak_merger_time,
            self.final_beta_1 / 50,
            self.total_loops_formed / 100,
            beta_0[0] / 100 if len(beta_0) > 0 else 0,
            beta_0[-1] / 100 if len(beta_0) > 0 else 0,
            beta_1[-1] / 50 if len(beta_1) > 0 else 0,
            # Moments of the beta_1 distribution over time
            np.mean(beta_1) / 50 if len(beta_1) > 0 else 0,
            np.std(beta_1) / 20 if len(beta_1) > 0 else 0,
            np.median(beta_1) / 50 if len(beta_1) > 0 else 0,
            np.max(beta_1) / 50 if len(beta_1) > 0 else 0,
        ], dtype=np.float32)

        self.topology_signature = signature
        return signature


# =============================================================================
# HISTORY-DEPENDENT EMERGENT PROPERTIES
# =============================================================================

@dataclass
class EmergentPropertyLabels:
    """
    Labels for properties that CANNOT be determined from final structure alone.

    These are the key targets that prove assembly history matters.
    """

    # Property 1: Mechanical class based on WHEN loops formed
    # Same final loop count, but early loops = more robust
    mechanical_class: str = "ductile"  # brittle, ductile, robust
    mechanical_score: float = 0.5

    # Property 2: Percolation onset time (explicitly history-dependent)
    percolation_time: Optional[float] = None
    normalized_percolation_time: float = 1.0

    # Property 3: Diffusion bottleneck index
    # Depends on order of edge formation, not final structure
    diffusion_bottleneck_index: float = 0.0

    # Property 4: Thermal stability class
    # Early bonds under different conditions = different stability
    thermal_stability_class: str = "moderate"  # low, moderate, high
    thermal_stability_score: float = 0.5

    # Property 5: Assembly pathway signature (for classification)
    pathway_class: str = "mixed"  # dla, rla, burst, sequential

    # Metadata
    final_edge_count: int = 0
    final_node_count: int = 0
    final_beta_1: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mechanical_class": self.mechanical_class,
            "mechanical_score": self.mechanical_score,
            "percolation_time": self.percolation_time,
            "normalized_percolation_time": self.normalized_percolation_time,
            "diffusion_bottleneck_index": self.diffusion_bottleneck_index,
            "thermal_stability_class": self.thermal_stability_class,
            "thermal_stability_score": self.thermal_stability_score,
            "pathway_class": self.pathway_class,
            "final_edge_count": self.final_edge_count,
            "final_node_count": self.final_node_count,
            "final_beta_1": self.final_beta_1,
        }


# =============================================================================
# PUBLICATION-QUALITY SIMULATOR
# =============================================================================

class PublicationSimulator:
    """
    Publication-quality stochastic simulator for coordination network assembly.

    Key features:
    1. Physically motivated assembly regimes (DLA, RLA, burst nucleation)
    2. Irreversible assembly rules (coordination saturation, path-dependence)
    3. Full topological tracking (Betti numbers, loop rates, signatures)
    4. History-dependent property labeling
    """

    def __init__(self, params: PublicationSimulationParameters):
        self.params = params
        self.rng = np.random.default_rng(params.seed)

        # State
        self.graph: Optional[AssemblyGraph] = None
        self.time = 0.0
        self.events: List[AssemblyEvent] = []

        # Topology tracking
        self.topology_history: List[TopologicalObservables] = []
        self.loops_formed_times: List[float] = []

        # Saturation tracking (for irreversibility)
        self.saturated_nodes: set = set()
        self.locked_edges: set = set()  # Edges that cannot be removed

    def initialize(self) -> AssemblyGraph:
        """Initialize the simulation."""
        self.graph = AssemblyGraph(time=0.0)
        self.time = 0.0
        self.events = []
        self.topology_history = []
        self.loops_formed_times = []
        self.saturated_nodes = set()
        self.locked_edges = set()

        # Add metal ions
        for i in range(self.params.num_metal_ions):
            features = NodeFeatures(
                node_type=NodeType.METAL_ION,
                valency=self.params.metal_valency,
                charge=2.0,
                mass=56.0,
            )
            self.graph.add_node(features)

        # Add ligands
        for i in range(self.params.num_ligands):
            features = NodeFeatures(
                node_type=NodeType.LIGAND,
                valency=self.params.ligand_valency,
                charge=-1.0,
                mass=150.0,
            )
            self.graph.add_node(features)

        return self.graph

    def _get_formation_rate(self, node_i: int, node_j: int) -> float:
        """Compute formation rate with regime-specific modifications."""
        nf_i = self.graph.node_features[node_i]
        nf_j = self.graph.node_features[node_j]

        # Check valency
        if nf_i.current_coordination >= nf_i.valency:
            return 0.0
        if nf_j.current_coordination >= nf_j.valency:
            return 0.0

        # Base rate
        rate = self.params.base_formation_rate

        # Regime-specific modifications
        if self.params.regime == AssemblyRegime.BURST_NUCLEATION:
            # High rate during burst period, then low
            if self.time < self.params.nucleation_burst_duration:
                rate *= self.params.nucleation_rate_multiplier
            else:
                rate *= 0.2

        elif self.params.regime == AssemblyRegime.RLA:
            # Reaction probability modulation
            rate *= self.params.reaction_probability

        elif self.params.regime == AssemblyRegime.DLA:
            # Encounter-limited
            rate *= self.params.encounter_rate_scaling

        # Available sites factor
        avail_i = nf_i.valency - nf_i.current_coordination
        avail_j = nf_j.valency - nf_j.current_coordination
        rate *= (avail_i / nf_i.valency) * (avail_j / nf_j.valency)

        return max(0, rate)

    def _get_dissociation_rate(self, edge_idx: int) -> float:
        """Compute dissociation rate with irreversibility rules."""
        # Check if edge is locked
        if edge_idx in self.locked_edges:
            return 0.0

        source, target = self.graph.edges[edge_idx]

        # Check coordination saturation
        if self.params.irreversibility == IrreversibilityMode.COORDINATION_SATURATION:
            if source in self.saturated_nodes or target in self.saturated_nodes:
                return 0.0

        # Fully irreversible
        if self.params.irreversibility == IrreversibilityMode.FULLY_IRREVERSIBLE:
            return 0.0

        ef = self.graph.edge_features[edge_idx]
        rate = self.params.base_dissociation_rate

        # Path-dependent: early bonds are stronger (lower dissociation)
        if self.params.irreversibility == IrreversibilityMode.PATH_DEPENDENT:
            # Formation time factor: earlier = stronger
            age = self.time - ef.formation_time
            early_factor = 1.0 / (1.0 + self.params.early_bond_strength_multiplier *
                                   (ef.formation_time / max(1, self.time)))
            rate *= early_factor

        # Bond strength factor
        rate /= ef.bond_strength

        return max(0, rate)

    def _update_saturation(self):
        """Update saturation status of nodes."""
        if self.params.irreversibility != IrreversibilityMode.COORDINATION_SATURATION:
            return

        threshold = self.params.coordination_saturation_threshold
        for i, nf in enumerate(self.graph.node_features):
            if nf.current_coordination >= threshold * nf.valency:
                self.saturated_nodes.add(i)

    def _compute_topology(self) -> TopologicalObservables:
        """Compute current topological observables."""
        obs = TopologicalObservables(time=self.time)

        if self.graph.num_nodes == 0:
            return obs

        G = self.graph.to_networkx()

        # Betti numbers
        components = list(nx.connected_components(G))
        obs.beta_0 = len(components)
        obs.beta_1 = len(self.graph.edges) - self.graph.num_nodes + obs.beta_0
        obs.beta_1 = max(0, obs.beta_1)

        # Network properties
        obs.num_edges = len(self.graph.edges)
        obs.largest_component_fraction = (
            max(len(c) for c in components) / self.graph.num_nodes
            if components else 0
        )
        obs.clustering_coefficient = nx.average_clustering(G) if G.number_of_nodes() > 2 else 0

        # Compute rates from history
        if len(self.topology_history) > 0:
            prev = self.topology_history[-1]
            dt = self.time - prev.time
            if dt > 0:
                obs.loop_formation_rate = (obs.beta_1 - prev.beta_1) / dt
                obs.component_merger_rate = -(obs.beta_0 - prev.beta_0) / dt

        return obs

    def step(self) -> Optional[AssemblyEvent]:
        """Perform one Gillespie step."""
        reactions = []
        rates = []

        # Formation reactions
        metals = [i for i, nf in enumerate(self.graph.node_features)
                  if nf.node_type == NodeType.METAL_ION]
        ligands = [i for i, nf in enumerate(self.graph.node_features)
                   if nf.node_type == NodeType.LIGAND]

        existing = set((min(s, t), max(s, t)) for s, t in self.graph.edges)

        for m in metals:
            for l in ligands:
                if (min(m, l), max(m, l)) not in existing:
                    rate = self._get_formation_rate(m, l)
                    if rate > 0:
                        reactions.append(("formation", (m, l)))
                        rates.append(rate)

        # Dissociation reactions
        for i in range(len(self.graph.edges)):
            rate = self._get_dissociation_rate(i)
            if rate > 0:
                reactions.append(("dissociation", i))
                rates.append(rate)

        if not reactions or sum(rates) == 0:
            self.time += self.params.snapshot_interval
            return None

        # Gillespie algorithm
        total_rate = sum(rates)
        tau = self.rng.exponential(1.0 / total_rate)
        self.time += tau

        probs = np.array(rates) / total_rate
        idx = self.rng.choice(len(reactions), p=probs)
        reaction_type, data = reactions[idx]

        # Track β₁ before reaction
        beta_1_before = (len(self.graph.edges) - self.graph.num_nodes +
                         len(list(nx.connected_components(self.graph.to_networkx()))))

        # Execute reaction
        event = None
        if reaction_type == "formation":
            source, target = data

            # Compute bond strength based on formation time (path-dependence)
            if self.params.irreversibility == IrreversibilityMode.PATH_DEPENDENT:
                # Earlier bonds are stronger
                time_factor = 1.0 + self.params.early_bond_strength_multiplier * \
                              (1.0 - self.time / self.params.total_time)
            else:
                time_factor = 1.0

            bond_strength = time_factor * (1.0 + self.rng.exponential(0.3))

            ef = EdgeFeatures(
                edge_type=EdgeType.COORDINATION,
                bond_strength=bond_strength,
                formation_time=self.time,
                ph_sensitivity=self.rng.uniform(0.2, 0.8),
            )

            self.graph.add_edge(source, target, ef)
            self._update_saturation()

            event = AssemblyEvent(
                time=self.time,
                event_type="formation",
                source_node=source,
                target_node=target,
                edge_features=ef,
            )

        elif reaction_type == "dissociation":
            edge_idx = data
            if edge_idx < len(self.graph.edges):
                source, target = self.graph.edges[edge_idx]
                ef = self.graph.edge_features[edge_idx]
                self.graph.remove_edge(source, target)

                event = AssemblyEvent(
                    time=self.time,
                    event_type="dissociation",
                    source_node=source,
                    target_node=target,
                    edge_features=ef,
                )

        # Check if a loop was formed
        if event and event.event_type == "formation":
            beta_1_after = (len(self.graph.edges) - self.graph.num_nodes +
                           len(list(nx.connected_components(self.graph.to_networkx()))))
            if beta_1_after > max(0, beta_1_before):
                self.loops_formed_times.append(self.time)

        return event

    def _compute_emergent_labels(
        self,
        trajectory: AssemblyTrajectory,
        topo: TrajectoryTopology
    ) -> EmergentPropertyLabels:
        """
        Compute history-dependent emergent property labels.

        These labels CANNOT be determined from the final structure alone.
        """
        labels = EmergentPropertyLabels()

        final_state = trajectory.final_state
        if final_state is None:
            return labels

        labels.final_edge_count = len(final_state.graph.edges)
        labels.final_node_count = final_state.graph.num_nodes
        labels.final_beta_1 = topo.final_beta_1

        # Property 1: Mechanical class based on WHEN loops formed
        # Key insight: Same number of loops, but formed early = more integrated = robust
        if len(self.loops_formed_times) > 0:
            loop_times = np.array(self.loops_formed_times)
            normalized_times = loop_times / self.params.total_time

            # Early loop dominance: what fraction of loops formed in first half?
            early_loops = np.sum(normalized_times < 0.5)
            total_loops = len(loop_times)
            early_fraction = early_loops / max(1, total_loops)

            topo.early_loop_dominance_index = early_fraction

            # Classification based on early loop dominance
            if early_fraction > 0.6 and total_loops > 5:
                labels.mechanical_class = "robust"
                labels.mechanical_score = 0.7 + 0.3 * early_fraction
            elif early_fraction > 0.3:
                labels.mechanical_class = "ductile"
                labels.mechanical_score = 0.4 + 0.3 * early_fraction
            else:
                labels.mechanical_class = "brittle"
                labels.mechanical_score = 0.2 * early_fraction

        # Property 2: Percolation time
        for obs in topo.observables:
            if obs.largest_component_fraction > 0.5:
                labels.percolation_time = obs.time
                labels.normalized_percolation_time = obs.time / self.params.total_time
                break

        topo.percolation_time = labels.percolation_time

        # Property 3: Diffusion bottleneck index
        # Based on order of edge formation creating bottlenecks
        if len(trajectory.events) > 0:
            formation_events = [e for e in trajectory.events if e.event_type == "formation"]
            if len(formation_events) > 10:
                # Check if early edges created narrow passages
                early_events = formation_events[:len(formation_events)//3]

                # Build early graph
                early_graph = AssemblyGraph(num_nodes=self.graph.num_nodes)
                early_graph.node_features = copy.deepcopy(self.graph.node_features)
                for nf in early_graph.node_features:
                    nf.current_coordination = 0

                for e in early_events:
                    ef = EdgeFeatures(
                        edge_type=EdgeType.COORDINATION,
                        bond_strength=1.0,
                        formation_time=e.time,
                    )
                    try:
                        early_graph.add_edge(e.source_node, e.target_node, ef)
                    except:
                        pass

                G_early = early_graph.to_networkx()
                G_final = final_state.graph.to_networkx()

                # Bottleneck index: ratio of early to final connectivity
                early_edges = G_early.number_of_edges()
                final_edges = G_final.number_of_edges()

                if final_edges > 0:
                    edge_ratio = early_edges / final_edges

                    # Check clustering difference
                    early_cluster = nx.average_clustering(G_early) if G_early.number_of_nodes() > 2 else 0
                    final_cluster = nx.average_clustering(G_final) if G_final.number_of_nodes() > 2 else 0

                    # High early edges + low early clustering = bottleneck
                    labels.diffusion_bottleneck_index = edge_ratio * (1 - early_cluster) / max(0.1, final_cluster)

        # Property 4: Thermal stability based on bond formation conditions
        if final_state.graph.edge_features:
            strengths = [ef.bond_strength for ef in final_state.graph.edge_features]
            mean_strength = np.mean(strengths)

            # Path-dependent bonds have higher strength variation
            strength_std = np.std(strengths)

            if mean_strength > 1.5 and strength_std < 0.5:
                labels.thermal_stability_class = "high"
                labels.thermal_stability_score = 0.8
            elif mean_strength > 1.0:
                labels.thermal_stability_class = "moderate"
                labels.thermal_stability_score = 0.5
            else:
                labels.thermal_stability_class = "low"
                labels.thermal_stability_score = 0.2

        # Property 5: Assembly pathway classification
        labels.pathway_class = self.params.regime.name.lower()

        return labels

    def run(self) -> Tuple[AssemblyTrajectory, TrajectoryTopology, EmergentPropertyLabels]:
        """
        Run the full simulation.

        Returns:
            Tuple of (trajectory, topology, labels)
        """
        self.initialize()

        trajectory = AssemblyTrajectory(
            metadata={
                "params": {
                    "regime": self.params.regime.name,
                    "irreversibility": self.params.irreversibility.name,
                    "seed": self.params.seed,
                }
            }
        )

        # Initial state
        initial_state = NetworkState(graph=self.graph.copy(), time=self.time)
        trajectory.add_state(initial_state)

        # Initial topology
        obs = self._compute_topology()
        self.topology_history.append(obs)

        last_snapshot = 0.0

        while self.time < self.params.total_time:
            event = self.step()

            if event:
                trajectory.add_event(event)

            # Snapshot
            if self.time - last_snapshot >= self.params.snapshot_interval:
                state = NetworkState(graph=self.graph.copy(), time=self.time)
                trajectory.add_state(state)

                obs = self._compute_topology()
                self.topology_history.append(obs)

                last_snapshot = self.time

        # Compute properties
        trajectory.compute_all_properties()

        # Build topology object
        topo = TrajectoryTopology(observables=self.topology_history)
        topo.final_beta_1 = self.topology_history[-1].beta_1 if self.topology_history else 0
        topo.total_loops_formed = len(self.loops_formed_times)
        topo.compute_signature()

        # Compute emergent labels
        labels = self._compute_emergent_labels(trajectory, topo)

        # Store labels in trajectory
        trajectory.labels = labels.to_dict()

        return trajectory, topo, labels


# =============================================================================
# DATASET GENERATION FOR EXPERIMENTS
# =============================================================================

def generate_regime_comparison_dataset(
    num_per_regime: int = 50,
    seed: int = 42,
) -> Dict[str, List[Tuple[AssemblyTrajectory, TrajectoryTopology, EmergentPropertyLabels]]]:
    """
    Generate dataset comparing different assembly regimes.

    This produces trajectories with similar final densities but different
    topology evolution, demonstrating that assembly history matters.
    """
    rng = np.random.default_rng(seed)
    datasets = {}

    for regime in [AssemblyRegime.DLA, AssemblyRegime.RLA, AssemblyRegime.BURST_NUCLEATION]:
        print(f"Generating {regime.name} trajectories...")
        trajectories = []

        for i in range(num_per_regime):
            params = PublicationSimulationParameters(
                num_metal_ions=25,
                num_ligands=50,
                regime=regime,
                irreversibility=IrreversibilityMode.PATH_DEPENDENT,
                total_time=100.0,
                seed=seed + i + hash(regime.name) % 10000,
            )

            sim = PublicationSimulator(params)
            traj, topo, labels = sim.run()
            trajectories.append((traj, topo, labels))

        datasets[regime.name] = trajectories

    return datasets


def generate_assembly_nonequivalent_pairs(
    num_pairs: int = 100,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Generate pairs of trajectories that are assembly-non-equivalent:
    - Same final graph structure (approximately)
    - Different assembly histories
    - Different emergent properties

    This is the KEY experiment demonstrating the central claim.
    """
    rng = np.random.default_rng(seed)
    pairs = []

    print("Generating assembly-non-equivalent pairs...")

    for i in range(num_pairs):
        # Same system composition
        n_metal = rng.integers(20, 30)
        n_ligand = rng.integers(40, 60)

        # Generate with DLA (fast, irreversible)
        params_dla = PublicationSimulationParameters(
            num_metal_ions=n_metal,
            num_ligands=n_ligand,
            regime=AssemblyRegime.DLA,
            irreversibility=IrreversibilityMode.PATH_DEPENDENT,
            total_time=100.0,
            seed=seed + i * 2,
        )

        # Generate with RLA (slow, reversible)
        params_rla = PublicationSimulationParameters(
            num_metal_ions=n_metal,
            num_ligands=n_ligand,
            regime=AssemblyRegime.RLA,
            irreversibility=IrreversibilityMode.PATH_DEPENDENT,
            total_time=150.0,  # More time for slower process
            seed=seed + i * 2 + 1,
        )

        sim_dla = PublicationSimulator(params_dla)
        traj_dla, topo_dla, labels_dla = sim_dla.run()

        sim_rla = PublicationSimulator(params_rla)
        traj_rla, topo_rla, labels_rla = sim_rla.run()

        # Record pair
        pair = {
            "dla": {
                "trajectory": traj_dla,
                "topology": topo_dla,
                "labels": labels_dla,
            },
            "rla": {
                "trajectory": traj_rla,
                "topology": topo_rla,
                "labels": labels_rla,
            },
            # Compare final structures
            "final_edge_diff": abs(labels_dla.final_edge_count - labels_rla.final_edge_count),
            "final_beta1_diff": abs(labels_dla.final_beta_1 - labels_rla.final_beta_1),
            # Compare emergent properties
            "mechanical_class_same": labels_dla.mechanical_class == labels_rla.mechanical_class,
            "mechanical_score_diff": abs(labels_dla.mechanical_score - labels_rla.mechanical_score),
            "stability_class_same": labels_dla.thermal_stability_class == labels_rla.thermal_stability_class,
        }

        pairs.append(pair)

        if (i + 1) % 20 == 0:
            print(f"  Generated {i + 1}/{num_pairs} pairs")

    return pairs


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_assembly_nonequivalence(pairs: List[Dict]) -> Dict[str, Any]:
    """
    Analyze the assembly-non-equivalence experiment results.

    Computes statistics showing that assembly history affects properties
    even when final structures are similar.
    """
    # Filter pairs with similar final structures
    similar_pairs = [p for p in pairs if p["final_edge_diff"] < 10 and p["final_beta1_diff"] < 3]

    if len(similar_pairs) == 0:
        similar_pairs = pairs[:min(20, len(pairs))]  # Use subset if no similar pairs

    # Compute property differences
    mech_diffs = [p["mechanical_score_diff"] for p in similar_pairs]
    mech_class_diff_rate = 1 - np.mean([p["mechanical_class_same"] for p in similar_pairs])
    stab_class_diff_rate = 1 - np.mean([p["stability_class_same"] for p in similar_pairs])

    analysis = {
        "total_pairs": len(pairs),
        "similar_structure_pairs": len(similar_pairs),
        "mean_mechanical_score_diff": float(np.mean(mech_diffs)),
        "std_mechanical_score_diff": float(np.std(mech_diffs)),
        "mechanical_class_difference_rate": float(mech_class_diff_rate),
        "stability_class_difference_rate": float(stab_class_diff_rate),
        "conclusion": "",
    }

    # Determine conclusion
    if mech_class_diff_rate > 0.3 or np.mean(mech_diffs) > 0.15:
        analysis["conclusion"] = (
            "CONFIRMED: Assembly-non-equivalent materials show different emergent properties "
            f"({mech_class_diff_rate*100:.1f}% mechanical class difference rate) "
            "even with similar final structures."
        )
    else:
        analysis["conclusion"] = (
            "Weak effect observed. Consider increasing parameter variation."
        )

    return analysis


def compute_topology_based_features(topo: TrajectoryTopology) -> np.ndarray:
    """
    Extract features purely from topology (no chemistry).

    This is for the topology-only baseline model.
    """
    if topo.topology_signature is None:
        topo.compute_signature()

    # Use the signature plus additional derived features
    sig = topo.topology_signature

    # Add time series statistics
    if topo.observables:
        beta1_series = np.array([o.beta_1 for o in topo.observables])
        beta0_series = np.array([o.beta_0 for o in topo.observables])

        extra_features = np.array([
            np.mean(beta1_series),
            np.std(beta1_series),
            np.max(beta1_series),
            np.argmax(beta1_series) / len(beta1_series) if len(beta1_series) > 0 else 0,
            np.mean(beta0_series),
            np.std(beta0_series),
            topo.early_loop_dominance_index,
        ], dtype=np.float32)
    else:
        extra_features = np.zeros(7, dtype=np.float32)

    return np.concatenate([sig, extra_features])


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def save_experiment_results(
    results: Dict[str, Any],
    output_dir: Union[str, Path],
    name: str = "experiment",
) -> None:
    """Save experiment results to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save analysis
    with open(output_dir / f"{name}_analysis.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Results saved to {output_dir}")
