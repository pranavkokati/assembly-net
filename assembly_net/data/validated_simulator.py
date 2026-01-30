"""
Validated Stochastic Simulator for Coordination Network Assembly.

This module implements a physically-grounded Gillespie (SSA) simulator with:
- Rigorous kinetic rate calculations based on transition state theory
- pH-dependent bond formation following Henderson-Hasselbalch
- Proper statistical mechanics foundations
- Comprehensive validation through unit tests

Algorithm: Gillespie Stochastic Simulation Algorithm (SSA)
----------------------------------------------------------

The Gillespie algorithm provides exact stochastic simulation of coupled
chemical reactions. For a system with M possible reactions:

1. Initialize: Set t = 0, define initial state
2. Calculate propensities: a_j for j = 1, ..., M
3. Calculate total propensity: a_0 = Σ_j a_j
4. Generate τ ~ Exp(a_0): time to next reaction
5. Select reaction j with probability a_j / a_0
6. Execute reaction j, update state
7. Update t = t + τ
8. Repeat until t > T_max or no reactions possible

References
----------
[1] Gillespie, D. T. (1977). Exact stochastic simulation of coupled chemical
    reactions. J. Phys. Chem. 81(25), 2340-2361.
[2] Eyring, H. (1935). The activated complex in chemical reactions.
    J. Chem. Phys. 3(2), 107-115.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np

# Import theory module using direct path import to avoid torch dependency
# from the main assembly_net package
import sys
import os
import importlib.util

_theory_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'theory')
_definitions_path = os.path.join(_theory_dir, 'definitions.py')

_spec = importlib.util.spec_from_file_location("theory_definitions", _definitions_path)
_theory_module = importlib.util.module_from_spec(_spec)
sys.modules["theory_definitions"] = _theory_module  # Register in sys.modules for dataclass support
_spec.loader.exec_module(_theory_module)

CoordinationBondModel = _theory_module.CoordinationBondModel
AssemblyKineticsModel = _theory_module.AssemblyKineticsModel
TopologicalInvariantsDefinition = _theory_module.TopologicalInvariantsDefinition
EmergentPropertyTheory = _theory_module.EmergentPropertyTheory


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class NodeType(Enum):
    """Types of nodes in coordination networks."""
    METAL_ION = auto()
    LIGAND = auto()


class EdgeType(Enum):
    """Types of edges (bonds) in coordination networks."""
    COORDINATION = auto()


class AssemblyRegime(Enum):
    """Assembly kinetic regimes."""
    DLA = auto()   # Diffusion-Limited Aggregation
    RLA = auto()   # Reaction-Limited Aggregation
    BURST = auto() # Burst Nucleation


@dataclass
class NodeFeatures:
    """Features of a node in the assembly graph."""
    node_type: NodeType
    valency: int  # Maximum coordination number
    current_coordination: int = 0

    def available_sites(self) -> int:
        """Return number of available coordination sites."""
        return max(0, self.valency - self.current_coordination)


@dataclass
class EdgeFeatures:
    """Features of an edge (bond) in the assembly graph."""
    edge_type: EdgeType
    bond_strength: float = 1.0  # Relative strength
    formation_time: float = 0.0  # When the bond formed


@dataclass
class TopologicalSnapshot:
    """
    Topological state at a single time point.

    Attributes
    ----------
    time : float
        Simulation time.
    beta_0 : int
        Number of connected components (Betti-0).
    beta_1 : int
        Number of independent cycles (Betti-1).
    num_edges : int
        Total number of edges.
    largest_component_frac : float
        Fraction of nodes in largest component.
    """
    time: float
    beta_0: int
    beta_1: int
    num_edges: int
    largest_component_frac: float


@dataclass
class SimulationResult:
    """
    Complete result of a simulation run.

    Attributes
    ----------
    final_num_edges : int
        Number of edges in final graph.
    final_beta_1 : int
        Number of cycles in final graph.
    topology_history : List[TopologicalSnapshot]
        Topology at each snapshot time.
    loop_formation_times : List[float]
        Times when new cycles formed.
    percolation_time : Optional[float]
        Time when percolation occurred (largest component > 50%).
    early_loop_dominance : float
        Fraction of loops formed in first half of simulation.
    mechanical_score : float
        History-dependent mechanical score.
    mechanical_class : str
        Classification: "robust", "ductile", or "brittle".
    """
    final_num_edges: int
    final_beta_1: int
    topology_history: List[TopologicalSnapshot]
    loop_formation_times: List[float]
    percolation_time: Optional[float]
    early_loop_dominance: float
    mechanical_score: float
    mechanical_class: str


# =============================================================================
# ASSEMBLY GRAPH
# =============================================================================

class AssemblyGraph:
    """
    A graph representing the coordination network at a single time point.

    Implements efficient operations for:
    - Node/edge addition and removal
    - Connected component tracking
    - Cycle counting using Euler characteristic
    """

    def __init__(self):
        self.num_nodes: int = 0
        self.node_features: List[NodeFeatures] = []
        self.edges: List[Tuple[int, int]] = []
        self.edge_features: List[EdgeFeatures] = []

        # Union-find structure for efficient component tracking
        self._parent: List[int] = []
        self._rank: List[int] = []

    def add_node(self, features: NodeFeatures) -> int:
        """Add a node and return its index."""
        idx = self.num_nodes
        self.node_features.append(features)
        self._parent.append(idx)
        self._rank.append(0)
        self.num_nodes += 1
        return idx

    def _find(self, x: int) -> int:
        """Find root with path compression."""
        if self._parent[x] != x:
            self._parent[x] = self._find(self._parent[x])
        return self._parent[x]

    def _union(self, x: int, y: int) -> bool:
        """
        Union two sets.

        Returns True if x and y were in different components (edge creates no cycle).
        Returns False if x and y were already connected (edge creates a cycle).
        """
        px, py = self._find(x), self._find(y)
        if px == py:
            return False  # Already connected -> creates cycle

        # Union by rank
        if self._rank[px] < self._rank[py]:
            px, py = py, px
        self._parent[py] = px
        if self._rank[px] == self._rank[py]:
            self._rank[px] += 1

        return True  # Components merged

    def add_edge(self, source: int, target: int, features: EdgeFeatures) -> bool:
        """
        Add an edge between two nodes.

        Returns True if a new cycle was formed by this edge.
        """
        self.edges.append((source, target))
        self.edge_features.append(features)
        self.node_features[source].current_coordination += 1
        self.node_features[target].current_coordination += 1

        # Check if this edge creates a cycle
        creates_cycle = not self._union(source, target)
        return creates_cycle

    def remove_edge(self, edge_index: int) -> Tuple[int, int]:
        """
        Remove an edge by index.

        Note: This invalidates the union-find structure for cycle detection.
        A full recomputation is needed after removal.
        """
        source, target = self.edges[edge_index]
        self.edges.pop(edge_index)
        self.edge_features.pop(edge_index)
        self.node_features[source].current_coordination -= 1
        self.node_features[target].current_coordination -= 1

        # Rebuild union-find (expensive but necessary for correctness)
        self._rebuild_union_find()

        return source, target

    def _rebuild_union_find(self):
        """Rebuild union-find from scratch after edge removal."""
        self._parent = list(range(self.num_nodes))
        self._rank = [0] * self.num_nodes
        for s, t in self.edges:
            self._union(s, t)

    def num_components(self) -> int:
        """Return number of connected components (β_0)."""
        if self.num_nodes == 0:
            return 0
        return len(set(self._find(i) for i in range(self.num_nodes)))

    def num_cycles(self) -> int:
        """
        Return number of independent cycles (β_1).

        Uses Euler characteristic: β_1 = |E| - |V| + β_0
        """
        return max(0, len(self.edges) - self.num_nodes + self.num_components())

    def largest_component_fraction(self) -> float:
        """Return fraction of nodes in the largest component."""
        if self.num_nodes == 0:
            return 0.0

        component_sizes: Dict[int, int] = {}
        for i in range(self.num_nodes):
            root = self._find(i)
            component_sizes[root] = component_sizes.get(root, 0) + 1

        return max(component_sizes.values()) / self.num_nodes

    def copy(self) -> "AssemblyGraph":
        """Create a deep copy of the graph."""
        g = AssemblyGraph()
        g.num_nodes = self.num_nodes
        g.node_features = [
            NodeFeatures(nf.node_type, nf.valency, nf.current_coordination)
            for nf in self.node_features
        ]
        g.edges = self.edges.copy()
        g.edge_features = [
            EdgeFeatures(ef.edge_type, ef.bond_strength, ef.formation_time)
            for ef in self.edge_features
        ]
        g._parent = self._parent.copy()
        g._rank = self._rank.copy()
        return g


# =============================================================================
# VALIDATED GILLESPIE SIMULATOR
# =============================================================================

class ValidatedGillespieSimulator:
    """
    Physically-grounded Gillespie simulator for coordination network assembly.

    This simulator implements the exact Stochastic Simulation Algorithm (SSA)
    with rate constants derived from transition state theory.

    Key Features
    ------------
    1. Exact stochastic dynamics via Gillespie algorithm
    2. Rate constants based on Eyring equation
    3. pH-dependent formation rates following Henderson-Hasselbalch
    4. Three assembly regimes (DLA, RLA, Burst)
    5. Path-dependent bond strengths for irreversibility
    6. Comprehensive topology tracking

    Parameters
    ----------
    num_metal : int
        Number of metal ion nodes.
    num_ligand : int
        Number of ligand nodes.
    metal_valency : int
        Coordination number of metal ions.
    ligand_valency : int
        Number of binding sites per ligand.
    regime : AssemblyRegime
        Assembly kinetic regime.
    temperature : float
        Temperature in Kelvin (affects rates via Eyring equation).
    ph : float
        Solution pH (affects formation rates).
    total_time : float
        Maximum simulation time.
    snapshot_interval : float
        Interval between topology snapshots.
    seed : Optional[int]
        Random seed for reproducibility.
    """

    # Physical constants
    k_B = 1.380649e-23  # Boltzmann constant (J/K)
    h = 6.62607015e-34  # Planck constant (J·s)
    R = 8.314462618     # Gas constant (J/mol·K)

    def __init__(
        self,
        num_metal: int = 25,
        num_ligand: int = 50,
        metal_valency: int = 6,
        ligand_valency: int = 2,
        regime: AssemblyRegime = AssemblyRegime.RLA,
        temperature: float = 298.15,
        ph: float = 7.0,
        pKa: float = 8.0,
        total_time: float = 100.0,
        snapshot_interval: float = 2.0,
        seed: Optional[int] = None,
    ):
        self.num_metal = num_metal
        self.num_ligand = num_ligand
        self.metal_valency = metal_valency
        self.ligand_valency = ligand_valency
        self.regime = regime
        self.temperature = temperature
        self.ph = ph
        self.pKa = pKa
        self.total_time = total_time
        self.snapshot_interval = snapshot_interval

        # Random state
        self.rng = np.random.default_rng(seed)

        # Configure regime-specific kinetics
        self._configure_regime()

        # Simulation state
        self.graph: Optional[AssemblyGraph] = None
        self.time: float = 0.0
        self.topology_history: List[TopologicalSnapshot] = []
        self.loop_formation_times: List[float] = []

    def _configure_regime(self):
        """Configure kinetic parameters based on assembly regime."""
        # Base rate constants (Eyring-like prefactor at 298 K)
        # k = (k_B T / h) * exp(-ΔG‡ / RT) ≈ 6.2e12 * exp(-ΔG‡ / RT)
        # For ΔG‡ ~ 50 kJ/mol: k ~ 0.1 s^-1 (order of magnitude)

        if self.regime == AssemblyRegime.DLA:
            # Diffusion-limited: fast formation, minimal dissociation
            # ΔG‡_on ~ 40 kJ/mol, ΔG‡_off ~ 80 kJ/mol
            self.delta_G_on = 40000.0   # J/mol
            self.delta_G_off = 80000.0  # J/mol
        elif self.regime == AssemblyRegime.RLA:
            # Reaction-limited: slower formation, reversible
            # ΔG‡_on ~ 55 kJ/mol, ΔG‡_off ~ 65 kJ/mol
            self.delta_G_on = 55000.0
            self.delta_G_off = 65000.0
        else:  # BURST
            # Burst: initially fast, then slow
            self.delta_G_on = 45000.0
            self.delta_G_off = 70000.0
            self.burst_duration = 20.0
            self.burst_delta_G_on = 35000.0  # Lower barrier during burst

    def _formation_rate_constant(self) -> float:
        """
        Compute formation rate constant using Eyring equation.

        k_on = (k_B T / h) * exp(-ΔG‡_on / RT)
        """
        prefactor = (self.k_B * self.temperature) / self.h
        delta_G = self.delta_G_on

        # Burst regime modification
        if self.regime == AssemblyRegime.BURST and self.time < getattr(self, 'burst_duration', 0):
            delta_G = self.burst_delta_G_on

        exponent = -delta_G / (self.R * self.temperature)
        k_on = prefactor * math.exp(exponent)

        # Scale to reasonable simulation timescale
        # Real chemistry: ~10^6 s^-1, simulation: ~1 s^-1
        return k_on * 1e-6

    def _dissociation_rate_constant(self) -> float:
        """
        Compute dissociation rate constant using Eyring equation.

        k_off = (k_B T / h) * exp(-ΔG‡_off / RT)
        """
        prefactor = (self.k_B * self.temperature) / self.h
        exponent = -self.delta_G_off / (self.R * self.temperature)
        k_off = prefactor * math.exp(exponent)

        return k_off * 1e-6

    def _ph_modulation(self) -> float:
        """
        pH-dependent modulation of formation rate.

        Models protonation equilibrium of ligand binding sites:
        f(pH) = 1 / (1 + 10^(pKa - pH))

        At pH < pKa: ligand protonated -> reduced binding
        At pH > pKa: ligand deprotonated -> enhanced binding
        """
        return 1.0 / (1.0 + 10**(self.pKa - self.ph))

    def _formation_propensity(self, node_i: int, node_j: int) -> float:
        """
        Compute propensity for bond formation between nodes i and j.

        a_form = k_on * f(pH) * (avail_i/total_i) * (avail_j/total_j)
        """
        nf_i = self.graph.node_features[node_i]
        nf_j = self.graph.node_features[node_j]

        avail_i = nf_i.available_sites()
        avail_j = nf_j.available_sites()

        if avail_i <= 0 or avail_j <= 0:
            return 0.0

        k_on = self._formation_rate_constant()
        ph_factor = self._ph_modulation()

        # Site availability factors
        f_i = avail_i / nf_i.valency
        f_j = avail_j / nf_j.valency

        return k_on * ph_factor * f_i * f_j

    def _dissociation_propensity(self, edge_idx: int) -> float:
        """
        Compute propensity for bond dissociation.

        a_dissoc = k_off / bond_strength

        Stronger bonds have lower dissociation rates.
        """
        ef = self.graph.edge_features[edge_idx]

        k_off = self._dissociation_rate_constant()

        # Stronger bonds dissociate slower
        return k_off / ef.bond_strength

    def _enumerate_reactions(self) -> Tuple[List[Tuple[str, Any]], List[float]]:
        """
        Enumerate all possible reactions and their propensities.

        Returns
        -------
        reactions : List[Tuple[str, Any]]
            List of (reaction_type, data) tuples.
        propensities : List[float]
            List of propensity values.
        """
        reactions = []
        propensities = []

        # Get node indices by type
        metals = [i for i, nf in enumerate(self.graph.node_features)
                  if nf.node_type == NodeType.METAL_ION]
        ligands = [i for i, nf in enumerate(self.graph.node_features)
                   if nf.node_type == NodeType.LIGAND]

        # Existing edges (for duplicate checking)
        existing = set((min(s, t), max(s, t)) for s, t in self.graph.edges)

        # Formation reactions: metal-ligand pairs
        for m in metals:
            for l in ligands:
                edge_key = (min(m, l), max(m, l))
                if edge_key not in existing:
                    prop = self._formation_propensity(m, l)
                    if prop > 0:
                        reactions.append(("formation", (m, l)))
                        propensities.append(prop)

        # Dissociation reactions
        for idx in range(len(self.graph.edges)):
            prop = self._dissociation_propensity(idx)
            if prop > 0:
                reactions.append(("dissociation", idx))
                propensities.append(prop)

        return reactions, propensities

    def _execute_formation(self, source: int, target: int) -> bool:
        """
        Execute a bond formation reaction.

        Returns True if a new cycle was formed.
        """
        # Path-dependent bond strength: early bonds are stronger
        time_factor = 1.0 - 0.5 * (self.time / self.total_time)
        strength = (1.0 + time_factor) * (1.0 + self.rng.exponential(0.2))

        ef = EdgeFeatures(
            edge_type=EdgeType.COORDINATION,
            bond_strength=strength,
            formation_time=self.time,
        )

        creates_cycle = self.graph.add_edge(source, target, ef)
        return creates_cycle

    def _execute_dissociation(self, edge_idx: int):
        """Execute a bond dissociation reaction."""
        self.graph.remove_edge(edge_idx)

    def _record_topology(self):
        """Record current topological state."""
        snapshot = TopologicalSnapshot(
            time=self.time,
            beta_0=self.graph.num_components(),
            beta_1=self.graph.num_cycles(),
            num_edges=len(self.graph.edges),
            largest_component_frac=self.graph.largest_component_fraction(),
        )
        self.topology_history.append(snapshot)

    def initialize(self):
        """Initialize the simulation."""
        self.graph = AssemblyGraph()
        self.time = 0.0
        self.topology_history = []
        self.loop_formation_times = []

        # Add metal nodes
        for _ in range(self.num_metal):
            self.graph.add_node(NodeFeatures(NodeType.METAL_ION, self.metal_valency))

        # Add ligand nodes
        for _ in range(self.num_ligand):
            self.graph.add_node(NodeFeatures(NodeType.LIGAND, self.ligand_valency))

        # Record initial state
        self._record_topology()

    def step(self) -> bool:
        """
        Execute one Gillespie step.

        Returns True if a reaction occurred, False if no reactions possible.
        """
        reactions, propensities = self._enumerate_reactions()

        if not reactions or sum(propensities) == 0:
            return False

        # Total propensity
        a_0 = sum(propensities)

        # Time to next reaction (exponential distribution)
        tau = self.rng.exponential(1.0 / a_0)
        self.time += tau

        # Select reaction
        probs = np.array(propensities) / a_0
        idx = self.rng.choice(len(reactions), p=probs)
        reaction_type, data = reactions[idx]

        # Execute reaction
        if reaction_type == "formation":
            source, target = data
            creates_cycle = self._execute_formation(source, target)
            if creates_cycle:
                self.loop_formation_times.append(self.time)
        else:  # dissociation
            self._execute_dissociation(data)

        return True

    def run(self) -> SimulationResult:
        """
        Run the complete simulation.

        Returns
        -------
        SimulationResult
            Complete simulation results including topology evolution
            and emergent property labels.
        """
        self.initialize()

        last_snapshot = 0.0

        while self.time < self.total_time:
            if not self.step():
                # No reactions possible, advance time
                self.time += self.snapshot_interval
                if self.time >= self.total_time:
                    break

            # Record topology at intervals
            if self.time - last_snapshot >= self.snapshot_interval:
                self._record_topology()
                last_snapshot = self.time

        # Final snapshot
        self._record_topology()

        # Compute derived quantities
        percolation_time = None
        for snap in self.topology_history:
            if snap.largest_component_frac > 0.5:
                percolation_time = snap.time
                break

        # Early loop dominance index
        if self.loop_formation_times:
            midpoint = self.total_time / 2.0
            early_count = sum(1 for t in self.loop_formation_times if t < midpoint)
            early_loop_dominance = early_count / len(self.loop_formation_times)
        else:
            early_loop_dominance = 0.0

        # Compute mechanical score using theory module
        final_beta_1 = self.graph.num_cycles()
        mechanical_score = EmergentPropertyTheory.mechanical_score(
            early_loop_dominance=early_loop_dominance,
            percolation_time=percolation_time,
            total_time=self.total_time,
            beta_1_final=final_beta_1,
        )
        mechanical_class = EmergentPropertyTheory.mechanical_class(mechanical_score)

        return SimulationResult(
            final_num_edges=len(self.graph.edges),
            final_beta_1=final_beta_1,
            topology_history=self.topology_history,
            loop_formation_times=self.loop_formation_times,
            percolation_time=percolation_time,
            early_loop_dominance=early_loop_dominance,
            mechanical_score=mechanical_score,
            mechanical_class=mechanical_class,
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def generate_validated_dataset(
    num_samples: int,
    regime: AssemblyRegime = AssemblyRegime.RLA,
    seed: int = 42,
    **kwargs,
) -> List[SimulationResult]:
    """
    Generate a dataset of simulation results.

    Parameters
    ----------
    num_samples : int
        Number of simulations to run.
    regime : AssemblyRegime
        Assembly regime for all simulations.
    seed : int
        Base random seed.
    **kwargs
        Additional arguments passed to ValidatedGillespieSimulator.

    Returns
    -------
    List[SimulationResult]
        List of simulation results.
    """
    results = []
    for i in range(num_samples):
        sim = ValidatedGillespieSimulator(
            regime=regime,
            seed=seed + i,
            **kwargs,
        )
        results.append(sim.run())
    return results
