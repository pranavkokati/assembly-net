"""
Formal Mathematical Definitions for Assembly-Net.

This module provides rigorous mathematical definitions for the theoretical
framework underlying assembly-centric materials learning.

References
----------
[1] Gillespie, D. T. (1977). Exact stochastic simulation of coupled chemical
    reactions. J. Phys. Chem. 81(25), 2340-2361.
[2] Edelsbrunner, H., & Harer, J. (2008). Persistent homology - a survey.
    Contemporary Mathematics, 453, 257-282.
[3] Thorpe, M. F. (1983). Continuous deformations in random networks.
    J. Non-Cryst. Solids, 57(3), 355-370.
[4] Caruso, M. M. et al. (2009). Mechanically-induced chemical changes in
    polymeric materials. Chem. Rev. 109(11), 5755-5798.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# SECTION 1: ASSEMBLY GRAPH DEFINITIONS
# =============================================================================

@dataclass
class AssemblyGraphDefinition:
    """
    Definition 1 (Assembly Graph).

    An assembly graph at time t is a tuple G(t) = (V, E(t), X_V, X_E(t)) where:

    - V = {v_1, ..., v_n} is the fixed set of nodes (chemical species)
    - E(t) ⊆ V × V is the time-dependent edge set (bonds at time t)
    - X_V: V → R^d_v is the node feature function (chemical properties)
    - X_E(t): E(t) → R^d_e is the edge feature function (bond properties)

    The edge set E(t) evolves according to stochastic kinetics (see Section 2).
    """

    @staticmethod
    def formal_definition() -> str:
        return r"""
        G(t) = (V, E(t), X_V, X_E(t))

        where:
        - V: Fixed vertex set (n = |V| chemical species)
        - E(t) ⊆ V × V: Time-dependent edge set
        - X_V: V → R^{d_v}: Node features (chemical identity)
        - X_E(t): E(t) → R^{d_e}: Edge features (bond properties)
        """


@dataclass
class AssemblyTrajectoryDefinition:
    """
    Definition 2 (Assembly Trajectory).

    An assembly trajectory T is a sequence of assembly graphs:

        T = {G(t_0), G(t_1), ..., G(t_T)}

    where t_0 < t_1 < ... < t_T, capturing the temporal evolution of the network.

    Key Property: Two trajectories T_1 and T_2 may satisfy G_1(t_T) ≅ G_2(t_T)
    (isomorphic final graphs) while T_1 ≠ T_2 (different assembly histories).

    This is the fundamental observation motivating assembly-centric learning.
    """

    @staticmethod
    def formal_definition() -> str:
        return r"""
        T = {G(t_i)}_{i=0}^{N}  where  t_0 < t_1 < ... < t_N = T

        Assembly Non-Equivalence Theorem:
        ∃ T_1, T_2 such that:
            G_1(T) ≅ G_2(T)  (isomorphic final structures)
            but P(T_1) ≠ P(T_2)  (different emergent properties)

        where P: T → R^k maps trajectories to emergent property vectors.
        """


# =============================================================================
# SECTION 2: KINETIC MODELS
# =============================================================================

class CoordinationBondModel:
    """
    Physical Model for Coordination Bond Formation.

    Models metal-ligand coordination following the associative mechanism:

        M + L ⇌ ML

    with rate constants derived from transition state theory.

    Formation rate constant (Eyring equation):
        k_on = (k_B T / h) exp(-ΔG‡_on / RT)

    Dissociation rate constant:
        k_off = (k_B T / h) exp(-ΔG‡_off / RT)

    where:
        - k_B: Boltzmann constant (1.38 × 10^-23 J/K)
        - h: Planck constant (6.63 × 10^-34 J·s)
        - T: Temperature (K)
        - R: Gas constant (8.314 J/mol·K)
        - ΔG‡: Activation free energy (J/mol)

    References
    ----------
    [4] Constable, E. C. (2019). The Coordination Chemistry of Metallic
        Elements in Plants. Coord. Chem. Rev. 384, 1-13.
    """

    # Physical constants
    k_B = 1.380649e-23  # Boltzmann constant (J/K)
    h = 6.62607015e-34  # Planck constant (J·s)
    R = 8.314462618     # Gas constant (J/mol·K)

    def __init__(
        self,
        delta_G_on: float = 50000.0,    # J/mol (≈12 kcal/mol)
        delta_G_off: float = 80000.0,   # J/mol (≈19 kcal/mol)
        temperature: float = 298.15,     # K
    ):
        """
        Initialize coordination bond model.

        Parameters
        ----------
        delta_G_on : float
            Activation free energy for bond formation (J/mol).
            Typical: 40-60 kJ/mol for metal-phenolic coordination.
        delta_G_off : float
            Activation free energy for bond dissociation (J/mol).
            Typical: 60-100 kJ/mol (higher than formation).
        temperature : float
            Temperature in Kelvin.
        """
        self.delta_G_on = delta_G_on
        self.delta_G_off = delta_G_off
        self.temperature = temperature

    def formation_rate_constant(self) -> float:
        """
        Compute k_on using Eyring equation.

        Returns
        -------
        float
            Formation rate constant (s^-1).
        """
        prefactor = (self.k_B * self.temperature) / self.h
        exponent = -self.delta_G_on / (self.R * self.temperature)
        return prefactor * math.exp(exponent)

    def dissociation_rate_constant(self) -> float:
        """
        Compute k_off using Eyring equation.

        Returns
        -------
        float
            Dissociation rate constant (s^-1).
        """
        prefactor = (self.k_B * self.temperature) / self.h
        exponent = -self.delta_G_off / (self.R * self.temperature)
        return prefactor * math.exp(exponent)

    def equilibrium_constant(self) -> float:
        """
        Compute equilibrium constant K_eq = k_on / k_off.

        Returns
        -------
        float
            Equilibrium constant (dimensionless).
        """
        return self.formation_rate_constant() / self.dissociation_rate_constant()

    @staticmethod
    def ph_modulation(ph: float, pKa: float = 7.0) -> float:
        """
        pH-dependent modulation of formation rate.

        Models protonation equilibrium of ligand binding sites:

            k_on(pH) = k_on × [1 / (1 + 10^(pKa - pH))]

        At low pH, ligand protonation reduces coordination.

        Parameters
        ----------
        ph : float
            Solution pH.
        pKa : float
            pKa of ligand binding site (default: 7.0).

        Returns
        -------
        float
            Modulation factor in [0, 1].
        """
        return 1.0 / (1.0 + 10**(pKa - ph))


class AssemblyKineticsModel:
    """
    Stochastic Kinetic Model for Network Assembly.

    Implements the Gillespie (Stochastic Simulation Algorithm, SSA) for
    exact stochastic simulation of the coupled bond formation/dissociation
    reactions.

    Algorithm (Gillespie SSA)
    -------------------------

    Given current state (graph G, time t):

    1. Enumerate all possible reactions R = {r_1, ..., r_m}
       - Formation: (v_i, v_j) where edge (v_i, v_j) ∉ E(t)
       - Dissociation: (v_i, v_j) where edge (v_i, v_j) ∈ E(t)

    2. Compute propensities a_k for each reaction r_k:
       - Formation: a_k = k_on × c_i × c_j × f_avail(v_i) × f_avail(v_j)
       - Dissociation: a_k = k_off × w_ij

       where:
       - c_i, c_j: Effective concentrations
       - f_avail(v): Fraction of available coordination sites
       - w_ij: Bond weight (inverse strength)

    3. Compute total propensity: a_0 = Σ_k a_k

    4. Sample time to next reaction:
       τ ~ Exp(a_0)   (exponential distribution with rate a_0)

    5. Select reaction k with probability a_k / a_0

    6. Execute reaction, update G(t + τ)

    7. Repeat until t > T_max or no reactions possible

    References
    ----------
    [1] Gillespie, D. T. (1977). Exact stochastic simulation of coupled
        chemical reactions. J. Phys. Chem. 81(25), 2340-2361.
    """

    @staticmethod
    def propensity_formation(
        k_on: float,
        concentration_i: float,
        concentration_j: float,
        available_sites_i: int,
        total_sites_i: int,
        available_sites_j: int,
        total_sites_j: int,
    ) -> float:
        """
        Compute propensity for bond formation reaction.

        a_form = k_on × c_i × c_j × (avail_i / total_i) × (avail_j / total_j)

        Parameters
        ----------
        k_on : float
            Formation rate constant.
        concentration_i, concentration_j : float
            Effective concentrations of species.
        available_sites_i, total_sites_i : int
            Available and total coordination sites for node i.
        available_sites_j, total_sites_j : int
            Available and total coordination sites for node j.

        Returns
        -------
        float
            Propensity (units: inverse time).
        """
        if available_sites_i <= 0 or available_sites_j <= 0:
            return 0.0

        f_i = available_sites_i / total_sites_i
        f_j = available_sites_j / total_sites_j

        return k_on * concentration_i * concentration_j * f_i * f_j

    @staticmethod
    def propensity_dissociation(
        k_off: float,
        bond_strength: float = 1.0,
    ) -> float:
        """
        Compute propensity for bond dissociation reaction.

        a_dissoc = k_off / bond_strength

        Stronger bonds have lower dissociation propensity.

        Parameters
        ----------
        k_off : float
            Dissociation rate constant.
        bond_strength : float
            Relative bond strength (≥1.0 for strong bonds).

        Returns
        -------
        float
            Propensity (units: inverse time).
        """
        return k_off / max(bond_strength, 0.1)

    @staticmethod
    def sample_reaction_time(total_propensity: float, rng: np.random.Generator) -> float:
        """
        Sample time to next reaction from exponential distribution.

        τ ~ Exp(a_0)  =>  τ = -ln(U) / a_0  where U ~ Uniform(0,1)

        Parameters
        ----------
        total_propensity : float
            Sum of all reaction propensities.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        float
            Time to next reaction.
        """
        if total_propensity <= 0:
            return float('inf')
        return rng.exponential(1.0 / total_propensity)

    @staticmethod
    def select_reaction(propensities: List[float], rng: np.random.Generator) -> int:
        """
        Select reaction by propensity-weighted sampling.

        P(reaction k) = a_k / a_0

        Parameters
        ----------
        propensities : List[float]
            List of reaction propensities.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        int
            Index of selected reaction.
        """
        probs = np.array(propensities) / sum(propensities)
        return rng.choice(len(propensities), p=probs)


# =============================================================================
# SECTION 3: TOPOLOGICAL INVARIANTS
# =============================================================================

class TopologicalInvariantsDefinition:
    """
    Formal Definitions of Topological Invariants.

    Definition 3 (Betti Numbers).

    For a graph G = (V, E), the Betti numbers are:

    - β_0(G): Number of connected components
      β_0 = |V| - rank(L)
      where L is the graph Laplacian.

    - β_1(G): Number of independent cycles (cyclomatic complexity)
      β_1 = |E| - |V| + β_0
      (Euler characteristic relation)

    Definition 4 (Betti Evolution).

    For an assembly trajectory T, define the Betti evolution functions:

        β_0: [0, T] → Z^+    (component count over time)
        β_1: [0, T] → Z^≥0   (cycle count over time)

    These capture the topological evolution of the network.

    Definition 5 (Early Loop Dominance Index).

    Let τ_1, τ_2, ..., τ_k be the times when cycles form (β_1 increases).
    The early loop dominance index is:

        ELDI = (# of τ_i < T/2) / k

    ELDI ∈ [0, 1] quantifies what fraction of loops formed in the first
    half of assembly. High ELDI correlates with mechanical robustness.

    This is a KEY history-dependent feature that cannot be computed
    from the final structure alone.

    References
    ----------
    [2] Edelsbrunner, H., & Harer, J. (2008). Persistent homology - a survey.
        Contemporary Mathematics, 453, 257-282.
    """

    @staticmethod
    def beta_0(num_nodes: int, num_edges: int, edges: List[Tuple[int, int]]) -> int:
        """
        Compute β_0: number of connected components.

        Uses union-find for O(n α(n)) complexity.

        Parameters
        ----------
        num_nodes : int
            Number of nodes in graph.
        num_edges : int
            Number of edges.
        edges : List[Tuple[int, int]]
            Edge list.

        Returns
        -------
        int
            Number of connected components.
        """
        if num_nodes == 0:
            return 0

        # Union-find implementation
        parent = list(range(num_nodes))
        rank = [0] * num_nodes

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1

        for s, t in edges:
            union(s, t)

        return len(set(find(i) for i in range(num_nodes)))

    @staticmethod
    def beta_1(num_nodes: int, num_edges: int, beta_0: int) -> int:
        """
        Compute β_1: number of independent cycles.

        Uses Euler characteristic: β_1 = |E| - |V| + β_0

        Parameters
        ----------
        num_nodes : int
            Number of nodes.
        num_edges : int
            Number of edges.
        beta_0 : int
            Number of connected components.

        Returns
        -------
        int
            Number of independent cycles.
        """
        return max(0, num_edges - num_nodes + beta_0)

    @staticmethod
    def early_loop_dominance_index(
        loop_formation_times: List[float],
        total_time: float,
    ) -> float:
        """
        Compute the early loop dominance index (ELDI).

        ELDI = |{τ_i : τ_i < T/2}| / |{τ_i}|

        Parameters
        ----------
        loop_formation_times : List[float]
            Times at which new cycles formed.
        total_time : float
            Total assembly duration T.

        Returns
        -------
        float
            ELDI in [0, 1].
        """
        if not loop_formation_times:
            return 0.0

        midpoint = total_time / 2.0
        early_count = sum(1 for t in loop_formation_times if t < midpoint)

        return early_count / len(loop_formation_times)


# =============================================================================
# SECTION 4: EMERGENT PROPERTY THEORY
# =============================================================================

class EmergentPropertyTheory:
    """
    Theoretical Framework for History-Dependent Emergent Properties.

    Definition 6 (Emergent Property).

    An emergent property P of a material is a macroscopic observable that
    arises from the collective behavior of the network, not attributable
    to individual components.

    =========================================================================
    THEOREM 1: ASSEMBLY NON-EQUIVALENCE
    =========================================================================

    Statement:
    ----------
    There exist pairs of assembly trajectories T_1, T_2 such that:

        G_1(T_final) ≅ G_2(T_final)  (isomorphic final structures)

    but

        P(T_1) ≠ P(T_2)  (different emergent properties)

    REQUIRED ASSUMPTIONS (A1-A5):
    -----------------------------

    (A1) IRREVERSIBILITY: Bond dissociation rate k_off << k_on.
         Bonds, once formed, persist for timescales longer than assembly.
         If fully reversible (k_off ≈ k_on), the system reaches equilibrium
         and trajectory history is erased. The theorem does NOT hold.

    (A2) FINITE VALENCY: Nodes have maximum coordination numbers.
         Metal ions: typically 4-6 coordination sites.
         Ligands: typically 2-3 binding sites.
         This creates geometric frustration that makes assembly order matter.

    (A3) NON-EQUILIBRIUM KINETICS: The system is driven by kinetic rates,
         not thermodynamic equilibrium. Assembly proceeds via stochastic
         reactions governed by Gillespie dynamics.

    (A4) PATH-DEPENDENT BOND PROPERTIES: Bond strength or other properties
         depend on WHEN the bond formed during assembly (e.g., early bonds
         form in less crowded environments → stronger).

    (A5) TOPOLOGY-PROPERTY COUPLING: Emergent properties depend on
         topological features (cycles, connectivity) and their formation
         timing, not just the final count.

    COROLLARY (When Theorem Does NOT Hold):
    --------------------------------------
    If reversibility is allowed (A1 violated):
        The system can reach thermodynamic equilibrium.
        All trajectories leading to the same final structure become equivalent.
        P(T_1) = P(T_2) for G_1(T) ≅ G_2(T).

    If valency is unlimited (A2 violated):
        No geometric frustration → all assembly paths equivalent.

    Proof Sketch:
    ------------
    The mechanical score M(T) depends on the Early Loop Dominance Index (ELDI),
    which is determined by WHEN cycles formed during assembly. Two trajectories
    can reach isomorphic final graphs G_1(T) ≅ G_2(T) with identical β_1,
    but with different loop formation times:

        T_1: Loops form at t ∈ [0, T/2]    → ELDI_1 ≈ 1
        T_2: Loops form at t ∈ [T/2, T]    → ELDI_2 ≈ 0

    Since M depends on ELDI: M(T_1) ≠ M(T_2).  ∎

    Physical Justification (Mechanical Properties)
    ----------------------------------------------

    The mechanical robustness of a coordination network depends on:

    1. Loop density (β_1 / |V|): More cycles = more redundant load paths

    2. Early loop formation: Cycles formed early become "load-bearing"
       as subsequent bonds form around them, creating hierarchical
       mechanical reinforcement.

    3. Percolation timing: Early percolation allows stress distribution
       across the entire network.

    These factors are captured by the mechanical score function:

        M(T) = w_1 × ELDI + w_2 × (1 - t_perc/T) + w_3 × min(1, β_1/β_1^*)

    where:
        - ELDI: Early loop dominance index
        - t_perc: Percolation time
        - β_1^*: Target cycle count for robust material
        - w_1, w_2, w_3: Weights (empirically: 0.5, 0.3, 0.2)

    References
    ----------
    [4] Caruso, M. M. et al. (2009). Mechanically-induced chemical changes
        in polymeric materials. Chem. Rev. 109(11), 5755-5798.
    [5] Meakin, P. (1988). Fractal aggregates and their fractal measures.
        Phase Transitions, 12(3), 151-203.
    [6] Lin, M. Y. et al. (1989). Universality in colloid aggregation.
        Nature, 339(6223), 360-362.
    """

    # Empirically determined weights for mechanical score
    W_EARLY_LOOP = 0.5
    W_PERCOLATION = 0.3
    W_CYCLE_DENSITY = 0.2

    # Target cycle count for "robust" classification
    BETA_1_TARGET = 20.0

    # Classification thresholds
    ROBUST_THRESHOLD = 0.65
    DUCTILE_THRESHOLD = 0.35

    # Theorem assumptions for reference
    THEOREM_ASSUMPTIONS = {
        'A1': {
            'name': 'Irreversibility',
            'statement': 'k_off << k_on: bonds persist once formed',
            'violation_consequence': 'System reaches equilibrium; trajectory history erased',
        },
        'A2': {
            'name': 'Finite Valency',
            'statement': 'Nodes have maximum coordination numbers',
            'violation_consequence': 'No geometric frustration; all paths equivalent',
        },
        'A3': {
            'name': 'Non-equilibrium Kinetics',
            'statement': 'Assembly driven by stochastic kinetics, not equilibrium',
            'violation_consequence': 'Final state determined by thermodynamics alone',
        },
        'A4': {
            'name': 'Path-dependent Bond Properties',
            'statement': 'Bond properties depend on formation time',
            'violation_consequence': 'All bonds identical; no history encoding',
        },
        'A5': {
            'name': 'Topology-Property Coupling',
            'statement': 'Properties depend on topology formation timing',
            'violation_consequence': 'Only final topology matters; static analysis sufficient',
        },
    }

    @classmethod
    def list_theorem_assumptions(cls) -> str:
        """Return formatted list of theorem assumptions."""
        lines = ["THEOREM 1 ASSUMPTIONS:", "=" * 50]
        for key, assumption in cls.THEOREM_ASSUMPTIONS.items():
            lines.append(f"\n({key}) {assumption['name']}")
            lines.append(f"    Statement: {assumption['statement']}")
            lines.append(f"    If violated: {assumption['violation_consequence']}")
        return "\n".join(lines)

    @classmethod
    def check_assumption_violated(cls, assumption_key: str) -> str:
        """Return what happens if a specific assumption is violated."""
        if assumption_key in cls.THEOREM_ASSUMPTIONS:
            a = cls.THEOREM_ASSUMPTIONS[assumption_key]
            return f"If {a['name']} violated: {a['violation_consequence']}"
        return f"Unknown assumption: {assumption_key}"

    @classmethod
    def mechanical_score(
        cls,
        early_loop_dominance: float,
        percolation_time: Optional[float],
        total_time: float,
        beta_1_final: int,
    ) -> float:
        """
        Compute the history-dependent mechanical score.

        M(T) = 0.5×ELDI + 0.3×(1 - t_perc/T) + 0.2×min(1, β_1/20)

        Parameters
        ----------
        early_loop_dominance : float
            ELDI in [0, 1].
        percolation_time : Optional[float]
            Time of percolation (None if not percolated).
        total_time : float
            Total assembly time.
        beta_1_final : int
            Final cycle count.

        Returns
        -------
        float
            Mechanical score in [0, 1].
        """
        # Early loop contribution
        term1 = cls.W_EARLY_LOOP * early_loop_dominance

        # Percolation timing contribution
        # Clamp to [0, 1] to handle edge cases where perc_time > total_time
        if percolation_time is not None and total_time > 0:
            perc_factor = max(0.0, min(1.0, 1.0 - percolation_time / total_time))
            term2 = cls.W_PERCOLATION * perc_factor
        else:
            term2 = 0.0

        # Cycle density contribution
        term3 = cls.W_CYCLE_DENSITY * min(1.0, max(0.0, beta_1_final / cls.BETA_1_TARGET))

        return term1 + term2 + term3

    @classmethod
    def mechanical_class(cls, score: float) -> str:
        """
        Classify mechanical behavior from score.

        Parameters
        ----------
        score : float
            Mechanical score in [0, 1].

        Returns
        -------
        str
            One of: "robust", "ductile", "brittle"
        """
        if score >= cls.ROBUST_THRESHOLD:
            return "robust"
        elif score >= cls.DUCTILE_THRESHOLD:
            return "ductile"
        else:
            return "brittle"

    @staticmethod
    def information_theoretic_gap() -> str:
        """
        Formal statement of the information-theoretic limitation of static models.

        Returns
        -------
        str
            Mathematical statement of the theorem.
        """
        return r"""
        Theorem 2 (Information-Theoretic Limitation of Static Models).

        Let f: G → R be any function computable from the final graph G(T) alone.
        Let M: T → R be the mechanical score function.

        There exist trajectories T_1, T_2 with G(T_1, T) ≅ G(T_2, T) such that:

            f(G(T_1, T)) = f(G(T_2, T))    (static features are identical)
            M(T_1) ≠ M(T_2)                (mechanical scores differ)

        Consequence: No static model can perfectly predict mechanical properties
        for assembly-driven materials. Temporal information is necessary.

        Proof: ELDI is a component of M that depends on loop formation times,
        which are not encoded in the final structure G(T). ∎
        """
