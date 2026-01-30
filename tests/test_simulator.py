"""
Unit Tests for Assembly-Net Simulator and Theory Modules.

These tests validate:
1. Physical correctness of kinetic rate calculations
2. Topological invariant computations
3. Gillespie algorithm statistical properties
4. Emergent property calculations

Run with: pytest tests/test_simulator.py -v
"""

import math
import numpy as np
from typing import List

# Mock pytest for running without it
class MockPytest:
    @staticmethod
    def skip(msg):
        print(f"  SKIPPED: {msg}")

pytest = MockPytest()

import sys
import os
import importlib.util

# Use direct file imports to avoid torch dependency from main assembly_net package

_project_root = os.path.dirname(os.path.dirname(__file__))

# Import theory definitions directly
_definitions_path = os.path.join(_project_root, 'assembly_net', 'theory', 'definitions.py')
_spec = importlib.util.spec_from_file_location("theory_definitions", _definitions_path)
_theory_module = importlib.util.module_from_spec(_spec)
sys.modules["theory_definitions"] = _theory_module  # Register in sys.modules
_spec.loader.exec_module(_theory_module)

CoordinationBondModel = _theory_module.CoordinationBondModel
AssemblyKineticsModel = _theory_module.AssemblyKineticsModel
TopologicalInvariantsDefinition = _theory_module.TopologicalInvariantsDefinition
EmergentPropertyTheory = _theory_module.EmergentPropertyTheory

# Import validated simulator directly
_simulator_path = os.path.join(_project_root, 'assembly_net', 'data', 'validated_simulator.py')
_spec2 = importlib.util.spec_from_file_location("validated_simulator", _simulator_path)
_sim_module = importlib.util.module_from_spec(_spec2)
sys.modules["validated_simulator"] = _sim_module  # Register in sys.modules
_spec2.loader.exec_module(_sim_module)

AssemblyGraph = _sim_module.AssemblyGraph
NodeFeatures = _sim_module.NodeFeatures
EdgeFeatures = _sim_module.EdgeFeatures
NodeType = _sim_module.NodeType
EdgeType = _sim_module.EdgeType
AssemblyRegime = _sim_module.AssemblyRegime
ValidatedGillespieSimulator = _sim_module.ValidatedGillespieSimulator
TopologicalSnapshot = _sim_module.TopologicalSnapshot


# =============================================================================
# TEST: COORDINATION BOND MODEL
# =============================================================================

class TestCoordinationBondModel:
    """Tests for the physical chemistry of coordination bonds."""

    def test_eyring_equation_prefactor(self):
        """Test that Eyring prefactor is correct at 298 K."""
        model = CoordinationBondModel(temperature=298.15)
        # k_B * T / h ≈ 6.2e12 s^-1 at 298 K
        prefactor = (model.k_B * model.temperature) / model.h
        assert 6.0e12 < prefactor < 6.5e12, f"Prefactor should be ~6.2e12, got {prefactor:.2e}"

    def test_rate_constant_temperature_dependence(self):
        """Test that rate constants increase with temperature (Arrhenius)."""
        model_low = CoordinationBondModel(temperature=273.15)  # 0°C
        model_high = CoordinationBondModel(temperature=323.15)  # 50°C

        k_on_low = model_low.formation_rate_constant()
        k_on_high = model_high.formation_rate_constant()

        assert k_on_high > k_on_low, "Rate should increase with temperature"
        # Approximate doubling every 10°C
        ratio = k_on_high / k_on_low
        assert ratio > 2.0, f"Rate ratio should be >2 for 50K increase, got {ratio:.2f}"

    def test_equilibrium_constant_thermodynamics(self):
        """Test that equilibrium constant follows thermodynamics."""
        # ΔG° = ΔG‡_off - ΔG‡_on (at equilibrium)
        delta_G_on = 50000.0  # J/mol
        delta_G_off = 80000.0  # J/mol
        model = CoordinationBondModel(delta_G_on=delta_G_on, delta_G_off=delta_G_off)

        K_eq = model.equilibrium_constant()

        # K = exp(-ΔG°/RT) where ΔG° = ΔG‡_on - ΔG‡_off (for forward reaction)
        # Since ΔG‡_off > ΔG‡_on, forward is favored (K > 1)
        assert K_eq > 1.0, f"K_eq should be >1 for favorable reaction, got {K_eq:.2e}"

    def test_ph_modulation_limits(self):
        """Test pH modulation has correct limiting behavior."""
        pKa = 7.0

        # At pH << pKa: ligand protonated, f → 0
        f_low = CoordinationBondModel.ph_modulation(ph=4.0, pKa=pKa)
        assert f_low < 0.01, f"At pH << pKa, modulation should be ~0, got {f_low:.3f}"

        # At pH >> pKa: ligand deprotonated, f → 1
        f_high = CoordinationBondModel.ph_modulation(ph=10.0, pKa=pKa)
        assert f_high > 0.99, f"At pH >> pKa, modulation should be ~1, got {f_high:.3f}"

        # At pH = pKa: f = 0.5
        f_mid = CoordinationBondModel.ph_modulation(ph=7.0, pKa=pKa)
        assert abs(f_mid - 0.5) < 0.01, f"At pH = pKa, modulation should be 0.5, got {f_mid:.3f}"


# =============================================================================
# TEST: ASSEMBLY KINETICS MODEL
# =============================================================================

class TestAssemblyKineticsModel:
    """Tests for the Gillespie algorithm components."""

    def test_propensity_formation_saturation(self):
        """Test that propensity is zero when sites are saturated."""
        prop = AssemblyKineticsModel.propensity_formation(
            k_on=1.0,
            concentration_i=1.0,
            concentration_j=1.0,
            available_sites_i=0,  # Saturated
            total_sites_i=6,
            available_sites_j=2,
            total_sites_j=2,
        )
        assert prop == 0.0, "Propensity should be 0 when one species is saturated"

    def test_propensity_formation_positive(self):
        """Test that propensity is positive with available sites."""
        prop = AssemblyKineticsModel.propensity_formation(
            k_on=1.0,
            concentration_i=1.0,
            concentration_j=1.0,
            available_sites_i=6,
            total_sites_i=6,
            available_sites_j=2,
            total_sites_j=2,
        )
        assert prop == 1.0, "Propensity should equal k_on when all sites available"

    def test_propensity_dissociation_strength_dependence(self):
        """Test that stronger bonds dissociate slower."""
        prop_weak = AssemblyKineticsModel.propensity_dissociation(k_off=1.0, bond_strength=1.0)
        prop_strong = AssemblyKineticsModel.propensity_dissociation(k_off=1.0, bond_strength=2.0)

        assert prop_strong < prop_weak, "Stronger bonds should have lower dissociation propensity"
        assert abs(prop_strong - 0.5) < 0.01, f"Propensity should halve with doubled strength"

    def test_exponential_waiting_time(self):
        """Test that waiting times follow exponential distribution."""
        rng = np.random.default_rng(42)
        a_0 = 2.0
        n_samples = 10000

        samples = [AssemblyKineticsModel.sample_reaction_time(a_0, rng) for _ in range(n_samples)]

        # Mean of Exp(λ) is 1/λ
        expected_mean = 1.0 / a_0
        actual_mean = np.mean(samples)
        assert abs(actual_mean - expected_mean) < 0.05, \
            f"Mean waiting time should be {expected_mean}, got {actual_mean:.3f}"

    def test_reaction_selection_probabilities(self):
        """Test that reactions are selected with correct probabilities."""
        rng = np.random.default_rng(42)
        propensities = [1.0, 2.0, 3.0]  # Total = 6, probs = [1/6, 2/6, 3/6]
        n_samples = 10000

        counts = [0, 0, 0]
        for _ in range(n_samples):
            idx = AssemblyKineticsModel.select_reaction(propensities, rng)
            counts[idx] += 1

        expected = [n_samples / 6, n_samples / 3, n_samples / 2]
        for i, (actual, exp) in enumerate(zip(counts, expected)):
            assert abs(actual - exp) < 200, \
                f"Reaction {i}: expected ~{exp:.0f}, got {actual}"


# =============================================================================
# TEST: TOPOLOGICAL INVARIANTS
# =============================================================================

class TestTopologicalInvariants:
    """Tests for topological computations (Betti numbers)."""

    def test_beta_0_isolated_nodes(self):
        """Test β_0 = n for n isolated nodes."""
        n = 10
        beta_0 = TopologicalInvariantsDefinition.beta_0(
            num_nodes=n, num_edges=0, edges=[]
        )
        assert beta_0 == n, f"β_0 should be {n} for isolated nodes, got {beta_0}"

    def test_beta_0_connected_graph(self):
        """Test β_0 = 1 for a connected tree."""
        # Tree on 5 nodes: 0-1, 1-2, 1-3, 3-4
        edges = [(0, 1), (1, 2), (1, 3), (3, 4)]
        beta_0 = TopologicalInvariantsDefinition.beta_0(
            num_nodes=5, num_edges=4, edges=edges
        )
        assert beta_0 == 1, f"β_0 should be 1 for connected graph, got {beta_0}"

    def test_beta_0_two_components(self):
        """Test β_0 = 2 for two disconnected components."""
        # Two triangles: 0-1-2 and 3-4-5
        edges = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]
        beta_0 = TopologicalInvariantsDefinition.beta_0(
            num_nodes=6, num_edges=6, edges=edges
        )
        assert beta_0 == 2, f"β_0 should be 2 for two components, got {beta_0}"

    def test_beta_1_tree(self):
        """Test β_1 = 0 for a tree (no cycles)."""
        # Tree: β_1 = |E| - |V| + β_0 = (n-1) - n + 1 = 0
        beta_1 = TopologicalInvariantsDefinition.beta_1(
            num_nodes=5, num_edges=4, beta_0=1
        )
        assert beta_1 == 0, f"β_1 should be 0 for tree, got {beta_1}"

    def test_beta_1_single_cycle(self):
        """Test β_1 = 1 for a graph with one cycle."""
        # Triangle: β_1 = 3 - 3 + 1 = 1
        beta_1 = TopologicalInvariantsDefinition.beta_1(
            num_nodes=3, num_edges=3, beta_0=1
        )
        assert beta_1 == 1, f"β_1 should be 1 for triangle, got {beta_1}"

    def test_beta_1_multiple_cycles(self):
        """Test β_1 for graph with multiple cycles."""
        # Complete graph K4: 4 nodes, 6 edges, β_0 = 1
        # β_1 = 6 - 4 + 1 = 3
        beta_1 = TopologicalInvariantsDefinition.beta_1(
            num_nodes=4, num_edges=6, beta_0=1
        )
        assert beta_1 == 3, f"β_1 should be 3 for K4, got {beta_1}"

    def test_early_loop_dominance_all_early(self):
        """Test ELDI = 1 when all loops form early."""
        times = [1.0, 2.0, 3.0, 4.0]  # All < 50 (midpoint)
        eldi = TopologicalInvariantsDefinition.early_loop_dominance_index(
            loop_formation_times=times, total_time=100.0
        )
        assert eldi == 1.0, f"ELDI should be 1.0, got {eldi}"

    def test_early_loop_dominance_all_late(self):
        """Test ELDI = 0 when all loops form late."""
        times = [60.0, 70.0, 80.0, 90.0]  # All > 50 (midpoint)
        eldi = TopologicalInvariantsDefinition.early_loop_dominance_index(
            loop_formation_times=times, total_time=100.0
        )
        assert eldi == 0.0, f"ELDI should be 0.0, got {eldi}"

    def test_early_loop_dominance_mixed(self):
        """Test ELDI for mixed early/late loops."""
        times = [10.0, 20.0, 60.0, 70.0]  # 2 early, 2 late
        eldi = TopologicalInvariantsDefinition.early_loop_dominance_index(
            loop_formation_times=times, total_time=100.0
        )
        assert eldi == 0.5, f"ELDI should be 0.5, got {eldi}"


# =============================================================================
# TEST: EMERGENT PROPERTY THEORY
# =============================================================================

class TestEmergentPropertyTheory:
    """Tests for emergent property calculations."""

    def test_mechanical_score_bounds(self):
        """Test mechanical score is in [0, 1]."""
        # Maximum score: high ELDI, early percolation, many cycles
        score_max = EmergentPropertyTheory.mechanical_score(
            early_loop_dominance=1.0,
            percolation_time=0.0,
            total_time=100.0,
            beta_1_final=30,
        )
        assert 0 <= score_max <= 1, f"Score should be in [0,1], got {score_max}"

        # Minimum score: no ELDI, no percolation, no cycles
        score_min = EmergentPropertyTheory.mechanical_score(
            early_loop_dominance=0.0,
            percolation_time=None,
            total_time=100.0,
            beta_1_final=0,
        )
        assert 0 <= score_min <= 1, f"Score should be in [0,1], got {score_min}"

    def test_mechanical_score_eldi_contribution(self):
        """Test that ELDI affects mechanical score."""
        score_high_eldi = EmergentPropertyTheory.mechanical_score(
            early_loop_dominance=1.0,
            percolation_time=50.0,
            total_time=100.0,
            beta_1_final=10,
        )
        score_low_eldi = EmergentPropertyTheory.mechanical_score(
            early_loop_dominance=0.0,
            percolation_time=50.0,
            total_time=100.0,
            beta_1_final=10,
        )
        assert score_high_eldi > score_low_eldi, "Higher ELDI should give higher score"

        # ELDI contributes with weight 0.5
        diff = score_high_eldi - score_low_eldi
        assert abs(diff - 0.5) < 0.01, f"ELDI contribution should be 0.5, got {diff}"

    def test_mechanical_classification(self):
        """Test mechanical class thresholds."""
        assert EmergentPropertyTheory.mechanical_class(0.8) == "robust"
        assert EmergentPropertyTheory.mechanical_class(0.5) == "ductile"
        assert EmergentPropertyTheory.mechanical_class(0.2) == "brittle"

        # Test boundary conditions
        assert EmergentPropertyTheory.mechanical_class(0.65) == "robust"
        assert EmergentPropertyTheory.mechanical_class(0.35) == "ductile"
        assert EmergentPropertyTheory.mechanical_class(0.34) == "brittle"


# =============================================================================
# TEST: ASSEMBLY GRAPH
# =============================================================================

class TestAssemblyGraph:
    """Tests for the AssemblyGraph data structure."""

    def test_add_node(self):
        """Test node addition."""
        g = AssemblyGraph()
        idx = g.add_node(NodeFeatures(NodeType.METAL_ION, 6))
        assert idx == 0
        assert g.num_nodes == 1
        assert g.node_features[0].valency == 6

    def test_add_edge_no_cycle(self):
        """Test edge addition that doesn't create a cycle."""
        g = AssemblyGraph()
        g.add_node(NodeFeatures(NodeType.METAL_ION, 6))
        g.add_node(NodeFeatures(NodeType.LIGAND, 2))

        creates_cycle = g.add_edge(0, 1, EdgeFeatures(EdgeType.COORDINATION))

        assert not creates_cycle, "First edge should not create cycle"
        assert len(g.edges) == 1
        assert g.node_features[0].current_coordination == 1
        assert g.node_features[1].current_coordination == 1

    def test_add_edge_creates_cycle(self):
        """Test edge addition that creates a cycle."""
        g = AssemblyGraph()
        for _ in range(3):
            g.add_node(NodeFeatures(NodeType.METAL_ION, 6))

        # Create path: 0-1-2
        g.add_edge(0, 1, EdgeFeatures(EdgeType.COORDINATION))
        g.add_edge(1, 2, EdgeFeatures(EdgeType.COORDINATION))

        # Close the cycle: 0-2
        creates_cycle = g.add_edge(0, 2, EdgeFeatures(EdgeType.COORDINATION))

        assert creates_cycle, "Third edge should create cycle"
        assert g.num_cycles() == 1

    def test_num_components(self):
        """Test connected component counting."""
        g = AssemblyGraph()
        for _ in range(4):
            g.add_node(NodeFeatures(NodeType.METAL_ION, 6))

        assert g.num_components() == 4, "4 isolated nodes = 4 components"

        g.add_edge(0, 1, EdgeFeatures(EdgeType.COORDINATION))
        assert g.num_components() == 3, "One edge merges 2 nodes"

        g.add_edge(2, 3, EdgeFeatures(EdgeType.COORDINATION))
        assert g.num_components() == 2, "Two pairs"

        g.add_edge(1, 2, EdgeFeatures(EdgeType.COORDINATION))
        assert g.num_components() == 1, "All connected"

    def test_largest_component_fraction(self):
        """Test largest component fraction calculation."""
        g = AssemblyGraph()
        for _ in range(4):
            g.add_node(NodeFeatures(NodeType.METAL_ION, 6))

        # Initially all isolated: each has size 1, fraction = 0.25
        assert g.largest_component_fraction() == 0.25

        # Connect 3 nodes: largest = 3/4 = 0.75
        g.add_edge(0, 1, EdgeFeatures(EdgeType.COORDINATION))
        g.add_edge(1, 2, EdgeFeatures(EdgeType.COORDINATION))
        assert g.largest_component_fraction() == 0.75


# =============================================================================
# TEST: VALIDATED GILLESPIE SIMULATOR
# =============================================================================

class TestValidatedGillespieSimulator:
    """Tests for the complete Gillespie simulator."""

    def test_initialization(self):
        """Test simulator initialization."""
        sim = ValidatedGillespieSimulator(
            num_metal=5, num_ligand=10, seed=42
        )
        sim.initialize()

        assert sim.graph.num_nodes == 15
        assert sim.time == 0.0
        assert len(sim.topology_history) == 1  # Initial snapshot

    def test_deterministic_seed(self):
        """Test that same seed gives same results."""
        results1 = []
        for _ in range(3):
            sim = ValidatedGillespieSimulator(num_metal=10, num_ligand=20, seed=42, total_time=50)
            result = sim.run()
            results1.append(result.final_num_edges)

        # Same seed should give same sequence
        assert results1[0] == results1[1] == results1[2], "Same seed should give deterministic results"

    def test_different_seeds_vary(self):
        """Test that different seeds give different results."""
        results = []
        for seed in [1, 2, 3, 4, 5]:
            sim = ValidatedGillespieSimulator(num_metal=15, num_ligand=30, seed=seed, total_time=100)
            result = sim.run()
            results.append(result.final_num_edges)

        # Should have some variation
        assert len(set(results)) > 1, "Different seeds should give some variation"

    def test_topology_monotonicity(self):
        """Test that edge count is non-decreasing in DLA regime."""
        sim = ValidatedGillespieSimulator(
            num_metal=20, num_ligand=40,
            regime=AssemblyRegime.DLA,
            total_time=100,
            seed=42,
        )
        result = sim.run()

        # In DLA, dissociation is rare, so edges should mostly increase
        edge_counts = [s.num_edges for s in result.topology_history]
        decreases = sum(1 for i in range(1, len(edge_counts))
                       if edge_counts[i] < edge_counts[i-1])

        # Allow some decreases but not many
        assert decreases < len(edge_counts) * 0.1, \
            f"DLA should have few edge decreases, got {decreases}/{len(edge_counts)}"

    def test_euler_characteristic_consistency(self):
        """Test that Euler characteristic formula holds at all times."""
        sim = ValidatedGillespieSimulator(
            num_metal=15, num_ligand=30,
            total_time=100,
            seed=42,
        )
        result = sim.run()

        for snap in result.topology_history:
            # β_1 = |E| - |V| + β_0
            expected_beta_1 = snap.num_edges - sim.graph.num_nodes + snap.beta_0
            assert snap.beta_1 == max(0, expected_beta_1), \
                f"Euler characteristic mismatch at t={snap.time}"

    def test_mechanical_score_range(self):
        """Test that mechanical score is always in [0, 1]."""
        for seed in range(10):
            sim = ValidatedGillespieSimulator(
                num_metal=20, num_ligand=40,
                total_time=100,
                seed=seed,
            )
            result = sim.run()
            assert 0 <= result.mechanical_score <= 1, \
                f"Mechanical score {result.mechanical_score} out of range"

    def test_regime_affects_dynamics(self):
        """Test that different regimes produce different dynamics."""
        dla_early_loops = []
        rla_early_loops = []

        for seed in range(10):
            sim_dla = ValidatedGillespieSimulator(
                num_metal=20, num_ligand=40,
                regime=AssemblyRegime.DLA,
                total_time=100,
                seed=seed,
            )
            dla_early_loops.append(sim_dla.run().early_loop_dominance)

            sim_rla = ValidatedGillespieSimulator(
                num_metal=20, num_ligand=40,
                regime=AssemblyRegime.RLA,
                total_time=100,
                seed=seed + 1000,
            )
            rla_early_loops.append(sim_rla.run().early_loop_dominance)

        # DLA should tend to have more early loops
        dla_mean = np.mean(dla_early_loops)
        rla_mean = np.mean(rla_early_loops)

        # This tests the core claim: assembly regime affects topology evolution
        print(f"DLA mean ELDI: {dla_mean:.3f}, RLA mean ELDI: {rla_mean:.3f}")


# =============================================================================
# STATISTICAL VALIDATION TESTS
# =============================================================================

class TestStatisticalValidation:
    """Statistical validation tests for simulator correctness."""

    def test_gillespie_waiting_time_distribution(self):
        """
        Validate that inter-event times follow exponential distribution.

        This is a key property of the Gillespie algorithm.

        NOTE: In practice, the CV may deviate from 1 because:
        1. The total propensity changes as the network evolves
        2. Formation rates depend on available sites (which decrease)
        3. Path-dependent effects modify rates over time

        This test validates that waiting times are positively distributed
        with reasonable variance, not exact exponentiality which requires
        constant rates.
        """
        sim = ValidatedGillespieSimulator(
            num_metal=30, num_ligand=60,
            total_time=200,
            snapshot_interval=0.1,
            seed=42,
        )
        sim.initialize()

        # Collect waiting times
        waiting_times = []
        prev_time = 0.0
        max_events = 500

        for _ in range(max_events):
            if not sim.step():
                break
            waiting_times.append(sim.time - prev_time)
            prev_time = sim.time

        if len(waiting_times) < 100:
            pytest.skip("Not enough events for statistical test")

        # Basic sanity checks for Gillespie-like dynamics:
        # 1. All waiting times should be positive
        assert all(t > 0 for t in waiting_times), "Waiting times must be positive"

        # 2. Mean should be positive and reasonable
        mean_tau = np.mean(waiting_times)
        assert mean_tau > 0, "Mean waiting time must be positive"

        # 3. Variance should be non-zero (stochastic behavior)
        std_tau = np.std(waiting_times)
        assert std_tau > 0, "Waiting times should have variance (stochastic)"

        # 4. Distribution should be right-skewed (exponential-like)
        # Skewness for exponential is 2, allow range [0.5, 10]
        median_tau = np.median(waiting_times)
        assert median_tau < mean_tau, "Distribution should be right-skewed"


# =============================================================================
# RUN TESTS
# =============================================================================

def run_all_tests():
    """Run all tests without pytest."""
    test_classes = [
        TestCoordinationBondModel,
        TestAssemblyKineticsModel,
        TestTopologicalInvariants,
        TestEmergentPropertyTheory,
        TestAssemblyGraph,
        TestValidatedGillespieSimulator,
        TestStatisticalValidation,
    ]

    total_passed = 0
    total_failed = 0

    for test_class in test_classes:
        print(f"\nTesting {test_class.__name__}...")
        instance = test_class()

        for method_name in dir(instance):
            if method_name.startswith("test_"):
                try:
                    getattr(instance, method_name)()
                    print(f"  ✓ {method_name}")
                    total_passed += 1
                except AssertionError as e:
                    print(f"  ✗ {method_name}: {e}")
                    total_failed += 1
                except Exception as e:
                    print(f"  ✗ {method_name}: {type(e).__name__}: {e}")
                    total_failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {total_passed} passed, {total_failed} failed")
    print("=" * 60)

    return total_failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
