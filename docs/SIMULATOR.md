# Simulator Documentation

## Overview

The `ValidatedGillespieSimulator` implements exact stochastic simulation of coordination network assembly using the Gillespie Stochastic Simulation Algorithm (SSA).

## Algorithm

### Gillespie SSA

For a system with M possible reactions:

1. **Initialize**: Set t = 0, define initial state
2. **Calculate propensities**: aⱼ for j = 1, ..., M
3. **Calculate total propensity**: a₀ = Σⱼ aⱼ
4. **Generate τ ~ Exp(a₀)**: Time to next reaction
5. **Select reaction** j with probability aⱼ / a₀
6. **Execute reaction** j, update state
7. **Update** t = t + τ
8. **Repeat** until t > T_max or no reactions possible

### Rate Constants

Rate constants are derived from transition state theory (Eyring equation):

```
k = (k_B T / h) × exp(-ΔG‡ / RT)
```

where:
- k_B: Boltzmann constant (1.38 × 10⁻²³ J/K)
- h: Planck constant (6.63 × 10⁻³⁴ J·s)
- T: Temperature (K)
- R: Gas constant (8.314 J/mol·K)
- ΔG‡: Activation free energy (J/mol)

## Parameters

### Network Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `num_metal` | int | Number of metal ion nodes | 25 |
| `num_ligand` | int | Number of ligand nodes | 50 |
| `metal_valency` | int | Coordination number of metal ions | 6 |
| `ligand_valency` | int | Binding sites per ligand | 2 |

### Kinetic Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `regime` | AssemblyRegime | Assembly kinetic regime | RLA |
| `temperature` | float | Temperature in Kelvin | 298.15 |
| `ph` | float | Solution pH | 7.0 |
| `pKa` | float | pKa of ligand binding site | 8.0 |

### Simulation Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `total_time` | float | Maximum simulation time | 100.0 |
| `snapshot_interval` | float | Interval between topology snapshots | 2.0 |
| `seed` | int | Random seed for reproducibility | None |

## Assembly Regimes

### DLA (Diffusion-Limited Aggregation)

- **Physics**: Fast bond formation, minimal dissociation
- **Kinetics**: ΔG‡_on ~ 40 kJ/mol, ΔG‡_off ~ 80 kJ/mol
- **Result**: Branching, early loop formation, high ELDI

### RLA (Reaction-Limited Aggregation)

- **Physics**: Slow bond formation, significant dissociation
- **Kinetics**: ΔG‡_on ~ 55 kJ/mol, ΔG‡_off ~ 65 kJ/mol
- **Result**: Compact structures, late loop formation, low ELDI

### BURST (Burst Nucleation)

- **Physics**: Initial rapid nucleation, then slow growth
- **Kinetics**: ΔG‡_on starts low (35 kJ/mol), increases after burst
- **Result**: Many small clusters that gradually merge

## pH Dependence

Ligand binding is modulated by pH via Henderson-Hasselbalch:

```
f(pH) = 1 / (1 + 10^(pKa - pH))
```

- At pH < pKa: Ligand protonated → reduced binding
- At pH > pKa: Ligand deprotonated → enhanced binding

## Usage

### Basic Usage

```python
from assembly_net.data.validated_simulator import (
    ValidatedGillespieSimulator,
    AssemblyRegime,
)

sim = ValidatedGillespieSimulator(
    num_metal=25,
    num_ligand=50,
    regime=AssemblyRegime.DLA,
    total_time=100.0,
    seed=42,
)

result = sim.run()

print(f"Final edges: {result.final_num_edges}")
print(f"Final cycles: {result.final_beta_1}")
print(f"ELDI: {result.early_loop_dominance:.3f}")
print(f"Mechanical score: {result.mechanical_score:.3f}")
print(f"Mechanical class: {result.mechanical_class}")
```

### Varying Parameters

```python
# High temperature (faster kinetics)
sim = ValidatedGillespieSimulator(
    temperature=323.15,  # 50°C
    regime=AssemblyRegime.RLA,
)

# Low pH (reduced ligand binding)
sim = ValidatedGillespieSimulator(
    ph=5.0,
    pKa=8.0,
)

# Large network
sim = ValidatedGillespieSimulator(
    num_metal=50,
    num_ligand=100,
    total_time=200.0,
)
```

### Batch Generation

```python
from assembly_net.data.validated_simulator import generate_validated_dataset

# Generate 100 RLA trajectories
results = generate_validated_dataset(
    num_samples=100,
    regime=AssemblyRegime.RLA,
    seed=42,
)

scores = [r.mechanical_score for r in results]
print(f"Mean score: {np.mean(scores):.3f}")
```

## Output

### SimulationResult

| Field | Type | Description |
|-------|------|-------------|
| `final_num_edges` | int | Number of edges in final graph |
| `final_beta_1` | int | Number of cycles in final graph |
| `topology_history` | List[TopologicalSnapshot] | Topology at each snapshot |
| `loop_formation_times` | List[float] | Times when new cycles formed |
| `percolation_time` | Optional[float] | Time when largest component > 50% |
| `early_loop_dominance` | float | ELDI in [0, 1] |
| `mechanical_score` | float | History-dependent score in [0, 1] |
| `mechanical_class` | str | "robust", "ductile", or "brittle" |

### TopologicalSnapshot

| Field | Type | Description |
|-------|------|-------------|
| `time` | float | Simulation time |
| `beta_0` | int | Connected components |
| `beta_1` | int | Independent cycles |
| `num_edges` | int | Total edges |
| `largest_component_frac` | float | Fraction in largest component |

## Validation

The simulator is validated by 34 unit tests covering:

1. Physical chemistry (Eyring equation, pH modulation)
2. Kinetic model (propensities, waiting times)
3. Topological invariants (Betti numbers, ELDI)
4. Emergent properties (mechanical score)

Run tests:
```bash
python tests/test_simulator.py
```
