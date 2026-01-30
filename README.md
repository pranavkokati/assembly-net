# Assembly-Net

**Topology-Aware Learning of Emergent Material Properties from Coordination-Network Assembly Graphs**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/pranavkokati/assembly-net/actions/workflows/tests.yml/badge.svg)](https://github.com/pranavkokati/assembly-net/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

Assembly-Net demonstrates that **assembly history affects emergent material properties in ways not captured by final structure alone**. This is the key scientific claim, validated through rigorous experiments.

### Core Insight

Static ML models are **information-theoretically insufficient** (under assumptions A1-A5) for predicting properties of assembly-driven materials. Temporal models that observe the assembly process can access information unavailable to static models.

### Scope of Claims

We claim that static models are **insufficient under assumptions A1-A5** (irreversibility, finite valency, non-equilibrium kinetics, path-dependent bonds, topology-property coupling). If these assumptions are violated (e.g., fully reversible assembly reaching thermodynamic equilibrium), the theorem does not hold. This precision reflects the actual scope of our results—we make no claims beyond what the theory supports.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/pranavkokati/assembly-net.git
cd assembly-net

# Install minimal dependencies (no GPU required)
pip install -r requirements.txt

# Run all publication experiments (~3-5 minutes)
python run_all_critical_experiments.py

# Run unit tests (34 tests)
python tests/test_simulator.py
```

## Reproduce All Results

**Single command to reproduce all publication results:**

```bash
python run_all_critical_experiments.py
```

**Expected runtime:** 3-5 minutes on a standard laptop (no GPU required)

**Output files:**
```
publication_results/
├── critical_experiments.json    # Main experiment results
├── figure1_data.json           # Data for Figure 1
├── figure1_plot.py             # Script to generate Figure 1
├── experiment_log.txt          # Execution log with timing
└── all_results_summary.json    # Summary with timestamps
```

**Generate publication Figure 1:**
```bash
pip install matplotlib
python publication_results/figure1_plot.py
```

## Key Experiments

| Experiment | Purpose | Key Result |
|------------|---------|------------|
| Isomorphic Failure | Show static models fail on identical-structure pairs | ELDI difference > 0.5 between matched pairs |
| Topology vs Simple Stats | Compare β₁, ELDI against degree, clustering | Topology R² > Simple R² |
| Literature Validation | Validate against known DLA/RLA behavior | Regime affects mechanical class |
| Figure 1 | Visualize β₁(t) curves for non-equivalent trajectories | Same final β₁, different formation timing |

## Simulator Parameters

| Parameter | Physical Meaning | Typical Range | Effect |
|-----------|-----------------|---------------|--------|
| `num_metal` | Metal ion count | 15-50 | Network size |
| `num_ligand` | Ligand count | 30-100 | Available binding sites |
| `metal_valency` | Coordination number | 4-6 | Network connectivity |
| `ligand_valency` | Binding sites per ligand | 2-3 | Crosslinking potential |
| `regime` | Assembly kinetics | DLA/RLA/BURST | Controls loop formation timing |
| `temperature` | Reaction temperature (K) | 273-373 | Rate constants via Eyring |
| `ph` | Solution pH | 4-10 | Ligand protonation state |
| `total_time` | Simulation duration | 50-200 | Assembly completion |

**Assembly Regimes:**
- **DLA** (Diffusion-Limited): Fast formation, minimal dissociation → early loops
- **RLA** (Reaction-Limited): Slow formation, reversible → late loops
- **BURST**: Rapid nucleation, then slow growth → many small clusters

## Theoretical Foundation

The framework is built on five assumptions (A1-A5):

| Assumption | Statement | If Violated |
|------------|-----------|-------------|
| A1: Irreversibility | k_off << k_on | Equilibrium erases history |
| A2: Finite Valency | Max coordination numbers | No geometric frustration |
| A3: Non-equilibrium | Gillespie kinetics | Thermodynamics determines state |
| A4: Path-dependent Bonds | Bond strength depends on formation time | No history encoding |
| A5: Topology-Property Coupling | Properties depend on topology timing | Static analysis sufficient |

See `assembly_net/theory/definitions.py` for formal definitions and proofs.

## Project Structure

```
assembly-net/
├── assembly_net/
│   ├── theory/              # Formal definitions, physical constants
│   ├── data/                # Simulator, data structures
│   ├── topology/            # Persistent homology, rigidity
│   ├── models/              # GNN architectures (requires PyTorch)
│   └── experiments/         # Publication experiments
├── tests/                   # Unit tests (34 tests)
├── publication_results/     # Generated results
├── docs/                    # Detailed documentation
├── requirements.txt         # Core dependencies
├── requirements-full.txt    # Full ML dependencies
└── run_all_critical_experiments.py  # Main reproducibility script
```

## Requirements

**Core (no GPU):**
- Python 3.8+
- NumPy
- NetworkX

**Full ML Pipeline:**
- PyTorch >= 2.0
- PyTorch Geometric >= 2.3
- giotto-tda >= 0.6

Install core dependencies only:
```bash
pip install -r requirements.txt
```

Install full ML dependencies:
```bash
pip install -r requirements-full.txt
```

## Known Limitations

1. **No spatial embedding** - Nodes have no 3D coordinates
2. **No atomistic resolution** - No quantum/orbital information
3. **No solvent dynamics** - Implicit medium only
4. **Simplified kinetics** - Single reaction type
5. **Graph abstraction** - No bond order/geometry

See `assembly_net/experiments/critical_experiments.py` for full limitations discussion.

## Reproducibility Checklist

- [x] All dependencies listed with versions (`requirements.txt`)
- [x] All experiments runnable with single command
- [x] Figures reproducible from code
- [x] Random seeds fixed (default: 42)
- [x] Unit tests provided (34 tests, all passing)
- [x] Expected outputs documented
- [x] Runtime estimates provided (~3-5 min)
- [x] No external data downloads required

## Citing

```bibtex
@software{assembly_net_2024,
  title = {Assembly-Net: Topology-Aware Learning of Emergent Material Properties},
  author = {Assembly-Net Contributors},
  year = {2024},
  url = {https://github.com/pranavkokati/assembly-net},
  note = {Static models are information-theoretically insufficient for assembly-driven materials}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## References

1. Gillespie, D. T. (1977). Exact stochastic simulation of coupled chemical reactions. *J. Phys. Chem.* 81(25), 2340-2361.
2. Edelsbrunner, H., & Harer, J. (2008). Persistent homology - a survey. *Contemporary Mathematics*, 453, 257-282.
3. Meakin, P. (1988). Fractal aggregates and their fractal measures. *Phase Transitions*, 12(3), 151-203.
4. Ejima, H. et al. (2013). One-step assembly of coordination complexes. *Science* 341(6142), 154-157.
