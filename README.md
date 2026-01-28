# Assembly-Net

**Topology-Aware Learning of Emergent Material Properties from Coordination-Network Assembly Graphs**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

Assembly-Net is a research framework for predicting emergent material properties from the **assembly process** itself, rather than just the final structure. This represents a paradigm shift from structure-centric to **process-centric** materials machine learning.

### Key Innovation

Most materials ML treats materials as static objects. However, many materials of current interest (metal–phenolic networks, MOFs, gels, supramolecular assemblies, bio-inspired networks) derive their properties from **how they assemble**, not just what they are.

This framework learns material properties from **time-evolving coordination graphs** representing the assembly process, incorporating **topological data analysis** as a first-class feature.

## Scientific Novelty

1. **Assembly-centric representation**: Materials are represented as sequences of graph states showing how the network grows over time
2. **Topology-aware descriptors**: Betti numbers, persistence diagrams, loop formation rates, and rigidity metrics serve as primary ML inputs
3. **Emergent property prediction**: Predict mechanical resilience, diffusion barriers, optical properties, and stimulus responsiveness
4. **Hybrid model architecture**: Combines Graph Neural Networks (local chemistry) with temporal attention (assembly dynamics) and topology injection (persistent homology)

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/assembly-net.git
cd assembly-net

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"

# For notebooks
pip install -e ".[notebooks]"
```

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- PyTorch Geometric ≥ 2.3
- giotto-tda ≥ 0.6 (for persistent homology)
- NetworkX ≥ 2.6

## Quick Start

### 1. Generate Assembly Trajectories

```python
from assembly_net.data.synthetic_assembly_simulator import (
    CoordinationNetworkSimulator,
    SimulationParameters,
    generate_dataset,
)

# Configure simulation
params = SimulationParameters(
    num_metal_ions=25,
    num_ligands=50,
    metal_valency=6,
    ligand_valency=2,
    ph=7.0,
    total_time=100.0,
    seed=42,
)

# Run simulation
simulator = CoordinationNetworkSimulator(params)
trajectory = simulator.run()

print(f"Generated {len(trajectory.states)} states")
print(f"Labels: {trajectory.labels}")
```

### 2. Compute Topological Features

```python
from assembly_net.topology import (
    PersistentHomologyComputer,
    RigidityAnalyzer,
    TopologicalDescriptorExtractor,
)

# Persistent homology
ph_computer = PersistentHomologyComputer(max_dimension=1)
features = ph_computer.compute(trajectory.final_state.graph)
print(f"Betti numbers: {features.betti_numbers}")

# Rigidity analysis
rigidity = RigidityAnalyzer(dimension=3)
metrics = rigidity.analyze(trajectory.final_state.graph)
print(f"Is rigid: {metrics.is_rigid}")
print(f"Mean coordination: {metrics.mean_coordination}")
```

### 3. Train a Model

```python
from assembly_net.models import TemporalAssemblyGNN, create_temporal_gnn
from assembly_net.models.property_heads import MechanicalPropertyHead, AssemblyPropertyPredictor
from assembly_net.models.training import Trainer, TrainingConfig
from assembly_net.data.dataset import AssemblyDataset, collate_trajectories
from torch.utils.data import DataLoader

# Generate dataset
trajectories = generate_dataset(num_samples=1000, seed=42)

# Create dataset and loader
dataset = AssemblyDataset(
    trajectories=trajectories,
    target_property="mechanical_class",
    num_timesteps=32,
)
train_loader = DataLoader(dataset, batch_size=32, collate_fn=collate_trajectories)

# Create model
encoder = create_temporal_gnn(use_topology=True)
head = MechanicalPropertyHead(input_dim=256, num_classes=3)
model = AssemblyPropertyPredictor(encoder, head)

# Train
config = TrainingConfig(num_epochs=50, learning_rate=1e-4)
trainer = Trainer(model, config, train_loader)
history = trainer.train()
```

## Project Structure

```
assembly_net/
├── data/
│   ├── core.py                    # Core data structures (AssemblyGraph, NetworkState)
│   ├── dataset.py                 # PyTorch Dataset implementation
│   └── synthetic_assembly_simulator/
│       └── simulator.py           # Stochastic coordination network simulator
├── topology/
│   ├── persistent_homology.py     # Betti numbers, persistence diagrams
│   ├── graph_rigidity.py          # Maxwell counting, rigidity analysis
│   └── descriptors.py             # Full topological descriptor extraction
├── models/
│   ├── temporal_gnn.py            # Temporal Assembly GNN
│   ├── baseline_static_gnn.py     # Static GNN baselines
│   ├── property_heads.py          # Property prediction heads
│   └── training.py                # Training utilities
├── experiments/
│   ├── experiment_runner.py       # Experiment management
│   ├── ablation_topology_vs_structure.py  # Topology ablation study
│   └── assembly_order_randomization.py    # Assembly order experiment
├── utils/
│   ├── config.py                  # Configuration management
│   ├── visualization.py           # Plotting utilities
│   └── metrics.py                 # Evaluation metrics
└── notebooks/
    ├── visualization_of_network_growth.ipynb
    └── topology_evolution.ipynb
```

## Key Research Questions

This framework enables investigation of fundamental questions in materials science:

1. **Does topology evolution outperform final-structure ML?**
   - Run ablation studies comparing temporal GNN with static baselines

2. **Which topological invariants matter most?**
   - Analyze feature importance across Betti numbers, persistence, rigidity

3. **Can assembly history distinguish identical final structures?**
   - Test with paired trajectories that reach the same endpoint differently

## Experiments

### Ablation: Topology vs Structure

```bash
python -m assembly_net.experiments.ablation_topology_vs_structure
```

This compares:
- Full temporal GNN with topology injection
- Temporal GNN without topology
- Static GNN on final structure
- Topology-only model

### Assembly Order Randomization

```bash
python -m assembly_net.experiments.assembly_order_randomization
```

Tests whether models can distinguish materials that have identical final structures but different assembly histories.

## Property Labels

The simulator generates ground-truth labels based on network topology:

| Property | Classes | Physics Basis |
|----------|---------|---------------|
| Mechanical | Brittle, Ductile, Robust | Loop density, percolation, rigidity |
| Diffusion | Fast, Slow, Barrier | Network density, largest component |
| Optical | Transparent, Narrow, Broadband | Metal content, coordination variance |
| Responsiveness | Stable, pH-responsive, Ion-responsive | Bond pH sensitivity |

## Model Architecture

```
                    ┌─────────────────────────────────────────┐
                    │         Assembly Trajectory              │
                    │    [G₀, G₁, G₂, ..., Gₜ] + Topology     │
                    └────────────────┬────────────────────────┘
                                     │
                    ┌────────────────▼────────────────────────┐
                    │           GNN Encoder (per timestep)     │
                    │         GAT/GCN/GIN + Pooling            │
                    └────────────────┬────────────────────────┘
                                     │
                    ┌────────────────▼────────────────────────┐
                    │        Topology Injection (FiLM)         │
                    │   Persistent homology + Rigidity         │
                    └────────────────┬────────────────────────┘
                                     │
                    ┌────────────────▼────────────────────────┐
                    │     Temporal Transformer/LSTM            │
                    │        Assembly dynamics                 │
                    └────────────────┬────────────────────────┘
                                     │
                    ┌────────────────▼────────────────────────┐
                    │        Property Prediction Head          │
                    │   Classification + Regression            │
                    └─────────────────────────────────────────┘
```

## Data Generation

The stochastic simulator models:

- **Metal-ligand coordination**: Kinetic Monte Carlo with Gillespie algorithm
- **pH-dependent binding**: Protonation equilibria affecting bond formation
- **Concentration effects**: Mass-action kinetics
- **Network evolution**: Edge formation/dissociation over time

Labels are computed from physics-motivated rules:
- Percolation → gel-like mechanical behavior
- High loop density → mechanical robustness
- Sparse tree-like networks → brittle behavior

## Citing

If you use Assembly-Net in your research, please cite:

```bibtex
@software{assembly_net,
  title = {Assembly-Net: Topology-Aware Learning of Emergent Material Properties},
  year = {2024},
  url = {https://github.com/your-username/assembly-net}
}
```

## Contributing

Contributions are welcome! Please see our contributing guidelines for:
- Adding new simulation parameters
- Implementing additional topological descriptors
- Adding new model architectures
- Contributing experimental protocols

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

This framework builds on foundational work in:
- Topological data analysis for materials science
- Graph neural networks for molecular property prediction
- Temporal modeling for dynamic systems
