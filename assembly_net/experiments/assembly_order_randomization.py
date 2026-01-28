"""
Experiment: Assembly Order Randomization.

This experiment addresses the key research question:
"Can two materials with identical final graphs but different assembly
histories be distinguished?"

We test this by:
1. Generating trajectories with different assembly orderings
2. Training models to distinguish between assembly histories
3. Comparing temporal-aware vs structure-only models
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from assembly_net.data.core import (
    AssemblyGraph,
    AssemblyTrajectory,
    NetworkState,
    EdgeFeatures,
    EdgeType,
)
from assembly_net.data.synthetic_assembly_simulator.simulator import (
    CoordinationNetworkSimulator,
    SimulationParameters,
)
from assembly_net.experiments.experiment_runner import (
    ExperimentConfig,
    ExperimentRunner,
)


@dataclass
class OrderRandomizationConfig:
    """Configuration for assembly order randomization experiment."""

    # Experiment settings
    name: str = "assembly_order_randomization"
    seed: int = 42
    num_seeds: int = 3

    # Data
    num_pairs: int = 500  # Number of trajectory pairs
    target_property: str = "assembly_order"  # Custom binary classification

    # Training
    num_epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4

    # Output
    output_dir: str = "experiments/order_randomization"


def generate_trajectory_with_order(
    final_graph: AssemblyGraph,
    order: str = "sequential",  # 'sequential', 'random', 'reverse'
    num_steps: int = 50,
    seed: int = 42,
) -> AssemblyTrajectory:
    """
    Generate a trajectory that builds to a target final graph
    using a specified assembly order.

    Args:
        final_graph: The target final graph.
        order: Assembly order ('sequential', 'random', 'reverse').
        num_steps: Number of timesteps.
        seed: Random seed.

    Returns:
        AssemblyTrajectory building to the final graph.
    """
    rng = np.random.default_rng(seed)

    # Get edges from final graph
    edges = list(final_graph.edges)
    edge_features = list(final_graph.edge_features)

    # Order edges based on strategy
    if order == "sequential":
        # Add edges in original order
        edge_order = list(range(len(edges)))
    elif order == "random":
        # Random permutation
        edge_order = rng.permutation(len(edges)).tolist()
    elif order == "reverse":
        # Reverse order
        edge_order = list(range(len(edges) - 1, -1, -1))
    else:
        raise ValueError(f"Unknown order: {order}")

    # Create trajectory
    trajectory = AssemblyTrajectory()

    # Start with nodes only (no edges)
    current_graph = AssemblyGraph(
        num_nodes=final_graph.num_nodes,
        node_features=[nf.__class__(**nf.__dict__) for nf in final_graph.node_features],
        edges=[],
        edge_features=[],
        time=0.0,
    )
    # Reset coordination counts
    for nf in current_graph.node_features:
        nf.current_coordination = 0

    # Save initial state
    initial_state = NetworkState(graph=current_graph.copy(), time=0.0)
    initial_state.compute_properties()
    trajectory.add_state(initial_state)

    # Add edges incrementally
    edges_per_step = max(1, len(edges) // num_steps)
    edge_idx = 0
    time = 0.0
    dt = 100.0 / num_steps

    for step in range(num_steps):
        time += dt

        # Add some edges
        edges_to_add = min(edges_per_step, len(edges) - edge_idx)
        for _ in range(edges_to_add):
            if edge_idx >= len(edges):
                break

            i = edge_order[edge_idx]
            src, tgt = edges[i]
            ef = edge_features[i]

            # Update formation time
            new_ef = EdgeFeatures(
                edge_type=ef.edge_type,
                bond_strength=ef.bond_strength,
                formation_time=time,
                ph_sensitivity=ef.ph_sensitivity,
                formation_rate=ef.formation_rate,
            )

            current_graph.add_edge(src, tgt, new_ef)
            edge_idx += 1

        # Save state
        current_graph.time = time
        state = NetworkState(graph=current_graph.copy(), time=time)
        state.compute_properties()
        trajectory.add_state(state)

    return trajectory


def generate_paired_trajectories(
    num_pairs: int,
    seed: int = 42,
) -> List[Tuple[AssemblyTrajectory, AssemblyTrajectory, int]]:
    """
    Generate pairs of trajectories with same final structure but
    different assembly orders.

    Args:
        num_pairs: Number of pairs to generate.
        seed: Random seed.

    Returns:
        List of (trajectory_1, trajectory_2, same_order_label) tuples.
        same_order_label is 1 if trajectories have same order, 0 otherwise.
    """
    rng = np.random.default_rng(seed)
    pairs = []

    for i in range(num_pairs):
        # Generate a base trajectory
        params = SimulationParameters(
            num_metal_ions=rng.integers(15, 30),
            num_ligands=rng.integers(30, 60),
            ph=7.0,
            total_time=100.0,
            seed=seed + i,
        )

        simulator = CoordinationNetworkSimulator(params)
        base_trajectory = simulator.run()
        final_graph = base_trajectory.final_state.graph

        # Generate two trajectories with different orders
        if rng.random() > 0.5:
            # Same order (label = 1)
            order1 = "sequential"
            order2 = "sequential"
            label = 1
        else:
            # Different orders (label = 0)
            orders = ["sequential", "random", "reverse"]
            order1, order2 = rng.choice(orders, size=2, replace=False)
            label = 0

        traj1 = generate_trajectory_with_order(
            final_graph, order=order1, seed=seed + i * 2
        )
        traj2 = generate_trajectory_with_order(
            final_graph, order=order2, seed=seed + i * 2 + 1
        )

        # Store order information in labels
        traj1.labels = {"assembly_order": order1, "pair_id": i}
        traj2.labels = {"assembly_order": order2, "pair_id": i}

        pairs.append((traj1, traj2, label))

    return pairs


class PairedTrajectoryDataset(Dataset):
    """Dataset for paired trajectory classification."""

    def __init__(
        self,
        pairs: List[Tuple[AssemblyTrajectory, AssemblyTrajectory, int]],
        num_timesteps: int = 32,
        embed_dim: int = 32,
    ):
        self.pairs = pairs
        self.num_timesteps = num_timesteps
        self.embed_dim = embed_dim

    def __len__(self):
        return len(self.pairs) * 2  # Each pair gives two samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pair_idx = idx // 2
        is_second = idx % 2

        traj1, traj2, label = self.pairs[pair_idx]

        # Select trajectory
        traj = traj2 if is_second else traj1

        # Subsample
        subsampled = traj.subsample(self.num_timesteps)

        # Convert to PyG format
        graphs = subsampled.to_pyg_sequence(self.embed_dim)

        # Compute topology features
        topology = self._compute_topology(subsampled)

        # For this experiment, we want to classify assembly order
        # Use binary label: 0 = sequential, 1 = non-sequential
        order = traj.labels.get("assembly_order", "sequential")
        target = 0 if order == "sequential" else 1

        return {
            "graphs": graphs,
            "times": torch.tensor([s.time for s in subsampled.states], dtype=torch.float),
            "target": torch.tensor(target, dtype=torch.long),
            "topology": topology,
            "num_timesteps": len(graphs),
            "idx": idx,
        }

    def _compute_topology(self, trajectory: AssemblyTrajectory) -> torch.Tensor:
        """Compute topology features for trajectory."""
        features = []
        for state in trajectory.states:
            if state.num_components is None:
                state.compute_properties()

            num_nodes = max(1, state.graph.num_nodes)
            feat = [
                state.num_components / num_nodes,
                (state.largest_component_size or 0) / num_nodes,
                (state.num_cycles or 0) / num_nodes,
                state.clustering_coefficient or 0.0,
                state.density or 0.0,
                state.mean_degree or 0.0,
                1.0 if state.is_percolated else 0.0,
            ]
            features.append(feat)

        return torch.tensor(features, dtype=torch.float)


class AssemblyOrderRandomization:
    """
    Experiment to test if models can distinguish assembly orders.

    Key question: Can temporal models distinguish between trajectories
    that reach the same final state through different paths?
    """

    def __init__(self, config: OrderRandomizationConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}

    def run(self) -> Dict[str, Any]:
        """Run the experiment."""
        print("=" * 60)
        print("EXPERIMENT: Assembly Order Randomization")
        print("=" * 60)

        # Generate paired data
        print(f"\nGenerating {self.config.num_pairs} trajectory pairs...")
        pairs = generate_paired_trajectories(
            self.config.num_pairs,
            seed=self.config.seed,
        )

        # Split into train/val/test
        n_train = int(0.7 * len(pairs))
        n_val = int(0.15 * len(pairs))

        train_pairs = pairs[:n_train]
        val_pairs = pairs[n_train : n_train + n_val]
        test_pairs = pairs[n_train + n_val :]

        print(f"Train pairs: {len(train_pairs)}")
        print(f"Val pairs: {len(val_pairs)}")
        print(f"Test pairs: {len(test_pairs)}")

        # Create datasets
        from assembly_net.data.dataset import collate_trajectories

        train_dataset = PairedTrajectoryDataset(train_pairs)
        val_dataset = PairedTrajectoryDataset(val_pairs)
        test_dataset = PairedTrajectoryDataset(test_pairs)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_trajectories,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_trajectories,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_trajectories,
        )

        # Test different models
        model_types = [
            ("temporal_gnn", True),  # Temporal with topology
            ("temporal_gnn", False),  # Temporal without topology
            ("static_gnn", True),  # Static with topology
            ("static_gnn", False),  # Static without topology
        ]

        for model_type, use_topology in model_types:
            name = f"{model_type}_topo{use_topology}"
            print(f"\n--- Testing: {name} ---")

            result = self._train_and_evaluate(
                name=name,
                model_type=model_type,
                use_topology=use_topology,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
            )
            self.results[name] = result

        # Save and summarize
        self._save_results()
        self._print_summary()

        return self.results

    def _train_and_evaluate(
        self,
        name: str,
        model_type: str,
        use_topology: bool,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
    ) -> Dict[str, Any]:
        """Train and evaluate a model."""
        from assembly_net.models.temporal_gnn import create_temporal_gnn
        from assembly_net.models.baseline_static_gnn import (
            FinalStructureBaseline,
            StaticGNNConfig,
        )
        from assembly_net.models.property_heads import (
            PropertyPredictionHead,
            AssemblyPropertyPredictor,
        )
        from assembly_net.models.training import Trainer, TrainingConfig
        import torch.nn as nn

        # Create model
        if model_type == "temporal_gnn":
            encoder = create_temporal_gnn(use_topology=use_topology)
        else:
            config = StaticGNNConfig()
            encoder = FinalStructureBaseline(config, use_final_topology=use_topology)

        # Binary classification head
        head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2),
        )

        class SimplePredictor(nn.Module):
            def __init__(self, enc, h):
                super().__init__()
                self.encoder = enc
                self.head = h

            def forward(self, graphs, topology=None, mask=None):
                emb = self.encoder(graphs, topology=topology, mask=mask)
                return {"logits": self.head(emb)}

        model = SimplePredictor(encoder, head)

        # Device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Train
        train_config = TrainingConfig(
            num_epochs=self.config.num_epochs,
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            device=str(device),
        )

        trainer = Trainer(
            model=model,
            config=train_config,
            train_loader=train_loader,
            val_loader=val_loader,
        )
        history = trainer.train()

        # Evaluate
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_loader:
                graphs = [g.to(device) for g in batch["graphs"]]
                targets = batch["targets"].to(device)
                topology = batch["topology"].to(device) if batch["topology"] is not None else None
                mask = batch["mask"].to(device) if "mask" in batch else None

                output = model(graphs, topology=topology, mask=mask)
                preds = torch.argmax(output["logits"], dim=-1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        accuracy = 100.0 * correct / total

        return {
            "accuracy": accuracy,
            "history": history,
            "model_type": model_type,
            "use_topology": use_topology,
        }

    def _save_results(self) -> None:
        """Save results."""
        with open(self.output_dir / "results.json", "w") as f:
            # Convert history to serializable format
            serializable = {}
            for name, result in self.results.items():
                serializable[name] = {
                    "accuracy": result["accuracy"],
                    "model_type": result["model_type"],
                    "use_topology": result["use_topology"],
                }
            json.dump(serializable, f, indent=2)

    def _print_summary(self) -> None:
        """Print summary."""
        print("\n" + "=" * 60)
        print("ASSEMBLY ORDER RANDOMIZATION RESULTS")
        print("=" * 60)

        print(f"\n{'Model':<35} {'Accuracy':<15}")
        print("-" * 50)

        # Sort by accuracy
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1]["accuracy"],
            reverse=True,
        )

        for name, result in sorted_results:
            print(f"{name:<35} {result['accuracy']:.2f}%")

        # Analysis
        temporal_with_topo = self.results.get("temporal_gnn_topoTrue", {}).get("accuracy", 0)
        static_with_topo = self.results.get("static_gnn_topoTrue", {}).get("accuracy", 0)

        print(f"\nKey Finding:")
        print(f"  Temporal model advantage: {temporal_with_topo - static_with_topo:.2f}%")

        if temporal_with_topo > 60 and static_with_topo < 55:
            print("  -> Temporal models CAN distinguish assembly orders!")
            print("  -> Static models perform near random (50%)")
        elif temporal_with_topo > static_with_topo:
            print("  -> Temporal models show some advantage in distinguishing orders")


def run_order_randomization_experiment(
    num_pairs: int = 500,
    num_epochs: int = 50,
    output_dir: str = "experiments/order_randomization",
) -> Dict[str, Any]:
    """
    Convenience function to run order randomization experiment.

    Args:
        num_pairs: Number of trajectory pairs.
        num_epochs: Training epochs.
        output_dir: Output directory.

    Returns:
        Results dictionary.
    """
    config = OrderRandomizationConfig(
        num_pairs=num_pairs,
        num_epochs=num_epochs,
        output_dir=output_dir,
    )

    experiment = AssemblyOrderRandomization(config)
    return experiment.run()


if __name__ == "__main__":
    results = run_order_randomization_experiment(
        num_pairs=200,
        num_epochs=20,
    )
