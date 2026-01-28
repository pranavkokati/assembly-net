"""
Ablation Study: Topology vs Structure.

This experiment investigates the key research question:
"Does topology evolution outperform final-structure ML in predicting properties?"

We compare:
1. Full temporal GNN with topology injection
2. Temporal GNN without topology
3. Static GNN on final structure
4. Static GNN with final topology
5. Topology-only model (no GNN)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from assembly_net.experiments.experiment_runner import (
    ExperimentConfig,
    ExperimentRunner,
)


@dataclass
class AblationConfig:
    """Configuration for topology vs structure ablation."""

    # Experiment settings
    base_name: str = "ablation_topology_vs_structure"
    seed: int = 42
    num_seeds: int = 3  # Number of random seeds for averaging

    # Data
    num_train_samples: int = 1000
    num_val_samples: int = 200
    num_test_samples: int = 200
    target_property: str = "mechanical_class"

    # Training
    num_epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4

    # Output
    output_dir: str = "experiments/ablation_topology"


class TopologyVsStructureAblation:
    """
    Ablation study comparing topology-aware vs structure-only models.

    This is a key experiment to demonstrate the value of:
    1. Temporal/assembly information
    2. Topological features
    """

    # Model configurations to compare
    MODEL_CONFIGS = [
        {
            "name": "temporal_gnn_with_topology",
            "model_type": "temporal_gnn",
            "use_topology": True,
            "description": "Full model with temporal GNN and topology injection",
        },
        {
            "name": "temporal_gnn_no_topology",
            "model_type": "temporal_gnn",
            "use_topology": False,
            "description": "Temporal GNN without topology features",
        },
        {
            "name": "static_gnn_final_structure",
            "model_type": "static_gnn",
            "use_topology": False,
            "description": "Static GNN on final structure only",
        },
        {
            "name": "static_gnn_with_topology",
            "model_type": "static_gnn",
            "use_topology": True,
            "description": "Static GNN with final topology features",
        },
        {
            "name": "topology_only",
            "model_type": "topology_only",
            "use_topology": True,
            "description": "Topology evolution only, no GNN",
        },
    ]

    def __init__(self, config: AblationConfig):
        """
        Initialize the ablation study.

        Args:
            config: Ablation configuration.
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results: Dict[str, List[Dict]] = {}

    def run(self) -> Dict[str, Any]:
        """
        Run the complete ablation study.

        Returns:
            Aggregated results.
        """
        print("=" * 60)
        print("ABLATION STUDY: Topology vs Structure")
        print("=" * 60)

        # Run each model configuration
        for model_config in self.MODEL_CONFIGS:
            model_name = model_config["name"]
            print(f"\n--- Running: {model_name} ---")
            print(f"Description: {model_config['description']}")

            self.results[model_name] = []

            # Run with multiple seeds
            for seed_idx in range(self.config.num_seeds):
                seed = self.config.seed + seed_idx
                print(f"\n  Seed {seed_idx + 1}/{self.config.num_seeds} (seed={seed})")

                # Create experiment config
                exp_config = ExperimentConfig(
                    name=f"{model_name}_seed{seed}",
                    description=model_config["description"],
                    seed=seed,
                    num_train_samples=self.config.num_train_samples,
                    num_val_samples=self.config.num_val_samples,
                    num_test_samples=self.config.num_test_samples,
                    target_property=self.config.target_property,
                    model_type=model_config["model_type"],
                    use_topology=model_config["use_topology"],
                    num_epochs=self.config.num_epochs,
                    batch_size=self.config.batch_size,
                    learning_rate=self.config.learning_rate,
                    output_dir=str(self.output_dir / model_name),
                    save_model=False,  # Don't save individual models
                )

                # Run experiment
                runner = ExperimentRunner(exp_config)
                result = runner.run()
                self.results[model_name].append(result)

        # Aggregate results
        aggregated = self._aggregate_results()

        # Save
        self._save_results(aggregated)

        # Print summary
        self._print_summary(aggregated)

        return aggregated

    def _aggregate_results(self) -> Dict[str, Any]:
        """Aggregate results across seeds."""
        aggregated = {}

        for model_name, seed_results in self.results.items():
            accuracies = [r["test_metrics"]["accuracy"] for r in seed_results]

            aggregated[model_name] = {
                "accuracy_mean": np.mean(accuracies),
                "accuracy_std": np.std(accuracies),
                "accuracy_min": np.min(accuracies),
                "accuracy_max": np.max(accuracies),
                "num_seeds": len(seed_results),
                "all_accuracies": accuracies,
            }

            # Per-class accuracies
            for key in seed_results[0]["test_metrics"]:
                if key.startswith("accuracy_class_"):
                    values = [r["test_metrics"][key] for r in seed_results]
                    aggregated[model_name][f"{key}_mean"] = np.mean(values)
                    aggregated[model_name][f"{key}_std"] = np.std(values)

        return aggregated

    def _save_results(self, aggregated: Dict[str, Any]) -> None:
        """Save results to file."""
        # Save aggregated results
        with open(self.output_dir / "aggregated_results.json", "w") as f:
            json.dump(aggregated, f, indent=2, default=float)

        # Save raw results
        raw_results = {}
        for model_name, seed_results in self.results.items():
            raw_results[model_name] = [
                {
                    "seed": r["config"]["seed"],
                    "test_metrics": r["test_metrics"],
                    "duration": r.get("duration_seconds", 0),
                }
                for r in seed_results
            ]

        with open(self.output_dir / "raw_results.json", "w") as f:
            json.dump(raw_results, f, indent=2, default=float)

    def _print_summary(self, aggregated: Dict[str, Any]) -> None:
        """Print summary of results."""
        print("\n" + "=" * 60)
        print("ABLATION STUDY RESULTS")
        print("=" * 60)

        # Sort by accuracy
        sorted_models = sorted(
            aggregated.items(),
            key=lambda x: x[1]["accuracy_mean"],
            reverse=True,
        )

        print(f"\n{'Model':<35} {'Accuracy':<20}")
        print("-" * 55)

        for model_name, stats in sorted_models:
            acc_str = f"{stats['accuracy_mean']:.2f} +/- {stats['accuracy_std']:.2f}%"
            print(f"{model_name:<35} {acc_str:<20}")

        # Key findings
        best_model = sorted_models[0][0]
        worst_model = sorted_models[-1][0]
        best_acc = sorted_models[0][1]["accuracy_mean"]
        worst_acc = sorted_models[-1][1]["accuracy_mean"]

        print(f"\nKey Findings:")
        print(f"  Best model: {best_model} ({best_acc:.2f}%)")
        print(f"  Worst model: {worst_model} ({worst_acc:.2f}%)")
        print(f"  Improvement: {best_acc - worst_acc:.2f}%")

        # Topology contribution
        if "temporal_gnn_with_topology" in aggregated and "temporal_gnn_no_topology" in aggregated:
            with_topo = aggregated["temporal_gnn_with_topology"]["accuracy_mean"]
            without_topo = aggregated["temporal_gnn_no_topology"]["accuracy_mean"]
            print(f"\n  Topology contribution (temporal): {with_topo - without_topo:+.2f}%")

        # Temporal contribution
        if "temporal_gnn_with_topology" in aggregated and "static_gnn_with_topology" in aggregated:
            temporal = aggregated["temporal_gnn_with_topology"]["accuracy_mean"]
            static = aggregated["static_gnn_with_topology"]["accuracy_mean"]
            print(f"  Temporal contribution: {temporal - static:+.2f}%")


def run_topology_ablation(
    num_train_samples: int = 1000,
    num_epochs: int = 50,
    num_seeds: int = 3,
    output_dir: str = "experiments/ablation_topology",
) -> Dict[str, Any]:
    """
    Convenience function to run topology ablation study.

    Args:
        num_train_samples: Number of training samples.
        num_epochs: Number of training epochs.
        num_seeds: Number of random seeds for averaging.
        output_dir: Output directory.

    Returns:
        Aggregated results dictionary.
    """
    config = AblationConfig(
        num_train_samples=num_train_samples,
        num_epochs=num_epochs,
        num_seeds=num_seeds,
        output_dir=output_dir,
    )

    ablation = TopologyVsStructureAblation(config)
    return ablation.run()


if __name__ == "__main__":
    # Run ablation study with default settings
    results = run_topology_ablation(
        num_train_samples=500,
        num_epochs=30,
        num_seeds=2,
    )
