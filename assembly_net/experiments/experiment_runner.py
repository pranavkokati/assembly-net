"""
Experiment Runner for Assembly-Net.

Provides utilities for running reproducible experiments with
logging, checkpointing, and result aggregation.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from assembly_net.data.dataset import AssemblyDataset, collate_trajectories
from assembly_net.data.synthetic_assembly_simulator.simulator import (
    generate_dataset,
    SimulationParameters,
)
from assembly_net.models.training import Trainer, TrainingConfig


@dataclass
class ExperimentConfig:
    """Configuration for a complete experiment."""

    # Experiment metadata
    name: str = "experiment"
    description: str = ""
    seed: int = 42

    # Data
    num_train_samples: int = 1000
    num_val_samples: int = 200
    num_test_samples: int = 200
    target_property: str = "mechanical_class"
    num_timesteps: int = 32

    # Model
    model_type: str = "temporal_gnn"  # 'temporal_gnn', 'static_gnn', 'topology_only'
    use_topology: bool = True
    gnn_hidden_dim: int = 128
    gnn_num_layers: int = 3
    temporal_hidden_dim: int = 256
    temporal_num_layers: int = 2

    # Training
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    early_stopping_patience: int = 15

    # Output
    output_dir: str = "experiments/results"
    save_model: bool = True
    save_predictions: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        """Create from dictionary."""
        return cls(**d)


class ExperimentRunner:
    """
    Runs reproducible experiments with the Assembly-Net framework.

    Handles:
    - Data generation/loading
    - Model creation
    - Training
    - Evaluation
    - Result logging
    """

    def __init__(self, config: ExperimentConfig):
        """
        Initialize the experiment runner.

        Args:
            config: Experiment configuration.
        """
        self.config = config
        self.output_dir = Path(config.output_dir) / config.name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set seeds
        self._set_seeds(config.seed)

        # Initialize tracking
        self.results: Dict[str, Any] = {}
        self.start_time = None
        self.end_time = None

    def _set_seeds(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def generate_data(self) -> Tuple[List, List, List]:
        """
        Generate synthetic training, validation, and test data.

        Returns:
            Tuple of (train_trajectories, val_trajectories, test_trajectories)
        """
        total_samples = (
            self.config.num_train_samples
            + self.config.num_val_samples
            + self.config.num_test_samples
        )

        print(f"Generating {total_samples} trajectories...")

        trajectories = generate_dataset(
            num_samples=total_samples,
            seed=self.config.seed,
        )

        # Split
        train_end = self.config.num_train_samples
        val_end = train_end + self.config.num_val_samples

        train_data = trajectories[:train_end]
        val_data = trajectories[train_end:val_end]
        test_data = trajectories[val_end:]

        return train_data, val_data, test_data

    def create_dataloaders(
        self,
        train_data: List,
        val_data: List,
        test_data: List,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create PyTorch DataLoaders."""
        train_dataset = AssemblyDataset(
            trajectories=train_data,
            target_property=self.config.target_property,
            num_timesteps=self.config.num_timesteps,
        )
        val_dataset = AssemblyDataset(
            trajectories=val_data,
            target_property=self.config.target_property,
            num_timesteps=self.config.num_timesteps,
        )
        test_dataset = AssemblyDataset(
            trajectories=test_data,
            target_property=self.config.target_property,
            num_timesteps=self.config.num_timesteps,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_trajectories,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_trajectories,
            num_workers=0,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_trajectories,
            num_workers=0,
        )

        return train_loader, val_loader, test_loader

    def create_model(self):
        """Create model based on configuration."""
        from assembly_net.models.temporal_gnn import create_temporal_gnn
        from assembly_net.models.baseline_static_gnn import (
            create_static_baseline,
            FinalStructureBaseline,
            TopologyOnlyBaseline,
        )
        from assembly_net.models.property_heads import (
            MechanicalPropertyHead,
            AssemblyPropertyPredictor,
        )

        if self.config.model_type == "temporal_gnn":
            encoder = create_temporal_gnn(
                hidden_dim=self.config.gnn_hidden_dim,
                num_gnn_layers=self.config.gnn_num_layers,
                num_temporal_layers=self.config.temporal_num_layers,
                use_topology=self.config.use_topology,
            )
        elif self.config.model_type == "static_gnn":
            base_gnn = create_static_baseline(
                hidden_dim=self.config.gnn_hidden_dim,
                num_layers=self.config.gnn_num_layers,
            )
            encoder = FinalStructureBaseline(base_gnn.config)
        elif self.config.model_type == "topology_only":
            encoder = TopologyOnlyBaseline(
                hidden_dim=self.config.temporal_hidden_dim,
            )
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

        # Create prediction head
        head = MechanicalPropertyHead(
            input_dim=256,  # Default output dim
            num_classes=3,
        )

        model = AssemblyPropertyPredictor(encoder, head)
        return model

    @torch.no_grad()
    def evaluate(
        self,
        model,
        test_loader: DataLoader,
        device: torch.device,
    ) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Returns:
            Dictionary of evaluation metrics.
        """
        model.eval()
        all_preds = []
        all_targets = []
        all_probs = []

        for batch in test_loader:
            graphs = [g.to(device) for g in batch["graphs"]]
            targets = batch["targets"].to(device)

            topology = None
            if batch["topology"] is not None:
                topology = batch["topology"].to(device)

            mask = batch["mask"].to(device) if "mask" in batch else None

            predictions = model(graphs, topology=topology, mask=mask)

            if isinstance(predictions, dict):
                logits = predictions["logits"]
            else:
                logits = predictions

            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)

        # Compute metrics
        accuracy = 100.0 * (all_preds == all_targets).mean()

        # Per-class metrics
        metrics = {"accuracy": accuracy}

        for c in np.unique(all_targets):
            mask = all_targets == c
            if mask.sum() > 0:
                # Class accuracy
                metrics[f"accuracy_class_{c}"] = (
                    100.0 * (all_preds[mask] == c).mean()
                )
                # Precision
                pred_c = all_preds == c
                if pred_c.sum() > 0:
                    metrics[f"precision_class_{c}"] = (
                        100.0 * (all_targets[pred_c] == c).mean()
                    )

        # Confusion matrix
        num_classes = len(np.unique(all_targets))
        confusion = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(all_targets, all_preds):
            confusion[t, p] += 1
        metrics["confusion_matrix"] = confusion.tolist()

        return metrics

    def run(self) -> Dict[str, Any]:
        """
        Run the complete experiment.

        Returns:
            Dictionary of results.
        """
        self.start_time = datetime.now()
        print(f"Starting experiment: {self.config.name}")
        print(f"Output directory: {self.output_dir}")

        # Save config
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Generate data
        train_data, val_data, test_data = self.generate_data()
        train_loader, val_loader, test_loader = self.create_dataloaders(
            train_data, val_data, test_data
        )

        # Create model
        model = self.create_model()
        print(f"Model type: {self.config.model_type}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Device
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Device: {device}")

        # Training config
        train_config = TrainingConfig(
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            batch_size=self.config.batch_size,
            num_epochs=self.config.num_epochs,
            patience=self.config.early_stopping_patience,
            device=str(device),
        )

        # Train
        trainer = Trainer(
            model=model,
            config=train_config,
            train_loader=train_loader,
            val_loader=val_loader,
        )
        history = trainer.train()

        # Evaluate
        test_metrics = self.evaluate(model, test_loader, device)
        print(f"\nTest Results:")
        print(f"  Accuracy: {test_metrics['accuracy']:.2f}%")

        # Save results
        self.results = {
            "config": self.config.to_dict(),
            "history": history,
            "test_metrics": test_metrics,
            "model_params": sum(p.numel() for p in model.parameters()),
        }

        self.end_time = datetime.now()
        self.results["duration_seconds"] = (
            self.end_time - self.start_time
        ).total_seconds()

        # Save
        with open(self.output_dir / "results.json", "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        if self.config.save_model:
            trainer.save(self.output_dir / "model.pt")

        print(f"\nExperiment completed in {self.results['duration_seconds']:.1f}s")
        print(f"Results saved to {self.output_dir}")

        return self.results


def log_metrics(
    metrics: Dict[str, float],
    step: int,
    prefix: str = "",
    writer=None,
) -> None:
    """Log metrics to tensorboard and console."""
    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            full_name = f"{prefix}/{name}" if prefix else name
            if writer is not None:
                writer.add_scalar(full_name, value, step)


def run_experiment(
    config: Union[ExperimentConfig, Dict],
) -> Dict[str, Any]:
    """
    Convenience function to run an experiment.

    Args:
        config: Experiment configuration or dictionary.

    Returns:
        Experiment results.
    """
    if isinstance(config, dict):
        config = ExperimentConfig.from_dict(config)

    runner = ExperimentRunner(config)
    return runner.run()


def compare_experiments(
    results: List[Dict[str, Any]],
    metric: str = "accuracy",
) -> None:
    """
    Compare results from multiple experiments.

    Args:
        results: List of experiment result dictionaries.
        metric: Metric to compare.
    """
    print(f"\nExperiment Comparison ({metric}):")
    print("-" * 50)

    for result in results:
        name = result["config"]["name"]
        value = result["test_metrics"].get(metric, "N/A")
        if isinstance(value, float):
            print(f"  {name}: {value:.2f}")
        else:
            print(f"  {name}: {value}")
