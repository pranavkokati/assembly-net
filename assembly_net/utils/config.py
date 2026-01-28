"""
Configuration management utilities for Assembly-Net.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Union

import yaml

T = TypeVar("T", bound="Config")


@dataclass
class Config:
    """
    Base configuration class with save/load functionality.

    Supports JSON and YAML formats.
    """

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def save(self, path: Union[str, Path]) -> None:
        """
        Save configuration to file.

        Supports .json and .yaml/.yml extensions.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict()

        if path.suffix == ".json":
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        elif path.suffix in [".yaml", ".yml"]:
            with open(path, "w") as f:
                yaml.safe_dump(data, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    @classmethod
    def load(cls: Type[T], path: Union[str, Path]) -> T:
        """
        Load configuration from file.

        Supports .json and .yaml/.yml extensions.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        if path.suffix == ".json":
            with open(path) as f:
                data = json.load(f)
        elif path.suffix in [".yaml", ".yml"]:
            with open(path) as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create configuration from dictionary."""
        # Filter to only valid fields
        valid_fields = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)

    def update(self, **kwargs) -> "Config":
        """Create a new config with updated values."""
        data = self.to_dict()
        data.update(kwargs)
        return self.__class__.from_dict(data)


@dataclass
class DataConfig(Config):
    """Configuration for data generation and loading."""

    # Simulation parameters
    num_metal_ions: int = 20
    num_ligands: int = 40
    metal_valency: int = 6
    ligand_valency: int = 2
    ph_range: tuple = (4.0, 10.0)
    ionic_strength_range: tuple = (0.01, 1.0)

    # Trajectory parameters
    total_time: float = 100.0
    snapshot_interval: float = 1.0
    num_timesteps: int = 32

    # Dataset sizes
    num_train: int = 1000
    num_val: int = 200
    num_test: int = 200

    # Processing
    embed_dim: int = 32
    compute_topology: bool = True


@dataclass
class ModelConfig(Config):
    """Configuration for model architecture."""

    # Model type
    model_type: str = "temporal_gnn"

    # GNN parameters
    node_feature_dim: int = 42
    edge_feature_dim: int = 12
    gnn_hidden_dim: int = 128
    gnn_num_layers: int = 3
    gnn_type: str = "gat"
    gnn_heads: int = 4
    gnn_dropout: float = 0.1

    # Temporal parameters
    temporal_hidden_dim: int = 256
    temporal_num_layers: int = 2
    temporal_type: str = "transformer"
    temporal_heads: int = 8
    temporal_dropout: float = 0.1

    # Topology
    use_topology: bool = True
    topology_feature_dim: int = 7

    # Output
    output_dim: int = 256
    num_classes: int = 3


@dataclass
class TrainConfig(Config):
    """Configuration for training."""

    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    num_epochs: int = 100
    optimizer: str = "adamw"

    # Learning rate schedule
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    min_lr: float = 1e-6

    # Loss
    classification_weight: float = 1.0
    regression_weight: float = 0.5
    label_smoothing: float = 0.1

    # Regularization
    gradient_clip: float = 1.0

    # Early stopping
    early_stopping: bool = True
    patience: int = 15
    min_delta: float = 1e-4

    # Device
    device: str = "cuda"


@dataclass
class ExperimentConfig(Config):
    """Full experiment configuration."""

    # Metadata
    name: str = "experiment"
    description: str = ""
    seed: int = 42

    # Sub-configs
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # Output
    output_dir: str = "experiments/results"
    save_model: bool = True
    save_predictions: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to nested dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "seed": self.seed,
            "data": self.data.to_dict(),
            "model": self.model.to_dict(),
            "train": self.train.to_dict(),
            "output_dir": self.output_dir,
            "save_model": self.save_model,
            "save_predictions": self.save_predictions,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Create from nested dictionary."""
        return cls(
            name=data.get("name", "experiment"),
            description=data.get("description", ""),
            seed=data.get("seed", 42),
            data=DataConfig.from_dict(data.get("data", {})),
            model=ModelConfig.from_dict(data.get("model", {})),
            train=TrainConfig.from_dict(data.get("train", {})),
            output_dir=data.get("output_dir", "experiments/results"),
            save_model=data.get("save_model", True),
            save_predictions=data.get("save_predictions", True),
        )


def load_config(path: Union[str, Path]) -> Config:
    """Load configuration from file (auto-detect type)."""
    path = Path(path)

    if path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
    elif path.suffix in [".yaml", ".yml"]:
        with open(path) as f:
            data = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")

    # Detect config type
    if "data" in data and "model" in data:
        return ExperimentConfig.from_dict(data)
    elif "gnn_hidden_dim" in data:
        return ModelConfig.from_dict(data)
    elif "learning_rate" in data:
        return TrainConfig.from_dict(data)
    elif "num_metal_ions" in data:
        return DataConfig.from_dict(data)
    else:
        return Config.from_dict(data)


def save_config(config: Config, path: Union[str, Path]) -> None:
    """Save configuration to file."""
    config.save(path)


def merge_configs(*configs: Config) -> Dict[str, Any]:
    """Merge multiple configurations into one dictionary."""
    merged = {}
    for config in configs:
        merged.update(config.to_dict())
    return merged
