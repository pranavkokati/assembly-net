"""
Dataset classes for Assembly-Net.

Provides PyTorch Dataset implementations for assembly trajectories.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from assembly_net.data.core import AssemblyTrajectory, NetworkState


class AssemblyDataset(Dataset):
    """
    PyTorch Dataset for assembly trajectories.

    Supports:
    - Loading from files or in-memory trajectories
    - On-the-fly feature computation
    - Trajectory subsampling
    - Multiple prediction targets
    """

    def __init__(
        self,
        trajectories: Optional[List[AssemblyTrajectory]] = None,
        data_dir: Optional[Union[str, Path]] = None,
        transform: Optional[Callable] = None,
        target_property: str = "mechanical_class",
        num_timesteps: int = 32,
        embed_dim: int = 32,
        compute_topology: bool = True,
    ):
        """
        Initialize the dataset.

        Args:
            trajectories: List of AssemblyTrajectory objects.
            data_dir: Directory containing saved trajectories.
            transform: Optional transform to apply to samples.
            target_property: Property to predict (from labels).
            num_timesteps: Number of timesteps to sample from each trajectory.
            embed_dim: Dimension of node embeddings.
            compute_topology: Whether to compute topological features.
        """
        self.transform = transform
        self.target_property = target_property
        self.num_timesteps = num_timesteps
        self.embed_dim = embed_dim
        self.compute_topology = compute_topology

        # Load data
        if trajectories is not None:
            self.trajectories = trajectories
        elif data_dir is not None:
            self.trajectories = self._load_from_dir(Path(data_dir))
        else:
            raise ValueError("Must provide either trajectories or data_dir")

        # Precompute class mappings
        self._compute_class_mappings()

    def _load_from_dir(self, data_dir: Path) -> List[AssemblyTrajectory]:
        """Load trajectories from a directory."""
        trajectories = []
        for file_path in sorted(data_dir.glob("*.pkl")):
            with open(file_path, "rb") as f:
                traj = pickle.load(f)
                trajectories.append(traj)
        return trajectories

    def _compute_class_mappings(self) -> None:
        """Compute mappings for classification targets."""
        from assembly_net.data.synthetic_assembly_simulator.simulator import PropertyClass

        # Map PropertyClass enum to integers
        self.class_to_idx = {pc: i for i, pc in enumerate(PropertyClass)}
        self.idx_to_class = {i: pc for pc, i in self.class_to_idx.items()}
        self.num_classes = len(PropertyClass)

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.

        Returns:
            Dictionary containing:
            - 'graphs': List of PyG Data objects
            - 'times': Tensor of timestamps
            - 'target': Target value/class
            - 'topology': Optional topology features
            - 'metadata': Additional information
        """
        trajectory = self.trajectories[idx]

        # Subsample trajectory
        subsampled = trajectory.subsample(self.num_timesteps)

        # Convert to PyG format
        graphs = subsampled.to_pyg_sequence(self.embed_dim)

        # Get timestamps
        times = torch.tensor(
            [state.time for state in subsampled.states], dtype=torch.float
        )

        # Get target
        target = self._get_target(trajectory)

        # Compute topology features if requested
        topology = None
        if self.compute_topology:
            topology = self._compute_topology_features(subsampled)

        sample = {
            "graphs": graphs,
            "times": times,
            "target": target,
            "topology": topology,
            "num_timesteps": len(graphs),
            "idx": idx,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def _get_target(self, trajectory: AssemblyTrajectory) -> torch.Tensor:
        """Extract target value from trajectory labels."""
        from assembly_net.data.synthetic_assembly_simulator.simulator import PropertyClass

        if self.target_property not in trajectory.labels:
            raise ValueError(f"Target property '{self.target_property}' not in labels")

        value = trajectory.labels[self.target_property]

        # Handle classification vs regression
        if isinstance(value, PropertyClass):
            return torch.tensor(self.class_to_idx[value], dtype=torch.long)
        elif isinstance(value, bool):
            return torch.tensor(int(value), dtype=torch.long)
        else:
            return torch.tensor(value, dtype=torch.float)

    def _compute_topology_features(
        self, trajectory: AssemblyTrajectory
    ) -> torch.Tensor:
        """
        Compute topological features for each timestep.

        Features include:
        - Betti numbers (b0, b1)
        - Normalized cycle count
        - Largest component fraction
        - Clustering coefficient
        """
        features = []

        for state in trajectory.states:
            if state.num_components is None:
                state.compute_properties()

            num_nodes = max(1, state.graph.num_nodes)

            feat = [
                state.num_components / num_nodes,  # Normalized component count
                (state.largest_component_size or 0) / num_nodes,  # LCC fraction
                (state.num_cycles or 0) / num_nodes,  # Normalized cycles
                state.clustering_coefficient or 0.0,
                state.density or 0.0,
                state.mean_degree or 0.0,
                1.0 if state.is_percolated else 0.0,
            ]
            features.append(feat)

        return torch.tensor(features, dtype=torch.float)

    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for imbalanced classification."""
        from assembly_net.data.synthetic_assembly_simulator.simulator import PropertyClass

        if "class" not in self.target_property:
            return None

        counts = torch.zeros(self.num_classes)
        for traj in self.trajectories:
            value = traj.labels[self.target_property]
            if isinstance(value, PropertyClass):
                counts[self.class_to_idx[value]] += 1

        # Inverse frequency weighting
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * self.num_classes

        return weights


def collate_trajectories(
    batch: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Custom collate function for batching trajectories.

    Handles variable-length graph sequences by padding.
    """
    # Find max timesteps in batch
    max_timesteps = max(sample["num_timesteps"] for sample in batch)

    # Collate graphs
    batched_graphs = []
    for t in range(max_timesteps):
        graphs_at_t = []
        for sample in batch:
            if t < len(sample["graphs"]):
                graphs_at_t.append(sample["graphs"][t])
            else:
                # Pad with last graph
                graphs_at_t.append(sample["graphs"][-1])
        batched_graphs.append(Batch.from_data_list(graphs_at_t))

    # Collate times (pad with last time)
    times = []
    for sample in batch:
        t = sample["times"]
        if len(t) < max_timesteps:
            pad = t[-1].expand(max_timesteps - len(t))
            t = torch.cat([t, pad])
        times.append(t)
    times = torch.stack(times)

    # Collate targets
    targets = torch.stack([sample["target"] for sample in batch])

    # Collate topology features if present
    topology = None
    if batch[0]["topology"] is not None:
        topology = []
        for sample in batch:
            topo = sample["topology"]
            if len(topo) < max_timesteps:
                pad = topo[-1:].expand(max_timesteps - len(topo), -1)
                topo = torch.cat([topo, pad])
            topology.append(topo)
        topology = torch.stack(topology)

    # Mask for valid timesteps
    mask = torch.zeros(len(batch), max_timesteps, dtype=torch.bool)
    for i, sample in enumerate(batch):
        mask[i, : sample["num_timesteps"]] = True

    return {
        "graphs": batched_graphs,
        "times": times,
        "targets": targets,
        "topology": topology,
        "mask": mask,
        "batch_size": len(batch),
        "max_timesteps": max_timesteps,
    }


class BalancedSampler(torch.utils.data.Sampler):
    """Sampler that balances classes for classification tasks."""

    def __init__(
        self,
        dataset: AssemblyDataset,
        num_samples: Optional[int] = None,
    ):
        self.dataset = dataset
        self.num_samples = num_samples or len(dataset)

        # Group indices by class
        self.class_indices = {}
        for idx in range(len(dataset)):
            target = dataset[idx]["target"].item()
            if target not in self.class_indices:
                self.class_indices[target] = []
            self.class_indices[target].append(idx)

        self.num_classes = len(self.class_indices)

    def __iter__(self):
        indices = []
        samples_per_class = self.num_samples // self.num_classes

        for class_idx in self.class_indices:
            class_samples = self.class_indices[class_idx]
            # Sample with replacement if needed
            sampled = np.random.choice(
                class_samples,
                size=samples_per_class,
                replace=len(class_samples) < samples_per_class,
            )
            indices.extend(sampled.tolist())

        np.random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples


def save_dataset(
    trajectories: List[AssemblyTrajectory],
    output_dir: Union[str, Path],
    prefix: str = "trajectory",
) -> None:
    """Save trajectories to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, traj in enumerate(trajectories):
        file_path = output_dir / f"{prefix}_{i:05d}.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(traj, f)


def load_dataset(
    data_dir: Union[str, Path],
    target_property: str = "mechanical_class",
    **kwargs,
) -> AssemblyDataset:
    """Convenience function to load a dataset from disk."""
    return AssemblyDataset(
        data_dir=data_dir,
        target_property=target_property,
        **kwargs,
    )
