"""
Models module for Assembly-Net.

Contains neural network architectures for learning from assembly trajectories,
including temporal graph neural networks and baseline static models.
"""

from assembly_net.models.temporal_gnn import (
    TemporalAssemblyGNN,
    TemporalMessagePassing,
    TopologyInjectionLayer,
)
from assembly_net.models.baseline_static_gnn import (
    StaticGNN,
    StaticGNNConfig,
)
from assembly_net.models.property_heads import (
    PropertyPredictionHead,
    MechanicalPropertyHead,
    DiffusionPropertyHead,
    OpticalPropertyHead,
    ResponsivenessHead,
)
from assembly_net.models.training import (
    Trainer,
    TrainingConfig,
    EarlyStopping,
)

__all__ = [
    # Temporal GNN
    "TemporalAssemblyGNN",
    "TemporalMessagePassing",
    "TopologyInjectionLayer",
    # Static GNN baseline
    "StaticGNN",
    "StaticGNNConfig",
    # Property heads
    "PropertyPredictionHead",
    "MechanicalPropertyHead",
    "DiffusionPropertyHead",
    "OpticalPropertyHead",
    "ResponsivenessHead",
    # Training
    "Trainer",
    "TrainingConfig",
    "EarlyStopping",
]
