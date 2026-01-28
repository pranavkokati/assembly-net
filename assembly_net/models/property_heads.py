"""
Property Prediction Heads for Assembly-Net.

These modules convert encoder outputs to specific material property predictions.
Each head is designed for a specific property type with appropriate outputs.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PropertyPredictionHead(nn.Module):
    """
    Base class for property prediction heads.

    Converts encoder output to property prediction.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        layers = []
        in_dim = input_dim

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.in_dim = in_dim  # For subclasses

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP."""
        return self.mlp(x)


class MechanicalPropertyHead(PropertyPredictionHead):
    """
    Head for predicting mechanical properties.

    Outputs:
    - Classification: Brittle, Ductile, Robust
    - Regression: Mechanical resilience score [0, 1]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 3,
        predict_regression: bool = True,
    ):
        super().__init__(input_dim, hidden_dim)
        self.num_classes = num_classes
        self.predict_regression = predict_regression

        # Classification head
        self.classifier = nn.Linear(self.in_dim, num_classes)

        # Regression head
        if predict_regression:
            self.regressor = nn.Sequential(
                nn.Linear(self.in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),  # Score in [0, 1]
            )

    def forward(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Encoder output [batch_size, input_dim]

        Returns:
            Dictionary with 'logits' and optionally 'score'
        """
        h = super().forward(x)

        output = {"logits": self.classifier(h)}

        if self.predict_regression:
            output["score"] = self.regressor(h).squeeze(-1)

        return output


class DiffusionPropertyHead(PropertyPredictionHead):
    """
    Head for predicting diffusion barrier properties.

    Outputs:
    - Classification: Fast diffusion, Slow diffusion, Barrier
    - Regression: Relative diffusion coefficient
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 3,
    ):
        super().__init__(input_dim, hidden_dim)
        self.num_classes = num_classes

        # Classification head
        self.classifier = nn.Linear(self.in_dim, num_classes)

        # Regression head (log-scale diffusion coefficient)
        self.regressor = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        h = super().forward(x)

        return {
            "logits": self.classifier(h),
            "diffusion_coeff": F.softplus(self.regressor(h)).squeeze(-1),
        }


class OpticalPropertyHead(PropertyPredictionHead):
    """
    Head for predicting optical properties.

    Outputs:
    - Classification: Transparent, Narrow absorption, Broadband absorption
    - Regression: Absorption breadth
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 3,
    ):
        super().__init__(input_dim, hidden_dim)
        self.num_classes = num_classes

        # Classification head
        self.classifier = nn.Linear(self.in_dim, num_classes)

        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        h = super().forward(x)

        return {
            "logits": self.classifier(h),
            "absorption_breadth": self.regressor(h).squeeze(-1),
        }


class ResponsivenessHead(PropertyPredictionHead):
    """
    Head for predicting responsiveness to stimuli.

    Outputs:
    - Classification: Stable, pH responsive, Ion responsive
    - Regression: Response magnitude
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 3,
    ):
        super().__init__(input_dim, hidden_dim)
        self.num_classes = num_classes

        # Classification head
        self.classifier = nn.Linear(self.in_dim, num_classes)

        # Response magnitude regressor
        self.regressor = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        h = super().forward(x)

        return {
            "logits": self.classifier(h),
            "response_magnitude": self.regressor(h).squeeze(-1),
        }


class MultiTaskHead(nn.Module):
    """
    Multi-task head for predicting all properties simultaneously.

    Shares a common trunk and has separate heads for each property.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        trunk_layers: int = 2,
        head_hidden_dim: int = 128,
    ):
        super().__init__()

        # Shared trunk
        trunk = []
        in_dim = input_dim
        for _ in range(trunk_layers):
            trunk.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*trunk)

        # Property-specific heads
        self.mechanical = MechanicalPropertyHead(hidden_dim, head_hidden_dim)
        self.diffusion = DiffusionPropertyHead(hidden_dim, head_hidden_dim)
        self.optical = OpticalPropertyHead(hidden_dim, head_hidden_dim)
        self.responsiveness = ResponsivenessHead(hidden_dim, head_hidden_dim)

    def forward(
        self, x: torch.Tensor
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Forward pass through all heads.

        Args:
            x: Encoder output [batch_size, input_dim]

        Returns:
            Dictionary mapping property name to prediction dict
        """
        h = self.trunk(x)

        return {
            "mechanical": self.mechanical(h),
            "diffusion": self.diffusion(h),
            "optical": self.optical(h),
            "responsiveness": self.responsiveness(h),
        }


class AssemblyPropertyPredictor(nn.Module):
    """
    Complete model combining encoder and prediction heads.

    This is the main interface for property prediction from assembly trajectories.
    """

    def __init__(
        self,
        encoder: nn.Module,
        head: nn.Module,
        property_name: str = "mechanical",
    ):
        """
        Initialize the predictor.

        Args:
            encoder: Encoder module (TemporalAssemblyGNN or baseline)
            head: Prediction head module
            property_name: Name of the property being predicted
        """
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.property_name = property_name

    def forward(
        self,
        graphs: list,
        topology: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass from trajectory to property prediction.

        Args:
            graphs: List of graph batches (one per timestep)
            topology: Topology features [batch_size, seq_len, topology_dim]
            mask: Valid timestep mask

        Returns:
            Prediction dictionary
        """
        # Encode trajectory
        embedding = self.encoder(graphs, topology=topology, mask=mask)

        # Predict property
        return self.head(embedding)

    def predict(
        self,
        graphs: list,
        topology: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with class labels and scores.

        Returns:
            Tuple of (predicted_classes, confidence_scores)
        """
        output = self.forward(graphs, topology, mask)

        logits = output["logits"]
        probs = F.softmax(logits, dim=-1)
        classes = torch.argmax(probs, dim=-1)
        confidence = probs.max(dim=-1).values

        return classes, confidence


class MultiPropertyPredictor(nn.Module):
    """
    Multi-property predictor combining encoder with multi-task head.
    """

    def __init__(
        self,
        encoder: nn.Module,
        input_dim: int = 256,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.encoder = encoder
        self.head = MultiTaskHead(input_dim, hidden_dim)

    def forward(
        self,
        graphs: list,
        topology: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Forward pass for all properties."""
        embedding = self.encoder(graphs, topology=topology, mask=mask)
        return self.head(embedding)

    def predict_all(
        self,
        graphs: list,
        topology: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict all properties and return class labels.

        Returns:
            Dictionary mapping property name to predicted class
        """
        output = self.forward(graphs, topology, mask)

        predictions = {}
        for prop_name, prop_output in output.items():
            logits = prop_output["logits"]
            predictions[prop_name] = torch.argmax(logits, dim=-1)

        return predictions
