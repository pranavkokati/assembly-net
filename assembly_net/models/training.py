"""
Training utilities for Assembly-Net models.

Provides training loops, loss functions, and optimization utilities.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    num_epochs: int = 100
    optimizer: str = "adamw"  # 'adam', 'adamw'

    # Learning rate schedule
    scheduler: str = "cosine"  # 'cosine', 'plateau', 'none'
    warmup_epochs: int = 5
    min_lr: float = 1e-6

    # Loss
    classification_weight: float = 1.0
    regression_weight: float = 0.5
    use_class_weights: bool = True

    # Early stopping
    early_stopping: bool = True
    patience: int = 15
    min_delta: float = 1e-4

    # Regularization
    label_smoothing: float = 0.1
    gradient_clip: float = 1.0

    # Logging
    log_interval: int = 10
    eval_interval: int = 1
    save_best: bool = True

    # Device
    device: str = "cuda"


class EarlyStopping:
    """Early stopping handler."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current validation score.

        Returns:
            True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True

        return False

    def reset(self):
        """Reset the early stopping counter."""
        self.counter = 0
        self.best_score = None
        self.should_stop = False


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining classification and regression.

    Handles class imbalance and uncertainty weighting.
    """

    def __init__(
        self,
        classification_weight: float = 1.0,
        regression_weight: float = 0.5,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        use_uncertainty_weighting: bool = False,
    ):
        super().__init__()
        self.classification_weight = classification_weight
        self.regression_weight = regression_weight
        self.label_smoothing = label_smoothing
        self.use_uncertainty_weighting = use_uncertainty_weighting

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

        # Learnable uncertainty weights (log variance)
        if use_uncertainty_weighting:
            self.log_var_cls = nn.Parameter(torch.zeros(1))
            self.log_var_reg = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-task loss.

        Args:
            predictions: Dictionary with 'logits' and optionally 'score'
            targets: Dictionary with 'class' and optionally 'value'

        Returns:
            Tuple of (total_loss, loss_components)
        """
        losses = {}

        # Classification loss
        if "logits" in predictions and "class" in targets:
            logits = predictions["logits"]
            target_class = targets["class"]

            if self.label_smoothing > 0:
                loss_cls = F.cross_entropy(
                    logits,
                    target_class,
                    weight=self.class_weights,
                    label_smoothing=self.label_smoothing,
                )
            else:
                loss_cls = F.cross_entropy(
                    logits,
                    target_class,
                    weight=self.class_weights,
                )
            losses["classification"] = loss_cls

        # Regression loss
        if "score" in predictions and "value" in targets:
            pred_value = predictions["score"]
            target_value = targets["value"]
            loss_reg = F.mse_loss(pred_value, target_value)
            losses["regression"] = loss_reg

        # Combine losses
        if self.use_uncertainty_weighting:
            total_loss = 0.0
            if "classification" in losses:
                precision_cls = torch.exp(-self.log_var_cls)
                total_loss += precision_cls * losses["classification"] + self.log_var_cls
            if "regression" in losses:
                precision_reg = torch.exp(-self.log_var_reg)
                total_loss += precision_reg * losses["regression"] + self.log_var_reg
        else:
            total_loss = 0.0
            if "classification" in losses:
                total_loss += self.classification_weight * losses["classification"]
            if "regression" in losses:
                total_loss += self.regression_weight * losses["regression"]

        # Convert to scalars for logging
        loss_dict = {k: v.item() for k, v in losses.items()}
        loss_dict["total"] = total_loss.item()

        return total_loss, loss_dict


class Trainer:
    """
    Trainer for Assembly-Net models.

    Handles the complete training loop with validation, early stopping,
    and model checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        loss_fn: Optional[nn.Module] = None,
        property_name: str = "mechanical",
    ):
        """
        Initialize the trainer.

        Args:
            model: Model to train.
            config: Training configuration.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            loss_fn: Loss function (uses MultiTaskLoss if None).
            property_name: Name of property being predicted.
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.property_name = property_name

        # Device
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        # Loss function
        self.loss_fn = loss_fn or MultiTaskLoss(
            classification_weight=config.classification_weight,
            regression_weight=config.regression_weight,
            label_smoothing=config.label_smoothing,
        )

        # Optimizer
        if config.optimizer == "adamw":
            self.optimizer = AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        else:
            self.optimizer = Adam(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )

        # Scheduler
        if config.scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config.num_epochs - config.warmup_epochs,
                eta_min=config.min_lr,
            )
        elif config.scheduler == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=5,
                min_lr=config.min_lr,
            )
        else:
            self.scheduler = None

        # Early stopping
        if config.early_stopping:
            self.early_stopping = EarlyStopping(
                patience=config.patience,
                min_delta=config.min_delta,
            )
        else:
            self.early_stopping = None

        # Tracking
        self.best_model = None
        self.best_val_loss = float("inf")
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "lr": [],
        }

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        loss_components = {}

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            graphs = [g.to(self.device) for g in batch["graphs"]]
            targets = batch["targets"].to(self.device)

            topology = None
            if batch["topology"] is not None:
                topology = batch["topology"].to(self.device)

            mask = batch["mask"].to(self.device) if "mask" in batch else None

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(graphs, topology=topology, mask=mask)

            # Prepare targets
            target_dict = {"class": targets}
            if isinstance(predictions, dict) and "score" in predictions:
                # Get regression target if available
                target_dict["value"] = targets.float() / 2.0  # Normalize

            # Compute loss
            loss, batch_losses = self.loss_fn(predictions, target_dict)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip
                )

            self.optimizer.step()

            # Tracking
            total_loss += loss.item()

            if isinstance(predictions, dict):
                logits = predictions["logits"]
            else:
                logits = predictions
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

            # Accumulate loss components
            for k, v in batch_losses.items():
                loss_components[k] = loss_components.get(k, 0) + v

            # Update progress bar
            pbar.set_postfix(
                loss=total_loss / (batch_idx + 1),
                acc=100.0 * correct / total,
            )

        # Average losses
        n_batches = len(self.train_loader)
        metrics = {
            "loss": total_loss / n_batches,
            "accuracy": 100.0 * correct / total,
        }
        for k, v in loss_components.items():
            metrics[f"loss_{k}"] = v / n_batches

        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        for batch in self.val_loader:
            graphs = [g.to(self.device) for g in batch["graphs"]]
            targets = batch["targets"].to(self.device)

            topology = None
            if batch["topology"] is not None:
                topology = batch["topology"].to(self.device)

            mask = batch["mask"].to(self.device) if "mask" in batch else None

            predictions = self.model(graphs, topology=topology, mask=mask)

            target_dict = {"class": targets}
            if isinstance(predictions, dict) and "score" in predictions:
                target_dict["value"] = targets.float() / 2.0

            loss, _ = self.loss_fn(predictions, target_dict)
            total_loss += loss.item()

            if isinstance(predictions, dict):
                logits = predictions["logits"]
            else:
                logits = predictions
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

        metrics = {
            "loss": total_loss / len(self.val_loader),
            "accuracy": 100.0 * correct / total,
        }

        # Per-class accuracy
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        for c in np.unique(all_targets):
            mask = all_targets == c
            if mask.sum() > 0:
                metrics[f"accuracy_class_{c}"] = (
                    100.0 * (all_preds[mask] == c).sum() / mask.sum()
                )

        return metrics

    def train(self) -> Dict[str, List[float]]:
        """
        Run the full training loop.

        Returns:
            Training history.
        """
        for epoch in range(1, self.config.num_epochs + 1):
            # Warmup
            if epoch <= self.config.warmup_epochs:
                warmup_factor = epoch / self.config.warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.config.learning_rate * warmup_factor

            # Train
            train_metrics = self.train_epoch(epoch)
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])

            # Validate
            if self.val_loader is not None and epoch % self.config.eval_interval == 0:
                val_metrics = self.validate()
                self.history["val_loss"].append(val_metrics["loss"])
                self.history["val_acc"].append(val_metrics["accuracy"])

                print(
                    f"Epoch {epoch}: "
                    f"Train Loss={train_metrics['loss']:.4f}, "
                    f"Train Acc={train_metrics['accuracy']:.2f}%, "
                    f"Val Loss={val_metrics['loss']:.4f}, "
                    f"Val Acc={val_metrics['accuracy']:.2f}%"
                )

                # Save best model
                if val_metrics["loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    self.best_model = copy.deepcopy(self.model.state_dict())

                # Early stopping
                if self.early_stopping is not None:
                    if self.early_stopping(val_metrics["loss"]):
                        print(f"Early stopping at epoch {epoch}")
                        break

                # Scheduler step
                if self.scheduler is not None:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_metrics["loss"])
                    elif epoch > self.config.warmup_epochs:
                        self.scheduler.step()
            else:
                print(
                    f"Epoch {epoch}: "
                    f"Train Loss={train_metrics['loss']:.4f}, "
                    f"Train Acc={train_metrics['accuracy']:.2f}%"
                )

            self.history["lr"].append(
                self.optimizer.param_groups[0]["lr"]
            )

        # Restore best model
        if self.best_model is not None:
            self.model.load_state_dict(self.best_model)

        return self.history

    def save(self, path: Union[str, Path]) -> None:
        """Save model and training state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
                "history": self.history,
                "best_val_loss": self.best_val_loss,
            },
            path,
        )

    def load(self, path: Union[str, Path]) -> None:
        """Load model and training state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint["history"]
        self.best_val_loss = checkpoint["best_val_loss"]


def log_metrics(
    metrics: Dict[str, float],
    step: int,
    prefix: str = "",
    writer=None,
) -> None:
    """
    Log metrics to tensorboard and console.

    Args:
        metrics: Dictionary of metric names to values.
        step: Current step/epoch.
        prefix: Prefix for metric names.
        writer: Optional TensorBoard SummaryWriter.
    """
    for name, value in metrics.items():
        full_name = f"{prefix}/{name}" if prefix else name
        if writer is not None:
            writer.add_scalar(full_name, value, step)


def create_trainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    learning_rate: float = 1e-4,
    num_epochs: int = 100,
    device: str = "cuda",
    **kwargs,
) -> Trainer:
    """
    Convenience function to create a trainer.

    Args:
        model: Model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        learning_rate: Learning rate.
        num_epochs: Number of epochs.
        device: Device to use.
        **kwargs: Additional config options.

    Returns:
        Configured Trainer instance.
    """
    config = TrainingConfig(
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        device=device,
        **kwargs,
    )
    return Trainer(model, config, train_loader, val_loader)
