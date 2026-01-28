"""
Visualization utilities for Assembly-Net.

Provides plotting functions for:
- Network growth visualization
- Persistence diagrams
- Betti number evolution
- Training curves
- Property distributions
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx

from assembly_net.data.core import AssemblyGraph, AssemblyTrajectory, NetworkState


def plot_assembly_trajectory(
    trajectory: AssemblyTrajectory,
    num_snapshots: int = 6,
    figsize: Tuple[int, int] = (15, 10),
    node_size: int = 100,
    with_labels: bool = False,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot snapshots of network assembly over time.

    Args:
        trajectory: AssemblyTrajectory to visualize.
        num_snapshots: Number of snapshots to show.
        figsize: Figure size.
        node_size: Size of nodes.
        with_labels: Whether to show node labels.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib Figure object.
    """
    # Sample snapshots
    indices = np.linspace(0, len(trajectory.states) - 1, num_snapshots, dtype=int)
    states = [trajectory.states[i] for i in indices]

    # Create figure
    n_cols = min(3, num_snapshots)
    n_rows = (num_snapshots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).flatten()

    # Use consistent layout
    G_final = states[-1].graph.to_networkx()
    pos = nx.spring_layout(G_final, seed=42)

    for ax, state in zip(axes, states):
        G = state.graph.to_networkx()

        # Color nodes by type
        node_colors = []
        for i, nf in enumerate(state.graph.node_features):
            if nf.node_type.name == "METAL_ION":
                node_colors.append("#E74C3C")  # Red
            elif nf.node_type.name == "LIGAND":
                node_colors.append("#3498DB")  # Blue
            else:
                node_colors.append("#95A5A6")  # Gray

        # Edge weights by bond strength
        edge_widths = []
        for ef in state.graph.edge_features:
            edge_widths.append(ef.bond_strength)

        if not edge_widths:
            edge_widths = [1.0]

        # Draw
        nx.draw_networkx_nodes(
            G, pos, ax=ax, node_color=node_colors, node_size=node_size
        )
        nx.draw_networkx_edges(
            G, pos, ax=ax, width=edge_widths, alpha=0.6
        )

        if with_labels:
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)

        ax.set_title(f"t = {state.time:.1f}")
        ax.axis("off")

    # Hide unused axes
    for ax in axes[len(states) :]:
        ax.axis("off")

    fig.suptitle("Network Assembly Trajectory", fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_persistence_diagram(
    diagram: Dict[int, np.ndarray],
    max_dim: int = 1,
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot persistence diagram.

    Args:
        diagram: Dictionary mapping dimension to persistence pairs.
        max_dim: Maximum dimension to plot.
        figsize: Figure size.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib Figure object.
    """
    colors = ["#E74C3C", "#3498DB", "#2ECC71"]

    fig, axes = plt.subplots(1, max_dim + 1, figsize=figsize)
    if max_dim == 0:
        axes = [axes]

    for dim in range(max_dim + 1):
        ax = axes[dim]

        if dim in diagram and len(diagram[dim]) > 0:
            births = diagram[dim][:, 0]
            deaths = diagram[dim][:, 1]

            # Filter infinite values for plotting
            finite_mask = np.isfinite(deaths)

            # Plot points
            ax.scatter(
                births[finite_mask],
                deaths[finite_mask],
                c=colors[dim % len(colors)],
                alpha=0.7,
                label=f"H{dim}",
            )

            # Plot infinite points at top
            inf_mask = ~finite_mask
            if inf_mask.any():
                max_val = np.max(deaths[finite_mask]) if finite_mask.any() else 1.0
                ax.scatter(
                    births[inf_mask],
                    [max_val * 1.1] * inf_mask.sum(),
                    c=colors[dim % len(colors)],
                    marker="^",
                    alpha=0.7,
                )

        # Diagonal line
        lim = ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 1
        ax.plot([0, lim], [0, lim], "k--", alpha=0.3)

        ax.set_xlabel("Birth")
        ax.set_ylabel("Death")
        ax.set_title(f"H{dim} Persistence")
        ax.set_aspect("equal")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_betti_evolution(
    trajectory: AssemblyTrajectory,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot evolution of Betti numbers over assembly time.

    Args:
        trajectory: AssemblyTrajectory to analyze.
        figsize: Figure size.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib Figure object.
    """
    from assembly_net.topology.persistent_homology import compute_betti_numbers

    times = []
    betti_0 = []
    betti_1 = []

    for state in trajectory.states:
        times.append(state.time)
        betti = compute_betti_numbers(state.graph)
        betti_0.append(betti[0])
        betti_1.append(betti[1])

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(times, betti_0, "o-", label="β₀ (Components)", color="#E74C3C")
    ax.plot(times, betti_1, "s-", label="β₁ (Cycles)", color="#3498DB")

    ax.set_xlabel("Time")
    ax.set_ylabel("Betti Number")
    ax.set_title("Topological Evolution During Assembly")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_network_growth(
    trajectory: AssemblyTrajectory,
    properties: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot network properties over time.

    Args:
        trajectory: AssemblyTrajectory to analyze.
        properties: Properties to plot. Defaults to common properties.
        figsize: Figure size.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib Figure object.
    """
    if properties is None:
        properties = [
            "num_edges",
            "largest_component_fraction",
            "mean_degree",
            "clustering_coefficient",
        ]

    # Compute properties
    trajectory.compute_all_properties()

    times = [s.time for s in trajectory.states]
    n_props = len(properties)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    prop_data = {
        "num_edges": [len(s.graph.edges) for s in trajectory.states],
        "num_nodes": [s.graph.num_nodes for s in trajectory.states],
        "num_components": [s.num_components or 0 for s in trajectory.states],
        "largest_component_fraction": [
            (s.largest_component_size or 0) / max(1, s.graph.num_nodes)
            for s in trajectory.states
        ],
        "mean_degree": [s.mean_degree or 0 for s in trajectory.states],
        "clustering_coefficient": [
            s.clustering_coefficient or 0 for s in trajectory.states
        ],
        "density": [s.density or 0 for s in trajectory.states],
    }

    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#9B59B6"]

    for i, prop in enumerate(properties[:4]):
        ax = axes[i]
        if prop in prop_data:
            ax.plot(times, prop_data[prop], "-", color=colors[i], linewidth=2)
            ax.fill_between(times, 0, prop_data[prop], alpha=0.2, color=colors[i])

        ax.set_xlabel("Time")
        ax.set_ylabel(prop.replace("_", " ").title())
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def animate_assembly(
    trajectory: AssemblyTrajectory,
    interval: int = 200,
    figsize: Tuple[int, int] = (8, 8),
    save_path: Optional[str] = None,
) -> animation.FuncAnimation:
    """
    Create an animation of network assembly.

    Args:
        trajectory: AssemblyTrajectory to animate.
        interval: Milliseconds between frames.
        figsize: Figure size.
        save_path: Optional path to save animation (requires ffmpeg).

    Returns:
        Matplotlib FuncAnimation object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Compute layout from final state
    G_final = trajectory.final_state.graph.to_networkx()
    pos = nx.spring_layout(G_final, seed=42)

    def update(frame):
        ax.clear()
        state = trajectory.states[frame]
        G = state.graph.to_networkx()

        # Node colors
        node_colors = []
        for nf in state.graph.node_features:
            if nf.node_type.name == "METAL_ION":
                node_colors.append("#E74C3C")
            elif nf.node_type.name == "LIGAND":
                node_colors.append("#3498DB")
            else:
                node_colors.append("#95A5A6")

        nx.draw_networkx_nodes(
            G, pos, ax=ax, node_color=node_colors, node_size=100
        )
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.6)

        ax.set_title(f"t = {state.time:.1f}")
        ax.axis("off")

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(trajectory.states),
        interval=interval,
        blit=False,
    )

    if save_path:
        anim.save(save_path, writer="ffmpeg", fps=1000 // interval)

    return anim


def plot_training_curves(
    history: Dict[str, List[float]],
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot training curves from training history.

    Args:
        history: Dictionary with 'train_loss', 'val_loss', etc.
        figsize: Figure size.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib Figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Loss
    ax = axes[0]
    if "train_loss" in history:
        ax.plot(history["train_loss"], label="Train", color="#E74C3C")
    if "val_loss" in history:
        ax.plot(history["val_loss"], label="Val", color="#3498DB")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[1]
    if "train_acc" in history:
        ax.plot(history["train_acc"], label="Train", color="#E74C3C")
    if "val_acc" in history:
        ax.plot(history["val_acc"], label="Val", color="#3498DB")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Training and Validation Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_property_distribution(
    trajectories: List[AssemblyTrajectory],
    property_name: str = "mechanical_class",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot distribution of property labels.

    Args:
        trajectories: List of trajectories.
        property_name: Name of property to plot.
        figsize: Figure size.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib Figure object.
    """
    from collections import Counter

    values = []
    for traj in trajectories:
        if property_name in traj.labels:
            val = traj.labels[property_name]
            if hasattr(val, "name"):
                values.append(val.name)
            else:
                values.append(str(val))

    counts = Counter(values)

    fig, ax = plt.subplots(figsize=figsize)

    labels = list(counts.keys())
    sizes = list(counts.values())

    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#9B59B6", "#F1C40F"]
    ax.bar(labels, sizes, color=colors[: len(labels)])

    ax.set_xlabel(property_name.replace("_", " ").title())
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of {property_name.replace('_', ' ').title()}")

    # Add counts on bars
    for i, (label, count) in enumerate(zip(labels, sizes)):
        ax.text(i, count + 0.5, str(count), ha="center")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def confusion_matrix_plot(
    confusion_matrix: np.ndarray,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (8, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot confusion matrix.

    Args:
        confusion_matrix: Confusion matrix array.
        class_names: Names for classes.
        figsize: Figure size.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(confusion_matrix, cmap="Blues")

    n_classes = confusion_matrix.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    # Add text annotations
    thresh = confusion_matrix.max() / 2
    for i in range(n_classes):
        for j in range(n_classes):
            color = "white" if confusion_matrix[i, j] > thresh else "black"
            ax.text(
                j, i, str(confusion_matrix[i, j]),
                ha="center", va="center", color=color
            )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
