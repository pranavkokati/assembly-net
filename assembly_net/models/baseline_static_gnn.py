"""
Baseline Static GNN for comparison with Temporal GNN.

This model processes only the final graph structure without any
temporal or assembly history information. It serves as a baseline
to demonstrate the value of assembly-aware learning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    GINConv,
    GlobalAttention,
    Set2Set,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)


@dataclass
class StaticGNNConfig:
    """Configuration for Static GNN baseline."""

    # Input dimensions
    node_feature_dim: int = 42
    edge_feature_dim: int = 12

    # Architecture
    hidden_dim: int = 256
    num_layers: int = 5
    gnn_type: str = "gat"  # 'gcn', 'gat', 'gin'
    heads: int = 4  # For GAT
    dropout: float = 0.1

    # Pooling
    pooling: str = "attention"  # 'mean', 'max', 'add', 'attention', 'set2set'

    # Output
    output_dim: int = 256

    # Readout MLP
    use_mlp_readout: bool = True
    mlp_layers: int = 2


class StaticGNN(nn.Module):
    """
    Static Graph Neural Network baseline.

    Processes only the final graph structure without temporal information.
    Used for comparison to demonstrate the value of assembly-aware learning.
    """

    def __init__(self, config: Optional[StaticGNNConfig] = None):
        """
        Initialize the Static GNN.

        Args:
            config: Model configuration. Uses defaults if None.
        """
        super().__init__()
        self.config = config or StaticGNNConfig()

        # Input projections
        self.node_encoder = nn.Sequential(
            nn.Linear(self.config.node_feature_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
        )

        if self.config.edge_feature_dim > 0:
            self.edge_encoder = nn.Sequential(
                nn.Linear(self.config.edge_feature_dim, self.config.hidden_dim),
                nn.ReLU(),
            )
        else:
            self.edge_encoder = None

        # GNN layers
        self.gnn_layers = nn.ModuleList()
        self.gnn_norms = nn.ModuleList()

        for i in range(self.config.num_layers):
            if self.config.gnn_type == "gcn":
                layer = GCNConv(self.config.hidden_dim, self.config.hidden_dim)
            elif self.config.gnn_type == "gat":
                layer = GATConv(
                    self.config.hidden_dim,
                    self.config.hidden_dim // self.config.heads,
                    heads=self.config.heads,
                    dropout=self.config.dropout,
                    edge_dim=self.config.hidden_dim if self.edge_encoder else None,
                )
            elif self.config.gnn_type == "gin":
                mlp = nn.Sequential(
                    nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
                )
                layer = GINConv(mlp)
            else:
                raise ValueError(f"Unknown GNN type: {self.config.gnn_type}")

            self.gnn_layers.append(layer)
            self.gnn_norms.append(nn.LayerNorm(self.config.hidden_dim))

        self.dropout = nn.Dropout(self.config.dropout)

        # Pooling
        if self.config.pooling == "attention":
            gate_nn = nn.Sequential(
                nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.config.hidden_dim, 1),
            )
            self.pool = GlobalAttention(gate_nn)
        elif self.config.pooling == "set2set":
            self.pool = Set2Set(self.config.hidden_dim, processing_steps=3)
            self.pool_proj = nn.Linear(2 * self.config.hidden_dim, self.config.hidden_dim)
        else:
            self.pool = None

        # Readout MLP
        if self.config.use_mlp_readout:
            layers = []
            in_dim = self.config.hidden_dim
            for i in range(self.config.mlp_layers - 1):
                layers.extend([
                    nn.Linear(in_dim, self.config.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.config.dropout),
                ])
                in_dim = self.config.hidden_dim
            layers.append(nn.Linear(in_dim, self.config.output_dim))
            self.readout = nn.Sequential(*layers)
        else:
            self.readout = nn.Linear(self.config.hidden_dim, self.config.output_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through Static GNN.

        Args:
            x: Node features [num_nodes, node_feature_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_feature_dim]
            batch: Batch assignment [num_nodes]

        Returns:
            Graph embedding [batch_size, output_dim]
        """
        # Encode inputs
        h = self.node_encoder(x)

        if self.edge_encoder is not None and edge_attr is not None:
            e = self.edge_encoder(edge_attr)
        else:
            e = None

        # GNN message passing
        for gnn, norm in zip(self.gnn_layers, self.gnn_norms):
            if self.config.gnn_type == "gat" and e is not None:
                h_new = gnn(h, edge_index, edge_attr=e)
            else:
                h_new = gnn(h, edge_index)

            h_new = norm(h_new)
            h_new = F.relu(h_new)
            h_new = self.dropout(h_new)

            # Residual connection
            h = h + h_new

        # Graph pooling
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)

        if self.config.pooling == "mean":
            graph_embed = global_mean_pool(h, batch)
        elif self.config.pooling == "max":
            graph_embed = global_max_pool(h, batch)
        elif self.config.pooling == "add":
            graph_embed = global_add_pool(h, batch)
        elif self.config.pooling == "attention":
            graph_embed = self.pool(h, batch)
        elif self.config.pooling == "set2set":
            graph_embed = self.pool(h, batch)
            graph_embed = self.pool_proj(graph_embed)
        else:
            graph_embed = global_mean_pool(h, batch)

        # Readout
        return self.readout(graph_embed)

    def forward_batch(self, data: Data) -> torch.Tensor:
        """
        Forward pass with PyG Data object.

        Args:
            data: PyG Data or Batch object.

        Returns:
            Graph embedding [batch_size, output_dim]
        """
        return self.forward(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr if hasattr(data, "edge_attr") else None,
            batch=data.batch if hasattr(data, "batch") else None,
        )


class StaticGNNWithTopology(nn.Module):
    """
    Static GNN enhanced with topological features.

    Still ignores temporal information but uses topology as additional input.
    Used to isolate the contribution of topology vs temporal information.
    """

    def __init__(
        self,
        config: Optional[StaticGNNConfig] = None,
        topology_dim: int = 64,
    ):
        super().__init__()
        self.config = config or StaticGNNConfig()
        self.topology_dim = topology_dim

        # Base GNN
        self.gnn = StaticGNN(config)

        # Topology encoder
        self.topology_encoder = nn.Sequential(
            nn.Linear(topology_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
        )

        # Combiner
        self.combiner = nn.Sequential(
            nn.Linear(self.config.output_dim + self.config.hidden_dim, self.config.output_dim),
            nn.ReLU(),
            nn.Linear(self.config.output_dim, self.config.output_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        topology: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional topology features.

        Args:
            x: Node features
            edge_index: Edge indices
            edge_attr: Edge features
            batch: Batch assignment
            topology: Topology features [batch_size, topology_dim]

        Returns:
            Graph embedding [batch_size, output_dim]
        """
        # GNN encoding
        gnn_embed = self.gnn(x, edge_index, edge_attr, batch)

        if topology is not None:
            # Encode and combine topology
            topo_embed = self.topology_encoder(topology)
            combined = torch.cat([gnn_embed, topo_embed], dim=-1)
            return self.combiner(combined)
        else:
            return gnn_embed


class FinalStructureBaseline(nn.Module):
    """
    Baseline that uses trajectory's final structure only.

    Takes the same input format as TemporalAssemblyGNN but
    only processes the last graph in the sequence.
    """

    def __init__(
        self,
        config: Optional[StaticGNNConfig] = None,
        topology_dim: int = 7,
        use_final_topology: bool = True,
    ):
        super().__init__()
        self.config = config or StaticGNNConfig()
        self.use_final_topology = use_final_topology

        if use_final_topology:
            self.gnn = StaticGNNWithTopology(config, topology_dim=topology_dim)
        else:
            self.gnn = StaticGNN(config)

    def forward(
        self,
        graphs: list,
        topology: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass using only final graph.

        Args:
            graphs: List of PyG Batch objects (one per timestep)
            topology: Topology features [batch_size, seq_len, topology_dim]
            mask: Timestep mask (ignored, we always use last)

        Returns:
            Output embedding [batch_size, output_dim]
        """
        # Use only the final graph
        final_graph = graphs[-1]

        if self.use_final_topology and topology is not None:
            # Use final timestep topology
            final_topology = topology[:, -1, :]
            return self.gnn(
                x=final_graph.x,
                edge_index=final_graph.edge_index,
                edge_attr=final_graph.edge_attr if hasattr(final_graph, "edge_attr") else None,
                batch=final_graph.batch if hasattr(final_graph, "batch") else None,
                topology=final_topology,
            )
        else:
            return self.gnn.forward_batch(final_graph)


class TopologyOnlyBaseline(nn.Module):
    """
    Baseline that uses only topological features, no GNN.

    Processes the topology evolution over time without graph structure.
    Used to test if topology alone is sufficient.
    """

    def __init__(
        self,
        topology_dim: int = 7,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()

        # Temporal model for topology evolution
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.input_proj = nn.Linear(topology_dim, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, hidden_dim) * 0.02)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        graphs: list,  # Ignored
        topology: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass using only topology features.

        Args:
            graphs: Ignored (for API compatibility)
            topology: Topology features [batch_size, seq_len, topology_dim]
            mask: Timestep mask

        Returns:
            Output embedding [batch_size, output_dim]
        """
        batch_size, seq_len, _ = topology.shape

        # Project and add positional encoding
        h = self.input_proj(topology)
        h = h + self.pos_encoding[:, :seq_len, :]

        # Temporal processing
        if mask is not None:
            attn_mask = ~mask
        else:
            attn_mask = None

        h = self.temporal(h, src_key_padding_mask=attn_mask)

        # Pool over time
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            h = (h * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            h = h.mean(dim=1)

        return self.output_proj(h)


def create_static_baseline(
    node_dim: int = 42,
    edge_dim: int = 12,
    hidden_dim: int = 256,
    output_dim: int = 256,
    num_layers: int = 5,
    gnn_type: str = "gat",
) -> StaticGNN:
    """Create a configured Static GNN baseline."""
    config = StaticGNNConfig(
        node_feature_dim=node_dim,
        edge_feature_dim=edge_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        gnn_type=gnn_type,
        output_dim=output_dim,
    )
    return StaticGNN(config)
