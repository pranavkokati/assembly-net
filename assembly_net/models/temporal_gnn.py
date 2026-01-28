"""
Temporal Graph Neural Network for Assembly Trajectories.

This module implements the core model architecture that learns from
assembly processes rather than static structures.

The model combines:
1. GNN for local chemistry at each timestep
2. Temporal attention for assembly dynamics
3. Topology injection for persistent homology features
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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
class TemporalGNNConfig:
    """Configuration for TemporalAssemblyGNN."""

    # Input dimensions
    node_feature_dim: int = 42  # NodeType one-hot + numerical + embedding
    edge_feature_dim: int = 12  # EdgeType one-hot + numerical
    topology_feature_dim: int = 7  # Topological features per timestep

    # GNN architecture
    gnn_hidden_dim: int = 128
    gnn_num_layers: int = 3
    gnn_type: str = "gat"  # 'gcn', 'gat', 'gin'
    gnn_heads: int = 4  # For GAT
    gnn_dropout: float = 0.1

    # Temporal architecture
    temporal_hidden_dim: int = 256
    temporal_num_layers: int = 2
    temporal_type: str = "transformer"  # 'transformer', 'lstm', 'gru'
    temporal_heads: int = 8  # For transformer
    temporal_dropout: float = 0.1

    # Topology injection
    use_topology: bool = True
    topology_injection_layers: List[int] = None  # Which layers to inject

    # Pooling
    graph_pooling: str = "attention"  # 'mean', 'max', 'add', 'attention', 'set2set'

    # Output
    output_dim: int = 256

    def __post_init__(self):
        if self.topology_injection_layers is None:
            self.topology_injection_layers = [0, self.gnn_num_layers - 1]


class GNNEncoder(nn.Module):
    """
    Graph Neural Network encoder for individual graph snapshots.

    Supports multiple GNN architectures (GCN, GAT, GIN).
    """

    def __init__(self, config: TemporalGNNConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.node_encoder = nn.Linear(config.node_feature_dim, config.gnn_hidden_dim)
        self.edge_encoder = nn.Linear(config.edge_feature_dim, config.gnn_hidden_dim)

        # GNN layers
        self.gnn_layers = nn.ModuleList()
        self.gnn_norms = nn.ModuleList()

        for i in range(config.gnn_num_layers):
            in_dim = config.gnn_hidden_dim
            out_dim = config.gnn_hidden_dim

            if config.gnn_type == "gcn":
                self.gnn_layers.append(GCNConv(in_dim, out_dim))
            elif config.gnn_type == "gat":
                # Multi-head attention
                self.gnn_layers.append(
                    GATConv(
                        in_dim,
                        out_dim // config.gnn_heads,
                        heads=config.gnn_heads,
                        dropout=config.gnn_dropout,
                        edge_dim=config.gnn_hidden_dim,
                    )
                )
            elif config.gnn_type == "gin":
                mlp = nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.ReLU(),
                    nn.Linear(out_dim, out_dim),
                )
                self.gnn_layers.append(GINConv(mlp))
            else:
                raise ValueError(f"Unknown GNN type: {config.gnn_type}")

            self.gnn_norms.append(nn.LayerNorm(out_dim))

        self.dropout = nn.Dropout(config.gnn_dropout)

        # Graph pooling
        if config.graph_pooling == "attention":
            gate_nn = nn.Sequential(
                nn.Linear(config.gnn_hidden_dim, config.gnn_hidden_dim),
                nn.ReLU(),
                nn.Linear(config.gnn_hidden_dim, 1),
            )
            self.pool = GlobalAttention(gate_nn)
        elif config.graph_pooling == "set2set":
            self.pool = Set2Set(config.gnn_hidden_dim, processing_steps=3)
            self.pool_transform = nn.Linear(2 * config.gnn_hidden_dim, config.gnn_hidden_dim)
        else:
            self.pool = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through GNN encoder.

        Args:
            x: Node features [num_nodes, node_feature_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_feature_dim]
            batch: Batch assignment [num_nodes]

        Returns:
            Tuple of (node_embeddings, graph_embedding)
        """
        # Encode inputs
        h = self.node_encoder(x)

        if edge_attr is not None and edge_attr.numel() > 0:
            e = self.edge_encoder(edge_attr)
        else:
            e = None

        # GNN layers
        for i, (gnn, norm) in enumerate(zip(self.gnn_layers, self.gnn_norms)):
            if self.config.gnn_type == "gat" and e is not None:
                h_new = gnn(h, edge_index, edge_attr=e)
            else:
                h_new = gnn(h, edge_index)

            h_new = norm(h_new)
            h_new = F.relu(h_new)
            h_new = self.dropout(h_new)

            # Residual connection
            h = h + h_new

        # Graph-level pooling
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)

        if self.config.graph_pooling == "mean":
            graph_embed = global_mean_pool(h, batch)
        elif self.config.graph_pooling == "max":
            graph_embed = global_max_pool(h, batch)
        elif self.config.graph_pooling == "add":
            graph_embed = global_add_pool(h, batch)
        elif self.config.graph_pooling == "attention":
            graph_embed = self.pool(h, batch)
        elif self.config.graph_pooling == "set2set":
            graph_embed = self.pool(h, batch)
            graph_embed = self.pool_transform(graph_embed)
        else:
            graph_embed = global_mean_pool(h, batch)

        return h, graph_embed


class TopologyInjectionLayer(nn.Module):
    """
    Layer for injecting topological features into the representation.

    Topology features (Betti numbers, persistence, etc.) are processed
    and combined with graph embeddings.
    """

    def __init__(
        self,
        graph_dim: int,
        topology_dim: int,
        output_dim: int,
        injection_type: str = "concat",  # 'concat', 'film', 'gating'
    ):
        """
        Initialize topology injection layer.

        Args:
            graph_dim: Dimension of graph embeddings.
            topology_dim: Dimension of topology features.
            output_dim: Output dimension.
            injection_type: How to combine topology with graph features.
        """
        super().__init__()
        self.injection_type = injection_type

        # Topology encoder
        self.topology_encoder = nn.Sequential(
            nn.Linear(topology_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

        if injection_type == "concat":
            self.combiner = nn.Linear(graph_dim + output_dim, output_dim)
        elif injection_type == "film":
            # Feature-wise Linear Modulation
            self.gamma = nn.Linear(output_dim, graph_dim)
            self.beta = nn.Linear(output_dim, graph_dim)
            self.project = nn.Linear(graph_dim, output_dim)
        elif injection_type == "gating":
            self.gate = nn.Sequential(
                nn.Linear(graph_dim + output_dim, output_dim),
                nn.Sigmoid(),
            )
            self.transform = nn.Linear(graph_dim, output_dim)
        else:
            raise ValueError(f"Unknown injection type: {injection_type}")

    def forward(
        self, graph_embed: torch.Tensor, topology_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Inject topology features into graph embedding.

        Args:
            graph_embed: Graph embedding [batch_size, graph_dim]
            topology_features: Topology features [batch_size, topology_dim]

        Returns:
            Combined embedding [batch_size, output_dim]
        """
        topo_encoded = self.topology_encoder(topology_features)

        if self.injection_type == "concat":
            combined = torch.cat([graph_embed, topo_encoded], dim=-1)
            return self.combiner(combined)
        elif self.injection_type == "film":
            gamma = self.gamma(topo_encoded)
            beta = self.beta(topo_encoded)
            modulated = gamma * graph_embed + beta
            return self.project(modulated)
        elif self.injection_type == "gating":
            combined = torch.cat([graph_embed, topo_encoded], dim=-1)
            gate = self.gate(combined)
            transformed = self.transform(graph_embed)
            return gate * transformed + (1 - gate) * topo_encoded


class TemporalMessagePassing(nn.Module):
    """
    Temporal processing for sequences of graph embeddings.

    Supports Transformer and RNN-based temporal modeling.
    """

    def __init__(self, config: TemporalGNNConfig):
        super().__init__()
        self.config = config

        input_dim = config.gnn_hidden_dim
        if config.use_topology:
            input_dim += config.topology_feature_dim

        if config.temporal_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.temporal_hidden_dim,
                nhead=config.temporal_heads,
                dim_feedforward=config.temporal_hidden_dim * 4,
                dropout=config.temporal_dropout,
                activation="gelu",
                batch_first=True,
            )
            self.temporal = nn.TransformerEncoder(
                encoder_layer, num_layers=config.temporal_num_layers
            )
            self.input_proj = nn.Linear(input_dim, config.temporal_hidden_dim)

            # Learnable positional encoding
            self.pos_encoding = nn.Parameter(
                torch.randn(1, 1000, config.temporal_hidden_dim) * 0.02
            )

        elif config.temporal_type == "lstm":
            self.temporal = nn.LSTM(
                input_size=input_dim,
                hidden_size=config.temporal_hidden_dim,
                num_layers=config.temporal_num_layers,
                dropout=config.temporal_dropout if config.temporal_num_layers > 1 else 0,
                batch_first=True,
                bidirectional=True,
            )
            self.output_proj = nn.Linear(
                2 * config.temporal_hidden_dim, config.temporal_hidden_dim
            )

        elif config.temporal_type == "gru":
            self.temporal = nn.GRU(
                input_size=input_dim,
                hidden_size=config.temporal_hidden_dim,
                num_layers=config.temporal_num_layers,
                dropout=config.temporal_dropout if config.temporal_num_layers > 1 else 0,
                batch_first=True,
                bidirectional=True,
            )
            self.output_proj = nn.Linear(
                2 * config.temporal_hidden_dim, config.temporal_hidden_dim
            )

        # Output projection
        self.final_proj = nn.Linear(config.temporal_hidden_dim, config.output_dim)

    def forward(
        self,
        sequence: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process sequence of graph embeddings.

        Args:
            sequence: [batch_size, seq_len, input_dim]
            mask: [batch_size, seq_len] boolean mask for valid positions

        Returns:
            Temporal embedding [batch_size, output_dim]
        """
        batch_size, seq_len, _ = sequence.shape

        if self.config.temporal_type == "transformer":
            # Project to model dimension
            h = self.input_proj(sequence)

            # Add positional encoding
            h = h + self.pos_encoding[:, :seq_len, :]

            # Create attention mask
            if mask is not None:
                # Transformer uses True for positions to mask OUT
                attn_mask = ~mask
            else:
                attn_mask = None

            # Transformer encoding
            h = self.temporal(h, src_key_padding_mask=attn_mask)

            # Pool over sequence (mean of valid positions)
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()
                h = (h * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                h = h.mean(dim=1)

        else:
            # LSTM/GRU
            h, _ = self.temporal(sequence)

            if self.config.temporal_type in ["lstm", "gru"]:
                h = self.output_proj(h)

            # Pool over sequence
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()
                h = (h * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                h = h.mean(dim=1)

        return self.final_proj(h)


class TemporalAssemblyGNN(nn.Module):
    """
    Main model: Temporal Graph Neural Network for Assembly Trajectories.

    This model processes sequences of graph states to predict emergent
    material properties. It combines:
    1. GNN encoder for local chemistry at each timestep
    2. Topology injection for persistent homology features
    3. Temporal model for assembly dynamics
    """

    def __init__(self, config: Optional[TemporalGNNConfig] = None):
        """
        Initialize the Temporal Assembly GNN.

        Args:
            config: Model configuration. Uses defaults if None.
        """
        super().__init__()
        self.config = config or TemporalGNNConfig()

        # GNN encoder for individual graphs
        self.gnn_encoder = GNNEncoder(self.config)

        # Topology injection
        if self.config.use_topology:
            self.topology_injection = TopologyInjectionLayer(
                graph_dim=self.config.gnn_hidden_dim,
                topology_dim=self.config.topology_feature_dim,
                output_dim=self.config.gnn_hidden_dim,
                injection_type="film",
            )

        # Temporal processing
        self.temporal = TemporalMessagePassing(self.config)

    def encode_graph(
        self, data: Data
    ) -> torch.Tensor:
        """
        Encode a single graph (or batch of graphs).

        Args:
            data: PyG Data object.

        Returns:
            Graph embedding [batch_size, gnn_hidden_dim]
        """
        _, graph_embed = self.gnn_encoder(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr if hasattr(data, "edge_attr") else None,
            batch=data.batch if hasattr(data, "batch") else None,
        )
        return graph_embed

    def forward(
        self,
        graphs: List[Batch],
        topology: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the full model.

        Args:
            graphs: List of PyG Batch objects, one per timestep.
                   Each Batch contains graphs from all samples at that time.
            topology: Topology features [batch_size, seq_len, topology_dim]
            mask: Valid timestep mask [batch_size, seq_len]

        Returns:
            Output embedding [batch_size, output_dim]
        """
        batch_size = graphs[0].num_graphs
        seq_len = len(graphs)
        device = graphs[0].x.device

        # Encode each timestep
        graph_embeddings = []
        for t, batch in enumerate(graphs):
            embed = self.encode_graph(batch)  # [batch_size, gnn_hidden_dim]
            graph_embeddings.append(embed)

        # Stack to sequence [batch_size, seq_len, gnn_hidden_dim]
        sequence = torch.stack(graph_embeddings, dim=1)

        # Inject topology features
        if self.config.use_topology and topology is not None:
            # Apply topology injection at each timestep
            injected = []
            for t in range(seq_len):
                inj = self.topology_injection(sequence[:, t, :], topology[:, t, :])
                injected.append(inj)
            sequence = torch.stack(injected, dim=1)

            # Concatenate topology for temporal model
            sequence = torch.cat([sequence, topology], dim=-1)

        # Temporal processing
        output = self.temporal(sequence, mask=mask)

        return output

    def get_temporal_attention(
        self,
        graphs: List[Batch],
        topology: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get temporal attention weights for interpretability.

        Returns attention weights showing which timesteps are most
        important for the prediction.
        """
        if self.config.temporal_type != "transformer":
            return None

        batch_size = graphs[0].num_graphs
        seq_len = len(graphs)

        # Encode each timestep
        graph_embeddings = []
        for batch in graphs:
            embed = self.encode_graph(batch)
            graph_embeddings.append(embed)

        sequence = torch.stack(graph_embeddings, dim=1)

        if self.config.use_topology and topology is not None:
            injected = []
            for t in range(seq_len):
                inj = self.topology_injection(sequence[:, t, :], topology[:, t, :])
                injected.append(inj)
            sequence = torch.stack(injected, dim=1)
            sequence = torch.cat([sequence, topology], dim=-1)

        # Get attention from transformer
        h = self.temporal.input_proj(sequence)
        h = h + self.temporal.pos_encoding[:, :seq_len, :]

        # Extract attention weights from first layer
        # This requires accessing transformer internals
        attention_weights = None

        return attention_weights


def create_temporal_gnn(
    node_dim: int = 42,
    edge_dim: int = 12,
    topology_dim: int = 7,
    hidden_dim: int = 128,
    output_dim: int = 256,
    num_gnn_layers: int = 3,
    num_temporal_layers: int = 2,
    gnn_type: str = "gat",
    temporal_type: str = "transformer",
    use_topology: bool = True,
) -> TemporalAssemblyGNN:
    """
    Convenience function to create a TemporalAssemblyGNN.

    Args:
        node_dim: Node feature dimension.
        edge_dim: Edge feature dimension.
        topology_dim: Topology feature dimension.
        hidden_dim: Hidden dimension.
        output_dim: Output dimension.
        num_gnn_layers: Number of GNN layers.
        num_temporal_layers: Number of temporal layers.
        gnn_type: GNN architecture ('gcn', 'gat', 'gin').
        temporal_type: Temporal architecture ('transformer', 'lstm', 'gru').
        use_topology: Whether to use topology injection.

    Returns:
        Configured TemporalAssemblyGNN model.
    """
    config = TemporalGNNConfig(
        node_feature_dim=node_dim,
        edge_feature_dim=edge_dim,
        topology_feature_dim=topology_dim,
        gnn_hidden_dim=hidden_dim,
        gnn_num_layers=num_gnn_layers,
        gnn_type=gnn_type,
        temporal_hidden_dim=hidden_dim * 2,
        temporal_num_layers=num_temporal_layers,
        temporal_type=temporal_type,
        use_topology=use_topology,
        output_dim=output_dim,
    )

    return TemporalAssemblyGNN(config)
