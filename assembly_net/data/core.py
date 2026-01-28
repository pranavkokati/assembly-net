"""
Core data structures for representing assembly graphs and trajectories.

This module defines the fundamental representations for coordination network
assembly processes, including nodes, edges, graphs, and temporal trajectories.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch_geometric.data import Data


class NodeType(Enum):
    """Types of nodes in coordination networks."""

    METAL_ION = auto()  # Metal center (Fe, Cu, Zn, etc.)
    LIGAND = auto()  # Organic ligand
    MONOMER = auto()  # Polymer monomer unit
    FUNCTIONAL_MOTIF = auto()  # Functional group or motif
    CROSSLINKER = auto()  # Crosslinking agent
    SOLVENT = auto()  # Solvent molecule (for explicit solvation)


class EdgeType(Enum):
    """Types of edges (interactions) in coordination networks."""

    COORDINATION = auto()  # Metal-ligand coordination bond
    HYDROGEN_BOND = auto()  # Hydrogen bonding
    PI_STACKING = auto()  # Pi-pi stacking interaction
    COVALENT = auto()  # Covalent bond
    IONIC = auto()  # Ionic interaction
    VAN_DER_WAALS = auto()  # Van der Waals / hydrophobic
    CROSSLINK = auto()  # Crosslinking bond


@dataclass
class NodeFeatures:
    """Features associated with a node in the assembly graph."""

    node_type: NodeType
    valency: int  # Maximum coordination number
    current_coordination: int = 0  # Current number of bonds
    charge: float = 0.0  # Formal charge
    mass: float = 1.0  # Relative mass
    # Chemical identity embedding (learnable or pre-computed)
    chemical_embedding: Optional[np.ndarray] = None
    # Spatial position (if available)
    position: Optional[np.ndarray] = None
    # Additional properties
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_vector(self, embed_dim: int = 32) -> np.ndarray:
        """Convert node features to a fixed-size vector."""
        # One-hot encode node type
        type_vec = np.zeros(len(NodeType))
        type_vec[self.node_type.value - 1] = 1.0

        # Numerical features
        num_features = np.array(
            [
                self.valency / 10.0,  # Normalize valency
                self.current_coordination / 10.0,
                self.charge / 5.0,  # Normalize charge
                np.log1p(self.mass) / 5.0,  # Log-normalize mass
            ]
        )

        # Chemical embedding
        if self.chemical_embedding is not None:
            chem_embed = self.chemical_embedding[:embed_dim]
            if len(chem_embed) < embed_dim:
                chem_embed = np.pad(chem_embed, (0, embed_dim - len(chem_embed)))
        else:
            chem_embed = np.zeros(embed_dim)

        return np.concatenate([type_vec, num_features, chem_embed])


@dataclass
class EdgeFeatures:
    """Features associated with an edge in the assembly graph."""

    edge_type: EdgeType
    bond_strength: float = 1.0  # Relative bond strength
    formation_time: float = 0.0  # When the bond was formed
    # pH dependence (0 = stable across pH, 1 = highly pH sensitive)
    ph_sensitivity: float = 0.0
    # Kinetic properties
    formation_rate: float = 1.0  # Rate constant for formation
    dissociation_rate: float = 0.0  # Rate constant for dissociation
    # Additional properties
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_vector(self) -> np.ndarray:
        """Convert edge features to a fixed-size vector."""
        # One-hot encode edge type
        type_vec = np.zeros(len(EdgeType))
        type_vec[self.edge_type.value - 1] = 1.0

        # Numerical features
        num_features = np.array(
            [
                self.bond_strength,
                self.formation_time / 100.0,  # Normalize time
                self.ph_sensitivity,
                np.log1p(self.formation_rate),
                np.log1p(self.dissociation_rate),
            ]
        )

        return np.concatenate([type_vec, num_features])


@dataclass
class AssemblyEvent:
    """Represents a single assembly event (bond formation or breaking)."""

    time: float
    event_type: str  # "formation" or "dissociation"
    source_node: int
    target_node: int
    edge_features: EdgeFeatures
    # Optional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class AssemblyGraph:
    """
    Represents a single snapshot of the coordination network.

    This is the fundamental graph representation at a specific time point.
    """

    def __init__(
        self,
        num_nodes: int = 0,
        node_features: Optional[List[NodeFeatures]] = None,
        edges: Optional[List[Tuple[int, int]]] = None,
        edge_features: Optional[List[EdgeFeatures]] = None,
        time: float = 0.0,
    ):
        self.num_nodes = num_nodes
        self.node_features = node_features or []
        self.edges = edges or []
        self.edge_features = edge_features or []
        self.time = time

        # Validate consistency
        if len(self.node_features) != self.num_nodes:
            if self.node_features:
                raise ValueError("Node features length must match num_nodes")
        if len(self.edges) != len(self.edge_features):
            if self.edge_features:
                raise ValueError("Edge features length must match edges")

    def add_node(self, features: NodeFeatures) -> int:
        """Add a node to the graph. Returns the node index."""
        node_idx = self.num_nodes
        self.node_features.append(features)
        self.num_nodes += 1
        return node_idx

    def add_edge(
        self, source: int, target: int, features: EdgeFeatures
    ) -> None:
        """Add an edge between two nodes."""
        if source >= self.num_nodes or target >= self.num_nodes:
            raise ValueError("Node indices out of bounds")
        self.edges.append((source, target))
        self.edge_features.append(features)
        # Update coordination numbers
        self.node_features[source].current_coordination += 1
        self.node_features[target].current_coordination += 1

    def remove_edge(self, source: int, target: int) -> bool:
        """Remove an edge between two nodes. Returns True if edge existed."""
        for i, (s, t) in enumerate(self.edges):
            if (s == source and t == target) or (s == target and t == source):
                self.edges.pop(i)
                self.edge_features.pop(i)
                self.node_features[source].current_coordination -= 1
                self.node_features[target].current_coordination -= 1
                return True
        return False

    def get_adjacency_list(self) -> Dict[int, List[int]]:
        """Get adjacency list representation."""
        adj = {i: [] for i in range(self.num_nodes)}
        for s, t in self.edges:
            adj[s].append(t)
            adj[t].append(s)
        return adj

    def get_degree_sequence(self) -> np.ndarray:
        """Get the degree sequence of the graph."""
        degrees = np.zeros(self.num_nodes)
        for s, t in self.edges:
            degrees[s] += 1
            degrees[t] += 1
        return degrees

    def to_networkx(self):
        """Convert to NetworkX graph."""
        import networkx as nx

        G = nx.Graph()
        for i, nf in enumerate(self.node_features):
            G.add_node(i, **nf.__dict__)
        for (s, t), ef in zip(self.edges, self.edge_features):
            G.add_edge(s, t, **ef.__dict__)
        return G

    def to_pyg_data(self, embed_dim: int = 32) -> Data:
        """Convert to PyTorch Geometric Data object."""
        # Node features
        if self.node_features:
            x = torch.tensor(
                np.array([nf.to_vector(embed_dim) for nf in self.node_features]),
                dtype=torch.float,
            )
        else:
            x = torch.zeros((self.num_nodes, embed_dim + len(NodeType) + 4))

        # Edge index (bidirectional)
        if self.edges:
            edge_list = []
            for s, t in self.edges:
                edge_list.append([s, t])
                edge_list.append([t, s])
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

            # Edge features (duplicated for bidirectional)
            edge_attr = torch.tensor(
                np.array(
                    [ef.to_vector() for ef in self.edge_features for _ in range(2)]
                ),
                dtype=torch.float,
            )
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, len(EdgeType) + 5))

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def copy(self) -> AssemblyGraph:
        """Create a deep copy of the graph."""
        return AssemblyGraph(
            num_nodes=self.num_nodes,
            node_features=copy.deepcopy(self.node_features),
            edges=copy.deepcopy(self.edges),
            edge_features=copy.deepcopy(self.edge_features),
            time=self.time,
        )


@dataclass
class NetworkState:
    """
    Represents the state of the network at a specific time, including
    both the graph structure and derived properties.
    """

    graph: AssemblyGraph
    time: float

    # Cached topological properties
    num_components: Optional[int] = None
    largest_component_size: Optional[int] = None
    num_cycles: Optional[int] = None
    is_percolated: Optional[bool] = None

    # Cached structural properties
    mean_degree: Optional[float] = None
    clustering_coefficient: Optional[float] = None
    density: Optional[float] = None

    def compute_properties(self) -> None:
        """Compute and cache network properties."""
        import networkx as nx

        G = self.graph.to_networkx()

        # Component analysis
        components = list(nx.connected_components(G))
        self.num_components = len(components)
        self.largest_component_size = (
            max(len(c) for c in components) if components else 0
        )

        # Cycle analysis
        self.num_cycles = len(nx.cycle_basis(G))

        # Percolation (largest component > 50% of nodes)
        self.is_percolated = self.largest_component_size > 0.5 * self.graph.num_nodes

        # Structural properties
        degrees = self.graph.get_degree_sequence()
        self.mean_degree = np.mean(degrees) if len(degrees) > 0 else 0.0
        self.clustering_coefficient = nx.average_clustering(G) if G.number_of_nodes() > 0 else 0.0
        max_edges = self.graph.num_nodes * (self.graph.num_nodes - 1) / 2
        self.density = len(self.graph.edges) / max_edges if max_edges > 0 else 0.0


class AssemblyTrajectory:
    """
    Represents the complete assembly process as a sequence of graph states.

    This is the primary input to the temporal GNN model.
    """

    def __init__(
        self,
        states: Optional[List[NetworkState]] = None,
        events: Optional[List[AssemblyEvent]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.states = states or []
        self.events = events or []
        self.metadata = metadata or {}

        # Ground truth labels (to be set during data generation)
        self.labels: Dict[str, Any] = {}

    def add_state(self, state: NetworkState) -> None:
        """Add a state to the trajectory."""
        self.states.append(state)

    def add_event(self, event: AssemblyEvent) -> None:
        """Add an assembly event."""
        self.events.append(event)

    @property
    def initial_state(self) -> Optional[NetworkState]:
        """Get the initial network state."""
        return self.states[0] if self.states else None

    @property
    def final_state(self) -> Optional[NetworkState]:
        """Get the final network state."""
        return self.states[-1] if self.states else None

    @property
    def num_timesteps(self) -> int:
        """Number of timesteps in the trajectory."""
        return len(self.states)

    @property
    def duration(self) -> float:
        """Total duration of the assembly process."""
        if not self.states:
            return 0.0
        return self.states[-1].time - self.states[0].time

    def get_state_at_time(self, time: float) -> Optional[NetworkState]:
        """Get the network state at a specific time (or closest earlier state)."""
        if not self.states:
            return None
        for i, state in enumerate(self.states):
            if state.time > time:
                return self.states[max(0, i - 1)]
        return self.states[-1]

    def subsample(self, num_points: int) -> AssemblyTrajectory:
        """Subsample the trajectory to a fixed number of time points."""
        if num_points >= len(self.states):
            return self

        indices = np.linspace(0, len(self.states) - 1, num_points, dtype=int)
        new_states = [self.states[i] for i in indices]

        traj = AssemblyTrajectory(
            states=new_states,
            events=self.events,  # Keep all events
            metadata=self.metadata.copy(),
        )
        traj.labels = self.labels.copy()
        return traj

    def to_pyg_sequence(self, embed_dim: int = 32) -> List[Data]:
        """Convert trajectory to sequence of PyG Data objects."""
        return [state.graph.to_pyg_data(embed_dim) for state in self.states]

    def compute_all_properties(self) -> None:
        """Compute properties for all states in the trajectory."""
        for state in self.states:
            state.compute_properties()

    def get_property_evolution(self, property_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the evolution of a property over time.

        Returns:
            Tuple of (times, values) arrays.
        """
        times = []
        values = []
        for state in self.states:
            if hasattr(state, property_name):
                val = getattr(state, property_name)
                if val is not None:
                    times.append(state.time)
                    values.append(val)
        return np.array(times), np.array(values)


def create_empty_graph(num_nodes: int, node_type: NodeType, valency: int) -> AssemblyGraph:
    """
    Create an empty graph with unconnected nodes.

    Args:
        num_nodes: Number of nodes to create.
        node_type: Type of nodes to create.
        valency: Valency (max coordination) for each node.

    Returns:
        AssemblyGraph with specified nodes and no edges.
    """
    node_features = [
        NodeFeatures(node_type=node_type, valency=valency) for _ in range(num_nodes)
    ]
    return AssemblyGraph(num_nodes=num_nodes, node_features=node_features, time=0.0)


def merge_graphs(graphs: List[AssemblyGraph], time: float = 0.0) -> AssemblyGraph:
    """
    Merge multiple graphs into a single graph.

    Node indices are remapped to be unique across the merged graph.
    """
    merged = AssemblyGraph(time=time)
    offset = 0

    for graph in graphs:
        # Add nodes
        for nf in graph.node_features:
            merged.node_features.append(copy.deepcopy(nf))

        # Add edges with remapped indices
        for (s, t), ef in zip(graph.edges, graph.edge_features):
            merged.edges.append((s + offset, t + offset))
            merged.edge_features.append(copy.deepcopy(ef))

        offset += graph.num_nodes

    merged.num_nodes = offset
    return merged
