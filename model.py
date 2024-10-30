import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool

from layers import TopologicalGNNLayer
from etnn import utils, invariants

class TopologicalGNN(nn.Module):
    """
    The Topological Graph Neural Network (TopologicalGNN) model.
    Designed for hierarchical protein representations with separate ranks for
    amino acid nodes, secondary structure, and pocket-level tasks.
    """

    def __init__(
        self,
        num_features_per_rank: dict[int, int],
        num_hidden: int,
        num_out: int,
        num_layers: int,
        adjacencies: list[str],
        initial_features: str,
        normalize_invariants: bool,
        hausdorff_dists: bool = True,
        batch_norm: bool = False,
        dropout: float = 0.0,
        global_pool: bool = True,
    ) -> None:
        super().__init__()

        self.num_ranks = 3  # Hierarchical levels: amino acids, secondary structure, pocket
        self.initial_features = initial_features
        self.normalize_invariants = normalize_invariants
        self.dropout = dropout
        self.global_pool = global_pool

        # Invariant normalization per adjacency if required
        self.inv_normalizer = nn.ModuleDict({
            adj: nn.BatchNorm1d(5 if hausdorff_dists else 3, affine=False)
            for adj in adjacencies
        }) if normalize_invariants else None

        # Feature embedding for each rank
        self.feature_embedding = nn.ModuleDict({
            str(rank): nn.Linear(num_features_per_rank[rank], num_hidden)
            for rank in range(self.num_ranks)
        })

        # TopologicalGNN layers for each rank
        self.layers = nn.ModuleDict({
            str(rank): nn.ModuleList([
                TopologicalGNNLayer(adjacencies, [rank], num_hidden, batch_norm=batch_norm)
                for _ in range(num_layers)
            ]) for rank in range(self.num_ranks)
        })

        # Pooling layers for hierarchical aggregation
        self.pre_pool = nn.ModuleDict({
            str(rank): nn.Linear(num_hidden, num_hidden)
            for rank in range(self.num_ranks)
        })

        # Cross-rank connections for hierarchical message passing
        self.cross_rank_layers = nn.ModuleDict({
            f"{i}_{j}": TopologicalGNNLayer(adjacencies, [i, j], num_hidden, batch_norm=batch_norm)
            for i in range(self.num_ranks) for j in range(i + 1, self.num_ranks)
        })

        # Final global pooling
        self.global_pool_layer = nn.Sequential(
            nn.Linear(self.num_ranks * num_hidden, num_hidden),
            nn.SiLU(),
            nn.Linear(num_hidden, num_out)
        )

    def forward(self, graph: Data) -> Tensor:
        device = graph.pos.device

        # Initial feature embeddings for each rank
        x = {str(rank): self.feature_embedding[str(rank)](graph.x[rank]) for rank in range(self.num_ranks)}
        pos = graph.pos

        # Compute initial invariant features and normalize if necessary
        inv = invariants.compute_invariants(pos)
        if self.normalize_invariants:
            inv = {adj: self.inv_normalizer[adj](feature) for adj, feature in inv.items()}

        # Hierarchical TopologicalGNN layers with cross-rank connections
        for rank in range(self.num_ranks):
            for layer in self.layers[str(rank)]:
                x[str(rank)], pos = layer(x[str(rank)], graph.adjacencies[rank], inv, pos)
                if self.dropout > 0:
                    x[str(rank)] = nn.functional.dropout(x[str(rank)], p=self.dropout, training=self.training)

            # Cross-rank message passing
            for higher_rank in range(rank + 1, self.num_ranks):
                cross_layer = self.cross_rank_layers[f"{rank}_{higher_rank}"]
                x[str(higher_rank)], pos = cross_layer(x[str(rank)], graph.adjacencies[higher_rank], inv, pos)

        # Global pooling and final output
        pooled_x = [global_add_pool(x[str(rank)], graph.batch) for rank in range(self.num_ranks)]
        hierarchical_state = torch.cat(pooled_x, dim=-1)
        out = self.global_pool_layer(hierarchical_state)

        return out

    def __str__(self):
        return "TopologicalGNN for hierarchical protein representations"
