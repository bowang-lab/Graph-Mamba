import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_layer


class ETransformer(nn.Module):
    """Mostly Multi-Head Graph Attention Layer.

    Ported to PyG from original repo:
    https://github.com/DevinKreuzer/SAN/blob/main/layers/graph_transformer_layer.py
    """

    def __init__(self, in_dim, out_dim, num_heads, use_bias, edge_index='edge_index', use_edge_attr=False, edge_attr='edge_attr'):
        super().__init__()

        if out_dim % num_heads != 0:
            raise ValueError('hidden dimension is not dividable by the number of heads')
        self.out_dim = out_dim // num_heads
        self.num_heads = num_heads
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.use_edge_attr = use_edge_attr

        self.Q = nn.Linear(in_dim, self.out_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, self.out_dim * num_heads, bias=use_bias)
        if self.use_edge_attr:
            self.E = nn.Linear(in_dim, self.out_dim * num_heads, bias=use_bias)
        self.V = nn.Linear(in_dim, self.out_dim * num_heads, bias=use_bias)

    def propagate_attention(self, batch):
        edge_index = getattr(batch, self.edge_index)

        src = batch.K_h[edge_index[0]]  # (num real edges) x num_heads x out_dim
        dest = batch.Q_h[edge_index[1]]  # (num real edges) x num_heads x out_dim
        score = torch.mul(src, dest)  # element-wise multiplication

        # Scale scores by sqrt(d)
        score = score / np.sqrt(self.out_dim)

        # Use available edge features to modify the scores for edges
        if self.use_edge_attr:
            score = torch.mul(score, batch.E)  # (num real edges) x num_heads x out_dim
        score = torch.exp(score.sum(-1, keepdim=True).clamp(-5, 5))  # (num real edges) x num_heads x 1

        # Apply attention score to each source node to create edge messages
        msg = batch.V_h[edge_index[0]] * score  # (num real edges) x num_heads x out_dim
        # Add-up real msgs in destination nodes as given by batch.edge_index[1]
        batch.wV = torch.zeros_like(batch.V_h)  # (num nodes in batch) x num_heads x out_dim
        scatter(msg, edge_index[1], dim=0, out=batch.wV, reduce='add')

        # Compute attention normalization coefficient
        batch.Z = score.new_zeros(batch.size(0), self.num_heads, 1)  # (num nodes in batch) x num_heads x 1
        scatter(score, edge_index[1], dim=0, out=batch.Z, reduce='add')

    def forward(self, batch):
        edge_index = getattr(batch, self.edge_index)

        if edge_index is None:
            raise ValueError(f'edge index: f{self.edge_index} not found')
        
        if edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
            edge_index = torch.t(edge_index)
            setattr(batch, self.edge_index, edge_index)

        if self.use_edge_attr:
            edge_attr = getattr(batch, self.edge_attr)
            if edge_attr is None or edge_attr.shape[0] != edge_index.shape[1]:
                print('edge_attr shape does not match edge_index shape, ignoring edge_attr')
                self.use_edge_attr = False
        
        Q_h = self.Q(batch.x)
        K_h = self.K(batch.x)
        if self.use_edge_attr:
            E = self.E(edge_attr)
        V_h = self.V(batch.x)

        # Reshaping into [num_nodes, num_heads, feat_dim] to
        # get projections for multi-head attention
        batch.Q_h = Q_h.view(-1, self.num_heads, self.out_dim)
        batch.K_h = K_h.view(-1, self.num_heads, self.out_dim)
        if self.use_edge_attr:
            batch.E = E.view(-1, self.num_heads, self.out_dim)
        batch.V_h = V_h.view(-1, self.num_heads, self.out_dim)

        self.propagate_attention(batch)

        h_out = batch.wV / (batch.Z + 1e-6)

        h_out = h_out.view(-1, self.out_dim * self.num_heads)

        return h_out


register_layer('etransformer', ETransformer)