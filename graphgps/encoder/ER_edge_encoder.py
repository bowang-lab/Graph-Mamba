import torch
import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_edge_encoder


@register_edge_encoder('ERE')
class EREdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim, use_edge_attr=False, expand_edge_attr=False):
        super().__init__()

        dim_in = cfg.gt.dim_edge  # Expected final edge_dim
        
        pecfg = cfg.posenc_ERE
        n_layers = pecfg.layers  # Num. layers in PE encoder model
        self.pass_as_var = pecfg.pass_as_var  # Pass PE also as a separate variable

        self.use_edge_attr = use_edge_attr
        self.expand_edge_attr = expand_edge_attr
        if expand_edge_attr:
            self.linear_x = nn.Linear(dim_in, dim_in - emb_dim)

        if not self.use_edge_attr:
            assert emb_dim == dim_in
        
        layers = []
        layers.append(nn.Linear(1, emb_dim))
        layers.append(nn.ReLU())
        if n_layers > 1:
            for _ in range(n_layers - 1):
                layers.append(nn.Linear(emb_dim, emb_dim))
                layers.append(nn.ReLU())
        self.er_encoder = nn.Sequential(*layers)

    def forward(self, batch):
        ere = self.er_encoder(batch.er_edge)
        if self.expand_edge_attr:
            batch.edge_attr = self.linear_x(batch.edge_attr)
        
        if self.use_edge_attr:
            batch.edge_attr = torch.cat([batch.edge_attr, ere], dim=1)
        else:
            batch.edge_attr = ere
        
        if self.pass_as_var:
            batch.er_edge = ere
        
        return batch