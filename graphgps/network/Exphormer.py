from re import I
import torch
import torch.nn as nn
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import (new_layer_config,
                                                   BatchNorm1dNode)
from torch_geometric.graphgym.register import register_network

from graphgps.layer.Exphormer import ExphormerFullLayer
from graphgps.encoder.ER_edge_encoder import EREdgeEncoder
from graphgps.encoder.exp_edge_fixer import ExpanderEdgeFixer


class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """
    def __init__(self, dim_in):
        super(FeatureEncoder, self).__init__()
        self.dim_in = dim_in
        if cfg.dataset.node_encoder:
            # Encode integer node features via nn.Embeddings
            NodeEncoder = register.node_encoder_dict[
                cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.gnn.dim_inner)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_inner, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))
            # Update dim_in to reflect the new dimension fo the node features
            self.dim_in = cfg.gnn.dim_inner
        if cfg.dataset.edge_encoder:
            if cfg.gt.dim_edge is None:
                cfg.gt.dim_edge = cfg.gt.dim_hidden

            if cfg.dataset.edge_encoder_name == 'ER':
                self.edge_encoder = EREdgeEncoder(cfg.gt.dim_edge)
            elif cfg.dataset.edge_encoder_name.endswith('+ER'):
                EdgeEncoder = register.edge_encoder_dict[
                    cfg.dataset.edge_encoder_name[:-3]]
                self.edge_encoder = EdgeEncoder(cfg.gt.dim_edge - cfg.posenc_ERE.dim_pe)
                self.edge_encoder_er = EREdgeEncoder(cfg.posenc_ERE.dim_pe, use_edge_attr=True)
            else:
                EdgeEncoder = register.edge_encoder_dict[
                    cfg.dataset.edge_encoder_name]
                self.edge_encoder = EdgeEncoder(cfg.gt.dim_edge)

            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gt.dim_edge, -1, -1, has_act=False,
                                    has_bias=False, cfg=cfg))

        self.exp_edge_fixer = ExpanderEdgeFixer(add_edge_index=cfg.prep.add_edge_index, 
                                                num_virt_node=cfg.prep.num_virt_node)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


@register_network('Exphormer')
class Exphormer(torch.nn.Module):
    """Exphormer
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gt.dim_hidden, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gt.dim_hidden

        assert cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in, \
            "The inner and hidden dims must match."

        dim_h = cfg.gt.dim_hidden

        self.layers = []
        for _ in range(cfg.gt.layers):
            self.layers.append(ExphormerFullLayer(
                dim_h, dim_h, 
                cfg.gt.n_heads, 
                cfg.gt.dropout, 
                dim_edge=cfg.gt.dim_edge,
                layer_norm=cfg.gt.layer_norm, 
                batch_norm=cfg.gt.batch_norm, 
                activation=cfg.gt.activation,
                use_virt_nodes=cfg.prep.num_virt_node > 0
            ))
        self.layers = nn.Sequential(*self.layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=dim_h, dim_out=dim_out)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch
