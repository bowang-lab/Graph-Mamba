import torch
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import (new_layer_config,
                                                   BatchNorm1dNode)
from torch_geometric.graphgym.register import register_network

from graphgps.layer.multi_model_layer import MultiLayer, SingleLayer
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
            if not hasattr(cfg.gt, 'dim_edge') or cfg.gt.dim_edge is None:
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

        if 'Exphormer' in cfg.gt.layer_type:
            self.exp_edge_fixer = ExpanderEdgeFixer(add_edge_index=cfg.prep.add_edge_index, 
                                                    num_virt_node=cfg.prep.num_virt_node)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


class MultiModel(torch.nn.Module):
    """Multiple layer types can be combined here.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in, \
            "The inner and hidden dims must match."

        try:
            model_types = cfg.gt.layer_type.split('+')
        except:
            raise ValueError(f"Unexpected layer type: {cfg.gt.layer_type}")
        layers = []
        for _ in range(cfg.gt.layers):
            layers.append(MultiLayer(
                dim_h=cfg.gt.dim_hidden,
                model_types=model_types,
                num_heads=cfg.gt.n_heads,
                pna_degrees=cfg.gt.pna_degrees,
                equivstable_pe=cfg.posenc_EquivStableLapPE.enable,
                dropout=cfg.gt.dropout,
                attn_dropout=cfg.gt.attn_dropout,
                layer_norm=cfg.gt.layer_norm,
                batch_norm=cfg.gt.batch_norm,
                bigbird_cfg=cfg.gt.bigbird,
                exp_edges_cfg=cfg.prep
            ))
        self.layers = torch.nn.Sequential(*layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


class SingleModel(torch.nn.Module):
    """A single layer type can be used without FFN between the layers.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in, \
            "The inner and hidden dims must match."

        layers = []
        for _ in range(cfg.gt.layers):
            layers.append(SingleLayer(
                dim_h=cfg.gt.dim_hidden,
                model_type=cfg.gt.layer_type,
                num_heads=cfg.gt.n_heads,
                pna_degrees=cfg.gt.pna_degrees,
                equivstable_pe=cfg.posenc_EquivStableLapPE.enable,
                dropout=cfg.gt.dropout,
                attn_dropout=cfg.gt.attn_dropout,
                layer_norm=cfg.gt.layer_norm,
                batch_norm=cfg.gt.batch_norm,
                bigbird_cfg=cfg.gt.bigbird,
                exp_edges_cfg=cfg.prep
            ))
        self.layers = torch.nn.Sequential(*layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


register_network('MultiModel', MultiModel)
register_network('SingleModel', SingleModel)
