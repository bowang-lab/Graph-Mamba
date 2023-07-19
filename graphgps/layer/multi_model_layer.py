import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pygnn
from performer_pytorch import SelfAttention
from torch_geometric.data import Batch
from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.utils import to_dense_batch

from graphgps.layer.gatedgcn_layer import GatedGCNLayer
from graphgps.layer.gine_conv_layer import GINEConvESLapPE
from graphgps.layer.bigbird_layer import SingleBigBirdLayer
from graphgps.layer.ETransformer import ETransformer
from graphgps.layer.Exphormer import ExphormerAttention


class LocalModel(nn.Module):
    def __init__(self, dim_h, local_gnn_type, edge_type, edge_attr_type, num_heads,
                pna_degrees=None, equivstable_pe=False, dropout=0.0,
                layer_norm=False, batch_norm=True):
        super().__init__()

        self.dim_h = dim_h
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.equivstable_pe = equivstable_pe
        self.edge_type = edge_type
        self.edge_attr_type = edge_attr_type

        if self.edge_type == 'edge_index' or self.edge_type is None:
            self.edge_type = 'edge_index'
            self.edge_attr_type = 'edge_attr'
        elif self.edge_type == 'exp':
            self.edge_type = 'expander_edge_index'
            self.edge_attr_type = 'expander_edge_attr'

        # Local message-passing model.
        if local_gnn_type == 'GENConv':
            self.local_model = pygnn.GENConv(dim_h, dim_h)
        elif local_gnn_type == 'GCN':
            self.local_model = pygnn.GCNConv(dim_h, dim_h)
        elif local_gnn_type == 'GINE':
            gin_nn = nn.Sequential(Linear_pyg(dim_h, dim_h),
                                   nn.ReLU(),
                                   Linear_pyg(dim_h, dim_h))
            if self.equivstable_pe:  # Use specialised GINE layer for EquivStableLapPE.
                self.local_model = GINEConvESLapPE(gin_nn)
            else:
                self.local_model = pygnn.GINEConv(gin_nn)
        elif local_gnn_type == 'GAT':
            self.local_model = pygnn.GATConv(in_channels=dim_h,
                                             out_channels=dim_h // num_heads,
                                             heads=num_heads,
                                             edge_dim=dim_h)
        elif local_gnn_type == 'PNA':
            # Defaults from the paper.
            # aggregators = ['mean', 'min', 'max', 'std']
            # scalers = ['identity', 'amplification', 'attenuation']
            aggregators = ['mean', 'max', 'sum']
            scalers = ['identity']
            deg = torch.from_numpy(np.array(pna_degrees))
            self.local_model = pygnn.PNAConv(dim_h, dim_h,
                                             aggregators=aggregators,
                                             scalers=scalers,
                                             deg=deg,
                                             edge_dim=16,  # dim_h,
                                             towers=1,
                                             pre_layers=1,
                                             post_layers=1,
                                             divide_input=False)
        elif local_gnn_type == 'CustomGatedGCN':
            self.local_model = GatedGCNLayer(dim_h, dim_h,
                                             dropout=dropout,
                                             residual=True,
                                             equivstable_pe=equivstable_pe)
        else:
            raise ValueError(f"Unsupported local GNN model: {local_gnn_type}")
        self.local_gnn_type = local_gnn_type

        if self.layer_norm and self.batch_norm:
            raise ValueError("Cannot apply two types of normalization together")

        # Normalization for MPNN.
        if self.layer_norm:
            self.norm1_local = pygnn.norm.GraphNorm(dim_h)
        if self.batch_norm:
            self.norm1_local = nn.BatchNorm1d(dim_h)
        self.dropout_local = nn.Dropout(dropout)

    def forward(self, batch):
        h = batch.x

        h_in1 = h  # for first residual connection

        edge_index = getattr(batch, self.edge_type)
        edge_attr = getattr(batch, self.edge_attr_type)
        if edge_index is None:
            raise ValueError(f'edge type {self.edge_type} is not stored in the data!')

        self.local_model: pygnn.conv.MessagePassing  # Typing hint.
        if self.local_gnn_type == 'CustomGatedGCN':
            es_data = None
            if self.equivstable_pe:
                es_data = batch.pe_EquivStableLapPE
            local_out = self.local_model(Batch(batch=batch,
                                               x=h,
                                               edge_index=edge_index,
                                               edge_attr=edge_attr,
                                               pe_EquivStableLapPE=es_data))
            # GatedGCN does residual connection and dropout internally.
            h_local = local_out.x
            setattr(batch, self.edge_attr_type, local_out.edge_attr)
        else:
            if self.equivstable_pe:
                h_local = self.local_model(h, edge_index, edge_attr,
                                           batch.pe_EquivStableLapPE)
            elif self.local_gnn_type == 'GCN':
                h_local = self.local_model(h, edge_index)
            else:
                h_local = self.local_model(h, edge_index, edge_attr)
            h_local = self.dropout_local(h_local)
            h_local = h_in1 + h_local  # Residual connection.

        if self.layer_norm:
            h_local = self.norm1_local(h_local, batch.batch)
        if self.batch_norm:
            h_local = self.norm1_local(h_local)

        return h_local


class GlobalModel(nn.Module):
    """
    Attention layer
    """

    def __init__(self, dim_h, global_model_type, edge_type, use_edge_attr, edge_attr_type, num_heads,
                dropout=0.0, attn_dropout=0.0, layer_norm=False,
                batch_norm=True, bigbird_cfg=None, exp_edges_cfg=None):

        super().__init__()

        self.dim_h = dim_h
        self.attn_dropout = attn_dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.num_heads = num_heads
        self.edge_type = edge_type
        self.edge_attr_type = edge_attr_type

        # Global attention transformer-style model.
        if global_model_type == 'Transformer':
            self.self_attn = torch.nn.MultiheadAttention(
                dim_h, num_heads, dropout=self.attn_dropout, batch_first=True)
        elif global_model_type == 'ETransformer':
            self.self_attn = ETransformer(dim_h, dim_h, num_heads, 
                                          use_bias=False, 
                                          edge_index=self.edge_type, 
                                          use_edge_attr=use_edge_attr, 
                                          edge_attr=self.edge_attr_type)
        elif global_model_type == 'Exphormer':
            self.self_attn = ExphormerAttention(dim_h, dim_h, num_heads,
                                          use_bias=False, 
                                          use_virt_nodes= exp_edges_cfg.num_virt_node > 0)
        elif global_model_type == 'Performer':
            self.self_attn = SelfAttention(
                dim=dim_h, heads=num_heads,
                dropout=self.attn_dropout, causal=False)
        elif global_model_type == "BigBird":
            bigbird_cfg.dim_hidden = dim_h
            bigbird_cfg.n_heads = num_heads
            bigbird_cfg.dropout = dropout
            self.self_attn = SingleBigBirdLayer(bigbird_cfg)
        else:
            raise ValueError(f"Unsupported global x-former model: "
                             f"{global_model_type}")
        self.global_model_type = global_model_type

        if self.layer_norm and self.batch_norm:
            raise ValueError("Cannot apply two types of normalization together")

        # Normalization for Self-Attention representation.
        if self.layer_norm:
            self.norm1_attn = pygnn.norm.GraphNorm(dim_h)
        if self.batch_norm:
            self.norm1_attn = nn.BatchNorm1d(dim_h)
        self.dropout_attn = nn.Dropout(dropout)

    def forward(self, batch):
        h = batch.x
        h_in1 = h  # for first residual connection

        # Multi-head attention.
        if self.global_model_type in ['ETransformer', 'Exphormer']:
            h_attn = self.self_attn(batch)
        else:
            h_dense, mask = to_dense_batch(h, batch.batch)
            if self.global_model_type == 'Transformer':
                h_attn = self._sa_block(h_dense, None, ~mask)[mask]
            elif self.global_model_type == 'Performer':
                h_attn = self.self_attn(h_dense, mask=mask)[mask]
            elif self.global_model_type == 'BigBird':
                h_attn = self.self_attn(h_dense, attention_mask=mask)
            else:
                raise RuntimeError(f"Unexpected {self.global_model_type}")

        h_attn = self.dropout_attn(h_attn)
        h_attn = h_in1 + h_attn  # Residual connection.
        if self.layer_norm:
            h_attn = self.norm1_attn(h_attn, batch.batch)
        if self.batch_norm:
            h_attn = self.norm1_attn(h_attn)
        return h_attn

    def _sa_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block.
        """
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return x


class MultiLayer(nn.Module):
    """Any combination of different models can be made here.
      Each layer can have several types of MPNN and Attention models combined.
      Examples:
      1. GCN
      2. GCN + Exphormer
      3. GINE + CustomGatedGCN
      4. GAT + CustomGatedGCN + Exphormer + Transformer
    """

    def __init__(self, dim_h,
                 model_types, num_heads,
                 pna_degrees=None, equivstable_pe=False, dropout=0.0,
                 attn_dropout=0.0, layer_norm=False, batch_norm=True,
                 bigbird_cfg=None, exp_edges_cfg=None):
        super().__init__()

        self.dim_h = dim_h
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.equivstable_pe = equivstable_pe
        self.model_types = model_types

        # Local message-passing models.
        self.models = []
        for layer_type in model_types:

            layer_type = layer_type.split('__')
            if len(layer_type) > 2:
                edge_attr_type = layer_type[2]
                edge_type = layer_type[1]
                layer_type = layer_type[0]
                use_edge_attr = True
            elif len(layer_type) == 2:
                edge_attr_type = None
                use_edge_attr = False
                edge_type = layer_type[1]
                layer_type = layer_type[0]
            else:
                edge_attr_type = 'edge_attr'
                edge_type = 'edge_index'
                use_edge_attr = True
                layer_type = layer_type[0]

            if layer_type in {'Transformer', 'ETransformer', 'Exphormer', 'Performer', 'BigBird'}:
                self.models.append(GlobalModel(dim_h=dim_h,
                                            global_model_type=layer_type,
                                            edge_type = edge_type,
                                            use_edge_attr = use_edge_attr,
                                            edge_attr_type = edge_attr_type,
                                            num_heads=self.num_heads,
                                            dropout=dropout,
                                            attn_dropout=self.attn_dropout,
                                            layer_norm=self.layer_norm,
                                            batch_norm=self.batch_norm,
                                            bigbird_cfg=bigbird_cfg,
                                            exp_edges_cfg=exp_edges_cfg))
                                            
            elif layer_type in {'GCN', 'GENConv', 'GINE', 'GAT', 'PNA', 'CustomGatedGCN'}:
                self.models.append(LocalModel(dim_h=dim_h,
                                            local_gnn_type=layer_type,
                                            edge_type = edge_type,
                                            edge_attr_type = edge_attr_type,
                                            num_heads=num_heads,
                                            pna_degrees=pna_degrees,
                                            equivstable_pe=self.equivstable_pe,
                                            dropout=dropout,
                                            layer_norm=self.layer_norm,
                                            batch_norm=self.batch_norm))
            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")

        self.models = nn.ModuleList(self.models)

        # Feed Forward block.
        self.activation = F.relu
        self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
        if self.layer_norm:
            # self.norm2 = pygnn.norm.LayerNorm(dim_h)
            self.norm2 = pygnn.norm.GraphNorm(dim_h)
            # self.norm2 = pygnn.norm.InstanceNorm(dim_h)
        if self.batch_norm:
            self.norm2 = nn.BatchNorm1d(dim_h)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)

    def forward(self, batch):
        h_out_list = []
        # representations from the models
        for model in self.models:
            h_out_list.append(model(batch))

        # Combine local and global outputs.
        # h = torch.cat(h_out_list, dim=-1)
        h = sum(h_out_list)

        # Feed Forward block.
        h = h + self._ff_block(h)
        if self.layer_norm:
            h = self.norm2(h, batch.batch)
        if self.batch_norm:
            h = self.norm2(h)

        batch.x = h
        return batch

    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.activation(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def extra_repr(self):
        s = f'summary: dim_h={self.dim_h}, ' \
            f'local_gnn_type={self.model_types}, ' \
            f'heads={self.num_heads}'
        return s


class SingleLayer(nn.Module):
    """Model just uses one layer type. 
    Difference with the Multi_Model is that after each layer there is no combining representations and Feed Forward network.
    """

    def __init__(self, dim_h,
                 model_type, num_heads,
                 pna_degrees=None, equivstable_pe=False, dropout=0.0,
                 attn_dropout=0.0, layer_norm=False, batch_norm=True,
                 bigbird_cfg=None, exp_edges_cfg=None):
        super().__init__()

        self.dim_h = dim_h
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.equivstable_pe = equivstable_pe
        self.model_type = model_type

        if self.model_type.endswith('__exp'):
            self.model_type = self.model_type[:-5]
            edge_type = 'expander_edge_index'
            edge_attr_type = 'expander_edge_attr'
        else:
            edge_type = 'edge_index'
            edge_attr_type = 'edge_attr'

        if self.model_type in {'Transformer', 'ETransformer', 'Exphormer', 'Performer', 'BigBird'}:
            self.model = GlobalModel(dim_h=dim_h,
                                        global_model_type=self.model_type,
                                        edge_type = edge_type,
                                        use_edge_attr = True,
                                        edge_attr_type = edge_attr_type,
                                        num_heads=self.num_heads,
                                        dropout=dropout,
                                        attn_dropout=self.attn_dropout,
                                        layer_norm=self.layer_norm,
                                        batch_norm=self.batch_norm,
                                        bigbird_cfg=bigbird_cfg,
                                        exp_edges_cfg=exp_edges_cfg)
                                        
        elif self.model_type in {'GENConv', 'GINE', 'GAT', 'PNA', 'CustomGatedGCN', 'GCN'}:
            self.model = LocalModel(dim_h=dim_h,
                                        local_gnn_type=self.model_type,
                                        edge_type = edge_type,
                                        edge_attr_type = edge_attr_type,
                                        num_heads=num_heads,
                                        pna_degrees=pna_degrees,
                                        equivstable_pe=self.equivstable_pe,
                                        dropout=dropout,
                                        layer_norm=self.layer_norm,
                                        batch_norm=self.batch_norm)
        else:
            raise ValueError(f"Unsupported layer type: {self.model_type}")

        self.activation = torch.nn.ReLU()


    def forward(self, batch):
        batch.x = self.activation(self.model(batch))
        return batch

    def extra_repr(self):
        s = f'summary: dim_h={self.dim_h}, ' \
            f'local_gnn_type={self.model_type}, ' \
            f'heads={self.num_heads}'
        return s
