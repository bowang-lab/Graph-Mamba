import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_layer


class ExpanderEdgeFixer(nn.Module):
    '''
        Gets the batch and sets new edge indices + global nodes
    '''
    def __init__(self, add_edge_index=False, num_virt_node=0):
        
        super().__init__()

        if not hasattr(cfg.gt, 'dim_edge') or cfg.gt.dim_edge is None:
            cfg.gt.dim_edge = cfg.gt.dim_hidden

        self.add_edge_index = add_edge_index
        self.num_virt_node = num_virt_node
        self.exp_edge_attr = nn.Embedding(1, cfg.gt.dim_edge)
        self.use_exp_edges = cfg.prep.use_exp_edges and cfg.prep.exp

        if self.num_virt_node > 0:
            self.virt_node_emb = nn.Embedding(self.num_virt_node, cfg.gt.dim_hidden)
            self.virt_edge_out_emb = nn.Embedding(self.num_virt_node, cfg.gt.dim_edge)
            self.virt_edge_in_emb = nn.Embedding(self.num_virt_node, cfg.gt.dim_edge)


    def forward(self, batch):
        edge_types = []
        device = self.exp_edge_attr.weight.device
        edge_index_sets = []
        edge_attr_sets = []
        if self.add_edge_index:
            edge_index_sets.append(batch.edge_index)
            edge_attr_sets.append(batch.edge_attr)
            edge_types.append(torch.zeros(batch.edge_index.shape[1], dtype=torch.long))


        num_node = batch.batch.shape[0]
        num_graphs = batch.num_graphs

        if self.use_exp_edges:
            if not hasattr(batch, 'expander_edges'):
                raise ValueError('expander edges not stored in data')

            data_list = batch.to_data_list()
            exp_edges = []
            cumulative_num_nodes = 0
            for data in data_list:
                exp_edges.append(data.expander_edges + cumulative_num_nodes)
                cumulative_num_nodes += data.num_nodes

            exp_edges = torch.cat(exp_edges, dim=0).t()
            edge_index_sets.append(exp_edges)
            edge_attr_sets.append(self.exp_edge_attr(torch.zeros(exp_edges.shape[1], dtype=torch.long).to(device)))
            edge_types.append(torch.zeros(exp_edges.shape[1], dtype=torch.long) + 1)

        if self.num_virt_node > 0:
            global_h = []
            virt_edges = []
            virt_edge_attrs = []
            for idx in range(self.num_virt_node):
                global_h.append(self.virt_node_emb(torch.zeros(num_graphs, dtype=torch.long).to(device)+idx))
                virt_edge_index = torch.cat([torch.arange(num_node).view(1, -1).to(device),
                                        (batch.batch+(num_node+idx*num_graphs)).view(1, -1)], dim=0)
                virt_edges.append(virt_edge_index)
                virt_edge_attrs.append(self.virt_edge_in_emb(torch.zeros(virt_edge_index.shape[1], dtype=torch.long).to(device)+idx))

                virt_edge_index = torch.cat([(batch.batch+(num_node+idx*num_graphs)).view(1, -1), 
                                                    torch.arange(num_node).view(1, -1).to(device)], dim=0)
                virt_edges.append(virt_edge_index)
                virt_edge_attrs.append(self.virt_edge_out_emb(torch.zeros(virt_edge_index.shape[1], dtype=torch.long).to(device)+idx))
                
            batch.virt_h = torch.cat(global_h, dim=0)
            batch.virt_edge_index = torch.cat(virt_edges, dim=1)
            batch.virt_edge_attr = torch.cat(virt_edge_attrs, dim=0)
            edge_types.append(torch.zeros(batch.virt_edge_index.shape[1], dtype=torch.long)+2)
        
        if len(edge_index_sets) > 1:
            edge_index = torch.cat(edge_index_sets, dim=1)
            edge_attr = torch.cat(edge_attr_sets, dim=0)
            edge_types = torch.cat(edge_types)
        else:
            edge_index = edge_index_sets[0]
            edge_attr = edge_attr_sets[0]
            edge_types = edge_types[0]

        del batch.expander_edges
        batch.expander_edge_index = edge_index
        batch.expander_edge_attr = edge_attr

        return batch