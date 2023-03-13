import torch
import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_node_encoder


@register_node_encoder('ERN')
class ERNodeEncoder(torch.nn.Module):
    """Effective Resistance Node Encoder

    ER of size dim_pe will get appended to each node feature vector.
    If `expand_x` set True, original node features will be first linearly
    projected to (dim_emb - dim_pe) size and the concatenated with ER.

    Args:
        dim_emb: Size of final node embedding
        expand_x: Expand node features `x` from dim_in to (dim_emb - dim_pe)
    """

    def __init__(self, dim_emb, expand_x=True):
        super().__init__()
        dim_in = cfg.share.dim_in  # Expected original input node features dim

        pecfg = cfg.posenc_ERN
        dim_pe = pecfg.dim_pe  # Size of Laplace PE embedding
        model_type = pecfg.model  # Encoder NN model type for DEs
        if model_type not in ['Transformer', 'DeepSet', 'Linear']:
            raise ValueError(f"Unexpected PE model {model_type}")
        self.model_type = model_type
        n_layers = pecfg.layers  # Num. layers in PE encoder model
        n_heads = pecfg.n_heads  # Num. attention heads in Trf PE encoder
        post_n_layers = pecfg.post_layers  # Num. layers to apply after pooling
        self.pass_as_var = pecfg.pass_as_var  # Pass PE also as a separate variable

        er_dim = pecfg.er_dim

        if dim_emb - dim_pe < 1:
            raise ValueError(f"ER_Node size {dim_pe} is too large for "
                             f"desired embedding size of {dim_emb}.")

        if expand_x:
            self.linear_x = nn.Linear(dim_in, dim_emb - dim_pe)
        self.expand_x = expand_x

        if model_type == 'Linear':
            self.pe_encoder = nn.Linear(er_dim, dim_pe)

        else:
            if model_type == 'Transformer':
                # Initial projection of each value of ER embedding
                self.linear_A = nn.Linear(1, dim_pe)
                # Transformer model for ER_Node
                encoder_layer = nn.TransformerEncoderLayer(d_model=dim_pe,
                                                        nhead=n_heads,
                                                        batch_first=True)
                self.pe_encoder = nn.TransformerEncoder(encoder_layer,
                                                        num_layers=n_layers)
            else:
                # DeepSet model for ER_Node
                layers = []
                if n_layers == 1:
                    layers.append(nn.ReLU())
                else:
                    self.linear_A = nn.Linear(1, dim_pe)
                    layers.append(nn.ReLU())
                    for _ in range(n_layers - 1):
                        layers.append(nn.Linear(dim_pe, dim_pe))
                        layers.append(nn.ReLU())
                self.pe_encoder = nn.Sequential(*layers)

        self.post_mlp = None
        if post_n_layers > 0:
            # MLP to apply post pooling
            layers = []
            if post_n_layers == 1:
                layers.append(nn.Linear(dim_pe, dim_pe))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(dim_pe, 2 * dim_pe))
                layers.append(nn.ReLU())
                for _ in range(post_n_layers - 2):
                    layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
                    layers.append(nn.ReLU())
                layers.append(nn.Linear(2 * dim_pe, dim_pe))
                layers.append(nn.ReLU())
            self.post_mlp = nn.Sequential(*layers)


    def forward(self, batch):
        if not hasattr(batch, 'er_emb'):
            raise ValueError("Precomputed ER embeddings required for calculating ER Node Encodings")
        
        pos_enc = batch.er_emb  # N * er_dim

        if self.training:
            pos_enc = pos_enc[:, torch.randperm(pos_enc.size()[1])]

        if self.model_type == 'Linear':
            pos_enc = self.pe_encoder(pos_enc)  #  N * er_dim -> N * dim_pe

        else:
            pos_enc = torch.unsqueeze(pos_enc, 2)
            pos_enc = self.linear_A(pos_enc)  # (Num nodes) x (er_dim) x dim_pe

            # PE encoder: a Transformer or DeepSet model
            if self.model_type == 'Transformer':
                pos_enc = self.pe_encoder(src=pos_enc)
            else:
                pos_enc = self.pe_encoder(pos_enc)

            # Sum pooling
            pos_enc = torch.sum(pos_enc, 1, keepdim=False)  # (Num nodes) x dim_pe

        # MLP post pooling
        if self.post_mlp is not None:
            pos_enc = self.post_mlp(pos_enc)  # (Num nodes) x dim_pe

        # Expand node features if needed
        if self.expand_x:
            h = self.linear_x(batch.x)
        else:
            h = batch.x
        # Concatenate final PEs to input embedding
        batch.x = torch.cat((h, pos_enc), 1)
        # Keep PE also separate in a variable (e.g. for skip connections to input)
        if self.pass_as_var:
            batch.pe_ern = pos_enc
        return batch
