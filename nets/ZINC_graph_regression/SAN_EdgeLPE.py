import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import numpy as np

"""
    Graph Transformer with edge features
    
"""
from layers.graph_transformer_layer import GraphTransformerLayer
from layers.mlp_readout_layer import MLPReadout

class SAN_EdgeLPE(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        
        num_atom_type = net_params['num_atom_type']
        num_bond_type = net_params['num_bond_type']
        
        full_graph = net_params['full_graph']
        gamma = net_params['gamma']
        
        LPE_layers = net_params['LPE_layers']
        LPE_dim = net_params['LPE_dim']
        LPE_n_heads = net_params['LPE_n_heads']
        
        GT_layers = net_params['GT_layers']
        GT_hidden_dim = net_params['GT_hidden_dim']
        GT_out_dim = net_params['GT_out_dim']
        GT_n_heads = net_params['GT_n_heads']
        
        self.residual = net_params['residual']
        self.readout = net_params['readout']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']

        self.readout = net_params['readout']
        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']

        self.device = net_params['device']
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.embedding_h = nn.Embedding(num_atom_type, GT_hidden_dim)
        self.embedding_e = nn.Embedding(num_bond_type, GT_hidden_dim-LPE_dim)#Remove some embedding dimensions to make room for concatenating laplace encoding
        self.linear_A = nn.Linear(3, LPE_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=LPE_dim, nhead=LPE_n_heads)
        self.PE_Transformer = nn.TransformerEncoder(encoder_layer, num_layers=LPE_layers)
        
        self.layers = nn.ModuleList([ GraphTransformerLayer(gamma, GT_hidden_dim, GT_hidden_dim, GT_n_heads, full_graph, dropout, self.layer_norm, self.batch_norm, self.residual) for _ in range(GT_layers-1) ])
        
        self.layers.append(GraphTransformerLayer(gamma, GT_hidden_dim, GT_out_dim, GT_n_heads, full_graph, dropout, self.layer_norm, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(GT_out_dim, 1)   # 1 out dim since regression problem        
        
        
    def forward(self, g, h, e, diff, product, EigVals):
        
        # input embedding
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        e = self.embedding_e(e)        
          
        PosEnc = torch.cat((diff, product, EigVals), 2) # (Num edges) x (Num Eigenvectors) x 3
        empty_mask = torch.isnan(PosEnc) # (Num edges) x (Num Eigenvectors) x 3
        PosEnc[empty_mask] = 0 # (Num edges) x (Num Eigenvectors) x 3

        PosEnc = torch.transpose(PosEnc, 0 ,1).float() # (Num Eigenvectors) x (Num edges) x 3
        PosEnc = self.linear_A(PosEnc) # (Num Eigenvectors) x (Num edges) x PE_dim
            
        #1st Transformer: Learned PE
        PosEnc = self.PE_Transformer(src=PosEnc, src_key_padding_mask=empty_mask[:,:,0]) 
        
        #remove masked sequences
        PosEnc[torch.transpose(empty_mask, 0 ,1)[:,:,0]] = float('nan') 

        #Sum pooling
        PosEnc = torch.nansum(PosEnc, 0, keepdim=False) # (Num edge) x PE_dim

        #Concatenate learned PE to input embedding
        e = torch.cat((e, PosEnc), 1)
        
        
        # GNN
        for conv in self.layers:
            h, e = conv(g, h, e)
        g.ndata['h'] = h
        
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes
            
        return self.MLP_layer(hg)
        
    def loss(self, scores, targets):

        loss = nn.L1Loss()(scores, targets)
        
        return loss
