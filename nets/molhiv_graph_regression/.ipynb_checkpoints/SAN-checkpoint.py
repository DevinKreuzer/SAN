import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import dgl

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

"""
    Graph Transformer with edge features
    
"""
from layers.graph_transformer_layer import GraphTransformerLayer
from layers.mlp_readout_layer import MLPReadout

class SAN(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        
        full_graph = net_params['full_graph']
        gamma = net_params['gamma']
        
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
        
        self.embedding_h = AtomEncoder(emb_dim = GT_hidden_dim)
        self.embedding_e = BondEncoder(emb_dim = GT_hidden_dim)

        
        self.layers = nn.ModuleList([ GraphTransformerLayer(gamma, GT_hidden_dim, GT_hidden_dim, GT_n_heads, full_graph, dropout, self.layer_norm, self.batch_norm, self.residual) for _ in range(GT_layers-1) ]) 
        
        self.layers.append(GraphTransformerLayer(gamma, GT_hidden_dim, GT_out_dim, GT_n_heads, full_graph, dropout, self.layer_norm, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(GT_out_dim, 1)   #  out dim for probability     
        
        
    def forward(self, g, h, e):
        
        # input embedding
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        e = self.embedding_e(e)  
          
        # Second Transformer
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
            
        sig = nn.Sigmoid()
    
        return sig(self.MLP_layer(hg))
        
    def loss(self, scores, targets):
        
        loss = nn.BCELoss()
        
        l = loss(scores.float(), targets.float())
        
        return l
