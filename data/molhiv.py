import torch
import pickle
import torch.utils.data
import time
import os
import numpy as np

import csv

import dgl
import torch.nn.functional as F

from scipy import sparse as sp
import numpy as np
import networkx as nx
import hashlib


def laplace_decomp(graph, max_freqs):
    g, label = graph

    # Laplacian
    n = g.number_of_nodes()
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVals, EigVecs = np.linalg.eigh(L.toarray())
    EigVals, EigVecs = EigVals[: max_freqs], EigVecs[:, :max_freqs]  # Keep up to the maximum desired number of frequencies

    # Normalize and pad EigenVectors
    EigVecs = torch.from_numpy(EigVecs).float()
    EigVecs = F.normalize(EigVecs, p=2, dim=1, eps=1e-12, out=None)

    if n < max_freqs:
        g.ndata['EigVecs'] = F.pad(EigVecs, (0, max_freqs - n), value=float('nan'))
    else:
        g.ndata['EigVecs'] = EigVecs

    # Save eigenvalues and pad
    EigVals = torch.from_numpy(np.sort(np.abs(np.real(
        EigVals))))  # Abs value is taken because numpy sometimes computes the first eigenvalue approaching 0 from the negative

    if n < max_freqs:
        EigVals = F.pad(EigVals, (0, max_freqs - n), value=float('nan')).unsqueeze(0)
    else:
        EigVals = EigVals.unsqueeze(0)

    # Save EigVals node features
    g.ndata['EigVals'] = EigVals.repeat(g.number_of_nodes(), 1).unsqueeze(2)

    return g, label


def make_full_graph(graph):
    g, label = graph

    full_g = dgl.from_networkx(nx.complete_graph(g.number_of_nodes()))

    # Copy over the node feature data and laplace eigvals/eigvecs
    full_g.ndata['feat'] = g.ndata['feat']

    try:
        full_g.ndata['EigVecs'] = g.ndata['EigVecs']
        full_g.ndata['EigVals'] = g.ndata['EigVals']
    except:
        pass

    # Initalize fake edge features w/ 0s
    full_g.edata['feat'] = torch.zeros(full_g.number_of_edges(), 3, dtype=torch.long)
    full_g.edata['real'] = torch.zeros(full_g.number_of_edges(), dtype=torch.long)

    # Copy real edge data over, and identify real edges!
    full_g.edges[g.edges(form='uv')[0].tolist(), g.edges(form='uv')[1].tolist()].data['feat'] = g.edata['feat']
    full_g.edges[g.edges(form='uv')[0].tolist(), g.edges(form='uv')[1].tolist()].data['real'] = torch.ones(
        g.edata['feat'].shape[0], dtype=torch.long)  # This indicates real edges

    return full_g, label


def add_edge_laplace_feats(graph):
    g, label = graph

    EigVals = g.ndata['EigVals'][0].flatten()

    source, dest = g.find_edges(g.edges(form='eid'))

    # Compute diffusion differences and Green function
    g.edata['diff'] = torch.abs(g.nodes[source].data['EigVecs'] - g.nodes[dest].data['EigVecs']).unsqueeze(2)
    g.edata['product'] = torch.mul(g.nodes[source].data['EigVecs'], g.nodes[dest].data['EigVecs']).unsqueeze(2)
    g.edata['EigVals'] = EigVals.repeat(g.number_of_edges(), 1).unsqueeze(2)

    # No longer need EigVecs and EigVals stored as node features
    del g.ndata['EigVecs']
    del g.ndata['EigVals']

    return g, label


from ogb.graphproppred import DglGraphPropPredDataset, collate_dgl


class MolHIVDataset(torch.utils.data.Dataset):

    def __init__(self, name):
        """
            Loading ZINC dataset
        """
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name

        dataset = DglGraphPropPredDataset(name='ogbg-molhiv')
        split_idx = dataset.get_idx_split()

        split_idx["train"] = split_idx["train"]
        split_idx["valid"] = split_idx["valid"]
        split_idx["test"] = split_idx["test"]

        self.train = dataset[split_idx["train"]]
        self.val = dataset[split_idx["valid"]]
        self.test = dataset[split_idx["test"]]

        print('train, test, val sizes :', len(self.train), len(self.test), len(self.val))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time() - start))

    def collate(self, samples):
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        labels = torch.stack(labels)

        return batched_graph, labels

    def _laplace_decomp(self, max_freqs):
        self.train = [laplace_decomp(graph, max_freqs) for graph in self.train]
        self.val = [laplace_decomp(graph, max_freqs) for graph in self.val]
        self.test = [laplace_decomp(graph, max_freqs) for graph in self.test]

    def _make_full_graph(self):
        self.train = [make_full_graph(graph) for graph in self.train]
        self.val = [make_full_graph(graph) for graph in self.val]
        self.test = [make_full_graph(graph) for graph in self.test]

    def _add_edge_laplace_feats(self):
        self.train = [add_edge_laplace_feats(graph) for graph in self.train]
        self.val = [add_edge_laplace_feats(graph) for graph in self.val]
        self.test = [add_edge_laplace_feats(graph) for graph in self.test]
