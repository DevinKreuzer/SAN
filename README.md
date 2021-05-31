# SAN

Implementation of Spectral Attention Networks, a powerful GNN that leverages key principles from spectral graph theory to enable full graph attention.

![full_method](https://user-images.githubusercontent.com/47570400/119883871-046aa280-befe-11eb-9063-108f4fe1a123.png)

# Overview

* ```nets``` contains the Node, Edge and no LPE architectures implemented with PyTorch.
* ```layers``` contains the multi-headed attention employed by the Main Graph Transformer implemented in DGL.
* ```train``` contains methods to train the models.
* ```data``` contains dataset classes and various methods used in precomputation.
* ```configs``` contains the various parameters used in the ablation and SOTA comparison studies.
* ```misc``` contains scripts from https://github.com/graphdeeplearning/graphtransformer to download datasets and setup environments.
* ```scripts``` contains scripts to reproduce ablation and SOTA comparison results. See ```scripts/reproduce.md``` for details.


