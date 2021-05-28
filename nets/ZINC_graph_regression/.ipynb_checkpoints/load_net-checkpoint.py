
from nets.ZINC_graph_regression.graph_transformer_net_node_PE import GraphTransformerNetNodePE
from nets.ZINC_graph_regression.graph_transformer_net_edge_PE import GraphTransformerNetEdgePE
from nets.ZINC_graph_regression.graph_transformer_net import GraphTransformerNet


def GraphTransformerNodePE(net_params):
    return GraphTransformerNetNodePE(net_params)

def GraphTransformerEdgePE(net_params):
    return GraphTransformerNetEdgePE(net_params)

def GraphTransformer(net_params):
    return GraphTransformerNet(net_params)

def gnn_model(LPE, net_params):
    model = {
        'edge': GraphTransformerEdgePE,
        'node': GraphTransformerNodePE,
        'none': GraphTransformer
    }
        
    return model[LPE](net_params)