import numpy as np
import torch
from torch.nn import init
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform
from torch_geometric.nn import global_mean_pool, global_add_pool
#import dgl

### GIN convolution along the graph structure
class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GINConv, self).__init__(aggr="add")
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim),
                                       torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))
        # self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))


    def forward(self, x, edge_index, edge_attr):
        # print(edge_attr)
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_attr))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


### GCN convolution along the graph structure
class GCNConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)

        row, col = edge_index

        # edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr=edge_attr, norm=norm) + F.relu(
            x + self.root_emb.weight) * 1. / deg.view(-1, 1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out



# pylint: enable=W0235
class EGATConv(torch.nn.Module):
    r"""

    Description
    -----------
    Apply Graph Attention Layer over input graph. EGAT is an extension
    of regular `Graph Attention Network <https://arxiv.org/pdf/1710.10903.pdf>`__
    handling edge features, detailed description is available in
    `Rossmann-Toolbox <https://pubmed.ncbi.nlm.nih.gov/34571541/>`__ (see supplementary data).
     The difference appears in the method how unnormalized attention scores :math:`e_{ij}`
     are obtain:

    .. math::
        e_{ij} &= \vec{F} (f_{ij}^{\prime})

        f_{ij}^{\prim} &= \mathrm{LeakyReLU}\left(A [ h_{i} \| f_{ij} \| h_{j}]\right)

    where :math:`f_{ij}^{\prim}` are edge features, :math:`\mathrm{A}` is weight matrix and
    :math: `\vec{F}` is weight vector. After that resulting node features
    :math:`h_{i}^{\prim}` are updated in the same way as in regular GAT.

    Parameters
    ----------
    in_node_feats : int
        Input node feature size :math:`h_{i}`.
    in_edge_feats : int
        Input edge feature size :math:`f_{ij}`.
    out_node_feats : int
        Output nodes feature size.
    out_edge_feats : int
        Output edge feature size.
    num_heads : int
        Number of attention heads.


    """

    def __init__(self,
                 in_node_feats,
                 in_edge_feats,
                 out_node_feats,
                 out_edge_feats,
                 num_heads,
                 **kw_args):
        super().__init__()
        self._num_heads = num_heads
        self._out_node_feats = out_node_feats
        self._out_edge_feats = out_edge_feats
        self.fc_nodes = torch.nn.Linear(in_node_feats, out_node_feats * num_heads, bias=True)
        self.fc_edges = torch.nn.Linear(in_edge_feats + 2 * in_node_feats, out_edge_feats * num_heads, bias=False)
        self.fc_attn = torch.nn.Linear(out_edge_feats, num_heads, bias=False)
        self.fc_out_node = torch.nn.Linear(out_node_feats * num_heads, out_node_feats, bias=True)
        self.fc_out_edge = torch.nn.Linear(out_edge_feats * num_heads, out_edge_feats, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reinitialize learnable parameters.
        """
        gain = init.calculate_gain('relu')
        init.xavier_normal_(self.fc_nodes.weight, gain=gain)
        init.xavier_normal_(self.fc_edges.weight, gain=gain)
        init.xavier_normal_(self.fc_attn.weight, gain=gain)

    def edge_attention(self, edges):
        # extract features
        h_src = edges.src['h']
        h_dst = edges.dst['h']
        f = edges.data['f']
        # stack h_i | f_ij | h_j
        stack = torch.cat([h_src, f, h_dst], dim=-1)
        # apply FC and activation
        f_out = self.fc_edges(stack)
        f_out = F.leaky_relu(f_out)
        f_out = f_out.view(-1, self._num_heads, self._out_edge_feats)
        # apply FC to reduce edge_feats to scalar
        a = self.fc_attn(f_out).sum(-1).unsqueeze(-1)

        return {'a': a, 'f': f_out}

    def message_func(self, edges):
        return {'h': edges.src['h'], 'a': edges.data['a']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['a'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['h'], dim=1)
        return {'h': h}

    def forward(self, nfeats,graph, efeats):
        r"""
        Compute new node and edge features.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        nfeats : torch.Tensor
            The input node feature of shape :math:`(*, D_{in})`
            where:
                :math:`D_{in}` is size of input node feature,
                :math:`*` is the number of nodes.
        efeats: torch.Tensor
             The input edge feature of shape :math:`(*, F_{in})`
             where:
                 :math:`F_{in}` is size of input node feauture,
                 :math:`*` is the number of edges.


        Returns
        -------
        pair of torch.Tensor
            node output features followed by edge output features
            The node output feature of shape :math:`(*, H, D_{out})`
            The edge output feature of shape :math:`(*, H, F_{out})`
            where:
                :math:`H` is the number of heads,
                :math:`D_{out}` is size of output node feature,
                :math:`F_{out}` is size of output edge feature.
        """

        with graph.local_scope():
            ##TODO allow node src and dst feats
            graph.edata['f'] = efeats
            graph.ndata['h'] = nfeats

            graph.apply_edges(self.edge_attention)

            nfeats_ = self.fc_nodes(nfeats)
            nfeats_ = nfeats_.view(-1, self._num_heads, self._out_node_feats)

            graph.ndata['h'] = nfeats_
            graph.update_all(message_func=self.message_func,
                             reduce_func=self.reduce_func)
            h = graph.ndata.pop('h')
            e = graph.edata.pop('f')
            h = self.fc_out_node(h.view(-1,self._num_heads * self._out_node_feats))
            e = self.fc_out_edge(e.view(-1,self._num_heads * self._out_node_feats))

            return h,e


### GNN to generate node embedding
class GNN_mol(torch.nn.Module):
    """

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """
    def __init__(self, num_layer, hidden_dim, gnn_type, JK='last',  drop_ratio=0):
        super(GNN_mol, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        #self.batch_norms = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        self.gnn_type = gnn_type

        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(hidden_dim))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(hidden_dim))
            elif gnn_type == "egat":
                self.gnns.append(EGATConv(hidden_dim,hidden_dim,hidden_dim,hidden_dim,2))
            '''
            elif gnn_type == "gat":
                self.gnns.append(GATConv(hidden_dim, heads = 4, out_channels=150))
            elif gnn_type == "sage":
                self.gnns.append(SAGEConv(self.nn, hidden_dim))
            '''
            #self.norms.append(torch.nn.BatchNorm1d(hidden_dim))
            self.norms.append(torch.nn.GroupNorm(num_groups=10, num_channels=hidden_dim))

    def forward(self, x, edge_index, edge_attr):

        if self.gnn_type == "egat":
            #print(edge_index )
            edge_index = edge_index.T
            #print(batch)
            u, v = edge_index[:, 0], edge_index[:, 1]
            #print(x.size(), edge_attr.size())
            graph = dgl.graph((u,v))
            h_list, e_list= [x], [edge_attr]

            for layer in range(self.num_layer):

                h,e = self.gnns[layer](h_list[layer],graph,e_list[layer])
                # h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
                h,e = self.norms[layer](h), self.norms[layer](e)
                if layer == self.num_layer - 1:
                    # remove relu for the last layer
                    h = F.dropout(h, self.drop_ratio, training=self.training)

                else:
                    h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

                h_list.append(h)
                e_list.append(e)

        else:
            h_list = [x]
            for layer in range(self.num_layer):
                h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
                h = self.norms[layer](h)
                if layer == self.num_layer - 1:
                    #remove relu for the last layer
                    h = F.dropout(h, self.drop_ratio, training = self.training)
                else:
                    h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
                h_list.append(h)


        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]

        return node_representation


class GNN_mol_virtual(torch.nn.Module):
    """

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """
    def __init__(self, num_layer, hidden_dim, gnn_type, JK='last',  drop_ratio=0):
        super(GNN_mol_virtual, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        #self.batch_norms = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        self.virtualnode_embedding = torch.nn.Embedding(1, hidden_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)
        self.mlp_virtualnode_list = torch.nn.ModuleList()
        self.gnn_type = gnn_type

        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(hidden_dim))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(hidden_dim))
            elif gnn_type == "egat":
                self.gnns.append(EGATConv(hidden_dim,hidden_dim,hidden_dim,hidden_dim,2))
            '''
            elif gnn_type == "gat":
                self.gnns.append(GATConv(hidden_dim, heads = 4, out_channels=150))
            elif gnn_type == "sage":
                self.gnns.append(SAGEConv(self.nn, hidden_dim))
            '''
            #self.norms.append(torch.nn.BatchNorm1d(hidden_dim))
            self.norms.append(torch.nn.GroupNorm(num_groups=10, num_channels=hidden_dim))

        #for layer in range(num_layer - 1):
            self.mlp_virtualnode_list.append(torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.BatchNorm1d(hidden_dim), torch.nn.ReLU(), \
                                                    torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.BatchNorm1d(hidden_dim), torch.nn.ReLU()))

    def forward(self, x, edge_index, edge_attr, batch):

        '''
            if len(argv) == 2:
                x, edge_index = argv[0], argv[1]
            elif len(argv) == 1:
                data = argv[0]
                x, edge_index = data.x, data.edge_index
            else:
                raise ValueError("unmatched number of arguments.")
        '''

        if self.gnn_type == "egat":
            #print(edge_index )
            edge_index = edge_index.T
            #print(batch)
            u, v = edge_index[:, 0], edge_index[:, 1]
            #print(x.size(), edge_attr.size())
            graph = dgl.graph((u,v))
            h_list, e_list= [x], [edge_attr]

            for layer in range(self.num_layer):

                h,e = self.gnns[layer](h_list[layer],graph,e_list[layer])
                # h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
                h,e = self.norms[layer](h), self.norms[layer](e)
                if layer == self.num_layer - 1:
                    # remove relu for the last layer
                    h = F.dropout(h, self.drop_ratio, training=self.training)

                else:
                    h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

                h_list.append(h)
                e_list.append(e)

        else:

            h_list = [x]

            virtualnode_embedding = self.virtualnode_embedding(
                torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))


            for layer in range(self.num_layer):

                h = self.gnns[layer](h_list[layer], edge_index, edge_attr) + virtualnode_embedding[batch]
                #h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
                h = self.norms[layer](h)

                if layer == self.num_layer - 1:
                    #remove relu for the last layer
                    h = F.dropout(h, self.drop_ratio, training = self.training)
                else:
                    h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
                h_list.append(h)


                if layer < self.num_layer - 1:
                    ### add message from graph nodes to virtual nodes
                    virtualnode_embedding_temp = global_add_pool(h_list[layer], batch) + virtualnode_embedding
                    virtualnode_embedding = F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp),
                                                                                         self.drop_ratio, training=self.training)


        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]

        return node_representation







