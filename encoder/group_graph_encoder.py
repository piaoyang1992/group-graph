#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# @Time : 2022/9/22 9:11 
# @Author : piaoyang cao 
# @File : group_graph_encoder.py

import torch
import torch.nn as nn
from torch_geometric.data import Data
#from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set

import os
import numpy as np
import pandas as pd

from graph_embed import GNN_mol
from gnn import GNN

from graph_load import SmilesDataset, DrugDataset
#from frag2graph import VocabDataset
from tqdm import tqdm
# 得到整个图的隐变量表示
from ogb.graphproppred.mol_encoder import AtomEncoder
#from MTL_BERT_model import EncoderForPrediction,PredictionModel


def smiles_to_batch_data(smiles_id, raw_smiles_data):
    data_list = []
    for id in smiles_id:
        raw_data = raw_smiles_data[id]
        data_list.append(raw_data)
    new_batch = DataLoader(data_list, batch_size=len(data_list), shuffle=False)
    new_batch = [bat for bat in new_batch][0]
    #new_batch = Batch().from_data_list(graph_data_batch)
    return new_batch

class HierGnnEncoder(nn.Module):

    def __init__(self, vocab_size,num_tasks, device, num_layers, emb_dim, drop_ratio, graph_pooling, gnn_type, JK):  # atom_sum 表示frag的最大原子数
        super(HierGnnEncoder, self).__init__()
        self.vocab = vocab_size
        self.num_tasks = num_tasks
        self.device = device
        self.num_layers = num_layers
        self.emb_dim = emb_dim
        self.drop_ratio = drop_ratio
        self.JK = JK
        #self.pretrain_file = pretrain_file


        #self.inter_gnn = GNN_mol(num_layers, emb_dim, gnn_type)
        self.graph_gnn = GNN_mol(num_layers, emb_dim, gnn_type)

        self.atom_encoder = AtomEncoder(int(0.5 * emb_dim))

        self.node_embed = torch.nn.Sequential(

            torch.nn.Dropout(drop_ratio),
            torch.nn.Linear(vocab_size[1] + emb_dim, emb_dim),
            #torch.nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
            #torch.nn.GroupNorm(num_groups=4, num_channels=emb_dim)
            torch.nn.BatchNorm1d(emb_dim),

        )

        self.mol_embed = torch.nn.Sequential(
            torch.nn.Dropout(drop_ratio),
            torch.nn.Linear(2*emb_dim, emb_dim),
            #torch.nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
            #torch.nn.GroupNorm(num_groups=4, num_channels=emb_dim),
            #torch.nn.Linear(emb_dim, num_tasks)
            torch.nn.BatchNorm1d(emb_dim),

        )

        # Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool

        #self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)
        self.graph_pred_linear = torch.nn.Sequential(
            #torch.nn.BatchNorm1d(emb_dim),
            #torch.nn.GroupNorm(num_groups=4, num_channels=emb_dim),
            torch.nn.Linear(self.emb_dim, self.num_tasks),
            #torch.nn.Dropout(drop_ratio),
            #torch.nn.Linear(emb_dim, num_tasks)
        )
        self.embedding = torch.nn.Embedding(vocab_size[0], emb_dim)


    def forward(self, vocab_datas, batch_data):


        x, edge_index, edge_attr, inter_tensor, smiles_id, batch = batch_data.x, batch_data.edge_index, batch_data.edge_attr, \
                                                                   batch_data.inter_tensor, batch_data.smiles_id, batch_data.batch

        x_embed = self.embedding(x.squeeze())

        x = x.to('cpu')

        x_input = vocab_datas.index_select(index=x.squeeze(), dim=0).to(self.device)

        x_tensor = self.node_embed(torch.cat([x_embed,x_input],dim=1))


        edge_attr_0 = torch.cat([inter_tensor[:,0,:], edge_attr[:, 0:2]], dim=1)
        edge_attr_1 = torch.cat([inter_tensor[:,1,:], edge_attr[:, 2:4]], dim=1)


        edge_attr_0 = self.atom_encoder(edge_attr_0)
        edge_attr_1 = self.atom_encoder(edge_attr_1)


        edge_attr = torch.cat([edge_attr_0, edge_attr_1], dim=1)
        #print(edge_attr.size())

        edge_attr_trans = torch.cat([edge_attr_1, edge_attr_0], dim=1)
        edge_attr = torch.cat([edge_attr, edge_attr_trans], dim=0)  #增加反向边信息
        #print(edge_attr.size())

        edge_index_trans = torch.stack([edge_index[1, :], edge_index[0, :]], dim=0)
        #print(edge_index_trans.size())
        edge_index = torch.cat([edge_index, edge_index_trans], dim=1) # 增加反向边
        #print(x[0].size(), vocab_tensor.size())
        node_representation = self.graph_gnn(x_tensor, edge_index, edge_attr)
        #print(x_tensor)
        #node_representation = self.graph_pred_linear(node_representation)
        node_representation = self.graph_pred_linear(node_representation)


        mol_pred = self.pool(node_representation, batch)



        return mol_pred,node_representation





