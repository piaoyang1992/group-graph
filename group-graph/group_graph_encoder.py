#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/22 9:11
# @Author : piaoyang cao
# @File : subgraph_encoder.py

import torch
import torch.nn as nn

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops

import os
import numpy as np
import pandas as pd

from graph_embed import GNN_mol, GCNConv




class HierGnnEncoder(nn.Module):

    def __init__(self, vocab_size, num_tasks, device, pretrain_file, args):  # atom_sum 表示frag的最大原子数
        super(HierGnnEncoder, self).__init__()
        self.vocab = vocab_size
        self.num_tasks = num_tasks
        self.device = device
        self.num_layers = args.num_layers
        self.emb_dim = args.embed_size
        self.drop_ratio = args.dropout
        self.JK = args.JK
        self.pretrain_file = pretrain_file
        self.gnn_type = args.gnn_type

        self.graph_gnn = GNN_mol(self.num_layers, self.emb_dim, self.gnn_type, drop_ratio=0)

        self.graph_pooling = args.graph_pooling


        self.fc_gat = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim, self.emb_dim),
            torch.nn.ReLU(),
        )
        self.fpn = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim, self.emb_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
        )
        self.attr_embed = torch.nn.Sequential(

            torch.nn.Linear(270, self.emb_dim),
            torch.nn.ReLU(),
            # torch.nn.Dropout(0.5),
            # torch.nn.GroupNorm(num_groups=4, num_channels=int(0.5 * emb_dim)),
            torch.nn.BatchNorm1d(self.emb_dim),
            torch.nn.Dropout(0.1),
        )

        self.node_embed = torch.nn.Sequential(
            torch.nn.Linear(vocab_size[1], self.emb_dim),
            # torch.nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
            # torch.nn.GroupNorm(num_groups=4, num_channels=emb_dim)
            torch.nn.BatchNorm1d(self.emb_dim),
            torch.nn.Dropout(0),

        )

        self.mol_embed = torch.nn.Sequential(
            torch.nn.Linear(2 * self.emb_dim, self.emb_dim),
            # torch.nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
            torch.nn.BatchNorm1d(self.emb_dim),
            # torch.nn.GroupNorm(num_groups=4, num_channels=emb_dim),
            # torch.nn.Linear(emb_dim, num_tasks)

            torch.nn.Dropout(self.drop_ratio),

        )

        # Different kind of graph pooling
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool

        # self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)
        self.graph_pred_linear = torch.nn.Sequential(
            # torch.nn.BatchNorm1d(emb_dim),
            # torch.nn.GroupNorm(num_groups=4, num_channels=emb_dim),
            torch.nn.Linear(self.emb_dim, self.emb_dim),
            nn.ReLU(),
            torch.nn.BatchNorm1d(self.emb_dim),
            torch.nn.Linear(self.emb_dim, num_tasks)
        )

    def forward(self, vocab_datas, batch_data):

        x, edge_index, edge_attr, inter_tensor, batch = batch_data.x, batch_data.edge_index, batch_data.edge_attr, \
                                                        batch_data.inter_tensor, batch_data.batch

        x = x.to('cpu')

        x_input_1 = vocab_datas.index_select(index=x.squeeze(), dim=0).to(self.device)


        edge_attr_0 = torch.cat([inter_tensor[:, 0, :], edge_attr[:, 0:2]], dim=1).to(torch.float)
        edge_attr_1 = torch.cat([inter_tensor[:, 1, :], edge_attr[:, 2:4]], dim=1).to(torch.float)
        edge_attr = torch.cat([edge_attr_0, edge_attr_1], dim=1)
        edge_attr_trans = torch.cat([edge_attr_1, edge_attr_0], dim=1)
        edge_attr = torch.cat([edge_attr, edge_attr_trans], dim=0)  # 增加反向边信息


        edge_index_trans = torch.stack([edge_index[1, :], edge_index[0, :]], dim=0)
        edge_index = torch.cat([edge_index, edge_index_trans], dim=1)  # 增加反向边
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        self_loop_attr = torch.zeros(x.size(0), 270)
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        x_tensor = self.node_embed(x_input_1)
        edge_attr = self.attr_embed(edge_attr)

        node_representation = self.graph_gnn(x_tensor, edge_index, edge_attr)
        gat_out = self.pool(node_representation, batch)

        mol_pred = self.graph_pred_linear(gat_out)


        return mol_pred, gat_out


class Predictor(nn.Module):
    def __init__(self, encoder, latent_dim=128, num_tasks=1):
        super().__init__()
        self.encoder = encoder
        self.emb_dim = latent_dim
        self.predictor = torch.nn.Sequential(
            # torch.nn.BatchNorm1d(emb_dim),
            # torch.nn.GroupNorm(num_groups=4, num_channels=emb_dim),
            torch.nn.Linear(self.emb_dim, self.emb_dim),
            nn.ReLU(),
            torch.nn.BatchNorm1d(self.emb_dim),
            torch.nn.Linear(self.emb_dim, num_tasks)
        )

    def forward(self, vocab_datas, batch_data1):
        _, emb1 = self.encoder(vocab_datas, batch_data1)
        out = self.predictor(emb1)

        return out

    def from_pretrained(self, model_path, device):
        pre_model = torch.load(model_path, map_location=device)['model_state_dict']
        # print(pre_model)
        self.encoder.load_state_dict(pre_model)


class DDIPredictor(nn.Module):
    def __init__(self, encoder, latent_dim=128, num_tasks=1):
        super().__init__()
        self.encoder = encoder
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim * 2),
            nn.ReLU(),
            # torch.nn.BatchNorm1d(latent_dim*2),
            nn.Linear(latent_dim * 2, num_tasks)
        )

    def forward(self, vocab_datas, batch_data1, batch_data2):
        _, emb1 = self.encoder(vocab_datas, batch_data1)
        _, emb2 = self.encoder(vocab_datas, batch_data2)
        emb = torch.cat((emb1, emb2), dim=-1)
        out = self.predictor(emb)

        return out

    def from_pretrained(self, model_path, device):
        pre_model = torch.load(model_path, map_location=device)['model_state_dict']
        # print(pre_model)
        self.encoder.load_state_dict(pre_model)


class SiameseNetwork(nn.Module):
    def __init__(self, encoder, latent_dim=300, dropout=0):
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4), nn.ReLU(), nn.Dropout(dropout), nn.Linear(latent_dim * 4, latent_dim)
        )

    def forward(self, vocab_datas, x):
        _, emb = self.encoder(vocab_datas, x)
        emb = self.projector(emb)
        return emb







