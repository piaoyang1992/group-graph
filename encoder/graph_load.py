#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# @Time : 2022/9/26 15:41 
# @Author : piaoyang cao 
# @File : graph_load.py

import sys
import os
import os.path as osp
from functools import partial

from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.mol import smiles2graph
from ogb.utils.url import decide_download, download_url, extract_zip
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch_geometric.data import Data

from torch.nn.utils.rnn import pad_sequence

from prepare_group_graph import MolSplit
from sklearn.metrics import roc_auc_score, mean_squared_error, precision_recall_curve, auc, r2_score
import torch.nn.functional as F
from fragment import sanitize,compute_similarity,get_mol,get_smiles,group_graph_2_mol,csv_2_list,get_sub_mol
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import PandasTools, QED, rdMolDescriptors,Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import MACCSkeys
import random
import networkx as nx
from torch_geometric.loader import DataLoader
#import pickle

from functools import partial
import time
from collections import Counter,defaultdict
import copy
import itertools
#from multiprocessing import Pool

#import matplotlib.pyplot as plt

def get_vocab_descriptors(vocab_list):
    vocab_tensor = []
    for v in vocab_list:
        mol = get_mol(v)
        des_list = [x[0] for x in Descriptors._descList]
        # print(des_list)
        calculator = MoleculeDescriptors.MolecularDescriptorCalculator(des_list)
        descriptors = list(calculator.CalcDescriptors(mol))
        #print(len(descriptors))
        vocab_tensor.append(descriptors)

    '''
    vocab_tensor = np.matrix(vocab_tensor)
    vocab_tensor[np.isnan(vocab_tensor)] = 0
    # vocab_tensor = torch.nan_to_num(vocab_tensor)
    vocab_tensor = preprocessing.scale(vocab_tensor)
    fa = FactorAnalysis(n_components=10)
    X_transformed = fa.fit_transform(vocab_tensor)
    '''
    vocab_tensor = torch.tensor(vocab_tensor).to(torch.float32)
    vocab_tensor = torch.nan_to_num(vocab_tensor)
    return vocab_tensor

def get_vocab_mordred(vocab_list):
    vocab_tensor = []
    for v in vocab_list:
        mol = get_mol(v)
        calc = Calculator(descriptors, ignore_3D=True) # 1826
        des = calc(mol)
        value = list(des.values())
        value = [float(x) for x in value]
        vocab_tensor.append(value)

    vocab_tensor = torch.tensor(vocab_tensor).to(torch.float32)
    vocab_tensor = torch.nan_to_num(vocab_tensor)
    '''
    vocab_tensor = np.matrix(vocab_tensor)
    vocab_tensor[np.isnan(vocab_tensor)] = 0
    #vocab_tensor = torch.nan_to_num(vocab_tensor)
    vocab_tensor = preprocessing.scale(vocab_tensor)
    fa = FactorAnalysis(n_components=100)
    X_transformed = fa.fit_transform(vocab_tensor)

    vocab_tensor = torch.from_numpy(X_transformed)
    '''
    return vocab_tensor

def get_vocab_macc(vocab_list):
    vocab_tensor = []
    for v in vocab_list:
        mol = get_mol(v)
        value = MACCSkeys.GenMACCSKeys(mol) # 1826
        vocab_tensor.append(value)

    vocab_tensor = torch.tensor(vocab_tensor).to(torch.float32)
    vocab_tensor = torch.nan_to_num(vocab_tensor)
    '''
    vocab_tensor = np.matrix(vocab_tensor)
    vocab_tensor[np.isnan(vocab_tensor)] = 0
    #vocab_tensor = torch.nan_to_num(vocab_tensor)
    vocab_tensor = preprocessing.scale(vocab_tensor)
    fa = FactorAnalysis(n_components=100)
    X_transformed = fa.fit_transform(vocab_tensor)

    vocab_tensor = torch.from_numpy(X_transformed)
    '''
    return vocab_tensor





class Vocab():
    def __init__(self, save_vocab_file):
        #self.vocab_list = vocab_list
        self.vocab_file = os.path.join(save_vocab_file, 'vocab.csv')
        self.processed_file_path =  os.path.join(save_vocab_file, 'vocab_info.pt')
        self.process()
    def get_mol(self,frag_id):
        return self.vocab_mol_list[frag_id]
    def get_smiles(self,frag_id):
        return self.vocab_list[frag_id]
    def get_vocab_mol(self):
        return self.vocab_mol_list
    def get_vocab_list(self):
        return self.vocab_list
    def get_vocab_tensor(self):
        return self.vocab_tensor

    def process(self):
        if os.path.exists(self.processed_file_path):
            datas = torch.load(self.processed_file_path)
            self.vocab_list,self.vocab_mol_list = datas['vocab_list'], datas['vocab_mol_list']
        else:
            self.vocab_list = pd.read_csv(self.vocab_file,header=0)['smiles'].tolist()
            self.vocab_mol_list = [get_mol(v) for v in self.vocab_list]
            #self.vocab_tensor = get_vocab_descriptors(self.vocab_list)
            vocab_info = {'vocab_list':self.vocab_list, 'vocab_mol_list':self.vocab_mol_list}
            torch.save(vocab_info,self.processed_file_path)


def get_data(vocab, batches):
    frag_data_l = []
    raw_data_l = []
    #print(smiles)
    for i,batch in enumerate(tqdm(batches, desc="Iteration")):
        #print(batch[0],batch[1],batch[2])
        s, s_id, v = batch[0], batch[1], batch[2]
        #print(v,torch.tensor(v))
        if sanitize(s) == None:
            print(s, s_id, v )
        else:
            mol_split = MolSplit(batch[0])
            frags_id, edge_index, edge_attr, inter_tensor, frag_idx_list = mol_split.get_mol_graph_data(vocab)
            graph = smiles2graph(s)
            raw_x, raw_edge_index, raw_edge_attr = torch.from_numpy(graph['node_feat']), torch.from_numpy(graph['edge_index']),\
                                                    torch.from_numpy(graph['edge_feat'])

            #print(raw_x.size(), raw_edge_attr.size())
            #frag_tensor =[vocab_tensor[vocab_list[frag_id]] for frag_id in  frags_id_list]
            frags_id = torch.from_numpy(frags_id).to(torch.int64)
            edge_index = torch.from_numpy(edge_index).to(torch.int64)
            edge_attr = torch.from_numpy(edge_attr).to(torch.int64)
            inter_tensor = torch.from_numpy(inter_tensor).to(torch.int64)
            smiles_id = torch.from_numpy(np.array([s_id]).astype(int)).to(torch.int64)
            pad_frag = torch.zeros([len(frag_idx_list), 100], dtype=torch.int64)
            for i,x in enumerate(frag_idx_list):
                for j in range(len(x)):
                    pad_frag[i, j] = x[j]
            #inters_tensor = torch.cat(inter_tensor_list, dim = 0)

            raw_smiles_data = Data(x=raw_x, edge_index=raw_edge_index, edge_attr=raw_edge_attr, smiles_id=smiles_id)

            #v = 0

            raw_smiles_data.y = torch.tensor(v).to(torch.float32)

            smiles_data = Data(x=frags_id, edge_index=edge_index, edge_attr=edge_attr, inter_tensor=inter_tensor,
                               smiles_id=smiles_id, frag_raw_id= pad_frag)
            smiles_data.y = torch.tensor(v).to(torch.float32)
            #print(smiles_data.y)
            frag_data_l.append(smiles_data)
            raw_data_l.append(raw_smiles_data)



    return  frag_data_l,raw_data_l




class SmilesDataset():
    def __init__(self, vocab, root, data_type='frag_datas'):
        #self.vocab_list = vocab.get_vocab_list()
        self.vocab = vocab
        self.data_type = data_type
        self.original_root = root
        #self.folder = osp.join(root, 'processed')
        self.pre_processed_file_path = osp.join(root, 'smiles_data.pt')

        super(SmilesDataset, self).__init__()
        self.data_df = pd.read_csv(osp.join(self.original_root, 'data_remove_chirality.csv'))
        self.smiles_list = self.data_df['smiles'].tolist()
        self.process()

    def process(self):
        if osp.exists(self.pre_processed_file_path):
            datas = torch.load(self.pre_processed_file_path)
            # if pre-processed file already exists
            self.datas = datas
            self.frag_data_l,self.raw_data_l= datas['frag_datas'], datas['raw_datas']
        else:
            #print(self.original_root)
            #data_df = pd.read_csv(osp.join(self.original_root, 'data_remove_chirality.csv'))
            #data_df = data_df.iloc[:,1:]
            #data_df.to_csv(osp.join(self.original_root, 'data_remove_chirality.csv'), index=False)
            data_df = self.data_df
            smiles_col = list(data_df.columns.values).index('smiles')
            #print(smiles_col)
            smiles_list = self.smiles_list
            smiles_id_list = data_df.index.tolist()

            value_list = np.array(data_df.iloc[:,smiles_col+1:]).tolist()
            #print(value_list[0])

            batches_list = list(zip(smiles_list, smiles_id_list,value_list))

            print('Converting SMILES strings into graphs...')

            self.frag_data_l,self.raw_data_l = get_data(self.vocab, batches_list)
            self.datas = {'frag_datas':self.frag_data_l, 'raw_datas':self.raw_data_l}
            # double-check prediction target
            #self.datas = datas
            #print(len(self.frag_data_l))

            print('Saving...')

            torch.save(self.datas, self.pre_processed_file_path)


    def get_idx_split(self):
        self.split_file_path = self.original_root + '/split/scaffold'
        # short-cut if split_dict.pt exists
        if os.path.isfile(os.path.join(self.original_root, 'split_dict.pt')):
            split_dict = torch.load(os.path.join(self.original_root, 'split_dict.pt'))
        else:
            train_idx = pd.read_csv(osp.join(self.split_file_path, 'train.csv.gz'), compression='gzip', header=None).values.T[0]
            valid_idx = pd.read_csv(osp.join(self.split_file_path, 'valid.csv.gz'), compression='gzip', header=None).values.T[0]
            test_idx = pd.read_csv(osp.join(self.split_file_path, 'test.csv.gz'), compression='gzip', header=None).values.T[0]

            split_dict =  {'train': torch.tensor(train_idx, dtype=torch.long),
                    'valid': torch.tensor(valid_idx, dtype=torch.long),
                    'test': torch.tensor(test_idx, dtype=torch.long)}

        #split_dict = replace_numpy_with_torchtensor(torch.load(osp.join(self.original_root, 'split_dict.pt')))
        return split_dict

    def get_raw_datas(self):
        return self.raw_data_l

    def get_frags_datas(self):
        return self.frag_data_l
    def get_vocab_datas(self):
        return self.vocab_data_l

    def __getitem__(self, id):
            return self.datas[self.data_type][id]

    def __len__(self):
        return len(self.datas[self.data_type])

    def __repr__(self):  # pragma: no cover
        return '{}({})'.format(self.__class__.__name__, len(self))




class Meter(object):
    """Track and summarize model performance on a dataset for
    (multi-label) binary classification."""
    def __init__(self):
        self.mask = []
        self.y_pred = []
        self.y_true = []
        self.mask = []

    def update(self, y_pred, y_true, mask):
        """Update for the result of an iteration
        Parameters
        ----------
        y_pred : float32 tensor
            Predicted molecule labels with shape (B, T),
            B for batch size and T for the number of tasks
        y_true : float32 tensor
            Ground truth molecule labels with shape (B, T)
        mask : float32 tensor
            Mask for indicating the existence of ground
            truth labels with shape (B, T)
        """
        self.y_pred.append(y_pred.detach().cpu())
        self.y_true.append(y_true.detach().cpu())

        if mask is None:
            self.mask.append(torch.ones_like(y_pred.detach().cpu()))
        else:
            self.mask.append(mask.detach().cpu())

    def roc_auc_score(self):
        """Compute roc-auc score for each task.
        Returns
        -------
        list of float
            roc-auc score for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        # Todo: support categorical classes
        # This assumes binary case only
        y_pred = torch.sigmoid(y_pred)
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(round(roc_auc_score(task_y_true, task_y_pred), 4))
        return scores

    def return_pred_true(self):
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        # Todo: support categorical classes
        return y_pred, y_true

    def l1_loss(self, reduction):
        """Compute l1 loss for each task.
        Returns
        -------
        list of float
            l1 loss for all tasks
        reduction : str
            * 'mean': average the metric over all labeled data points for each task
            * 'sum': sum the metric over all labeled data points for each task
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(F.l1_loss(task_y_true, task_y_pred, reduction=reduction).item())
        return scores

    def rmse(self):
        """Compute RMSE for each task.
        Returns
        -------
        list of float
            rmse for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(np.sqrt(F.mse_loss(task_y_pred, task_y_true).cpu().item()))
        return scores

    def mae(self):
        """Compute MAE for each task.
        Returns
        -------
        list of float
            mae for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(mean_squared_error(task_y_true, task_y_pred))
        return scores

    def r2(self):
        """Compute R2 for each task.
        Returns
        -------
        list of float
            r2 for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(round(r2_score(task_y_true, task_y_pred), 4))
        return scores

    def roc_precision_recall_score(self):
        """Compute AUC_PRC for each task.
        Returns
        -------
        list of float
            AUC_PRC for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        # Todo: support categorical classes
        # This assumes binary case only
        y_pred = torch.sigmoid(y_pred)
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            precision, recall, _thresholds = precision_recall_curve(task_y_true, task_y_pred)
            scores.append(auc(recall, precision))
        return scores



    def compute_metric(self, metric_name, reduction='mean'):
        """Compute metric for each task.
        Parameters
        ----------
        metric_name : str
            Name for the metric to compute.
        reduction : str
            Only comes into effect when the metric_name is l1_loss.
            * 'mean': average the metric over all labeled data points for each task
            * 'sum': sum the metric over all labeled data points for each task
        Returns
        -------
        list of float
            Metric value for each task
        """
        assert metric_name in ['roc_auc', 'l1', 'rmse', 'mae', 'roc_prc', 'r2', 'return_pred_true'], \
            'Expect metric name to be "roc_auc", "l1" or "rmse", "mae", "roc_prc", "r2", "return_pred_true", got {}'.format(metric_name)
        assert reduction in ['mean', 'sum']
        if metric_name == 'roc_auc':
            return self.roc_auc_score()
        if metric_name == 'l1':
            return self.l1_loss(reduction)
        if metric_name == 'rmse':
            return self.rmse()
        if metric_name == 'mae':
            return self.mae()
        if metric_name == 'roc_prc':
            return self.roc_precision_recall_score()
        if metric_name == 'r2':
            return self.r2()
        if metric_name == 'return_pred_true':
            return self.return_pred_true()





def dict_diff(dict_0,dict_1):
    diff_node = {}
    for x, y in dict_0.items():
        if x in dict_1 and dict_1[x] != y:
            diff_node[x] = abs(dict_1[x] - y)
        elif x not in dict_1:
            diff_node[x] = y
    for x,y in dict_1.items():
        if x not in dict_0:
            diff_node[x] = y

    return diff_node

def dict_diff_1(x0,inter):
    diff_node = {}
    #print(inter)
    for x, y in x0.items():
        if y in inter and y not in diff_node:
            ai = np.array(inter[y])
            #print(ai)
            a0 = list(set(ai[:,0].tolist()))
            a1 = list(set(ai[:,1].tolist()))
            if len(a0) > len(a1):
                p = (len(a0)-len(a1))/ len(a0)
                diff_node[y] =  [p if i in a0 else 0 for i in range(len(x0))]
        else:
            diff_node[y] = [1 if i == x else 0 for i in range(len(x0))]


    return diff_node


def dict_inter(dict_0,dict_1):
    diff_node = {}
    for x, y in dict_0.items():
        if x in dict_1:
            diff_node[x] = min([dict_1[x],y])
    return diff_node

def dict_inter_1(x0,x1):
    diff_node = {}
    #print(x0,x1)
    for x, y in x0.items():
        for a, b in x1.items():
            if y == b:
                if y in diff_node:
                    #print(diff_node)
                    diff_node[y] = diff_node[y] + [[x,a]]
                else:
                    diff_node[y] = [[x,a]]
    return diff_node



def dict_len(d):
    l=sum([sum(x) for i,x in d.items()])

    return l

def compare_graphs(data_df,dataset,vocab):
    smiles_list = data_df['smiles'].tolist()
    res_df = pd.DataFrame()
    print(smiles_list)
    for m in [0,1]:
        s_i = smiles_list[m]
        d0 = dataset[m]
        x0 = {k: v for k, v in enumerate(d0.x.squeeze().tolist())}
        #x0 = d0.x.squeeze().tolist()
        #x0 = [x0] if type(x0) == int else x0
        e0_list = {k:(x0[v[0]], x0[v[1]])for k, v in enumerate(d0.edge_index.T.tolist())}
        i0_list = d0.edge_attr[:, [1, 3]].tolist()
        for n in [2,3]:
        #for j in range(len(smiles_list)-1,i,-1):
            print(m,n)
            s_j = smiles_list[n]
            d1 = dataset[n]
            x1 = {k:v for k,v in enumerate(d1.x.squeeze().tolist())}
            #x1 = [x1] if type(x1) == int else x1
            e1_list = {k:(x1[v[0]], x1[v[1]]) for k, v in enumerate(d1.edge_index.T.tolist())}
            #i1_list = d1.edge_attr[:, [1, 3]].tolist()
            smiles_id_0,smiles_id_1 = m, n
            # 只比较节点数相同的图
            #d = []
            #label_0 = [attr['sub_smiles'] for n,attr in g0.nodes(data=True)]
            #dict_0 = dict(Counter(x0))
            #label_1 = [attr['sub_smiles'] for n,attr in g1.nodes(data=True)]
            #dict_1 = dict(Counter(x1))
            #print(dict(Counter(x0)),dict(Counter(x1)))
            inter_node = dict_inter_1(x0,x1)
            print(inter_node)
            #inter_node = dict_diff(Counter(x0),diff_node)

            diff_node_0 = dict_diff_1(x0,inter_node)
            diff_node_1 = dict_diff_1(x1,inter_node)
            print(diff_node_0,diff_node_1)

            #mw = Descriptors.MolWt(get_mol(s_i)) + Descriptors.MolWt(get_mol(s_j))

            inter_edge = dict_inter_1(e0_list,e1_list)
            diff_edge_0 = dict_diff_1(e0_list,inter_edge)
            diff_edge_1 = dict_diff_1(e1_list,inter_edge)

            #diff = set(x0).symmetric_difference(set(x1))
            if len(x0) == len(x1)  and dict_len(diff_node_0) == dict_len(diff_node_1) <=1 and \
                    abs(dict_len(diff_edge_0) - dict_len(diff_edge_1)) <=4:

                #constant_node_sum = sum([vocab.get_mol(x).GetNumHeavyAtoms() * len(n) for x, n in inter_node.items()])
                diff_node_0_sum = sum([vocab.get_mol(x).GetNumHeavyAtoms() * sum(n) for x, n in diff_node_0.items()])
                diff_node_1_sum = sum([vocab.get_mol(x).GetNumHeavyAtoms() * sum(n) for x, n in diff_node_1.items()])
                diff_0 = diff_node_0_sum/ get_mol(s_i).GetNumHeavyAtoms()
                diff_1 = diff_node_1_sum/ get_mol(s_j).GetNumHeavyAtoms()

                if diff_0 < 0.5 and diff_1 < 0.5 and  -6<=diff_node_0_sum - diff_node_1_sum <= 6:
                    #fps_i,fps_j = MACCSkeys.GenMACCSKeys(get_mol(s_i)), MACCSkeys.GenMACCSKeys(get_mol(s_j))
                    #sim =  DataStructs.FingerprintSimilarity(fps_i,fps_j)
                    '''
                    for dist in nx.algorithms.similarity.optimize_graph_edit_distance(g0, g1,
                                                                                      node_match=lambda a, b: a['label'] == b['label'],
                                                                                      edge_match=None):
                                                                                      #edge_match=lambda a, b: a['inter'] == b['inter']):
    
                        d.append(dist)
                    
                    min_d = min(d)
                    '''
                    node_0 = []
                    for x,y in diff_node_0.items():
                        for i,a in enumerate(y):
                            if a == 1:
                                node_0.append(i)

                    node_label_0 =  [vocab.get_smiles(x0[a]) for a in node_0]

                    node_1 = []
                    for x,y in diff_node_1.items():
                        for i,a in enumerate(y):
                            if a == 1:
                                node_1.append(i)

                    node_label_1 = [vocab.get_smiles(x1[a]) for a in node_1]
                    #diff_node_0 = [ for x,y in diff_node_0.items()]
                    #diff_node_1 = [(vocab.get_smiles(x),y) for x,y in diff_node_1.items()]
                    #print(diff_node_0)
                    res_df = res_df.append({'smiles_0':data_df.loc[smiles_id_0,'smiles'],
                                            'id_0':int(smiles_id_0),
                                            #'label_0':data_df.loc[smiles_id_0,'p_np'],
                                            'smiles_1':data_df.loc[smiles_id_1,'smiles'],
                                            'id_1':int(smiles_id_1),
                                            #'label_1': data_df.loc[smiles_id_1,'p_np'],
                                            'diff_node_num_0':node_0,
                                            'diff_node_label_0': node_label_0,
                                            'diff_node_num_1': node_1,
                                            'diff_node_label_1': node_label_1,
                                            #'similarity':sim,
                                            #'constant_num':constant_node_sum,
                                            },ignore_index=True)
            # diff = set(x0).symmetric_difference(set(x1))
            if abs(len(x0) - len(x1)) > 0 and (inter_node == dict(Counter(x0)) or inter_node == dict(Counter(x1))) and \
                    (inter_edge == dict(Counter(e0_list)) or inter_edge == dict(Counter(e1_list))):
                #constant_node_sum = sum([vocab.get_mol(x).GetNumHeavyAtoms() * sum(n) for x, n in inter_node.items()])
                diff_node_0_sum = sum([vocab.get_mol(x).GetNumHeavyAtoms() * sum(n) for x, n in diff_node_0.items()])
                diff_node_1_sum = sum([vocab.get_mol(x).GetNumHeavyAtoms() * sum(n) for x, n in diff_node_1.items()])
                diff_0 = diff_node_0_sum / get_mol(s_i).GetNumHeavyAtoms()
                diff_1 = diff_node_1_sum / get_mol(s_j).GetNumHeavyAtoms()

                if diff_0 < 0.5 and diff_1 < 0.5 and -6 <= diff_node_0_sum - diff_node_1_sum <= 6:
                    #fps_i,fps_j = MACCSkeys.GenMACCSKeys(get_mol(s_i)), MACCSkeys.GenMACCSKeys(get_mol(s_j))
                    #sim =  DataStructs.FingerprintSimilarity(fps_i,fps_j)
                    '''
                    for dist in nx.algorithms.similarity.optimize_graph_edit_distance(g0, g1,
                                                                                      node_match=lambda a, b: a['label'] == b['label'],
                                                                                      edge_match=None):
                                                                                      #edge_match=lambda a, b: a['inter'] == b['inter']):

                        d.append(dist)

                    min_d = min(d)
                    '''
                    #diff_node_0 = [(vocab.get_smiles(x), y) for x, y in diff_node_0.items()]
                    #diff_node_1 = [(vocab.get_smiles(x), y) for x, y in diff_node_1.items()]
                    #print(diff_node_0)
                    res_df = res_df.append({'smiles_0': data_df.loc[smiles_id_0, 'smiles'],
                                            'id_0': int(smiles_id_0),
                                            'label_0': data_df.iloc[smiles_id_0, -1],
                                            'smiles_1': data_df.loc[smiles_id_1, 'smiles'],
                                            'id_1': int(smiles_id_1),
                                            'label_1': data_df.iloc[smiles_id_1, -1],
                                            'diff_node_0': diff_node_0,
                                            'diff_node_1': diff_node_1,
                                            #'constant_num':constant_node_sum,
                                            #'similarity':sim
                                            }, ignore_index=True)


    return res_df




def mmpdb_deal(path):
    constant_smiles = pd.read_csv(os.path.join(path, 'constant_smiles.csv'), header=0)
    comp = pd.read_csv(os.path.join(path, 'mmpdb_compound_1.csv'), header=0)
    mmpdb_pair = pd.read_csv(os.path.join(path, 'mmpdb.csv'), header=0)

    mmpdb_pair['id_0'] = mmpdb_pair['compound1_id'].apply(lambda x: comp.loc[x, 'public_id'])
    mmpdb_pair['id_1'] = mmpdb_pair['compound2_id'].apply(lambda x: comp.loc[x, 'public_id'])
    mmpdb_pair['smiles_0'] = mmpdb_pair['id_0'].apply(lambda x: smiles_list[x])
    mmpdb_pair['smiles_1'] = mmpdb_pair['id_1'].apply(lambda x: smiles_list[x])
    mmpdb_pair['constant_smiles'] = mmpdb_pair['constant_id'].apply(lambda x: constant_smiles.loc[x, 'smiles'])
    mmpdb_pair['constant_num'] = mmpdb_pair['constant_smiles'].apply(lambda x: get_mol(x).GetNumHeavyAtoms())
    print(len(mmpdb_pair))
    mmpdb_pair.drop_duplicates(subset=['id_0', 'id_1'], keep='first', inplace=True)
    mmpdb_pair.reset_index(drop=True, inplace=True)
    drop_list = []
    print(len(mmpdb_pair))
    for i in range(len(mmpdb_pair)):
        for j in range(len(mmpdb_pair) - 1, i, -1):
            if mmpdb_pair.loc[i, 'id_0'] == mmpdb_pair.loc[j, 'id_1'] and mmpdb_pair.loc[i, 'id_1'] == mmpdb_pair.loc[
                j, 'id_0']:
                if mmpdb_pair.loc[i, 'id_0'] < mmpdb_pair.loc[j, 'id_0']:
                    drop_list.append(j)
                else:
                    drop_list.append(i)
    mmpdb_pair = mmpdb_pair.drop(mmpdb_pair.index[drop_list])

    mmpdb_pair.reset_index(drop=True, inplace=True)
    print(len(mmpdb_pair))
    sim_list = []
    for i in range(len(mmpdb_pair)):
        smi0, smi1 = mmpdb_pair.loc[i, 'smiles_0'], mmpdb_pair.loc[i, 'smiles_1']
        fps_i, fps_j = MACCSkeys.GenMACCSKeys(get_mol(smi0)), MACCSkeys.GenMACCSKeys(get_mol(smi1))
        sim = DataStructs.FingerprintSimilarity(fps_i, fps_j)
        sim_list.append(sim)

    mmpdb_pair['similarity'] = sim_list

    mmpdb_pair.to_csv(path + '/mmpdb_pair.csv', index=False)



def get_network(loader,vocab):
    node_df = pd.DataFrame(columns=['smiles_id','frag_id'])
    edge_df = pd.DataFrame(columns=['Source','Target'])
    with torch.no_grad():
        for step, batch in enumerate(tqdm(loader, desc="Iteration")):
            smiles_id = batch.smiles_id.index_select(index=batch.batch,dim=0).tolist()
            source = batch.edge_index[0,].squeeze().tolist()
            target = batch.edge_index[1,].squeeze().tolist()
            frag_id = batch.x.squeeze().tolist()
            edge_attr_list = batch.edge_attr[:, [1, 3]].tolist()

    inter_list = []
    for i, e in enumerate(batch.edge_index.T.tolist()):
        #e = sorted([x_list[e[0]],x_list[e[1]]])
        u,v = frag_id[e[0]],frag_id[e[1]]
        inter_u, inter_v = edge_attr_list[i][0], edge_attr_list[i][1]
        if u <= v:
            inter=str(inter_u)+'_'+ str(inter_v)
        else:
            inter=str(inter_v)+'_'+ str(inter_u)
        inter_list.append(inter)
    node_df['smiles_id'] = smiles_id
    node_df['frag_id'] = frag_id
    node_df['label'] = node_df['frag_id'].apply(lambda x: vocab.get_smiles(int(x)))
    edge_df['Source'] = source
    edge_df['Target'] = target
    edge_df['label'] = inter_list

    #print(data_df)

    #result = {'auc':auc, 'accuracy':accuracy, 'se':se, 'sp':sp, 'f1':f1, 'pre':pre, 'rec':rec, 'err':err, 'mcc':mcc}
    return node_df,edge_df


if __name__ == '__main__':

    #task_name_list = ['hiv', 'tox21', 'toxcast', 'sider', 'clintox','muv']
    #task_name_list = ['freesolv', 'esol', 'lipo', 'qm7', 'qm8', 'qm9']
    #task_name_list = ['Pgp-sub', 'HIA', 'F(20%)', 'F(30%)', 'FDAMDD', 'CYP1A2-sub', 'CYP2C19-sub', 'CYP2C9-sub',
                      #'CYP2D6-sub', 'CYP3A4-sub', 'T12', 'DILI', 'SkinSen', 'Carcinogenicity', 'Respiratory']
    root = '/home/pycao/hgraph2graph-master/hgraph2graph-master/dataset/dataset/class_task'
    smiles_list = ['[O-]C(=O)CCC(=O)C([O-])=O','C(C(C(=O)[O-])[NH3+])C(=O)[O-]','[NH3+]C(CCC([O-])=O)C([O-])=O','[O-]C(=O)CC(=O)C([O-])=O']
    task = pd.DataFrame(columns=['id', 'smiles', 'label'])
    task['smiles'] = smiles_list
    task['id'] = [0,1,2,3]
    task['label'] = [1, 1, -1, -1]
    path = os.path.join(root, 'bbbp')
    vocab = Vocab(path)
    frag_data_l, raw_data_l = get_data(vocab, list(zip(smiles_list, [0, 1,2, 3], [1, 1, 1, 1])))
    loader = DataLoader(frag_data_l, batch_size=len(frag_data_l), shuffle=False)
    node_df,edge_df = get_network(loader,vocab)
    node_df.to_csv(os.path.join(path, 'transamination_node.csv'), index=False)
    print(node_df)
    edge_df.to_csv(os.path.join(path, 'transamination_edge.csv'), index=False)
    mmp = compare_graphs(task, frag_data_l, vocab)
    mmp.to_csv(path + '/transamination_mmp.csv', sep=',', index=False, header=True)
    print(mmp)

