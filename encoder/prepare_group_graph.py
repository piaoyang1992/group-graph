#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/10/14 10:53
# @Author : piaoyang cao
# @File : prepare_group_graph.py


import os
import pandas as pd
import numpy
import numpy as np
import torch
import random

from fragment import *
# 处理每一个mol 形成大图
import networkx as nx
import itertools
import random

import scipy.sparse as sp

from functools import partial
from tqdm import tqdm
from multiprocessing import Pool
from ogb.utils.features import atom_to_feature_vector
import shutil
from get_data import remove_chirality
import time



def trans_adj_coo(adj):
    adj = sp.coo_matrix(adj)
    indices = np.vstack((adj.row, adj.col))  # 我们真正需要的coo形式  # PyG框架需要的coo形式
    return indices




class MolSplit(object):
    # 元素及其最大成键电子数
    def __init__(self, smiles):

        self.mol = get_mol(smiles)
        self.smiles = get_smiles(self.mol)

        # print(self.smiles)
        # self.mol_graph = self.build_mol_graph()
        self.cluster, self.atom_cls = find_clusters(self.mol)
        # print(self.cluster, self.atom_cls)
        self.inter_pair = get_inter(self.mol)
        self.max_inter_num = 4

        #self.vocab_mol_list = Vocab.get_vocab_mol()

    def get_inter_num(self, sub_mol, atoms_degree):
        inter_num_list = []

        for atom in sub_mol.GetAtoms():
            atom_idx = atom.GetIdx()

            atom_bond = {'SINGLE': 0, 'DOUBLE': 0, 'TRIPLE': 0}
            for i in atom.GetBonds():
                if i.GetBondType().name == 'SINGLE':
                    atom_bond['SINGLE'] = atom_bond['SINGLE'] + 1
                if i.GetBondType().name == 'DOUBLE':
                    atom_bond['DOUBLE'] = atom_bond['DOUBLE'] + 1
                if i.GetBondType().name == 'TRIPLE':
                    atom_bond['TRIPLE'] = atom_bond['TRIPLE'] + 1
            # 点上的label 保存原子符号 形式电荷 原子编号
            # print(atom.GetImplicitValence(), atom.GetTotalValence())

            inter_num = atoms_degree[atom_idx] - atom_bond['SINGLE'] - atom_bond['DOUBLE'] - atom_bond['TRIPLE']

            inter_num = inter_num if inter_num <= self.max_inter_num else self.max_inter_num
            # 节点的特征

            inter_num_list.append(inter_num)

        # inter_num_list = torch.IntTensor(inter_num_list)

        return inter_num_list

    def get_mol_graph(self):
        g1 = nx.MultiGraph()
        g1.graph['mol'] = self.mol
        mol = self.mol
        cluster, atom_cls = self.cluster, self.atom_cls

        cls = list(cluster.keys())

        inter_pair = self.inter_pair
        if len(inter_pair) == 0:
            #print( [ i for i in range(len(self.atom_cls))])
            g1.add_node(0, frag_idx = [i for i in range(len(self.atom_cls))],sub_smiles=self.smiles, sub_mol=mol)
        else:
            for i, c in enumerate(cls):
                sub_mol = get_sub_mol(mol, c)
                sub_smiles = get_smiles(sub_mol)
                g1.add_node(i, frag_idx=c, sub_smiles=sub_smiles, sub_mol=sub_mol)

                #frags_label[i] = {'sub_smiles': sub_smiles, 'frag_idx': tuple(c), 'sub_mol': sub_mol}
            for x, y in inter_pair:
                cls_x, cls_y = atom_cls[x],atom_cls[y]
                inter_pair = [cls_x, cls_y]
                if cls_x > cls_y:
                    inter_pair = [cls_y, cls_x]
                g1.add_edge(cls_x, cls_y,inter_pair=inter_pair)
        return g1


    # 作出整体的mol_graph, 每个节点是一个标记了inter的frag
    def get_mol_graph_data(self,vocab):
        # mol = get_mol(s)
        # 计算每个原子的总键个数
        self.vocab_list = vocab.get_vocab_list()
        mol = self.mol
        cluster, atom_cls = self.cluster, self.atom_cls

        atoms_degree = {}
        for atom in mol.GetAtoms():
            atoms_degree[atom.GetIdx()] = atom.GetTotalDegree()

        cls = list(cluster.keys())

        inter_pair = self.inter_pair
        # print(inter_pair)
        # 同一个frag不同的连接点拆分画全连接图
        inter_l = list(set(get_list(inter_pair)))

        # print(cluster, inter_pair)
        edge_index = []
        edge_attr = []
        inter_tensor = []

        frags_label = {}
        frag_tensor_list = []
        frag_id_list = []

        # 获取所有节点的信息
        # 如果该化合物没有连接点
        frag_idx_list = []

        if len(inter_l) == 0:

            inter_num_l = self.get_inter_num(self.mol, atoms_degree)
            if self.smiles not in self.vocab_list:
                self.smiles = self.vocab_list[-1]
                print(self.smiles, cls, 'frag_smiles not in vocab_list')

            frag_smiles_id = self.vocab_list.index(self.smiles)
            frag_id_list.append([frag_smiles_id])
            edge_index.append([0, 0])
            edge_attr.append([0, 0,0, 0])

            inter_tensor.append([[0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0]])

            # frag_tensor = vocab_tensor[self.smiles]
            # frag_tensor_list.append(frag_tensor)

            # inter_id = -1
            # inter_idx_list.append(inter_id)
            # frags_inter_id.append([self.mol.GetNumAtoms()/100, (inter_id+1)/self.mol.GetNumAtoms()])
            frags_label[0] = {'sub_smiles': self.smiles, 'frag_idx': cls[0]}
            # edge_index.append([0])
            # adj = sp.coo_matrix(edge_index)
            # edge_index_coo = np.vstack((adj.row, adj.col))
            new_c = [i + 1 for i in cls[0]]
            frag_idx_list.append(new_c)

        else:
            # 如果该节点有连接点
            for i, c in enumerate(cls):

                sub_mol = get_sub_mol(mol, c)
                sub_smiles = get_smiles(sub_mol)

                # frag_tensor = vocab_tensor[sub_smiles]
                # frag_tensor_list.append(frag_tensor)

                mol_idx = {a.GetIdx(): int(a.GetProp('mol_idx')) for a in sub_mol.GetAtoms()}
                atoms_degree_frag = {a.GetIdx(): atoms_degree[mol_idx[a.GetIdx()]] for a in
                                     sub_mol.GetAtoms()}  # frag中新原子
                inter_num_l = self.get_inter_num(sub_mol, atoms_degree_frag)

                if sub_smiles not in self.vocab_list:
                    print(self.smiles, sub_smiles, c, 'frag_smiles not in vocab_list')

                    sub_smiles = self.vocab_list[-1]

                frag_smiles_id = self.vocab_list.index(sub_smiles)

                frag_id_list.append([frag_smiles_id])

                frags_label[i] = {'sub_smiles': sub_smiles, 'frag_idx': tuple(c), 'sub_mol':sub_mol, 'vocab_id':frag_smiles_id}
                new_c =[i + 1 for i in c]
                frag_idx_list.append(list(new_c)) # 所有序号加1，避免与pad混淆
            #print(frags_label)
            for x, y in inter_pair:
                cls_x, cls_y = atom_cls[x],atom_cls[y]
                #sub_mol_x, sub_mol_y = get_sub_mol(mol, cls[atom_cls[x]]), get_sub_mol(mol, cls[atom_cls[y]])
                sub_mol_x, sub_mol_y = frags_label[cls_x]['sub_mol'], frags_label[cls_y]['sub_mol']
                #sub_smiles_x,sub_smiles_y = frags_label[atom_cls[x]]['sub_smiles'], frags_label[atom_cls[y]]['sub_smiles']
                inter_x = []
                inter_tensor_x = []
                for a in sub_mol_x.GetAtoms():
                    if int(a.GetProp('mol_idx')) == x:
                        inter_tensor_x.append(atom_to_feature_vector(a))
                        inter_x.append(a.GetIdx())
                if len(inter_x) == 0:
                    print(frags_label[cls_x]['sub_smiles'], 'do not match vocab_mol_list')
                    inter_x[0] = 0

                inter_y = []
                inter_tensor_y = []
                for a in sub_mol_y.GetAtoms():
                    if int(a.GetProp('mol_idx')) == y:
                        inter_tensor_y.append(atom_to_feature_vector(a))
                        inter_y.append(a.GetIdx())
                if len(inter_y) == 0:
                    print(frags_label[cls_y]['sub_smiles'], 'do not match vocab_mol_list')
                    inter_y[0] = 0


                #inter_x = [a.GetIdx() for a in sub_mol_x.GetAtoms() if int(a.GetProp('mol_idx')) == x]
                #inter_y = [a.GetIdx() for a in sub_mol_y.GetAtoms() if int(a.GetProp('mol_idx')) == y]
                #inter_tensor_x = [atom_to_feature_vector(a) for a in sub_mol_x.GetAtoms() if int(a.GetProp('mol_idx')) == x]
                #inter_tensor_y = [atom_to_feature_vector(a) for a in sub_mol_y.GetAtoms() if int(a.GetProp('mol_idx')) == y]

                #assert len(inter_x) == 1 and len(inter_y) == 1, print(self.smiles, 'inter error')
                if atom_cls[x] <= atom_cls[y]:
                    edge_index.append([atom_cls[x], atom_cls[y]])
                    edge_attr.append([sub_mol_x.GetNumAtoms(), inter_x[0],
                                       sub_mol_y.GetNumAtoms(), inter_y[0]])
                    inter_tensor.append([inter_tensor_x[0], inter_tensor_y[0]])
                else:
                    edge_index.append([atom_cls[y], atom_cls[x]])
                    edge_attr.append([sub_mol_y.GetNumAtoms(), inter_y[0],
                                      sub_mol_x.GetNumAtoms(), inter_x[0]])
                    inter_tensor.append([inter_tensor_y[0], inter_tensor_x[0]])

        # print(edge_index)

        edge_index = np.array(edge_index).T
        edge_index = edge_index.astype(int)

        edge_attr = np.array(edge_attr).astype(int)

        inter_tensor = np.array(inter_tensor).astype(int)

        frags_id = np.array(frag_id_list).astype(int)

        return frags_id, edge_index, edge_attr, inter_tensor, frag_idx_list


def update_vocab(vocab, smi_list):
    # vocab_df = pd.DataFrame(columns=['smiles','frequency'])
    vocab_dict = {}
    for smi in tqdm(smi_list):
        #if sanitize(smi) == None: continue  # 避免产生无意义的smiles
        mol = get_mol(smi)
        inter_pair = get_inter(mol)
        inter_l = list(set(get_list(inter_pair)))
        frags,atom_cls = find_clusters(mol)
        #frags_smiles = [get_smiles(get_sub_mol(mol, f)) for f in frags]
        #print(frags_smiles)
        # print(frags)
        #print(inter_pair)
        #print(frags)
        if len(inter_l) == 0 and len(frags) >= 1:
            # print(smi, frags, 'have no inter')
            if smi not in vocab and len(frags) == 1:
                # vocab_df = vocab_df.append({'smiles':smi, 'frequency':1},ignore_index=True)
                smi = get_smiles(mol)
                vocab_dict[smi] = 1
                vocab.add(smi)
            elif smi not in vocab and len(frags) > 1:  # 碎片之间不以共价键连接，无法找到两个cls之间的连接点
                print(smi, frags, 'split error')
                # vocab_df = vocab_df.append({'smiles': smi, 'frequency': 1}, ignore_index=True)
                smi = get_smiles(mol)
                vocab_dict[smi] = 1
                vocab.add(smi)
            elif smi in vocab:
                # freq = vocab_df[vocab_df['smiles'] == smi]['frequency']
                # vocab_df.loc[vocab_df['smiles'] == smi]['frequency'] = freq + 1
                vocab_dict[smi] = vocab_dict[smi] + 1
                # print(vocab_dict[smi])
            else:
                print('error')

        else:
            for frag in frags:
                frag_mol = get_sub_mol(mol, frag)
                # print(frag_mol)
                frag_smi = get_smiles(frag_mol)
                assert sanitize(frag_smi) != None, print(smi, frag_smi, frag, 'sanitize error')
                # s = get_frags(frag_mol)

                # assert len(s) < 2, print(frag_smi, frag, 'split error')
                if frag_smi not in vocab:
                    # vocab_df = vocab_df.append({'smiles': frag_smi, 'frequency': 1},ignore_index=True)

                    vocab_dict[frag_smi] = 1
                    vocab.add(frag_smi)
                elif frag_smi in vocab:
                    # freq = vocab_df[vocab_df['smiles'] == frag_smi]['frequency']
                    # vocab_df.loc[vocab_df['smiles'] == smi]['frequency'] = freq + 1
                    vocab_dict[frag_smi] = vocab_dict[frag_smi] + 1
                    # print(vocab_dict[frag_smi])
                else:
                    print('error')
        # print(vocab)
    return vocab_dict


def get_vocab(smiles_list, ncpu):
    vocab = set()
    vocab_df = pd.DataFrame()
    batch_size = len(smiles_list) // ncpu + 1
    batches = [smiles_list[i: i + batch_size] for i in range(0, len(smiles_list), batch_size)]
    pool = Pool(ncpu)
    func = partial(update_vocab, vocab)
    vocab_dict_list = pool.map(func, batches)
    # vocab_df = pd.concat(vocab_df,axis=0)
    pool.close()
    pool.join()
    # print(np_dict_list)
    for d in vocab_dict_list:
        for k, v in d.items():
            vocab_df = vocab_df.append({'smiles': k, 'frequency': v}, ignore_index=True)
            # print(vocab_df)

    vocab_df = vocab_df.groupby('smiles', as_index=False).sum()
    # print(vocab_df[vocab_df['smiles'] == 'C'])
    vocab_df = vocab_df.sort_values(by='frequency', ascending=False)
    vocab_df = vocab_df.reset_index(drop=True)

    vocab_df['frequency'] = vocab_df['frequency'].astype(int)

    return vocab_df





if __name__ == "__main__":
    root =  '/home/pycao/hgraph2graph-master/hgraph2graph-master/dataset/chembl'


    data_path = os.path.join(path, task + '.csv')
    data_df = pd.read_csv(data_path, header=0)

    #print(len(data_df))
    #data_df['smiles'] = data_df['smiles'].apply(
        #lambda x: standardize_smi(x, basicClean=True, clearCharge=True, clearFrag=True))

    #print(len(data_df))
    data_df.insert(data_df.shape[1], 'id',  list(range(0,len(data_df))))
    data_df = data_df.iloc[:,[2,1]]
    #data_df['id'] = list(range(0,len(data_df)))
    data_df.to_csv(path + '/bbbp.txt',sep='\t',index=False,header=True)

    l = len(data_df)
    data_df['smiles'] = data_df['smiles'].apply(
        lambda x: standardize_smi(x, basicClean=True, clearCharge=False, clearFrag=False))
    print(len(data_df))
    # data_df['smiles'] = data_df['smiles'].apply(lambda x: sanitize(x, kekulize=True, isomeric=False))

    data_df = data_df.replace(to_replace='None', value=np.nan).dropna(subset=['smiles'])
    # print(data_df)
    data_df.fillna(0.)
    print(task, l - len(data_df), ' smiles sanitize fail')

    data_df.to_csv(path + '/data_remove_chirality.csv')
    ncpu = 8
    vocab_df = get_vocab(data_df['smiles'], ncpu)
    vocab_df.to_csv(path + '/vocab.csv')







