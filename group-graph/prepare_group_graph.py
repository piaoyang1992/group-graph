#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/10/14 10:53
# @Author : piaoyang cao
# @File : prepare_mol_data.py


import os
import pandas as pd
import numpy
import numpy as np
import torch
import random

from molecule_fragment import *
# 处理每一个mol 形成大图
import networkx as nx
import itertools
import random

import scipy.sparse as sp
import time
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')


def trans_adj_coo(adj):
    adj = sp.coo_matrix(adj)
    indices = np.vstack((adj.row, adj.col))  # 我们真正需要的coo形式  # PyG框架需要的coo形式
    return indices


MAX_ATOMIC_NUM = 100
ATOM_FEATURES = {
    'atomic_num': list(range(MAX_ATOMIC_NUM)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

# Distance feature sizes
PATH_DISTANCE_BINS = list(range(10))
THREE_D_DISTANCE_MAX = 20
THREE_D_DISTANCE_STEP = 1
THREE_D_DISTANCE_BINS = list(range(0, THREE_D_DISTANCE_MAX + 1, THREE_D_DISTANCE_STEP))

# len(choices) + 1 to include room for uncommon values; + 2 at end for IsAromatic and mass
ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2
BOND_FDIM = 14


def onek_encoding_unk(value, choices):
    """
    Creates a one-hot encoding.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the value in a list of length len(choices) + 1.
    If value is not in the list of choices, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def atom_features(atom: Chem.rdchem.Atom, functional_groups=None):
    """
    Builds a feature vector for an atom.

    :param atom: An RDKit atom.
    :param functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
    :return: A list containing the atom features.
    """
    features = onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
               onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
               onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
               onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
               onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
               onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
               [1 if atom.GetIsAromatic() else 0] + \
               [atom.GetMass() * 0.01]  # scaled to about the same range as other features
    if functional_groups is not None:
        features += functional_groups
    return features


def bond_features(bond: Chem.rdchem.Bond):
    """
    Builds a feature vector for a bond.

    :param bond: A RDKit bond.
    :return: A list containing the bond features.
    """
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return fbond


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

        # self.vocab_mol_list = Vocab.get_vocab_mol()

    def get_mol_graph(self):
        g1 = nx.MultiGraph()
        g1.graph['mol'] = self.mol
        g1.graph['smiles'] = self.smiles
        mol = self.mol

        cluster, atom_cls = self.cluster, self.atom_cls
        g1.graph['atom_cls'] = atom_cls
        cls = list(cluster.keys())

        inter_pair = self.inter_pair
        Chem.Kekulize(self.mol)
        vocab_dict = {}
        if len(inter_pair) == 0 or len(cls) == 1:
            # print(self.smiles,'no inter',cls, get_smiles(self.mol))

            # print( [ i for i in range(len(self.atom_cls))])

            g1.add_node(0, frag_idx=[i + 1 for i in range(len(self.atom_cls))], sub_smiles=self.smiles,
                        sub_mol=self.mol)
            if self.smiles not in vocab_dict:
                vocab_dict[self.smiles] = 1
            else:
                vocab_dict[self.smiles] = vocab_dict[self.smiles] + 1
        else:
            for i, c in enumerate(cls):
                new_c = [idx + 1 for idx in c]
                sub_mol = get_sub_mol(mol, c)
                sub_smiles = get_smiles(sub_mol, isomeric=False)

                if get_mol(sub_smiles) != None:
                    g1.add_node(i, frag_idx=new_c, sub_smiles=sub_smiles, sub_mol=sub_mol)
                    if sub_smiles not in vocab_dict:
                        vocab_dict[sub_smiles] = 1
                    else:
                        vocab_dict[sub_smiles] = vocab_dict[sub_smiles] + 1
                else:
                    print(self.smiles, sub_smiles, c)
                    print(cls)

            for x, y in inter_pair:
                cls_x, cls_y = atom_cls[x], atom_cls[y]
                sub_mol_x, sub_mol_y = g1.nodes[cls_x]['sub_mol'], g1.nodes[cls_y]['sub_mol']
                # sub_smiles_x,sub_smiles_y = frags_label[atom_cls[x]]['sub_smiles'], frags_label[atom_cls[y]]['sub_smiles']
                inter_x = []
                inter_tensor_x = []
                for a in sub_mol_x.GetAtoms():
                    if int(a.GetProp('mol_idx')) == x:
                        inter_tensor_x.append(atom_features(a))
                        inter_x.append(a.GetIdx())
                if len(inter_x) == 0:
                    print(g1.nodes[cls_x]['sub_smiles'], 'do not match vocab_mol_list')
                    inter_x[0] = 0

                inter_y = []
                inter_tensor_y = []
                for a in sub_mol_y.GetAtoms():
                    if int(a.GetProp('mol_idx')) == y:
                        inter_tensor_y.append(atom_features(a))
                        inter_y.append(a.GetIdx())
                if len(inter_y) == 0:
                    print(g1.nodes[cls_x]['sub_smiles'], 'do not match vocab_mol_list')
                    inter_y[0] = 0

                if cls_x <= cls_y:
                    edge_attr = [sub_mol_x.GetNumAtoms(), inter_x[0],
                                 sub_mol_y.GetNumAtoms(), inter_y[0]]

                    inter_tensor = [inter_tensor_x[0], inter_tensor_y[0]]
                    g1.add_edge(cls_x, cls_y, edge_attr=edge_attr, inter_tensor=inter_tensor)

                else:
                    edge_attr = [sub_mol_y.GetNumAtoms(), inter_y[0],
                                 sub_mol_x.GetNumAtoms(), inter_x[0]]
                    inter_tensor = [inter_tensor_y[0], inter_tensor_x[0]]

                    g1.add_edge(cls_y, cls_x, edge_attr=edge_attr, inter_tensor=inter_tensor)

        g1.graph['vocab_dict'] = vocab_dict
        return g1



def update_vocab(vocab_dict, df):
    vocab = vocab_dict
    smiles_col = list(df.columns.values).index('smiles')
    # value_list = np.array(df.iloc[:,smiles_col+1:]).tolist()
    g_dict = {}
    for i in tqdm(range(len(df))):
        #s_list = df.iloc[i, smiles_col].split('*')
        s_list = [df.iloc[i, smiles_col]]
        for s in s_list:
            if get_mol(s) != None:
                mol_split = MolSplit(s)
                smiles_g = mol_split.get_mol_graph()
                smiles_g.graph['smiles'] = s
                smiles_g.graph['label'] = df.iloc[i, smiles_col + 1:].values
                #smiles_g.graph['label'] = df.loc[i, 'output']
                smiles_g.graph['smiles_id'] = i
                # print(i,smiles_g.graph['smiles'], smiles_g.graph['label'])
                # smiles_g.graph['smiles_id'] = df.loc[i,'smiles_id']
                # print(smiles_g.nodes(data=True))
                g_dict[s] = smiles_g
                vocab_s = smiles_g.graph['vocab_dict']
                for k in vocab_s.keys():
                    if k not in vocab:
                        vocab[k] = 1
                    else:
                        vocab[k] = vocab[k] + 1
            else:
                print(s, 'error')
                g_dict[s] = None
    return g_dict, vocab


def get_vocab(root, task, ncpu=8):
    data_path = os.path.join(root, task)
    # df = pd.read_csv(os.path.join(root,'raw.csv'))
    df = pd.read_csv(os.path.join(data_path, 'raw.csv'))

    # df['smiles_id'] = df.index.tolist()
    df = df.fillna(0)
    smiles_list = df['smiles'].tolist()
    print(len(smiles_list))
    vocab_dict = {}
    vocab_df = pd.DataFrame()
    g_list, vocab_dict = update_vocab(vocab_dict, df)
    for k, v in vocab_dict.items():
        vocab_df = vocab_df.append({'smiles': k, 'frequency': v}, ignore_index=True)

    # print(g_list[40].graph['vocab_dict'], g_list[40].graph['label'])
    vocab_df = vocab_df.groupby('smiles', as_index=False).sum()
    # print(vocab_df[vocab_df['smiles'] == 'C'])
    vocab_df = vocab_df.sort_values(by='frequency', ascending=False)
    vocab_df = vocab_df.reset_index(drop=True)

    vocab_df['frequency'] = vocab_df['frequency'].astype(int)
    torch.save(g_list, os.path.join(path, 'group_graph.pt'))
    vocab_df.to_csv(os.path.join(path, 'vocab.csv'))
    print(len(g_list))
    return g_list, vocab_df




if __name__ == "__main__":

    #s = 'C[C@@H](c1cc(O)ccc1O)N(C)C'
    s = 'C[C@]1(O)CC[C@H]2[C@@H]3CCC4=CC(=O)CC[C@]4(C)[C@H]3CC[C@]12C'
    print(s)
    ms = MolSplit(s)
    g = ms.get_mol_graph()
    print(g.graph['vocab_dict'])
    print(g.nodes(data=True))

    root = '/home/pycao/group-graph/dataset/class_task'
    # root = 'J:/hgraph2graph-master/hgraph2graph-master/dataset/chembl'

    task_name_list = ['bbbp']

    # root = '/home/pycao/GraphRXN-master/GraphRXN-master/data_scaler/Buchwald-Hartwig/random_split'
    for task in task_name_list:
        path = os.path.join(root, task)
        # path = root
        time_start = time.time()
        data_path = os.path.join(path, task + '.csv')
        data_df = pd.read_csv(data_path, header=0)

        l = len(data_df)
        #data_df['smiles'] = data_df['smiles'].apply(lambda x: sanitize(x))

        data_df = data_df.replace(to_replace='None', value=np.nan).dropna(subset=['smiles'])
        data_df.fillna(0.)

        data_df.to_csv(path + '/raw.csv', index=False)
        # print(data_df['smiles'].tolist())
        # smiles_list = data_df['smiles'].tolist()
        print(task, l - len(data_df), ' smiles sanitize fail')

        g_list, vocab_df = get_vocab(root, task, ncpu=8)
        # vocab_df.to_csv(path + '/vocab.csv')
        # pd.set_option('display.max_columns', None)
        print(vocab_df)




