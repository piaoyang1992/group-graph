#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/26 15:41
# @Author : piaoyang cao
# @File : pcqm4mv2_mol_split.py


import os
import os.path as osp

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch_geometric.data import Data
from sklearn import metrics

from torch.nn.utils.rnn import pad_sequence
from functools import partial


from molecule_fragment import sanitize, compute_similarity, get_mol, get_smiles, csv_2_list, get_sub_mol

from rdkit import Chem
from rdkit.Chem import PandasTools, QED, rdMolDescriptors, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import MACCSkeys, AllChem
import random
import networkx as nx
from torch_geometric.loader import DataLoader
# import pickle
from mordred import Calculator, descriptors
from functools import partial
import time
from collections import Counter, defaultdict
import copy
import itertools
from sklearn import preprocessing
from sklearn.decomposition import FactorAnalysis, PCA
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.AtomPairs import Torsions
# import matplotlib.pyplot as plt

from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from drfp import DrfpEncoder



def get_vocab_descriptors(vocab_list, n_components=10):
    vocab_tensor = []
    for v in vocab_list:
        mol = Chem.MolFromSmiles(v)
        # fp = Torsions.GetTopologicalTorsionFingerprintAsIntVect(mol)

        des_list = [x[0] for x in Descriptors._descList]
        # print(des_list)
        calculator = MoleculeDescriptors.MolecularDescriptorCalculator(des_list)
        fp = list(calculator.CalcDescriptors(mol))
        # print(len(descriptors))

        vocab_tensor.append(fp)

    vocab_tensor = np.matrix(vocab_tensor)
    vocab_tensor = np.nan_to_num(vocab_tensor)

    minmax_scaler = preprocessing.MinMaxScaler()
    vocab_tensor = minmax_scaler.fit_transform(vocab_tensor)


    vocab_tensor = torch.from_numpy(vocab_tensor).to(torch.float32)
    # vocab_tensor = torch.nan_to_num(vocab_tensor)
    return vocab_tensor


def get_vocab_mordred(vocab_list):
    vocab_tensor = []
    for v in vocab_list:
        mol = get_mol(v)
        calc = Calculator(descriptors, ignore_3D=True)  # 1826
        des = calc(mol)
        value = list(des.values())
        value = [float(x) for x in value]
        vocab_tensor.append(value)

    vocab_tensor = np.matrix(vocab_tensor)
    vocab_tensor = np.nan_to_num(vocab_tensor)
    minmax_scaler = preprocessing.MinMaxScaler()
    vocab_tensor = minmax_scaler.fit_transform(vocab_tensor)


    vocab_tensor = torch.from_numpy(vocab_tensor).to(torch.float32)
    vocab_tensor = torch.nan_to_num(vocab_tensor)
    return vocab_tensor


def get_vocab_macc(vocab_list, n_components=10):
    vocab_tensor = []
    for v in vocab_list:
        mol = get_mol(v)
        value = MACCSkeys.GenMACCSKeys(mol)  # 1826
        vocab_tensor.append(value)

    vocab_tensor = np.matrix(vocab_tensor)

    vocab_tensor = np.nan_to_num(vocab_tensor)
    vocab_tensor = torch.from_numpy(vocab_tensor).to(torch.float)


    return vocab_tensor


def get_vocab_morgan(vocab_list, n_components=10):
    from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect

    vocab_tensor = []
    for v in vocab_list:
        mol = get_mol(v)
        value = GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)  # 1826
        vocab_tensor.append(value)

    vocab_tensor = np.matrix(vocab_tensor)

    vocab_tensor = np.nan_to_num(vocab_tensor)
    vocab_tensor = torch.from_numpy(vocab_tensor).to(torch.float)

    return vocab_tensor





def count_vocab(vocab_df):
    atoms_df = pd.DataFrame()
    vocab_df['atoms_sum'] = vocab_df['smiles'].apply(lambda x: get_mol(x).GetNumAtoms())
    atom_sum_max = vocab_df['atoms_sum'].max()
    vocab_sum = len(vocab_df)
    vocab_ocuur_sum = vocab_df['frequency'].sum()
    # atoms_set = set(vocab_df['atoms_sum'])
    for i in range(1, atom_sum_max + 1):
        prop = len(vocab_df[vocab_df['atoms_sum'] == i]) / vocab_sum
        fre = sum(vocab_df[vocab_df['atoms_sum'] == i]['frequency']) / vocab_ocuur_sum
        atoms_df = atoms_df.append({'number of atoms': i, 'proportion': prop, 'frequency': fre}, ignore_index=True)
    return atoms_df





def graph_to_data(g, vocab):
    frag_id_list = [vocab.index(n['sub_smiles']) for i, n in g.nodes(data=True)]

    edge_attr = []
    edge_index = []
    inter_tensor = []

    if len(g.edges(data=True)) == 0:
        edge_index.append([0, 0])
        edge_attr.append([0, 0, 0, 0])
        inter_tensor.append([[0] * 133, [0] * 133])

    else:
        for u, v, attr in g.edges(data=True):
            edge_index.append([u, v])
            edge_attr.append(attr['edge_attr'])
            inter_tensor.append(attr['inter_tensor'])

    edge_index = np.array(edge_index).T
    edge_index = edge_index.astype(int)

    edge_attr = np.array(edge_attr).astype(int)

    inter_tensor = np.array(inter_tensor).astype(int)

    frags_id = np.array(frag_id_list).astype(int)


    return frags_id, edge_index, edge_attr, inter_tensor


def get_data(vocab, g_list):
    frag_data_l = []
    for s, g in tqdm(g_list.items(), desc="Iteration"):
        if g == None:
            print(s)
        else:
            s, v, id = g.graph['smiles'], g.graph['label'], g.graph['smiles_id']

            frags_id, edge_index, edge_attr, inter_tensor = graph_to_data(g,vocab)

            smiles_id = torch.from_numpy(np.array([id]).astype(int)).to(torch.int64)

            frags_id = torch.from_numpy(frags_id).to(torch.int64)
            edge_index = torch.from_numpy(edge_index).to(torch.int64)
            edge_attr = torch.from_numpy(edge_attr).to(torch.int64)
            inter_tensor = torch.from_numpy(inter_tensor).to(torch.int64)
            smiles_data = Data(x=frags_id,  edge_index=edge_index, edge_attr=edge_attr, inter_tensor=inter_tensor,
                               smiles_id=smiles_id)

            smiles_data.y = torch.from_numpy(np.array(v).astype(float)).to(torch.float32)
            frag_data_l.append(smiles_data)

    return frag_data_l


class SmilesDataset():
    def __init__(self, vocab, g_list, root, data_type='frag_datas'):
        # self.vocab_list = vocab.get_vocab_list()
        self.vocab = vocab
        self.data_type = data_type
        self.original_root = root
        # self.folder = osp.join(root, 'processed')
        self.pre_processed_file_path = osp.join(root, 'smiles_data.pt')

        super(SmilesDataset, self).__init__()
        self.g_list = g_list
        # self.data_df = pd.read_csv(osp.join(self.original_root, 'raw.csv'))
        # self.smiles_list = self.data_df['smiles'].tolist()
        self.process()

    def process(self):
        if osp.exists(self.pre_processed_file_path):
            datas = torch.load(self.pre_processed_file_path)
            # if pre-processed file already exists
            self.datas = datas
            self.frag_data_l = datas['frag_datas']
        else:
            print('Converting SMILES strings into graphs...')
            self.frag_data_l = get_data(self.vocab, self.g_list)
            self.datas = {'frag_datas': self.frag_data_l}
            print('Saving...')

            torch.save(self.datas, self.pre_processed_file_path)


    def get_frags_datas(self):
        return self.frag_data_l


    def __getitem__(self, id):
        return self.datas[self.data_type][id]

    def __len__(self):
        return len(self.datas[self.data_type])

    def __repr__(self):  # pragma: no cover
        return '{}({})'.format(self.__class__.__name__, len(self))


class DDIDataset():
    def __init__(self,
                 vocab,
                 g_list,
                 root,
                 ddi_filename='ZhangDDI_train.csv'):

        self.vocab = vocab
        self.g_list = g_list
        self.root = root
        self.pre_processed_file_path = osp.join(self.root, 'smiles_data.pt')
        self.ddi_fn = ddi_filename

        super(DDIDataset, self).__init__()

        # self.data_df = pd.read_csv(osp.join(self.original_root, 'raw.csv'))
        # self.smiles_list = self.data_df['smiles'].tolist()
        self.process()

        df = pd.read_csv(os.path.join(self.root, self.ddi_fn), usecols=['smiles_1', 'smiles_2', 'label'])
        self.ddi = df.values

        self.drugs = torch.load(self.pre_processed_file_path)

    def __getitem__(self, idx):
        id1, id2, label = self.ddi[idx]
        return self.drugs[id1], self.drugs[id2], torch.Tensor([float(label)])

    def __len__(self):
        return len(self.ddi)

    def process(self):
        if osp.exists(self.pre_processed_file_path):
            self.drugs = torch.load(self.pre_processed_file_path)

        else:
            data_dict = {}
            print('Converting SMILES strings into graphs...')

            for smiles, g in tqdm(self.g_list.items()):
                frags_id, edge_index, edge_attr, inter_tensor = graph_to_data(g, self.vocab)
                frags_id = torch.from_numpy(np.array(frags_id)).to(torch.int64)
                edge_index = torch.from_numpy(np.array(edge_index)).to(torch.int64)
                edge_attr = torch.from_numpy(np.array(edge_attr)).to(torch.int64)
                inter_tensor = torch.from_numpy(np.array(inter_tensor)).to(torch.int64)

                data = Data(x=frags_id, edge_index=edge_index, edge_attr=edge_attr, inter_tensor=inter_tensor)

                data_dict[smiles] = data

            self.drugs = data_dict
            print('Saving...')

            torch.save(self.drugs, self.pre_processed_file_path)












