#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# @Time : 2023/7/21 14:05 
# @Author : piaoyang cao 
# @File : class_task.py


import argparse

import torch

from torch_geometric.loader import DataLoader

from torch.optim.lr_scheduler import StepLR

import numpy as np

import torch.optim as optim

from tqdm import tqdm

from group_graph_encoder import HierGnnEncoder
from motif_prediction import MotifPrediction

from graph_load import Vocab,SmilesDataset, Meter,get_data,get_vocab_descriptors,get_vocab_datas,get_vocab_macc
from motif_encoder import MotifEncoder
from sklearn import metrics
import random
import time

import os
import pandas as pd
from ogb.lsc.pcqm4mv2 import PCQM4Mv2Evaluator

from torch.utils.data import random_split
from math import sqrt


from mtl_dataset import Prediction_Dataset
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch
from gnn_model import GNN, GNN_graphpred
#import networkx as nx
#import matplotlib.pyplot as plt


def sesp_score(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(y_true)):
        if int(y_true[i]) == y_pred[i] and int(y_true[i]) == 1:
            tp = tp + 1
        if int(y_true[i]) == y_pred[i] and int(y_true[i]) == 0:
            tn = tn + 1
        if int(y_true[i]) == 0 and int(y_pred[i]) == 1:
            fp = fp + 1
        if int(y_true[i]) == 1 and int(y_pred[i]) == 0:
            fn = fn + 1
    sensitivity = round(tp / (tp + fn), 4)
    specificity = round(tn / (tn + fp), 4)
    return sensitivity, specificity


def train(vocab_datas, raw_smiles_datas, model, device, loader, optimizer, loss_criterion):
    model.train()
    #train_meter = Meter()
    loss_accum = 0
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        #print(batch.y)
        pred,_ = model(vocab_datas, raw_smiles_datas, batch)
        if torch.any(torch.isnan(pred)):
            print(batch.smiles_id)
        #print(batch.y)
        loss = (loss_criterion(pred, batch.y.view(pred.shape).to(device)).float()).mean()
        #print(loss)
        #loss_accum += loss.detach().cpu().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #train_meter.update(pred, batch.y.to(args.device).float(), None)
        #torch.cuda.empty_cache()

    #print(train_meter.compute_metric('return_pred_true'))
    #train_score = np.mean(train_meter.compute_metric('roc_auc'))
    #return loss_accum / len(loader)

def eval(vocab_datas, raw_smiles_datas, model, device, loader):
    model.eval()
    eval_meter = Meter()
    with torch.no_grad():
        for step, batch in enumerate(tqdm(loader, desc="Iteration")):
            batch = batch.to(device)
            #print(frag_id)
            pred,node_presentation = model(vocab_datas, raw_smiles_datas, batch)
            eval_meter.update(pred, batch.y.view(pred.shape).to(device), mask=None)
            torch.cuda.empty_cache()

    y_pred, y_true = eval_meter.compute_metric('return_pred_true')
    #print(y_pred)
    #print(y_pred.size())
    y_true_list = y_true.squeeze().tolist()
    #print(y_true_list)
    y_pred_list = torch.sigmoid(y_pred.squeeze()).tolist()
    # save prediction

    #y_pred_label = torch.where(y_pred>=0.5,1,0)
    y_pred_label = [1 if x >= 0.5 else 0 for x in y_pred_list]
    #print(y_pred_label)
    auc = metrics.roc_auc_score(y_true_list, y_pred_list)
    accuracy = metrics.accuracy_score(y_true_list, y_pred_label)
    se, sp = sesp_score(y_true,y_pred_label)
    pre, rec, f1, sup = metrics.precision_recall_fscore_support(y_true_list, y_pred_label)
    mcc = metrics.matthews_corrcoef(y_true_list, y_pred_label)
    f1 = f1[1]
    rec = rec[1]
    pre = pre[1]
    err = 1 - accuracy
    result = [auc,accuracy,se,sp,f1,pre,rec,err,mcc]
    #result = {'auc':auc, 'accuracy':accuracy, 'se':se, 'sp':sp, 'f1':f1, 'pre':pre, 'rec':rec, 'err':err, 'mcc':mcc}
    return result

def compute_examples(vocab_datas, raw_smiles_datas, model, device, loader):
    model.eval()
    node_df = pd.DataFrame(columns=['smiles_id','frag_id','node_value'])
    edge_df = pd.DataFrame(columns=['Source','Target'])
    with torch.no_grad():
        for step, batch in enumerate(tqdm(loader, desc="Iteration")):
            batch = batch.to(device)
            smiles_id = batch.smiles_id.index_select(index=batch.batch,dim=0).tolist()
            source = batch.edge_index[0,].squeeze().tolist()
            target = batch.edge_index[1,].squeeze().tolist()
            frag_id = batch.x.squeeze().tolist()
            edge_attr_list = batch.edge_attr[:, [1, 3]].tolist()
            #print(frag_id)

            pred,node_presentation = model(vocab_datas, raw_smiles_datas, batch)

    y_pred_list = torch.sigmoid(pred.squeeze()).tolist()
    #node_presentation = torch.sigmoid(node_presentation.squeeze()).tolist()
    if type(y_pred_list) != list: y_pred_list = [y_pred_list]
    y_pred_label = [1 if x >= 0.5 else 0 for x in y_pred_list]
    print(y_pred_label)
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
    node_df['node_value'] = node_presentation
    node_df['node_value'] = node_df['node_value'].apply(lambda x: format(x, '.2f'))

    edge_df['Source'] = source
    edge_df['Target'] = target
    edge_df['label'] = inter_list

    #print(data_df)

    #result = {'auc':auc, 'accuracy':accuracy, 'se':se, 'sp':sp, 'f1':f1, 'pre':pre, 'rec':rec, 'err':err, 'mcc':mcc}
    return node_df,edge_df



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--times', type=int, default=1,
                        help='repeat times')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='number layers of gnn')
    parser.add_argument('--task_name_list', type=list,
                        # default=['bace'],
                        default=['bbbp'],
                        help='task_name_list')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 4)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--embed_size', type=int, default=300,
                        help='embedding dimensions (default: 256)')

    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout (default: 0)')

    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='concat,last,max,sum')
    parser.add_argument('--gnn_type', type=str, default='gin')

    parser.add_argument('--seed', type=int, default=42, help="Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help="Seed for minibatch selection, random initialization.")
    # parser.add_argument('--graph_pooling', type=int, default='mean', help="evaluating training or not")
    parser.add_argument('--log_dir', type=str, default='log/', help="tensorboard log directory")
    #parser.add_argument('--checkpoint_dir', type=str, default='/home/pycao/hgraph2graph-master/hgraph2graph-master/hgraph/check/', help="checkpoint directory")
    parser.add_argument('--checkpoint_dir', type=str, default='check/', help="checkpoint directory")
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataset loading')
    parser.add_argument('--input_pretrain_file', type=str, default='motif_pretrain_output/motif_pretrain.pt',
                        help='vocab_pretrain directory')
    args = parser.parse_args()

    print(args)

    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    #device = 'cpu'

    test_pd = pd.DataFrame(columns=args.task_name_list)
    time_task = {}
    root = '/home/pycao/hgraph2graph-master/hgraph2graph-master/dataset/dataset/class_task'
    #root = '../dataset/class_task'

    #result_path = '../result/class_task'
    result_path = '/home/pycao/hgraph2graph-master/hgraph2graph-master/dataset/dataset/result'

    for task in args.task_name_list:
        time_start = time.time()
        if task == 'qm8':
            num_tasks = 12
            met = 'mae'
        elif task == 'qm9':
            num_tasks = 3
        else:
            num_tasks = 1


        path =  os.path.join(root, task)
        vocab = Vocab(path)
        vocab_list = vocab.get_vocab_list()

        #vocab_df = pd.read_csv(vocab_path, sep=',', header=0)
        #vocab_list = vocab_df['smiles'].tolist()
        #vocab_size = len(vocab_list)

        '''
        vocab_dataset = Prediction_Dataset(vocab_df, num_tasks)
        vocab_tensor = [vocab_dataset[i] for i in range(0, len(vocab_dataset))]
        vocab_tensor = pad_sequence([torch.from_numpy(np.array(x)) for x in vocab_tensor],
                                     batch_first=True).long()
        
        
        
        vocab_datas = torch.load(os.path.join(root, 'pretrain_vocab_tensor.pt')).to('cpu')
        vocab_datas = torch.nan_to_num(vocab_datas)
        '''
        #vocab_datas = get_vocab_descriptors(vocab.get_vocab_list())
        #vocab_datas = torch.load(path+'/vocab_tensor.pt')
        #vocab_datas = get_vocab_datas(vocab_list)
        vocab_datas = get_vocab_macc(vocab_list)

        vocab_size = vocab_datas.shape
        print(vocab_size)

        '''
        data_path = os.path.join(root, task + '.csv')
        data_df = pd.read_csv(data_path, sep=',', header=0)

        train_dataset = Prediction_Dataset(data_df, num_tasks)

        raw_smiles_datas = [train_dataset[i] for i in range(0, len(train_dataset))]

        raw_smiles_datas = pad_sequence([torch.from_numpy(np.array(x)) for x in raw_smiles_datas], batch_first=True).long()
        '''

        #smiles_tensor = torch.load(os.path.join(root, 'tensor_pretrain.pt')).to('cpu')
        #smiles_tensor = torch.nan_to_num(smiles_tensor)

        dataset = SmilesDataset(vocab, path, data_type='frag_datas')
        split_idx = dataset.get_idx_split()
        raw_smiles_datas = dataset.get_raw_datas()
        #vocab_datas = dataset.get_vocab_datas()
        #vocab_size = [len(vocab_list), 9]

        result_pd = pd.DataFrame()
        result_pd['index'] = ['roc_auc', 'accuracy', 'sensitivity', 'specificity', 'f1-score', 'precision', 'recall',
                              'error rate', 'mcc']
        test_result = []
        '''
        train_dataset = [dataset[i] for i in split_idx['train']]
        valid_dataset = [dataset[i] for i in split_idx['valid']]
        test_dataset = [dataset[i] for i in split_idx['test']]
        '''
        #pretrain_path = '/home/pycao/hgraph2graph-master/hgraph2graph-master/hgraph/motif_pretrain_output'
        #pretrain_state_dict = torch.load(os.path.join(pretrain_path, task + '.pt'),device)
        for time_id in range(args.times):
            '''
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers)
            valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.num_workers)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers)

            '''
            train_dataset, valid_dataset, test_dataset = random_split(
                dataset=dataset,
                lengths=[int(0.8 * len(dataset)), int(0.1 * len(dataset)),
                         len(dataset) - int(0.8 * len(dataset)) - int(0.1 * len(dataset))],
                generator=torch.Generator().manual_seed(time_id)
            )
            
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
            valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)


            print("loading model")
            model = HierGnnEncoder(vocab_size, num_tasks, device, args.num_layers, args.embed_size, args.dropout,
                                   args.graph_pooling, args.gnn_type, args.JK)

            #model.smiles_gnn.load_state_dict(pretrain_state_dict)

            #model.to(device)
            num_params = sum(p.numel() for p in model.parameters())
            print(f'#Params: {num_params}')

            best_valid_loss = 0
            loss_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
            #loss_criterion = torch.nn.CrossEntropyLoss(size_average=False)
            for epoch in range(1, args.epochs + 1):

                params = model.parameters()
                optimizer = optim.Adam(params, lr=args.lr)
                scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
                model.to(device)


                train(vocab_datas, raw_smiles_datas, model, device, train_loader, optimizer, loss_criterion)
                scheduler.step()

                # print('Evaluating...')
                valid_loss = eval(vocab_datas, raw_smiles_datas, model, device, valid_loader)[0]
                test_loss = eval(vocab_datas, raw_smiles_datas, model, device, test_loader)[0]


                if valid_loss > best_valid_loss:
                    best_valid_loss = valid_loss
                    if args.checkpoint_dir == '':
                        os.makedirs(args.checkpoint_dir, exist_ok=True)
                    if args.checkpoint_dir != '':
                        # print('Saving checkpoint...')
                        checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                                      'optimizer_state_dict': optimizer.state_dict(),
                                      'scheduler_state_dict': scheduler.state_dict(), 'best_val_mae': best_valid_loss,
                                      'num_params': num_params}
                        torch.save(checkpoint, args.checkpoint_dir + task + '_checkpoint.pt')

                print(
                    '***************************************************************************************************')
                print('{}, {}/{} time {} valid_loss{} test_loss{}'.format(task, time_id + 1, args.times, epoch, valid_loss,
                                                                   test_loss))
                print(
                    '***************************************************************************************************')

            # 回归任务计算R2打分
            state_dict = torch.load(args.checkpoint_dir + task + '_checkpoint.pt')
            model.load_state_dict(state_dict['model_state_dict'])
            stop_train_list = eval(vocab_datas, raw_smiles_datas, model, device, train_loader)
            stop_val_list = eval(vocab_datas, raw_smiles_datas, model, device, valid_loader)
            stop_test_list = eval(vocab_datas, raw_smiles_datas, model, device, test_loader)

            test_result.append(stop_test_list[0])
            print('{}, {}/{} time  test_loss{}'.format(task, time_id + 1, args.times,
                                                               stop_test_list[0]))

            result_pd['train_' + str(time_id + 1)] = stop_train_list
            result_pd['val_' + str(time_id + 1)] = stop_val_list
            result_pd['test_' + str(time_id + 1)] = stop_test_list
        time_end = time.time()
        time_task[task] = (time_end - time_start)/ args.times

        test_pd[task] = test_result
        result_pd.to_csv(result_path + '/' + task + '_result.csv', index=False)
    test_pd = test_pd.append(time_task,ignore_index=True)
    test_pd.to_csv(result_path + '/class_task_result.csv', index=False)
    print(test_pd)

def compute_data_list():
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--times', type=int, default=1,
                        help='repeat times')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='number layers of gnn')
    parser.add_argument('--task_name_list', type=list,
                        default=['bbbp'],
                        # default=['freesolv', ', 'lipo', 'qm7', 'qm8', 'qm9'],
                        help='task_name_list')
    parser.add_argument('--batch_size', type=int, default=12,
                        help='input batch size for training (default: 4)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--embed_size', type=int, default=300,
                        help='embedding dimensions (default: 256)')

    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout (default: 0)')

    parser.add_argument('--graph_pooling', type=str, default="sum",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='concat,last,max,sum')

    parser.add_argument('--gnn_type', type=str, default='gin')
    # parser.add_argument('--depthT', type=int, default=15)
    parser.add_argument('--dataset', type=str, default='pcqm4m-v2',
                        help='root directory of dataset.')
    parser.add_argument('--vocab', type=str, default='',
                        help='root directory of vocab.')
    parser.add_argument('--seed', type=int, default=42, help="Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help="Seed for minibatch selection, random initialization.")
    # parser.add_argument('--graph_pooling', type=int, default='mean', help="evaluating training or not")
    parser.add_argument('--log_dir', type=str, default='log/', help="tensorboard log directory")
    parser.add_argument('--checkpoint_dir', type=str, default='/home/pycao/hgraph2graph-master/hgraph2graph-master/hgraph/check/', help="checkpoint directory")
    parser.add_argument('--save_test_dir', type=str, default='../result/class_task/', help="test directory")
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataset loading')
    parser.add_argument('--input_pretrain_file', type=str, default='motif_pretrain_output/motif_pretrain.pt',
                        help='vocab_pretrain directory')
    args = parser.parse_args()

    print(args)

    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")


    test_pd = pd.DataFrame(columns=args.task_name_list)
    # pretrain_file = 'motif_pretrain_output/motif_pretrain.pt'
    root = '/home/pycao/hgraph2graph-master/hgraph2graph-master/dataset/dataset/class_task'
    for task in args.task_name_list:
        if task == 'qm8':
            num_tasks = 12
            met = 'mae'
        elif task == 'qm9':
            num_tasks = 3
        else:
            num_tasks = 1

        path = os.path.join(root,task)

        # vocab_size = len(vocab_list)
        vocab = Vocab(path)
        vocab_list = vocab.get_vocab_list()
        '''
        vocab_dataset = Prediction_Dataset(vocab_df, num_tasks)
        vocab_tensor = [vocab_dataset[i] for i in range(0, len(vocab_dataset))]
        vocab_tensor = pad_sequence([torch.from_numpy(np.array(x)) for x in vocab_tensor],
                                     batch_first=True).long()

        vocab_datas = torch.load(os.path.join(root, 'pretrain_vocab_tensor.pt')).to('cpu')
        vocab_datas = torch.nan_to_num(vocab_datas)
        '''
        #vocab_datas = torch.load(os.path.join(path, 'vocab_tensors.pt'))
        #vocab_datas = torch.nan_to_num(vocab_datas)
        vocab_datas = get_vocab_macc(vocab_list)
        vocab_size = vocab_datas.shape

        print(vocab_datas.shape)


        '''
        data_path = os.path.join(root, task + '.csv')
        data_df = pd.read_csv(data_path, sep=',', header=0)

        train_dataset = Prediction_Dataset(data_df, num_tasks)

        raw_smiles_datas = [train_dataset[i] for i in range(0, len(train_dataset))]

        raw_smiles_datas = pad_sequence([torch.from_numpy(np.array(x)) for x in raw_smiles_datas], batch_first=True).long()
        '''

        # smiles_tensor = torch.load(os.path.join(root, 'tensor_pretrain.pt')).to('cpu')
        # smiles_tensor = torch.nan_to_num(smiles_tensor)

        dataset = SmilesDataset(vocab_list, path, data_type='frag_datas')
        raw_smiles_datas = dataset.get_raw_datas()
        # vocab_datas = dataset.get_vocab_datas()
        # vocab_size = [len(vocab_list), 9]
        #smiles_id_list = [491, 508, 509, 511, 513, 514, 515, 516, 517, 538, 541, 600]
        #train_dataset = [dataset[i] for i in smiles_id_list]
        # bbbp



        #smiles_list = ['CCOc1ccc2ccccc2c1C(=O)NC1C(=O)N2C1SC(C)(C)C2C(=O)O']
        smiles_list = ['CC1=C(C(=NN1C)C)NS(=O)(=O)C2=C(C=C(C=C2Cl)C3=CC(=NC=C3)N4CCNCC4)Cl',
                       'CC1=C(C=CC=N1)COC2=C(C=C(C=C2Cl)C3=CC(=NC=C3)N4CCNCC4)Cl',
                       'C1CC(CCC1NC2=NC=CC(=N2)C3=C4C=CC=NN4N=C3)O', 'C1=CC2=C(C=NN2N=C1)C3=NC(=NC=C3)NC4=CC=NC=C4',
                       'CC1=C(C(=NC(=N1)OCC(=O)N(C)C2CCN(CC2)CC3=CC=CC=C3)C)N',
                       'CC1=C(C(=NC(=N1)OCC(=O)N(C)C2CCN(CC2)CC3CC3(F)F)C)N']

        frag_data_l,raw_data_l = get_data(vocab,list(zip(smiles_list, [1,2,3,4,5,6],[1,1,1,1,1,1])))


        result_pd = pd.DataFrame()
        result_pd['index'] = ['roc_auc', 'accuracy', 'sensitivity', 'specificity', 'f1-score', 'precision', 'recall',
                              'error rate', 'mcc']
        test_result = []

        for time_id in range(args.times):
            '''
            train_dataset, valid_dataset, test_dataset = random_split(
                dataset=dataset,
                lengths=[int(0.8 * len(dataset)), int(0.1 * len(dataset)),
                         len(dataset) - int(0.8 * len(dataset)) - int(0.1 * len(dataset))],
                generator=torch.Generator().manual_seed(time_id)
            )
            '''
            print("loading model")
            # smiles_encoder_pretrained_dict = torch.load('weights/model/' + task + '_encoder_mtl_bert.pt')
            model = HierGnnEncoder(vocab_size, num_tasks, device, args.num_layers, args.embed_size, args.dropout,
                                   args.graph_pooling, args.gnn_type, args.JK)

            #model.get_smiles_encoder().load_state_dict(smiles_encoder_pretrained_dict)

            # model.to(device)
            print("get datas")
            train_loader = DataLoader(frag_data_l, batch_size=len(frag_data_l), shuffle=False)
            state_dict = torch.load(args.checkpoint_dir + task + '_checkpoint.pt','cpu')
            model.load_state_dict(state_dict['model_state_dict'])
            node_df, edge_df = compute_examples(vocab_datas, raw_smiles_datas, model, device, train_loader)
            node_df['label'] = node_df['frag_id'].apply(lambda x: vocab_list[x])
            node_df.to_csv(os.path.join(path,'pred_'+ task + '_node_improve.csv'),index=False)
            edge_df.to_csv(os.path.join(path,'pred_'+ task + '_edge_improve.csv'),index=False)

            print(node_df)
            print(edge_df)


if __name__ == "__main__":
    #main()
    compute_data_list()
