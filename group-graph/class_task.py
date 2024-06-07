#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/7/21 14:05
# @Author : piaoyang cao
# @File : run_class.py


import argparse

import torch
from torch import nn

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.optim.lr_scheduler import StepLR

import numpy as np

import torch.optim as optim

from tqdm import tqdm

from group_graph_encoder import HierGnnEncoder, Predictor
from data_loader import SmilesDataset, get_vocab_descriptors,  get_vocab_macc

import random
import time

import os
import pandas as pd

import copy

from torch_geometric.data import Batch

from sklearn.model_selection import KFold
from sklearn import metrics





def train(vocab_datas, model, device, loader, optimizer, loss_criterion):
    model.train()
    # train_meter = Meter()
    loss_accum = 0
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        pred = model(vocab_datas, batch)
        # pred, _ = model(batch)
        # print(pred.size)
        # print(batch.y)
        loss = (loss_criterion(pred, batch.y.view(pred.shape).to(device)).float()).mean()
        # print(loss)
        loss_accum += loss.detach().cpu().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # train_meter.update(pred, batch.y.to(args.device).float(), None)
        torch.cuda.empty_cache()

        # print(train_meter.compute_metric('return_pred_true'))
        # train_score = np.mean(train_meter.compute_metric('roc_auc'))
    return loss_accum / (step + 1)


def eval(vocab_datas, model, device, loader):
    model.eval()
    # eval_meter = Meter()
    y_pred_list = []
    y_true_list = []

    with torch.no_grad():
        for step, batch in enumerate(tqdm(loader, desc="Iteration")):
            batch = batch.to(device)
            pred = model(vocab_datas, batch)

            y_pred_list.append(pred.detach().cpu())
            y_true_list.append(batch.y.view(pred.shape).detach().cpu())
            # eval_meter.update(pred, batch.y.view(pred.shape).to(device), mask=None)
            torch.cuda.empty_cache()

    # y_pred, y_true = eval_meter.compute_metric('return_pred_true')
    y_pred = torch.cat(y_pred_list, dim=0)
    y_true = torch.cat(y_true_list, dim=0)
    # print(y_pred)
    # print(y_pred.size())
    y_true = torch.nan_to_num(y_true.squeeze())
    y_pred = torch.nan_to_num(y_pred.squeeze())

    y_pred_list = torch.sigmoid(y_pred)
    y_pred_label = torch.where(y_pred_list >= 0.5, 1, 0)

    auc = metrics.roc_auc_score(y_true.numpy(), y_pred.numpy())
    prc_auc = metrics.average_precision_score(y_true.numpy(), y_pred.numpy())
    accuracy = metrics.accuracy_score(y_true.numpy(), y_pred_label.numpy())
    # se, sp = sesp_score(y_true.numpy(),y_pred_label.numpy())
    pre, rec, f1, sup = metrics.precision_recall_fscore_support(y_true.numpy(), y_pred_label.numpy())
    # mcc = metrics.matthews_corrcoef(y_true.numpy(), y_pred_label.numpy())
    f1 = f1[1]
    result = {'roc_auc': auc, 'accuracy': accuracy, 'prc_auc': prc_auc, 'f1': f1}
    return result



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--times', type=int, default=5,
                        help='repeat times')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='number layers of gnn')
    parser.add_argument('--task_name_list', type=list,
                        default=['bbbp'],
                        # default=['bbbp','bace','hiv','clintox','toxcast', 'tox21','sider'],
                        help='task_name_list')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 4)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--embed_size', type=int, default=300,
                        help='embedding dimensions (default: 256)')

    parser.add_argument('--dropout', type=float, default=0,
                        help='dropout (default: 0)')

    parser.add_argument('--graph_pooling', type=str, default="sum",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='concat,last,max,sum')
    parser.add_argument('--gnn_type', type=str, default='gin')

    parser.add_argument('--seed', type=list, default=[42, 0, 1], help="Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help="Seed for minibatch selection, random initialization.")
    # parser.add_argument('--graph_pooling', type=int, default='mean', help="evaluating training or not")
    parser.add_argument('--log_dir', type=str, default='log/', help="tensorboard log directory")
    # parser.add_argument('--checkpoint_dir', type=str, default='/home/pycao/hgraph2graph-master/hgraph2graph-master/hgraph/check/', help="checkpoint directory")
    parser.add_argument('--checkpoint_dir', type=str, default='check/', help="checkpoint directory")
    parser.add_argument('--pretrain', type=bool, default=False, help='number of workers for dataset loading')
    parser.add_argument('--input_pretrain_file', type=str, default='motif_pretrain_output/motif_pretrain.pt',
                        help='vocab_pretrain directory')
    args = parser.parse_args()

    print(args)

    # device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    device = 'cpu'
    root = '/home/pycao/group-graph/dataset/class_task'
    # root = '../dataset/class_task'
    check_path = '/home/pycao/group-graph/group-graph/check/'
    result_path = '/home/pycao/group-graph/group-graph/result'
    # result_path = '../result/class_task'
    test_pd = pd.DataFrame(columns=args.task_name_list)
    time_task = {}

    for task in args.task_name_list:

        time_start = time.time()

        path = os.path.join(root, task)

        g_list = torch.load(os.path.join(path, 'group_graph.pt'))

        vocab_df = pd.read_csv(os.path.join(path, 'vocab.csv'), sep=',', header=0)
        data_df = pd.read_csv(os.path.join(path, 'raw.csv'), sep=',', header=0)

        vocab_list = vocab_df['smiles'].tolist()

        dataset = SmilesDataset(vocab_list, g_list, path, data_type='frag_datas')

        vocab_datas = get_vocab_descriptors(vocab_list)
        # vocab_datas = torch.load(os.path.join(path, 'vocab_descriptor.pt'))
        vocab_size = vocab_datas.shape

        raw_smiles_datas = []

        if task == "tox21":
            num_tasks = 12
        elif task == "pcba":
            num_tasks = 128
        elif task == "muv":
            num_tasks = 17
        elif task == "toxcast":
            num_tasks = 617
        elif task == "sider":
            num_tasks = 27
        elif task == "clintox":
            num_tasks = 2
        else:
            num_tasks = 1

        result_pd = pd.DataFrame()
        # result_pd['index'] = ['roc_auc']
        result_pd['index'] = ['roc_auc', 'accuracy', 'prc_auc', 'f1']
        test_result = []
        # loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        split = []
        kf = KFold(n_splits=args.times, shuffle=True, random_state=42)
        # split = torch.load(path + '/split.pt')
        for k, (train_index, test_index) in enumerate(kf.split(dataset)):
            # for k, (train_index, test_index) in enumerate(split):
            split.append((train_index, test_index))
            train_dataset = [dataset[i] for i in train_index]
            test_dataset = [dataset[i] for i in test_index]

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            # valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

            print("loading model")
            pretrain_file = None
            encoder = HierGnnEncoder(vocab_size, num_tasks, device, pretrain_file, args)

            model = Predictor(encoder=encoder, latent_dim=args.embed_size, num_tasks=num_tasks,
                              )
            best_valid_loss = 0

            loss_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

            params = model.parameters()
            optimizer = optim.Adam(params, lr=args.lr)
            scheduler = StepLR(optimizer, step_size=10, gamma=0.8)

            for epoch in range(1, args.epochs + 1):
                model.to(device)
                l = train(vocab_datas, model, device, train_loader, optimizer, loss_criterion)
                # print(l)
                scheduler.step()

                valid_loss = eval(vocab_datas, model, device, test_loader)['roc_auc']
                # test_loss = eval(vocab_datas, raw_smiles_datas, model, device, test_loader)['auc']

                if valid_loss > best_valid_loss:
                    best_valid_loss = valid_loss
                    if args.checkpoint_dir == '':
                        os.makedirs(args.checkpoint_dir, exist_ok=True)
                    if args.checkpoint_dir != '':
                        # print('Saving checkpoint...')
                        checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                                      'optimizer_state_dict': optimizer.state_dict(),
                                      'scheduler_state_dict': scheduler.state_dict(), 'best_val_mae': best_valid_loss,
                                      }
                        torch.save(checkpoint, check_path + task + '_checkpoint.pt')

                print(
                    '***************************************************************************************************')
                print('{}, {}/{} time {}  valid_loss{}'.format(task, k + 1, args.times, epoch,
                                                               valid_loss))
                print(
                    '***************************************************************************************************')

            # 回归任务计算R2打分

            state_dict = torch.load(args.checkpoint_dir + task + '_checkpoint.pt')
            model.load_state_dict(state_dict['model_state_dict'])
            stop_train_list = eval(vocab_datas, model, device, train_loader)
            # stop_val_list = eval(vocab_datas, raw_smiles_datas, model, device, valid_loader)
            stop_test_list = eval(vocab_datas, model, device, test_loader)
            test_result.append(stop_test_list['roc_auc'])

            result_pd['train_' + str(k + 1)] = list(stop_train_list.values())
            # result_pd['val_' + str(k + 1)] = list(stop_val_list.values())
            result_pd['test_' + str(k + 1)] = list(stop_test_list.values())

            print(result_pd)

        time_end = time.time()
        time_task[task] = (time_end - time_start) / args.times

        test_pd[task] = test_result
        result_pd.to_csv(result_path + '/' + task + '_result.csv', index=False)
        torch.save(split,path+'/split.pt')
    # test_pd = test_pd.append(time_task,ignore_index=True)
    test_pd.loc[len(test_pd)] = list(time_task.values())
    print(test_pd)
    test_pd.to_csv(result_path + '/' + 'class_result.csv', index=False)




if __name__ == "__main__":
    main()

