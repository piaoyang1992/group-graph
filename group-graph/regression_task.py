#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/6/21 15:50
# @Author : piaoyang cao
# @File : prop_pred.py
# from data_loader import GetLoader, get_data_list
import argparse

import torch

from torch_geometric.loader import DataLoader

from torch.optim.lr_scheduler import StepLR

import numpy as np

import torch.optim as optim

from tqdm import tqdm

from group_graph_encoder import HierGnnEncoder, Predictor
# from motif_prediction import MotifPrediction

from data_loader import SmilesDataset, get_vocab_descriptors, get_vocab_macc

import random

import os
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import time
from sklearn.model_selection import KFold



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
    y_pred_list = []
    y_true_list = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(loader, desc="Iteration")):
            batch = batch.to(device)

            pred = model(vocab_datas, batch)

            # pred = scaler.inverse_transform(pred.data.cpu().numpy())
            y = torch.nan_to_num(batch.y.view(pred.shape))

            y_pred_list.append(pred.detach().cpu())
            y_true_list.append(y.detach().cpu())
            torch.cuda.empty_cache()

    y_pred = torch.cat(y_pred_list, dim=0)
    y_true = torch.cat(y_true_list, dim=0)
    # print(y_pred)
    # print(y_pred.size())
    y_true = torch.nan_to_num(y_true).numpy()
    y_pred = torch.nan_to_num(y_pred).numpy()

    # y_true = torch.cat(y_true, dim = 0)
    # y_pred = torch.cat(y_pred, dim = 0)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    R2_score = r2_score(y_true, y_pred)
    # result = [mae, mse, rmse, R2_score]
    result = {'mae': mae, 'mse': mse, 'rmse': rmse, 'R2_score': R2_score}

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
                        default=['esol'],
                        # default=['esol','freesolv','lipo','qm7', 'qm8', 'qm9' ],
                        help='task_name_list')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 4)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01,
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
    # parser.add_argument('--depthT', type=int, default=15)
    parser.add_argument('--log_dir', type=str, default='log/', help="tensorboard log directory")
    parser.add_argument('--checkpoint_dir', type=str, default='check/', help="checkpoint directory")
    parser.add_argument('--save_test_dir', type=str, default='../result/regression_task/', help="test directory")
    parser.add_argument('--pretrain', type=bool, default=False, help='pretrain or not ')
    parser.add_argument('--input_pretrain_file', type=str, default='motif_pretrain_output/motif_pretrain.pt',
                        help='vocab_pretrain directory')
    args = parser.parse_args()

    print(args)

    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    # pretrain_file = 'motif_pretrain_output/motif_pretrain.pt'

    test_pd = pd.DataFrame(columns=args.task_name_list)
    time_task = {}
    root = '/home/pycao/group-graph/dataset/class_task'
    # root = '../dataset/class_task'
    check_path = '/home/pycao/group-graph/group-graph/check/'
    result_path = '/home/pycao/group-graph/group-graph/result'

    for task in args.task_name_list:
        time_start = time.time()
        path = os.path.join(root, task)

        smiles_list = pd.read_csv(path + '/raw.csv')['smiles'].tolist()
        vocab_list = pd.read_csv(path + '/vocab.csv')['smiles'].tolist()

        g_list = torch.load(os.path.join(path, 'group_graph.pt'))

        dataset = SmilesDataset(vocab_list, g_list, path, data_type='frag_datas')

        raw_smiles_datas = []

        vocab_datas = get_vocab_descriptors(vocab_list)

        vocab_size = vocab_datas.shape

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        if task == 'qm8':
            num_tasks = 12
            eval_metric = 'mae'
        elif task == 'qm9':
            num_tasks = 3
            eval_metric = 'mae'
        elif task == 'qm7':
            num_tasks = 1
            eval_metric = 'mae'
        else:
            num_tasks = 1
            eval_metric = 'rmse'

        result_pd = pd.DataFrame()
        result_pd['index'] = ['mae', 'mse', 'rmse', 'R2-score']
        test_result = []

        split = []
        for k, (train_index, test_index) in enumerate(kf.split(dataset)):
           #for k, (train_index, test_index) in enumerate(split):
            split.append((train_index, test_index))
            train_dataset = [dataset[i] for i in train_index]

            test_dataset = [dataset[i] for i in test_index]

            print("loading model")
            pretrain_file = None
            encoder = HierGnnEncoder(vocab_size, num_tasks, device, pretrain_file, args)

            model = Predictor(encoder=encoder, latent_dim=args.embed_size, num_tasks=num_tasks,
                              )


            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
            model.to(device)

            # valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

            num_params = sum(p.numel() for p in model.parameters())
            # print(f'#Params: {num_params}')

            best_valid_loss = 10000
            loss_criterion = torch.nn.L1Loss()
            params = model.parameters()
            optimizer = optim.Adam(params, lr=args.lr)
            scheduler = StepLR(optimizer, step_size=10, gamma=0.8)

            for epoch in range(1, args.epochs + 1):
                model.to(device)
                train(vocab_datas, raw_smiles_datas, model, device, train_loader, optimizer, loss_criterion)
                scheduler.step()

                # print('Evaluating...')
                # valid_loss = eval(vocab_datas, raw_smiles_datas, model, device, valid_loader)[eval_metric]
                test_loss = eval(vocab_datas, raw_smiles_datas, model, device, test_loader)[eval_metric]

                if test_loss < best_valid_loss:
                    best_valid_loss = test_loss
                    if args.checkpoint_dir == '':
                        os.makedirs(args.checkpoint_dir, exist_ok=True)
                    if args.checkpoint_dir != '':
                        # print('Saving checkpoint...')
                        checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                                      'optimizer_state_dict': optimizer.state_dict(),
                                      'best_val_mae': best_valid_loss,
                                      'num_params': num_params}
                        torch.save(checkpoint, check_path + task + '_checkpoint.pt')

                print(
                    '***************************************************************************************************')
                print('{}, {}/{} epoch {}  test_loss{}'.format(task, k + 1, 5, epoch,
                                                               test_loss))
                print(
                    '***************************************************************************************************')

            # 回归任务计算R2打分
            state_dict = torch.load(args.checkpoint_dir + task + '_checkpoint.pt')
            model.load_state_dict(state_dict['model_state_dict'])
            stop_train_list = eval(vocab_datas, raw_smiles_datas, model, device, train_loader)
            # stop_val_list = eval(vocab_datas, raw_smiles_datas, model, device, valid_loader)
            stop_test_list = eval(vocab_datas, raw_smiles_datas, model, device, test_loader)

            test_result.append(stop_test_list[eval_metric])

            result_pd['train_' + str(k + 1)] = list(stop_train_list.values())
            # result_pd['val_' + str(k + 1)] = list(stop_val_list.values())
            result_pd['test_' + str(k + 1)] = list(stop_test_list.values())
            print(result_pd)

        time_end = time.time()
        time_task[task] = (time_end - time_start) / 5

        test_pd[task] = test_result
        result_pd.to_csv(result_path + '/' + task + '_result.csv', index=False)
        torch.save(split, path + '/split.pt')

    test_pd.loc[len(test_pd)] = list(time_task.values())
    test_pd.to_csv(result_path + '/regression_task_result.csv', index=False)
    print(test_pd)


if __name__ == "__main__":
    main()
