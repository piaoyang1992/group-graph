"""
Downstream task: BACE: Quantitative (IC50) and qualitative (binary label) binding results for a set of inhibitors of human β-secretase 1(BACE-1)
Dataset source: DeepChem
"""

import os, sys, argparse, time
import json

import pandas as pd
from tqdm import tqdm
from rdkit import Chem
#import deepchem as dc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import DataLoader
sys.path.append('..')
from models.mol_predictor import MolPredictor
from models.series_gin_edge import SerGINE
from loader import MoleculeNetDataset
from chem import *
from utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    # environment arguments
    parser.add_argument('--cpu', default=True, action='store_true', help="train on cpu")
    parser.add_argument('--gpu', default=0, type=int, help="gpu id")
    # directory arguments
    parser.add_argument('--output_dir', default='result/BACE', type=str, help="output directory of task")
    parser.add_argument('--model_name', default='roc_best_model.pth', type=str, help="saved model name")
    parser.add_argument('--time', default=1, type=int, help="time of experiment")
    parser.add_argument('--task_name_list', type=list,
                        default=['bbbp','bace','hiv','clintox'],
                        # default=['bbbp','bace','hiv','clintox','toxcast', 'tox21','sider']
                        help='task_name_list')

    # network arguments
    parser.add_argument('--gnn', default='SerGINE', type=str, help="GNN architecture")
    parser.add_argument('--num_atom_layers', default=3, type=int, help="num of atom-level gnn layers")
    parser.add_argument('--num_fg_layers', default=2, type=int, help="num of FG-level gnn layers")
    parser.add_argument('--emb_dim', default=128, type=int, help="embedding dimension")
    parser.add_argument('--num_tasks', default=1, type=int, help="number of tasks")
    parser.add_argument('--dropout', default=0.5, type=float, help="dropout rate")
    # training arguments
    parser.add_argument('--from_scratch', default=False, action='store_true', help="train from scratch")
    parser.add_argument('--pretrain_dir', default='../pretrained_model_cl_zinc15_250k', type=str, help="directory of pretrained models")
    parser.add_argument('--pretrain_model_name', default='model.pth', type=str, help="pretrained model name")
    parser.add_argument('--metric', default='CosineSimilarity', type=str, help="criterion of embedding distance")
    parser.add_argument('--margin', default=1.0, type=float, help="margin of contrastive loss")
    parser.add_argument('--pre_lr', default=1e-3, type=float, help="learning rate of pretraining")
    parser.add_argument('--batch_size', default=32, type=int, help="batch size")
    parser.add_argument('--lr0', default=1e-3, type=float, help="learning rate of encoder")
    parser.add_argument('--lr1', default=1e-3, type=float, help="learning rate of predictor")
    parser.add_argument('--epochs', default=10, type=int, help="number of training epoch")
    parser.add_argument('--log_interval', default=10, type=int, help="log interval (batch/log)")
    parser.add_argument('--early_stop', default=False, action='store_true', help="use early stop strategy")
    parser.add_argument('--patience', default=20, type=int, help="num of waiting epoch")
    parser.add_argument('--weight_decay', default=0, type=float, help="weight decay")
    parser.add_argument('--splitter', default='random', choices=['scaffold', 'random'], help="Split method of dataset")

    args = parser.parse_args()

    return args



'''
def process_data():
    tasks, datasets, transformer = dc.molnet.load_bace_classification(data_dir='../data/MoleculeNet', save_dir='../data/MoleculeNet',                                                                  splitter=args.splitter)
    dataset = [[], [], []]
    err_cnt = 0
    for i in range(3):
        for X, y, w, ids in datasets[i].itersamples():
            mol = Chem.MolFromSmiles(ids)
            if mol is None:
                print(f"'{ids}' cannot be convert to graph")
                err_cnt += 1
                continue
            atom_features, bond_list, bond_features, fg_features, fg_edge_list, fg_edge_features, atom2fg_list = mol_to_graphs(mol)
            if fg_features == []:  # C
                err_cnt += 1
                print(f"{ids} cannot be converted to FG graph")
                continue
            dataset[i].append([atom_features, bond_list, bond_features, fg_features, fg_edge_list, fg_edge_features, atom2fg_list, y, w])
    print(f"{err_cnt} data can't be convert to graph")
    train_set, valid_set, test_set = dataset
    return train_set, valid_set, test_set
'''


def process_data(df,train_index):
    dataset = []
    err_cnt = 0
    for i in tqdm(train_index):
        smiles_col = list(df.columns.values).index('smiles')
        # value_list = np.array(df.iloc[:,smiles_col+1:]).tolist()
        ids = df.iloc[i, smiles_col]
        y = df.iloc[i, smiles_col + 1:].values

        mol = Chem.MolFromSmiles(ids)
        w = 1
        if mol is None:
            print(f"'{ids}' cannot be convert to graph")
            err_cnt += 1
            continue
        atom_features, bond_list, bond_features, fg_features, fg_edge_list, fg_edge_features, atom2fg_list = mol_to_graphs(mol)
        if fg_features == []:  # C
            err_cnt += 1
            print(f"{ids} cannot be converted to FG graph")
            continue
        dataset.append([atom_features, bond_list, bond_features, fg_features, fg_edge_list, fg_edge_features, atom2fg_list, y, w])
    print(f"{err_cnt} data can't be convert to graph")
    return dataset




def train(model, data_loader, optimizer, device):
    model.train()
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    log_loss = 0
    for i, mol in enumerate(data_loader):
        mol = mol.to(device)

        optimizer.zero_grad()
        output = model(mol)
        loss = criterion(output, mol.y.view(output.shape)).mean()
        #loss = torch.mean(loss * mol.w.reshape(-1, args.num_tasks))
        loss.backward()
        optimizer.step()
        log_loss += loss.item()

        # log
        if (i+1) % args.log_interval == 0:
            log_loss = log_loss/args.log_interval
            print(f"batch: {i+1}/{len(data_loader)} | loss: {log_loss :.8f} | time: {time.time()-start_time :.4f}")
            log_loss = 0


def test(model, data_loader, device):
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    with torch.no_grad():
        avg_loss = 0
        true_label, pred_score = [], []
        for mol in data_loader:
            mol = mol.to(device)
            output = model(mol)
            y_true = mol.y.view(output.shape)
            loss = criterion(output, y_true).mean()
            #loss = torch.mean(loss * mol.w.reshape(-1, args.num_tasks))
            avg_loss += loss.item()

            # output = sigmoid(output)
            true_label.append(y_true)
            pred_score.append(output)

            '''
            for true, pred in zip(y_true, output):
                true_label.append(true.item())
                pred_score.append(pred.item())
            '''
        pred_score = torch.cat(pred_score, dim=0)
        true_label = torch.cat(true_label, dim=0)

        avg_loss = avg_loss/len(data_loader)
        auc_roc = roc_auc_score(true_label.numpy(), pred_score.numpy())

    return avg_loss, auc_roc


args = parse_args()
start_time = time.time()

output_dir = args.gnn+'_dim'+str(args.emb_dim)
output_dir = os.path.join(args.output_dir, output_dir,
                          'margin'+str(args.margin) + '_lr0_'+str(args.lr0) + '_lr1_'+str(args.lr1) + '_dropout'+str(args.dropout),
                          'time'+str(args.time))
if args.from_scratch:
    output_dir = os.path.join(output_dir, 'scratch')
ext_setting = None
if args.weight_decay > 0:
    ext_setting = 'decay'+str(args.weight_decay)
if ext_setting is not None:
    output_dir = os.path.join(output_dir, ext_setting)


def main():
    root = '/home/pycao/group-graph/dataset/class_task'
    # root = '../dataset/class_task'
    check_path = '/home/pycao/group-graph/RelMole-main/check/'
    result_path = '/home/pycao/group-graph/RelMole-main/result'

    test_pd = pd.DataFrame(columns=args.task_name_list)
    time_task = {}
    for task in args.task_name_list:

        result_pd = pd.DataFrame()
        result_pd['index'] = ['roc_auc']
        test_result = []
        time_start = time.time()

        path =  os.path.join(root, task)

        if task == "tox21":
            args.num_tasks = 12
        elif task == "pcba":
            args.num_tasks = 128
        elif task == "muv":
            args.num_tasks = 17
        elif task == "toxcast":
            args.num_tasks = 617
        elif task == "sider":
            args.num_tasks = 27
        elif task == "clintox":
            args.num_tasks = 2
        else:
            args.num_tasks = 1
        '''
        logger = create_file_logger(os.path.join(path, 'log.txt'))
        logger.info("=======Setting=======")
        for k in args.__dict__:
            v = args.__dict__[k]
            logger.info(f"{k}: {v}")
        
        logger.info(f"\nUtilized device as {device}")

        # load data
        logger.info("\n=======Process Data=======")
        #train_set, valid_set, test_set = process_data()
        '''
        device = torch.device('cpu' if args.cpu else ('cuda:' + str(args.gpu)))
        split = torch.load(os.path.join(path, 'split.pt'))
        df = pd.read_csv(os.path.join(path, 'raw.csv'))
        for k, (train_index, test_index) in enumerate(split):
            #mol = Chem.MolFromSmiles(df.loc[int(train_index[28248]), 'smiles'])
            #mol_to_graphs(mol)
            #logger.info(f"train data num: {len(train_set)} | valid data num: {len(valid_set)} | test data num: {len(test_set)}")
            train_set, test_set= process_data(df,train_index), process_data(df,test_index)
            train_set = MoleculeNetDataset(train_set)
            #valid_set = MoleculeNetDataset(valid_set)
            test_set = MoleculeNetDataset(test_set)
            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, follow_batch=['fg_x'],drop_last=True)
            #valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, follow_batch=['fg_x'])
            test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, follow_batch=['fg_x'],drop_last=True)

            # define model
            if args.gnn == 'SerGINE':
                encoder = SerGINE(num_atom_layers=args.num_atom_layers, num_fg_layers=args.num_fg_layers, latent_dim=args.emb_dim,
                                  atom_dim=ATOM_DIM, fg_dim=FG_DIM, bond_dim=BOND_DIM, fg_edge_dim=FG_EDGE_DIM,
                                  dropout=args.dropout)
            # elif args.gnn == :  # more GNN
            else:
                raise ValueError("Undefined GNN!")
            model = MolPredictor(encoder=encoder, latent_dim=args.emb_dim, num_tasks=args.num_tasks, dropout=args.dropout)
            model.to(device)
            optimizer = optim.Adam([{'params': model.encoder.parameters(), 'lr': args.lr0},
                                    {'params': model.predictor.parameters()}], lr=args.lr1, weight_decay=args.weight_decay)
            # load pre-trained model
            if not args.from_scratch:
                #logger.info("\n=======Load Pre-trained Model=======")
                print("\n=======Load Pre-trained Model=======")
                pre_path = args.gnn
                pre_path += '_dim'+str(args.emb_dim) + '_'+args.metric + '_margin'+str(args.margin) + '_lr'+str(args.pre_lr)
                pre_path = os.path.join(args.pretrain_dir, pre_path, args.pretrain_model_name)
                model.from_pretrained(model_path=pre_path, device=device)
                #logger.info(f"Load pre-trained model from {pre_path}")

            best_valid_loss = 0
            for epoch in range(1, args.epochs + 1):
                model.to(device)
                train(model, train_loader, optimizer, device)

                # valid_loss = eval(vocab_datas, raw_smiles_datas, model, device, valid_loader)[0]
                _,test_loss = test(model, test_loader, device)

                if test_loss > best_valid_loss:
                    best_valid_loss = test_loss
                    if args.checkpoint_dir == '':
                        os.makedirs(args.checkpoint_dir, exist_ok=True)
                    if args.checkpoint_dir != '':
                        # print('Saving checkpoint...')
                        checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict()}
                        torch.save(checkpoint, args.checkpoint_dir + task + '_checkpoint.pt')

                print(
                    '***************************************************************************************************')
                print('{}, {}/{} time {}  test_loss{}'.format(task, k + 1, len(split), epoch,
                                                              test_loss))
                print(
                    '***************************************************************************************************')

            # 回归任务计算R2打分
            state_dict = torch.load(args.checkpoint_dir + task + '_checkpoint.pt',map_location=device)
            model.load_state_dict(state_dict['model_state_dict'])
            _, stop_train_list = test(model, train_loader, device)
            # stop_val_list = eval(vocab_datas, raw_smiles_datas, model, device, valid_loader)
            _, stop_test_list = test(model, test_loader, device)

            test_result.append(stop_test_list)

            result_pd['train_' + str(k + 1)] = stop_train_list
            # result_pd['val_' + str(k + 1)] = stop_val_list
            result_pd['test_' + str(k + 1)] = stop_test_list
            print(result_pd)

        time_end = time.time()
        time_task[task] = (time_end - time_start) / len(split)

        test_pd[task] = test_result
        result_pd.to_csv(result_path + '/' + task + '_result.csv', index=False)
    # test_pd = test_pd.append(time_task,ignore_index=True)
    test_pd.loc[len(test_pd)] = list(time_task.values())
    print(test_pd)
    test_pd.to_csv(result_path + '/' + 'class_result.csv', index=False)




if __name__ == '__main__':
    main()
