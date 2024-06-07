import torch
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from skimage.io import imread
from cairosvg import svg2png, svg2ps
import os
from torch_geometric.data import DataLoader
from rdkit import Chem
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import random
from collections import Counter,defaultdict

import argparse

import torch
from torch import nn

from torch_geometric.data import Data
from torch.optim.lr_scheduler import StepLR

import numpy as np

import torch.optim as optim

from tqdm import tqdm

from group_graph_encoder import HierGnnEncoder, Predictor, DDIPredictor

from data_loader import SmilesDataset, get_vocab_descriptors, get_vocab_macc

import random
import time

import os
import pandas as pd

import copy

from torch.nn import  functional as F
from gnn import GNN
from sklearn.model_selection import KFold
from sklearn import metrics
import networkx as nx


def train(vocab_datas, raw_smiles_datas,model, device, loader, optimizer, loss_criterion):
    model.train()
    #train_meter = Meter()
    loss_accum = 0
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        pred = model(vocab_datas, batch)
        #pred, _ = model(batch)
        #print(pred.size)
        #print(batch.y)
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

def eval(vocab_datas, raw_smiles_datas, model, device, loader):
    model.eval()
    #eval_meter = Meter()
    y_pred_list = []
    y_true_list = []

    with torch.no_grad():
        for step, batch in enumerate(tqdm(loader, desc="Iteration")):
            batch = batch.to(device)
            pred = model(vocab_datas, batch)

            y_pred_list.append(pred.detach().cpu())
            y_true_list.append(batch.y.view(pred.shape).detach().cpu())
            #eval_meter.update(pred, batch.y.view(pred.shape).to(device), mask=None)
            torch.cuda.empty_cache()

    #y_pred, y_true = eval_meter.compute_metric('return_pred_true')
    y_pred = torch.cat(y_pred_list, dim=0)
    y_true = torch.cat(y_true_list, dim=0)
    #print(y_pred)
    #print(y_pred.size())
    y_true = torch.nan_to_num(y_true.squeeze())
    y_pred = torch.nan_to_num(y_pred.squeeze())


    y_pred_list = torch.sigmoid(y_pred)
    y_pred_label = torch.where(y_pred_list >= 0.5, 1, 0)

    roc_auc = metrics.roc_auc_score(y_true.numpy(), y_pred.numpy())
    prc_auc = metrics.average_precision_score(y_true.numpy(), y_pred.numpy())
    accuracy = metrics.accuracy_score(y_true.numpy(), y_pred_label.numpy())
    #se, sp = sesp_score(y_true.numpy(),y_pred_label.numpy())
    pre, rec, f1, sup = metrics.precision_recall_fscore_support(y_true.numpy(), y_pred_label.numpy())
    #mcc = metrics.matthews_corrcoef(y_true.numpy(), y_pred_label.numpy())
    f1 = f1[1]
    #rec = rec[1]
    #pre = pre[1]
    #err = 1 - accuracy

    #result = [auc,accuracy,se,sp,f1,mcc]
    #result = [auc]
    result = {'roc_auc':roc_auc, 'accuracy':accuracy,'prc_auc':prc_auc, 'f1':f1}
    return result





def similarity(smi1, smi2, mol1, mol2):
    global s1, s2
    d1 = 1 - 0.1 * Levenshtein.distance(smi1, smi2)
    d2 = calculateMCStanimoto(mol1, mol2)[0]

    return max(d1, d2)

def img_for_mol(mol, atom_weights=[]):
    # print(atom_weights)
    highlight_kwargs = {}
    if len(atom_weights) > 0:
        norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
        cmap = cm.get_cmap('bwr', 128)
        '''
        bottom = cm.get_cmap('Oranges_r', 128)
        top = cm.get_cmap('Blues', 128)

        newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                               bottom(np.linspace(0, 1, 128))))
        newcmp = ListedColormap(newcolors, name='OrangeBlue')
        '''
        plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap)

        atom_colors = {
            i: plt_colors.to_rgba(atom_weights[i]) for i in range(len(atom_weights))
        }
        highlight_kwargs = {
            'highlightAtoms': list(range(len(atom_weights))),
            'highlightBonds': [],
            'highlightAtomColors': atom_colors
        }
        # print(highlight_kwargs)


    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(280, 280)
    drawer.SetFontSize(1)

    mol = rdMolDraw2D.PrepareMolForDrawing(mol)
    drawer.DrawMolecule(mol, **highlight_kwargs)
                        # highlightAtoms=list(range(len(atom_weights))),
                        # highlightBonds=[],
                        # highlightAtomColors=atom_colors)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    svg = svg.replace('svg:', '')
    svg2png(bytestring=svg, write_to='tmp.png', dpi=100)
    img = imread('tmp.png')
    os.remove('tmp.png')
    return img



def saliency_map(input_grads):
    # print('saliency_map')
    node_saliency_map = []
    for n in range(input_grads.shape[0]): # nth node
        node_grads = input_grads[n,:]
        node_saliency = torch.norm(F.relu(node_grads)).item()
        node_saliency_map.append(node_saliency)
    return node_saliency_map

def grad_cam(final_conv_acts, final_conv_grads):
    # print('grad_cam')
    node_heat_map = []
    alphas = torch.mean(final_conv_grads, axis=0) # mean gradient for each feature (512x1)
    for n in range(final_conv_acts.shape[0]): # nth node
        node_heat = F.relu(alphas @ final_conv_acts[n]).item()
        node_heat_map.append(node_heat)
    return node_heat_map

def ugrad_cam(final_conv_acts, final_conv_grads):
    # print('new_grad_cam')
    node_heat_map = []
    alphas = torch.mean(final_conv_grads, axis=0) # mean gradient for each feature (512x1)
    for n in range(final_conv_acts.shape[0]): # nth node
        node_heat = (alphas @ final_conv_acts[n]).item()
        node_heat_map.append(node_heat)

    node_heat_map = np.array(node_heat_map).reshape(-1, 1)
    pos_node_heat_map = MinMaxScaler(feature_range=(0,1)).fit_transform(node_heat_map*(node_heat_map >= 0)).reshape(-1,)
    neg_node_heat_map = MinMaxScaler(feature_range=(-1,0)).fit_transform(node_heat_map*(node_heat_map < 0)).reshape(-1,)
    return pos_node_heat_map + neg_node_heat_map

def plot_explanations(model, data, g_list):

    smiles_id = data.smiles_id
    g = g_list[smiles_id]
    #print(g.nodes(data=True))
    smiles = g.graph['smiles']
    #print(smiles)
    mol = g.graph['mol']
    # breakpoint()

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes[0][0].imshow(img_for_mol(mol))
    axes[0][0].set_title(smiles)

    
    axes[0][1].set_title('group graph')
    pos = nx.kamada_kawai_layout(g)
    #plt.rcParams['figure.figsize'] = (18, 8)
    nx.draw(g)
    node_labels = nx.get_node_attributes(g, 'sub_smiles')
    #nx.draw_networkx_labels(g, pos, labels=node_labels)
    axes[0][1].imshow(nx.draw_networkx_labels(g, pos, labels=node_labels))

    #nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=20)
    # edge_labels = nx.get_edge_attributes(G, 'name')
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    axes[0][2].set_title('Feature Matrix')
    axes[0][2].imshow(data.x.cpu().detach().numpy())
    

    axes[1][0].set_title('Saliency Map')
    input_grads = model.input.grad
    saliency_map_weights = saliency_map(input_grads)

    scaled_saliency_map_weights = MinMaxScaler(feature_range=(0, 1)).fit_transform(
        np.array(saliency_map_weights).reshape(-1, 1)).reshape(-1, )
    print(scaled_saliency_map_weights)
    atom_scaled_saliency_map_weights = [0] * mol.GetNumAtoms()
    for i in range(len(scaled_saliency_map_weights)):
        atom_idx = g.nodes[i]['frag_idx']
        #print(g.nodes[i]['sub_smiles'])

        for idx in atom_idx:
            #print(mol.GetAtomWithIdx(idx-1).GetSymbol(),scaled_saliency_map_weights[i])
            atom_scaled_saliency_map_weights[idx-1] = saliency_map_weights[i]


    axes[1][0].imshow(img_for_mol(mol, atom_weights=atom_scaled_saliency_map_weights))

    axes[1][1].set_title('Grad-CAM')
    final_conv_acts = model.final_conv_acts
    final_conv_grads = model.final_conv_grads
    grad_cam_weights = grad_cam(final_conv_acts, final_conv_grads)
    print(np.sum(grad_cam_weights))
    scaled_grad_cam_weights = MinMaxScaler(feature_range=(0, 1)).fit_transform(
        np.array(grad_cam_weights).reshape(-1, 1)).reshape(-1, )

    atom_scaled_grad_cam_weights = [0] * mol.GetNumAtoms()
    for i in range(len(scaled_grad_cam_weights)):
        atom_idx = g.nodes[i]['frag_idx']
        for idx in atom_idx:
            atom_scaled_grad_cam_weights[idx-1] = scaled_grad_cam_weights[i]
    axes[1][1].imshow(img_for_mol(mol, atom_weights=atom_scaled_grad_cam_weights))

    axes[1][2].set_title('UGrad-CAM')
    ugrad_cam_weights = ugrad_cam(final_conv_acts, final_conv_grads)
    atom_ugrad_cam_weights = [0] * mol.GetNumAtoms()
    for i in range(len(ugrad_cam_weights)):
        atom_idx = g.nodes[i]['frag_idx']
        for idx in atom_idx:
            atom_ugrad_cam_weights[idx-1] = ugrad_cam_weights[i]
    axes[1][2].imshow(img_for_mol(mol, atom_weights=atom_ugrad_cam_weights))

    plt.savefig(f'explanations/{smiles_id.item()}.png')
    plt.close('all')


def predict_label(vocab_datas, smiles_list, model, device, loader):
    model.eval()
    # eval_meter = Meter()
    y_pred_list = []
    y_true_list = []
    df = pd.DataFrame(columns=['smiles_id', 'smiles', 'true', 'pred'])
    smiles = []
    smiles_id = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(loader, desc="Iteration")):
            batch = batch.to(device)
            pred = model(vocab_datas, batch)
            y_pred_list.append(pred.detach().cpu())
            y_true_list.append(batch.y.view(pred.shape).detach().cpu())
            # eval_meter.update(pred, batch.y.view(pred.shape).to(device), mask=None)
            torch.cuda.empty_cache()
            batch_s = [smiles_list[i] for i in batch.smiles_id]
            smiles.extend(batch_s)
            smiles_id.extend(batch.smiles_id.tolist())

    # y_pred, y_true = eval_meter.compute_metric('return_pred_true')
    y_pred = torch.cat(y_pred_list, dim=0)
    y_true = torch.cat(y_true_list, dim=0)
    # print(y_pred)
    # print(y_pred.size())
    y_true = torch.nan_to_num(y_true.squeeze())
    y_pred = torch.nan_to_num(y_pred.squeeze())

    y_pred_list = torch.sigmoid(y_pred)
    y_pred_label = torch.where(y_pred_list >= 0.5, 1, 0)

    df['smiles'] = smiles
    df['smiles_id'] = smiles_id
    df['true'] = y_true.squeeze().tolist()
    df['pred'] = y_pred_label.squeeze().tolist()

    return df

def explain():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--times', type=int, default=5,
                        help='repeat times')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='number layers of gnn')
    parser.add_argument('--task_name_list', type=list,
                        default=['hiv'],
                        #default=['bbbp','bace','hiv','clintox','toxcast', 'tox21','sider'],
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

    parser.add_argument('--seed', type=list, default=[42,0,1], help="Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help="Seed for minibatch selection, random initialization.")
    # parser.add_argument('--graph_pooling', type=int, default='mean', help="evaluating training or not")
    parser.add_argument('--log_dir', type=str, default='log/', help="tensorboard log directory")
    #parser.add_argument('--checkpoint_dir', type=str, default='/home/pycao/hgraph2graph-master/hgraph2graph-master/hgraph/check/', help="checkpoint directory")
    parser.add_argument('--checkpoint_dir', type=str, default='check/', help="checkpoint directory")
    parser.add_argument('--pretrain', type=bool, default=False, help='number of workers for dataset loading')
    parser.add_argument('--input_pretrain_file', type=str, default='motif_pretrain_output/motif_pretrain.pt',
                        help='vocab_pretrain directory')
    args = parser.parse_args()

    print(args)


    # args.task_name_list = ['BIOSNAP', 'DrugBankDDI', 'BIOSNAP2']

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    # device = 'cpu'
    root = '/home/pycao/hgraph2graph-master/hgraph2graph-master/dataset/dataset/class_task'
    for task in args.task_name_list:

        path =  os.path.join(root, task)

        g_list = torch.load(os.path.join(path, 'group_graph.pt'))

        vocab_df = pd.read_csv(os.path.join(path, 'vocab.csv'), sep=',', header=0)
        data_df = pd.read_csv(os.path.join(path, 'raw.csv'), sep=',', header=0)

        vocab_list = vocab_df['smiles'].tolist()




        vocab_datas = torch.load(os.path.join(path, 'vocab_descriptor.pt'))
        vocab_size = vocab_datas.shape
        dataset = torch.load(os.path.join(path, 'smiles_data.pt'))
        dataset = dataset['frag_datas']
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
        #result_pd['index'] = ['roc_auc']
        result_pd['index'] = ['roc_auc','accuracy','prc_auc','f1']
        test_result = []
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        print("loading model")
        pretrain_file = None
        encoder = HierGnnEncoder(vocab_size, num_tasks, device, pretrain_file, args)

        model = Predictor(encoder=encoder, latent_dim=args.embed_size, num_tasks=num_tasks,
                          )

        state_dict = torch.load(args.checkpoint_dir + task + '_checkpoint.pt')
        model.load_state_dict(state_dict['model_state_dict'])
        # weigth_init(model)
        stop_all_list = eval(vocab_datas, raw_smiles_datas, model, device, loader)
        print(stop_all_list)

        df = predict_label(vocab_datas, data_df['smiles'].tolist(), model, device, loader)
        for g in g_list:
            #print(df[df['smiles_id'] == g.graph['smiles_id']]['pred'].item())
            g.graph['predicted_label'] = df[df['smiles_id'] == g.graph['smiles_id']]['pred'].item()

        auc = metrics.roc_auc_score(df['true'].tolist(), df['pred'].tolist())
        prc_auc = metrics.average_precision_score(df['true'].tolist(), df['pred'].tolist())
        print(auc,prc_auc)

        model.train()
        for m in model.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.eval()


        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        total_loss = 0


        for data in tqdm(loader):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(vocab_datas, data)
            loss = torch.nn.BCEWithLogitsLoss(reduction='none')(out, data.y.view(out.shape)).mean()
            loss.backward()

            final_conv_acts = model.encoder.graph_gnn.final_conv_acts

            final_conv_grads = model.encoder.graph_gnn.final_conv_grads


            batch = data.batch
            smiles_id = data.smiles_id
            for a, s_id in enumerate(smiles_id):
                #print(s_id)
                node_i = torch.where(batch == a)
                final_conv_acts_i = final_conv_acts[node_i[0], :]
                final_conv_grads_i = final_conv_grads[node_i[0], :]
                grad_cam_weights = ugrad_cam(final_conv_acts_i, final_conv_grads_i)
                #print(grad_cam_weights)
                for i, w in enumerate(grad_cam_weights):
                    g_list[s_id].nodes[i]['grad_cam_weight'] = w


            total_loss += loss.item()
        torch.save(g_list, path + '/gg_grad_cam.pt')
        return g_list


from rdkit.Chem import rdFMCS
def calculateMCStanimoto(ref_mol, target_mol):
    numAtomsRefCpd = float(ref_mol.GetNumAtoms())
    numAtomsTargetCpd = float(target_mol.GetNumAtoms())

    if numAtomsRefCpd < numAtomsTargetCpd:
        leastNumAtms = int(numAtomsRefCpd)
    else:
        leastNumAtms = int(numAtomsTargetCpd)

    pair_of_molecules = [ref_mol, target_mol]
    numCommonAtoms = rdFMCS.FindMCS(pair_of_molecules,
                                    atomCompare=rdFMCS.AtomCompare.CompareElements,
                                    bondCompare=rdFMCS.BondCompare.CompareOrderExact, matchValences=True).numAtoms
    mcsTanimoto = numCommonAtoms / ((numAtomsTargetCpd + numAtomsRefCpd) - numCommonAtoms)

    return mcsTanimoto, leastNumAtms


# Calculate the similarity of two molecules (with SMILE representations smi1 and smi2)
#  This is the maximum of the two functions above
import Levenshtein
def similarity(smi1, smi2, mol1, mol2):
    global s1, s2
    d1 = 1 - 0.1 * Levenshtein.distance(smi1, smi2)
    d2 = calculateMCStanimoto(mol1, mol2)[0]

    return max(d1, d2)


def compare_graphs(g_list):
    res_df = pd.DataFrame(
        columns=['smiles_0', 'id_0', 'label_true_predict_0', 'smiles_1', 'id_1', 'label_true_predict_1', 'diff_0',
                 'diff_0', 'grad_cam_0', 'grad_cam_1', 'sorted_0', 'sorted_1', 'similarity', 'node_sum'])
    for i in tqdm(range(len(g_list))):
        gi = g_list[i]
        for j in range(len(g_list) - 1, i, -1):
            gj = g_list[j]
            smiles_id_0, smiles_id_1 = i, j
            # 只比较节点数相同的图
            label_0 = [attr['sub_smiles'] for n, attr in gi.nodes(data=True)]
            weight_0 = sorted([attr['grad_cam_weight'] for n, attr in gi.nodes(data=True)])

            dict_0 = dict(Counter(label_0))

            label_1 = [attr['sub_smiles'] for n, attr in gj.nodes(data=True)]
            weight_1 = sorted([attr['grad_cam_weight'] for n, attr in gj.nodes(data=True)])
            dict_1 = dict(Counter(label_1))
            # diff_counter = tuple(set(dict_0.items()).symmetric_difference(set(dict_1.items())))
            diff_0 = tuple(set(dict_0.items()) - set(dict_1.items()))
            diff_1 = tuple(set(dict_1.items()) - set(dict_0.items()))
            # diff = tuple(set(label_0).symmetric_difference(set(label_1)))
            # len(diff) <= 2
            if gi.number_of_nodes() == gj.number_of_nodes() and gi.number_of_nodes() > 1 and len(diff_0) == len(
                    diff_1) == 1:

                sim = similarity(gi.graph['smiles'], gj.graph['smiles'], gi.graph['mol'], gj.graph['mol'])

                node_i, node_j = label_0.index(diff_0[0][0]), label_1.index(diff_1[0][0])
                sorted_i = weight_0.index(gi.nodes[node_i]['grad_cam_weight'])
                sorted_j = weight_1.index(gj.nodes[node_j]['grad_cam_weight'])
                weight_0 = [format(n, '.4f') for n in weight_0]
                weight_1 = [format(n, '.4f') for n in weight_1]
                res_df.loc[len(res_df)] = [gi.graph['smiles'], gi.graph['smiles_id'],
                                           [gi.graph['label'].item(), gi.graph['predicted_label']], gj.graph['smiles'],
                                           gj.graph['smiles_id'],
                                           [gj.graph['label'].item(), gj.graph['predicted_label']], diff_0, diff_1,
                                           weight_0, weight_1, sorted_i, sorted_j, sim, gi.number_of_nodes()]

    return res_df








if __name__ == "__main__":

    root = '/home/pycao/hgraph2graph-master/hgraph2graph-master/dataset/dataset/class_task'

    task_name_list = ['hiv']
    # root = 'J:/hgraph2graph-master/hgraph2graph-master/dataset/dataset/class_task'
    for task in task_name_list:
        # path = os.path.join(root, task)
        path = os.path.join(root, task)

        g_list = explain()
        res_df = compare_graphs(g_list)
        res_df.to_csv(path + '/gg_compare_result.csv')
        print(res_df)
