import pandas as pd
import rdkit.Chem as Chem

from fragment import *

from rdkit.Chem import Draw
#from rdkit.Chem.Draw import IPythonConsole  # Needed to show molecules
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions  # Only needed if modifying defaults
from rdkit.Chem.Draw import rdMolDraw2D

from rdkit.Chem import FragmentCatalog
import os
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib_venn import venn2
import seaborn as sns

def add_atomID(mol):
    atoms = mol.GetNumAtoms()
    for i in range(atoms):
        mol.GetAtomWithIdx(i).SetProp(
            'molAtomMapNumber', str(mol.GetAtomWithIdx(i).GetIdx())
        )

def draw_mol(mol,filename):
    # 将原子序号加到原子标签中
    opts = DrawingOptions()
    opts.atomLabelFontSize = 250
    #opts.includeAtomNumbers = True
    opts.dblBondLengthFrac = 1
    opts.dotsPerAngstrom = 1200

    if isinstance(mol,list):
        s = Draw.MolsToGridImage(
            mols=mol,  # mol对象
            # filename='F:/bioinfor/hgraph2graph-master/hgraph2graph-master/hgraph/' + name,  # 图片存储地址
            molsPerRow=4,
            subImgSize=(800, 400),
            #returnPNG=False,
            #legends=['' for x in mol]
        )
        #print(type(s))
        s.save(filename)
    else:
        #add_atomID(mol)
        Draw.MolToFile(
            mol=mol,  # mol对象
            size=(1000, 1000),
            options=opts,
            filename = filename
        )


# 得到官能团的mol
def get_group():
    DataDir = 'D:/anaconda/envs/env_protein/Library/share/RDKit/Data'
    func = os.path.join(DataDir,'FunctionalGroups.txt')
    gro = os.path.join(DataDir,'MolStandardize/acid_base_pairs2.txt')
    #print(func,gro)
    fparams = FragmentCatalog.FragCatParams(1, 6, gro)

    fparams.GetNumFuncGroups()
    mols=[]
    for i in range(fparams.GetNumFuncGroups()):
        mol = fparams.GetFuncGroup(i)
        if Chem.SanitizeMol(mol, catchErrors=True) == 0:
            mols.append(mol)
        else:
            print(i, Chem.MolToSmiles(mol))
    draw_mol(mols, 'gro.png')
    return mols

def plt_with_label(g,node_labels,edge_labels):
    #pos = nx.shell_layout(g)
    pos=nx.kamada_kawai_layout(g)
    plt.rcParams['figure.figsize'] = (18, 8)  # 设置画布大小
    nx.draw(g, node_size=800, pos=pos, node_color='r')

    #node_labels = nx.get_node_attributes(g, label)
    nx.draw_networkx_labels(g, pos, labels=node_labels)
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=20)
    #edge_labels = nx.get_edge_attributes(G, 'name')
    #nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.savefig("graph.png")



def draw_frag_mol(m,name):
    #add_atomID(m)
    opts = rdMolDraw2D.MolDrawOptions()
    split_frags, frags = get_frags(m)
    frags = split_frags + frags

    inter_pair = get_inter(m)
    #print(inter_pair)
    #print(split_frags, frags)
    atom_cols = {}
    bond_cols = {}
    #opts.elemDict = defaultdict(lambda: (0, 0, 0))


    for x,y in inter_pair:
        bind_idx = m.GetBondBetweenAtoms(x, y).GetIdx()
        bond_cols[bind_idx] = (0.95, 0.75, 0.1)  # 浅绿

        #opts.atomLabels[2] = 'OH group'

    for i, s in enumerate(frags):
        s = list(s)
        sub_mol = get_sub_mol(m, s)
        for a in sub_mol.GetAtoms():
            mol_idx = a.GetProp('mol_idx')
            #atom_cols[mol_idx] = (0.9, 0.9, 0.9)
            atom = m.GetAtomWithIdx(int(mol_idx))
            atom.SetProp('molAtomMapNumber', str(a.GetIdx()))
            '''
            for n in atom.GetNeighbors():
                if n.GetIdx() in s:
                    bind_idx = m.GetBondBetweenAtoms(int(mol_idx), n.GetIdx()).GetIdx()
                    bond_cols[bind_idx] = (0.9,0.9,0.9)
            '''


    for i, s in enumerate(frags):
        s = list(s)
        #print(at)
        for j in s:
            atom_cols[j] = (0.9, 0.9, 0.9) # 浅紫
            atom = m.GetAtomWithIdx(j)
            for n in atom.GetNeighbors():
                if n.GetIdx() in s:
                    bind_idx = m.GetBondBetweenAtoms(j, n.GetIdx()).GetIdx()
                    #hit_bonds.add(bind_idx)
                    bond_cols[bind_idx] = (0.9, 0.9, 0.9)

    #print(atom_cols)
    #print(bond_cols)

    d = rdMolDraw2D.MolDraw2DCairo(500, 500)

    opts.bondLineWidth = 1
    opts.dblBondLengthFrac = 0.8
    opts.fixedBondLength = 5
    #d.drawOptions().addAtomIndices = True

    #d.SetDrawOptions(opts)
    rdMolDraw2D.PrepareAndDrawMolecule(d, m, highlightAtoms=list(atom_cols.keys()),
                                        highlightAtomColors=atom_cols,
                                        highlightBonds=list(bond_cols.keys()),
                                        highlightBondColors=bond_cols
                                       )

    plt.show()
    d.WriteDrawingText(name)



root = 'J:/hgraph2graph-master/hgraph2graph-master/dataset'
s = 'CCOc1ccc2ccccc2c1C(=O)NC1C(=O)N2C1SC(C)(C)C2C(=O)O'
m = get_mol(s)
draw_frag_mol(m,'decoder_split.png')
#plt.show()




def draw_frags(m, name):
    split_frags, frags = get_frags(m)
    frags = split_frags + frags
    sub_mol_list = [get_sub_mol(m,s) for i, s in enumerate(frags)]
    '''
    for sub_mol in sub_mol_list:
        for atom in sub_mol.GetAtoms():
            atom.SetProp(
            'molAtomMapNumber', atom.GetProp('mol_idx')
        )
    '''
    for sub_mol in sub_mol_list:
        for atom in sub_mol.GetAtoms():
            atom.SetProp(
            'molAtomMapNumber',  str(atom.GetIdx())
        )


    #for atom in sub_mol.GetAtoms():
        #opts.atomLabels[atom.GetIdx()] = atom.GetProp('mol_idx')
    opts = DrawingOptions()
    opts.atomLabelFontSize = 100
    #opts.includeAtomNumbers = True
    opts.dblBondLengthFrac = 1
    opts.dotsPerAngstrom = 500


    s = Draw.MolsToGridImage(
        mols=sub_mol_list,  # mol对象
        # filename='F:/bioinfor/hgraph2graph-master/hgraph2graph-master/hgraph/' + name,  # 图片存储地址
        molsPerRow=4,
        subImgSize=(200, 200),
        #returnPNG=False,
        #legends=[Chem.MolToSmiles(x) for x in sub_mol_list]
    )
    #s.show()
    #print(type(s))
    s.save(name)



def draw_rdmol(mol,name='a.png'):
    # 将原子序号加到原子标签中
    #add_atomID(mol)
    d = rdMolDraw2D.MolDraw2DCairo(500,500)
    #tmp = rdMolDraw2D.PrepareMolForDrawing(mol)
    opts = rdMolDraw2D.MolDrawOptions()
    opts.bondLineWidth = 2
    opts.fixedBondLength = 50
    d.drawOptions().addAtomIndices = True

    d.SetDrawOptions(opts)
    #print(opts.addAtomIndices)

    #mol.GetAtomWithIdx(2).SetProp('atomNote', 'foo')
    #opts.includeAtomNumbers = True
    #mol.GetBondWithIdx(0).SetProp('bondNote', 'bar')


    rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
    #rdMolDraw2D.ContourAndDrawGrid(500,500)

    d.FinishDrawing()
    d.WriteDrawingText(name)


def draw_venn(subset,label):
    venn2(subset,
          set_labels=label,
          set_colors=('r', 'b'))


def draw_heatmap(df):
    sns.heatmap(data=df,
                cmap='Blues',
                vmax=0.5,

                #annot=True,
                #cmap=sns.diverging_palette(10, 220, sep=80, n=7),  # 区分度显著色盘
                )

def get_similar_node(data_df):
    m_pd = pd.DataFrame()
    m_list = []
    for i in range(data_df.shape[0]):
        m = 0
        sim_i = 0
        for j in range(data_df.shape[1] - 1, i, -1):
            if data_df.iloc[i, j] > m:
                m = data_df.iloc[i, j]
                sim_i = j
        m_pd = m_pd.append({'i': i, 'j': sim_i, 'smiles_i': data_df.columns[i],
                            'smiles_j': data_df.columns[sim_i], 'node_similarity': m}, ignore_index=True)
        m_list.append(m)

    # print(data_df.columns,data_df.index)
    print(len(m_pd))
    #print(m_pd)
    m_pd.to_csv(os.path.join(root, task, 'nodes_similarity_max.csv'))
    remove = [i for i ,x in enumerate(m_list) if x > 0.43]
    data_df = data_df.iloc[remove,remove] # 56
    print(m_pd.shape)
    return data_df


#root =  'F:/bioinfor/hgraph2graph-master-0/hgraph2graph-master/data'
#root = 'J:/fragment-based-dgm-master/utils/data/gdb'
root = 'J:/hgraph2graph-master/hgraph2graph-master/dataset/gdb'
#root = '/home/pycao/hgraph2graph-master/hgraph2graph-master/dataset/chembl'
#root = '/home/pycao/hgraph2graph-master/hgraph2graph-master/dataset/dataset/class_task'
task_list =  ['bbbp']
#a = [945, 946, 959, 960, 961, 962]
#b = [41, 127, 133, 192, 205]
'''
for task in task_list:
    #smiles_list = pd.read_csv(os.path.join(root,task, task + '.csv'))['smiles']
    #s = ''C1CC1C[]
    #s = ['c1(ccccc1)C([*])C([*])C([*])','c1(ccccc1)C([*])C([*])','c1(ccccc1)C([*])','c1(ccccc1)([*])','c1(c([*])c([*])c([*])cc1)C([*])C([*])']
    s = ['c1(CC(N2[C@H](CN(CC2)C(=O)C)C[N@]2CC[C@H](O)C2)=O)ccc(N(=O)=O)cc1','c1(ccc(c(c1)Cl)Cl)CC(N1[C@H](CN(CC1)C(=O)OC)CN1CCCC1)=O']

    #s_id = [142,203,210,280,327,426,452,536,718,818,1006,1173,1334,1454,1814,1866,1908,1946]

    # s_id = [21,138,209,406,450,683,836,1023,1156,1173,1437,1453,1667,1946,1947]

    mols = Chem.MolFromSmiles(s[1])

    #mols = [get_mol(smiles_list[i]) for i in s_id]
    draw_rdmol(mols,name='bbbp_hbd_0.png')

    #data_df = pd.read_csv(os.path.join(root,task, task + 'nodes_similarity.csv'),header=0,index_col=0)
    #m_pd = get_similar_node(data_df)
    #draw_heatmap(data_df)

   
    mmpdb_pair = pd.read_csv(os.path.join(root, 'bbbp','mmpdb_pair.csv'), header=0)
    mmpdb_pair = mmpdb_pair[mmpdb_pair['constant_num'] >= 6]
    mmpdb_pair = mmpdb_pair[mmpdb_pair['similarity'] != 1]
    #mmpdb_pair['sim'] =  [MACCSkeys.GenMACCSKeys(get_mol(s_i)), MACCSkeys.GenMACCSKeys(get_mol(s_j)) for i in len(mmpdb_pair)]
    gmmp_pair = pd.read_csv(os.path.join(root, 'bbbp','gmmp_pair.csv'), header=0)
    gmmp_pair = gmmp_pair[gmmp_pair['constant_num'] >= 6]
    gmmp_pair = gmmp_pair[gmmp_pair['similarity'] != 1]

    mmpdb_set = set(zip(mmpdb_pair['id_0'].tolist(),mmpdb_pair['id_1'].tolist()))
    print(len(mmpdb_set))
    gmmp_set = set(zip(gmmp_pair['id_0'].tolist(),gmmp_pair['id_1'].tolist()))
    print(len(gmmp_set))
    draw_venn([mmpdb_set,gmmp_set],['mmpdb','gmmp'])
    plt.savefig(os.path.join(root,task, 'mmp.png'))
    plt.show()

#plt.figure(figsize=(20,20))
#plt.subplots_adjust(left=0.095, bottom=0.08, right=0.96, top=0.98)

f,ax  = plt.subplots(figsize=(12,9))
plt.subplots_adjust(left=0.35, bottom=0.35, right=0.96, top=0.98)
h = draw_heatmap(m_pd)

label_y = ax.get_yticklabels()
plt.setp(label_y,  rotation = 360)
label_x = ax.get_xticklabels()
plt.setp(label_x , rotation = 90)
plt.savefig(os.path.join(root,task, 'nodes_heatmap.png'))
plt.show()




#root = '/home/pycao/hgraph2graph-master/hgraph2graph-master/dataset/dataset/class_task/'




#plt_with_label(g,node_labels,edge_labels)
'''






