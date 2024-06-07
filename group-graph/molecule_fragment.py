#from rdkit.Chem.Draw import SimilarityMaps
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import BRICS
from rdkit.Chem import Recap
from collections import Counter
import numpy as np
import numpy.random as npr
#import pubchempy as pcp
import re

from rdkit.Chem import MACCSkeys
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')




split_smart = ['[#6;X3&+]', '[#6;X3&-]','[#8;X2]', '[#8;-]', '[#8;+]', '[#16;X2]', '[#16;X3&+]', '[#16;X1&-]', '[#7;X3&+0]',
                '[#7;X4&+]', '[#7;X5&2+]',  '[#7;-]', '[#15;X4&+]',
                '[As;X5]', '[As;X3]','[As;X4&+]','[Fe]', '[Al]',
               '[Sn]', '[#0]', '[Mg]', '[Si]', '[Se]', '[F]', '[Cl]', '[Br]', '[I]', '[#5]', '[Ge]', '[Ni]', '[Se]',
               '[Ca]', '[Cu]','[Li]', '[Ru]', '[Co]', '[Pt]','[Ir]', '[Pd]', '[W]',
               '[#6]=[#6]', '[#6]#[#6]', '[#6]=[#7]', '[#6]=[#16]','[#6]#[#7]', '[#6]=[#15]','[#7;X3&+][#8;X1&+0]',
               '[#7]=[#7]', '[#7]#[#7]', '[#7]=[#8]','[#6]=[#8]', '[#16]=[#8]','[#16]=[#16]','[#16]=[#7]', '[#15]=[#8]',
               '[As]=[#16]','[As]=[#8]',
               '[#7](=[#8])(=[#8])', '[#6]=[#7;+]=[#7;-]', '[#7]=[#7;+]=[#7;-]',
               '[#16](=[#8])(=[#8])',  '[#15](=[#8])(=[#8])']

'''
split_smart = ['[#8;X2]', '[#8;-]', '[#16;X2]', '[#16;X3&+]', '[#16;X1&-]', '[#7;X3&+0]', '[#7;X4&+]', '[#6]=[#6]', '[#6]#[#6]', '[#6]=[#7]', '[#6]#[#7]',
               '[#6]=[#15]', '[#7]=[#7]', '[#7]#[#7]', '[#7]=[#8]', '[#7](=[#8])(=[#8])', '[#6](=[#8])', '[#15;X4&+]',
               '[#16](=[#8])(=[#8])', '[#16](=[#8])', '[#16](=[#7])', '[#15](=[#8])', '[#15](=[#8])(=[#8])', '[#6](=[#16])', '[Fe]', '[Al]',
               '[Sn]', '[#0]', '[Mg]', '[Si]', '[Se]', '[#9]', '[#17]', '[#35]', '[#53]', '[#5]']


split_smi_1 = ['[C+]','[C-]','[O]','[S]','[N]','[P+]',
            '[As]', '[As+]','[Fe]', '[Al]',
            '[Sn]', '[#0]', '[Mg]', '[Si]', '[Se]', '[F]', '[Cl]', '[Br]', '[I]', '[#5]', '[Ge]', '[Ni]', '[Se]',
            '[Ca]', '[Cu]','[Li]', '[Ru]', '[Co]', '[Pt]','[Ir]', '[Pd]', '[W]']
split_smi_2 = ['N','O','C=C','C#C','C=N', 'C=O','C=S' ,'C#N','C=P','[N+][O-]','N=N','N#N','N=O','S=O','S=S','S=N','P=O',
            'As=O','O=N=O','[N+]=[N-](=O)','[N+]=[N-]=N','O=S=O','O=P=O']

'''



get_list = lambda a: [j for i in a for j in i]


#mol_list = [Chem.MolFromSmarts(s) for s in split_smi_2]



def list_2_csv(l, file):  # 将列表保存为csv
    with open(file, 'w') as f:
        for i in l:
            f.write(str(i) + '\n')


def csv_2_list(file):
    l = []
    with open(file, 'r') as f:
        for line in f:
            if len(line) != 0:
                l.append(line.strip('\n'))
    return l

def get_mol(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        Chem.Kekulize(mol)
    except:
        mol = None
    return mol


def get_smiles(mol, kekulize=True, isomeric=False):
    return Chem.MolToSmiles(mol, kekuleSmiles=kekulize,isomericSmiles=isomeric)



def sanitize(smiles, kekulize=True, isomeric=False):  # kekulize不为True，拆出的芳香环sanitize错误
    try:
        mol = get_mol(smiles)
        new_smiles = get_smiles(mol,kekulize,isomeric)
        mol = get_mol(new_smiles)
        if mol == None:
            new_smiles = None
    except:
        new_smiles = None
    return new_smiles


def compute_similarity(m0,m1):
    f0 = MACCSkeys.GenMACCSKeys(m0)
    f1 = MACCSkeys.GenMACCSKeys(m1)
    s = DataStructs.FingerprintSimilarity(f0, f1)
    return s


def isRingAromatic(mol, ring):
    '''给定分子（MOL对象）和环中化学键的索引，若所有键都是芳香的，则这个环为芳香环。
    '''
    # print([mol.GetBondWithIdx(index).GetIsAromatic() for index in ring])
    if False in [mol.GetBondWithIdx(index).GetIsAromatic() for index in ring]:
        return False
    else:
        return True


def get_ar_rings(mol):
    '''给定分子(MOL对象）， 返回分子中所有芳香环的原子索引。
       返回一个元素为元组的列表，一个元组是分子中一个芳香环的原子索引。
    '''

    ar_rings = []
    rings = mol.GetRingInfo()

    if rings:
        bond_rings = rings.BondRings()  # 返回每个环键序号
        for i, x in enumerate(bond_rings):
            if isRingAromatic(mol, x):
                ar_rings.append(list(rings.AtomRings()[i]))
                #print(ar_rings)

    # 将共轭的芳香环合并

    c = Counter(get_list(ar_rings))
    a = [i for i, x in c.items() if x > 1]
    if len(a) > 0:
        new_ar_rings = ar_rings
        for i in a:  # 共同元素是a的合并
            dup = set()
            n_dup = set()
            # print(i,new_ar_rings)
            for j in new_ar_rings:  # 更新ar_rings
                if i in j:  # 有相同元素合并
                    dup.update(j)
                else:  # 无则不变
                    n_dup.add(tuple(j))
            n_dup.add(tuple(dup))
            new_ar_rings = n_dup
            # print(new_ar_rings)
    else:
        new_ar_rings = ar_rings

    new_ar_rings = list(new_ar_rings)

    for i, x in enumerate(new_ar_rings): new_ar_rings[i] = tuple(x)  # tuple能作为字典的键，而list不能
    # print(new_ar_rings)

    return new_ar_rings



def get_diff(a, b):
    new_b = []
    for i in a:
        for j in b:
            if not set(j).intersection(set(i)):
                new_b.append(j)
    return tuple(new_b)


def get_split_dict(mol):
    split_dict = {}
    # 芳香环不作为双键识别,将芳香环作为一个断点
    ar_rings = get_ar_rings(mol)

    ar_list = get_list(ar_rings)

    match_ar = set()
    for s in split_smart:
        # print(s)
        match_i = mol.GetSubstructMatches(Chem.MolFromSmarts(s))
        match_s = set()
        for i in match_i:
            # 芳香环可能有重合原子或键
            if not set(i).intersection(set(ar_list)):
                match_s.add(i)
            else:
                match_ar.add(i)
        if match_s:
            split_dict[s] = list(match_s)

    # 将重合的split合并。s=o,o=s=o; p=o,o=p=o;
    if '[#16](=[#8])(=[#8])' in split_dict.keys():
        split_dict['[#16]=[#8]'] = get_diff(split_dict['[#16](=[#8])(=[#8])'], split_dict['[#16]=[#8]'])
    if '[#15](=[#8])(=[#8])' in split_dict.keys():
        split_dict['[#15]=[#8]'] = get_diff(split_dict['[#15](=[#8])(=[#8])'], split_dict['[#15]=[#8]'])
    if '[#7](=[#8])(=[#8])' in split_dict.keys():
        split_dict['[#7]=[#8]'] = get_diff(split_dict['[#7](=[#8])(=[#8])'], split_dict['[#7]=[#8]'])
        split_dict['[#7;X3&+0]'] = get_diff(split_dict['[#7](=[#8])(=[#8])'], split_dict['[#7;X3&+0]'])
    if '[#6]=[#7;+]=[#7;-]' in split_dict.keys():
        split_dict['[#6]=[#7]'] = get_diff(split_dict['[#6]=[#7;+]=[#7;-]'], split_dict['[#6]=[#7]'])
        split_dict['[#7]=[#7]'] = get_diff(split_dict['[#6]=[#7;+]=[#7;-]'], split_dict['[#7]=[#7]'])
        #split_dict['[#7;X4+]'] = get_diff(split_dict['[#6]=[#7;+]=[#7;-]'], split_dict['[#7;X4+]'])
        split_dict['[#7;-]'] = get_diff(split_dict['[#6]=[#7;+]=[#7;-]'], split_dict['[#7;-]'])
    if '[#7]=[#7;+]=[#7;-]' in split_dict.keys():
        split_dict['[#7]=[#7]'] = get_diff(split_dict['[#7]=[#7;+]=[#7;-]'], split_dict['[#7]=[#7]'])
        #split_dict['#7;X4+'] = get_diff(split_dict['[#7]=[#7;+]=[#7;-]'], split_dict['#7;X4+'])
        split_dict['[#7;-]'] = get_diff(split_dict['[#7]=[#7;+]=[#7;-]'], split_dict['[#7;-]'])

    # print(match_ar)
    # 将与芳香环共轭的原子合并
    for m in match_ar:
        for i, ar in enumerate(ar_rings):
            if set(m).intersection(set(ar)):
                ar_rings[i] = tuple(set(m).union(set(ar)))

    # print(ar_rings)
    split_dict['AromaticRing'] = ar_rings
    return split_dict


# mol must be RWMol object # 将提取出来的frag_mol的原子序号设为原来的原子序号

# 如果只有一个原子
def get_sub_mol(mol, sub_atoms):
    Chem.Kekulize(mol, clearAromaticFlags=True)
    new_mol = Chem.RWMol()
    atom_map = {}
    for idx in sub_atoms:
        atom = mol.GetAtomWithIdx(idx)
        atom_map[idx] = new_mol.AddAtom(atom)  # 返回新加原子的序号
        new_atom = new_mol.GetAtomWithIdx(atom_map[idx])
        new_atom.SetUnsignedProp('mol_idx', atom.GetIdx())

    sub_atoms = set(sub_atoms)
    bond_i = -1
    for idx in sub_atoms:
        a = mol.GetAtomWithIdx(idx)
        for b in a.GetNeighbors():
            if b.GetIdx() not in sub_atoms: continue
            bond = mol.GetBondBetweenAtoms(a.GetIdx(), b.GetIdx())
            bt = bond.GetBondType()
            # print(bt)
            add_bond = []
            if a.GetIdx() < b.GetIdx():  # each bond is enumerated twice
                new_mol.AddBond(atom_map[a.GetIdx()], atom_map[b.GetIdx()], bt)
                bond_i = bond_i + 1
                add_bond = new_mol.GetBondWithIdx(bond_i)

                # begin_atom,end_atom = add_bond.GetBeginAtom(),add_bond.GetEndAtom()
                # begin_atom.SetUnsignedProp('mol_idx', a.GetIdx())
                # end_atom.SetUnsignedProp('mol_idx', b.GetIdx())

    #print(Chem.MolToSmiles(new_mol))

    #print(Chem.MolToSmiles(new_mol))
    sub_mol = new_mol.GetMol()
    Chem.Kekulize(sub_mol,True)


    #Chem.SanitizeMol(sub_mol)

    return sub_mol




get_nei_list = lambda idx, mol: [nei.GetIdx() for nei in mol.GetAtomWithIdx(idx).GetNeighbors()]


def get_next_idxs(before_idxs, cur_idxs, mol):
    inter_dict = {}
    for id in cur_idxs:
        next_idx = get_nei_list(id, mol)
        inter = set(next_idx).difference(set(cur_idxs))
        inter = inter.difference(before_idxs)
        if len(inter) > 0:
            inter_dict[id] = inter
    return inter_dict


# 'C=[N+]=[N-]', 'N=[N+]=[N-]'
def get_frags(mol):
    split = get_split_dict(mol)
    # print(split)
    split_set = set(get_list(get_list(split.values())))
    idx_set = set([n for n in range(mol.GetNumAtoms())])
    frag_set = idx_set.difference(set(split_set))

    frags = set()
    before_idxs = {-1}
    split_frags = get_list(split.values())
    # print(split_frags)
    # 所有frag都连在split_frag两边
    for cur_idxs in split_frags:
        next_dict = get_next_idxs(before_idxs, cur_idxs, mol)  # 找到split_frags的下一个节点
        inter = set(get_list(next_dict.values())).difference(split_set)  # 删除split_frag部分,作为连接点

        for next_inter in inter:  # 连接点可能有好几个，分别遍历
            n_inter = {next_inter}
            while True:  # 找到连接点后，循环直到两个split之间的frag全被找出来为止
                next_dict = get_next_idxs(cur_idxs, n_inter, mol)  # 找到连接点的下一个节点
                next_idxs = set(get_list(next_dict.values())).difference(split_set)  # 删除split_frag部分
                # print(next_inter)
                if len(next_idxs) != 0:  # 如果下一个节点有不在split的部分，说明该节点应该归属于frag部分
                    n_inter = n_inter.union(next_idxs)
                else:  # 如果下一个节点全是split，说明所有哦属于该frag的部分都被找到
                    n_inter = tuple(sorted(list(n_inter)))
                    frags.add(n_inter)
                    frag_set = frag_set.difference(set(n_inter))
                    break  # 遍历split_frag下一个连接点

    if len(split_frags) == 0:
        frags = [tuple(idx_set)]
        frag_set = []
    # assert len(frag_set) == 0, print(get_smiles(mol), frag_set, 'undistributed')

    if len(frag_set) > 0:
        frags.add(tuple(frag_set))

    x = get_list(split_frags + list(frags))

    if set(x) != idx_set:
        print(get_smiles(mol), frag_set, 'mol_split error')
    # print(split_frags, list(frags))
    return split_frags, list(frags)



def get_index(a, l):
    index = []
    x, y = a
    b = y, x
    for i, x in enumerate(l):
        if x == a or x == b:
            index.append(i)
    return index


def find_clusters(mol):
    node = get_frags(mol)  # mol_graph 保存的是键，因此原子会被重复，每个原子只属于一个cluster
    cluster = {}
    for i in node[0]: cluster[i] = 'split_frag'
    for j in node[1]: cluster[j] = 'frag'

    atom_cls = [i for i in range(mol.GetNumAtoms())]
    for i in atom_cls:
        for j, x in enumerate(cluster):
            if i in x:
                atom_cls[i] = j
    return cluster, atom_cls


# 找到所有连接对
def get_inter(mol):
    inter_pair = set()  # inter_pair去重
    cluster, atom_cls = find_clusters(mol)
    for i in list(cluster.keys()):
        inter = get_next_idxs({-1}, i, mol)
        for k in inter.items():
            for j in k[1]:
                c = tuple(sorted([k[0], j]))
                inter_pair.add(c)
    inter_pair = list(inter_pair)  # 将集合变为列表

    '''
    for i, j in inter_pair:
        cls_i, cls_j = atom_cls[i], atom_cls[j]
        inter_cls.append(tuple([cls_i, cls_j]))
    '''
    return inter_pair





def atom_equal(a1, a2):
    return a1.GetSymbol() == a2.GetSymbol() and a1.GetFormalCharge() == a2.GetFormalCharge()



