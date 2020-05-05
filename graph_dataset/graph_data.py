from torch_geometric.data import Data
import torch_geometric as pyg
import pandas as pd
# torch.utils.data.TensorDataset()
# torchvision.datasets.ImageFolder()
# torch.utils.data.DataLoader()
import rdkit
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from graph_dataset.node_edge_feature import *

from rdkit import Chem
import os
# print(rdkit.__version__)

# fp =  open("log.txt" , "a")
# # sys.stdout = fp , 或者sys.stdout = logger ,自定义logger,应该是实现write方法就行
# df = pd.read_csv("bbbp.csv" )
# print(df.columns)
# # print(df.sample(10))
# print(df["p_np"].describe())
# print(df["p_np"].value_counts() )
# fp.close()



# def create_feats_from_df(df, dataset_path):
#     raw_path = dataset_path+"bbbp/raw"
#     if not os.path.exists(raw_path):
#         os.mkdir(raw_path)

#     fp_edge_index = open(f"{raw_path}/bbbp_A.txt", "a")
#     fp_graph_indicator = open(f"{raw_path}/bbbp_graph_indicator.txt", "a")
#     fp_graph_labels = open(f"{raw_path}/bbbp_graph_labels.txt", "a")
#     fp_node_labels = open(f"{raw_path}/bbbp_node_labels.txt", "a")
#     fp_node_labels = open(f"{raw_path}/bbbp_node_labels.txt", "a")
#     fp_node_attrs = open(f"{raw_path}/bbbp_node_attributes.txt", "a")
#     fp_edge_attrs = open(f"{raw_path}/bbbp_edge_attributes.txt", "a")

#     # r defer close

#     # df = pd.read_csv("bbbp.csv")
#     cnt = 1
#     # max_nodes = 0
#     for idx, (smiles, g_label) in enumerate(zip(df.iloc[:, 0], df.iloc[:, 1])):
#         mol = Chem.MolFromSmiles(smiles)
#         num_nodes = len(mol.GetAtoms())
#         # if num_nodes > max_nodes :
#         #     max_nodes = num_nodes

#         dict = {}
#         for i, atom in enumerate(mol.GetAtoms()):
#             dict[atom.GetIdx()] = cnt + i
#             fp_node_labels.writelines(str(atom.GetAtomicNum())+"\n")
#             fp_node_attrs.writelines(str(atom_features(atom))[1:-1]+"\n")

#         bond_len = len(mol.GetBonds())
#         fp_graph_indicator.write(f"{idx+1}\n"*num_nodes)  # node_i to graph id
#         fp_graph_labels.write(str(g_label)+"\n")
#         for bond in mol.GetBonds():
#             node_1 = dict[bond.GetBeginAtomIdx()]
#             node_2 = dict[bond.GetEndAtomIdx()]
#             fp_edge_index.write(f"{node_1}, {node_2}\n{node_2}, {node_1}\n")
#             fp_edge_attrs.write(
#                 str([1 if i else 0 for i in bond_features(bond)])[1:-1]+"\n")
#             fp_edge_attrs.write(
#                 str([1 if i else 0 for i in bond_features(bond)])[1:-1]+"\n")

#         cnt += num_nodes
#         print(f"mol:{smiles[:6]}... ok , idx:{idx+1}")
#     # print(max_nodes)
#     print("Create feats from df OK!")

#     fp_graph_labels.close()
#     fp_node_labels.close()
#     fp_graph_indicator.close()
#     fp_edge_index.close()
#     fp_node_attrs.close()
#     fp_edge_attrs.close()


# def create_dataset(dataset_path, dataset_name):
#     dataset = TUDataset(root=dataset_path, name=dataset_name)
#     return dataset


# def create_dataloader(dataset, batch_size=1, shuffle=True):
#     dataloader = DataLoader(dataset, batch_size, shuffle)
#     return dataloader


def smiles_to_graphData(smiles: str, g_label: int):
    mol = Chem.MolFromSmiles(smiles)
    num_nodes = len(mol.GetAtoms())

    x = []
    for i, atom in enumerate(mol.GetAtoms()):
        node_label = atom.GetAtomicNum()
        node_attr = atom_features(atom)
        x.append(node_attr.append(node_label))

    edge_index = []
    edge_attrs = []
    for bond in mol.GetBonds():
        node_1 = bond.GetBeginAtomIdx()
        node_2 = bond.GetEndAtomIdx()

        edge_index.append((node_1, node_2))
        edge_index.append((node_2, node_1))

        edge_attr = [1 if i else 0 for i in bond_features(bond)]
        edge_attrs.append(edge_attr)
        edge_attrs.append(edge_attr)

    return Data(x, edge_index, edge_attrs, y=g_label)


def load_data_from_df(df, train_idxs, test_idxs,label_index):
    '''
    return DataList
    '''
    dataList = []
    for _, smiles, labels in df.itertuples():
        label = labels[label_index]
        if label == pd.nan:
            continue
        data = smiles_to_graphData(smiles,label)
        dataList.append((data, label))
    return dataList[train_idxs], dataList[test_idxs]


def construct_dataloader(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size, shuffle)
