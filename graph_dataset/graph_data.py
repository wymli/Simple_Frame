import numpy as np
from torch.utils.data import sampler
from torch_geometric.data import Data
import torch_geometric as pyg
import pandas as pd
# torch.utils.data.TensorDataset()
# torchvision.datasets.ImageFolder()
# torch.utils.data.DataLoader()
import rdkit
from torch_geometric.datasets import TUDataset
# from torch_geometric.data import DataLoader
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


# def smiles_to_graphData(smiles: str, g_label: int):
#     mol = Chem.MolFromSmiles(smiles)
#     num_nodes = len(mol.GetAtoms())

#     x = []
#     for i, atom in enumerate(mol.GetAtoms()):
#         node_label = atom.GetAtomicNum()
#         node_attr = atom_features(atom)
#         x.append(node_attr.append(node_label))

#     edge_index = []
#     edge_attrs = []
#     for bond in mol.GetBonds():
#         node_1 = bond.GetBeginAtomIdx()
#         node_2 = bond.GetEndAtomIdx()

#         edge_index.append((node_1, node_2))
#         edge_index.append((node_2, node_1))

#         edge_attr = [1 if i else 0 for i in bond_features(bond)]
#         edge_attrs.append(edge_attr)
#         edge_attrs.append(edge_attr)

#     return Data(x, edge_index, edge_attrs, y=g_label)


# def load_data_from_df(df, train_idxs, test_idxs,label_index):
#     '''
#     return DataList
#     '''
#     dataList = []
#     for _, smiles, labels in df.itertuples():
#         label = labels[label_index]
#         if label == pd.nan:
#             continue
#         data = smiles_to_graphData(smiles,label)
#         dataList.append((data, label))
#     return dataList[train_idxs], dataList[test_idxs]


# def construct_dataloader(dataset, batch_size, shuffle):
#     return DataLoader(dataset, batch_size, shuffle)


# original dataloader from gnn-comparison
from torch_geometric import data
import torch


class Batch(data.Batch):
    @staticmethod
    def from_data_list(data_list, follow_batch=[]):
        laplacians = None
        v_plus = None

        if 'laplacians' in data_list[0]:
            laplacians = [d.laplacians[:] for d in data_list]
            v_plus = [d.v_plus[:] for d in data_list]

        copy_data = []
        for d in data_list:
            copy_data.append(Data(x=d.x,
                                  y=d.y,
                                  edge_index=d.edge_index,
                                  edge_attr=d.edge_attr,
                                  v_outs=d.v_outs,
                                  g_outs=d.g_outs,
                                  e_outs=d.e_outs,
                                  o_outs=d.o_outs)
                             )

        batch = data.Batch.from_data_list(copy_data, follow_batch=follow_batch)
        batch['laplacians'] = laplacians
        batch['v_plus'] = v_plus

        return batch


class DataLoader(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.
    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (list or tuple, optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`[]`)
    """

    def __init__(self,
                 dataset,
                 batch_size=1,
                 shuffle=False,
                 follow_batch=[],
                 **kwargs):
        super(DataLoader, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: Batch.from_data_list(
                data_list, follow_batch),
            **kwargs)


class RandomSampler(sampler.RandomSampler):
    """
    This sampler saves the random permutation applied to the training data,
    so it is available for further use (e.g. for saving).
    The permutation is saved in the 'permutation' attribute.
    The DataLoader can now be instantiated as follows:

    >>> data = Dataset()
    >>> dataloader = DataLoader(dataset=data, batch_size=32, shuffle=False, sampler=RandomSampler(data))
    >>> for batch in dataloader:
    >>>     print(batch)
    >>> print(dataloader.sampler.permutation)

    For convenience, one can create a method in the dataloader class to access the random permutation directly, e.g:

    class MyDataLoader(DataLoader):
        ...
        def get_permutation(self):
            return self.sampler.permutation
        ...
    """

    def __init__(self, data_source, num_samples=None, replacement=False):
        super().__init__(data_source, replacement=replacement, num_samples=num_samples)
        self.permutation = None

    def __iter__(self):
        n = len(self.data_source)
        self.permutation = torch.randperm(n).tolist()
        return iter(self.permutation)


def construct_dataloader(trainset, testset, batch_size=1, shuffle=True):
    sampler = RandomSampler(trainset) if shuffle is True else None

    # 'shuffle' needs to be set to False when instantiating the DataLoader,
    # because pytorch  does not allow to use a custom sampler with shuffle=True.
    # Since our shuffler is a random shuffler, either one wants to do shuffling
    # (in which case he should instantiate the sampler and set shuffle=False in the
    # DataLoader) or he does not (in which case he should set sampler=None
    # and shuffle=False when instantiating the DataLoader)

    return DataLoader(trainset,
                      batch_size=batch_size,
                      sampler=sampler,
                      shuffle=False,  # if shuffle is not None, must stay false, ow is shuffle is false
                      pin_memory=True), DataLoader(testset,
                                                   batch_size=batch_size,
                                                   sampler=sampler,
                                                   shuffle=False,  # if shuffle is not None, must stay false, ow is shuffle is false
                                                   pin_memory=True)


# ---------------------------------------------------------------------------------------------
# class GraphDataset:
#     def __init__(self, data):
#         self.data = data

#     def __getitem__(self, index):
#         return self.data[index]

#     def __len__(self):
#         return len(self.data)

#     def get_targets(self):
#         targets = [d.y.item() for d in self.data]
#         return np.array(targets)

#     def get_data(self):
#         return self.data

#     def augment(self, v_outs=None, e_outs=None, g_outs=None, o_outs=None):
#         """
#         v_outs must have shape |G|x|V_g| x L x ? x ...
#         e_outs must have shape |G|x|E_g| x L x ? x ...
#         g_outs must have shape |G| x L x ? x ...
#         o_outs has arbitrary shape, it is a handle for saving extra things
#         where    L = |prev_outputs_to_consider|.
#         The graph order in which these are saved i.e. first axis, should reflect the ones in which
#         they are saved in the original dataset.
#         :param v_outs:
#         :param e_outs:
#         :param g_outs:
#         :param o_outs:
#         :return:
#         """
#         for index in range(len(self)):
#             if v_outs is not None:
#                 self[index].v_outs = v_outs[index]
#             if e_outs is not None:
#                 self[index].e_outs = e_outs[index]
#             if g_outs is not None:
#                 self[index].g_outs = g_outs[index]
#             if o_outs is not None:
#                 self[index].o_outs = o_outs[index]

# class GraphDatasetSubset(GraphDataset):
#     """
#     Subsets the dataset according to a list of indices.
#     """

#     def __init__(self, data, indices):
#         self.data = data
#         self.indices = indices

#     def __getitem__(self, index):
#         return self.data[self.indices[index]]

#     def __len__(self):
#         return len(self.indices)

#     def get_targets(self):
#         targets = [self.data[i].y.item() for i in self.indices]
#         return np.array(targets)

def load_data_from_pt(dataset_path, train_idxs, test_idxs, label_index):
    # dataset = GraphDataset(torch.load(dataset_path))
    dataset = torch.load(dataset_path)
    trainset, testset = dataset[train_idxs], dataset[test_idxs]
    # 或者有更高效的做法
    trainset = [(feat, label) for (feat, label)
                in trainset if label[label_index] != None]  # or nan
    testset = [(feat, label) for (feat, label)
               in testset if label[label_index] != None]  # or nan
    return trainset, testset
