import argparse
import torch
import os
from graph_dataset.node_edge_feature import *
from graph_dataset.tu_utils import *
import pandas as pd
from networkx import normalized_laplacian_matrix
from rdkit import Chem
import rdkit
# import glob
from pathlib import Path

class Graph2feats():
    def __init__(self, dataset_path , use_node_attrs = True ,use_node_degree= False,use_one=False, classification = True , max_reductions=10 ,precompute_kron_indices = True ):
        self.dataset_path = dataset_path
        self.folder_path = "/".join(self.dataset_path.split("/")[:-1])
        self.raw_dir = Path(os.path.join(self.folder_path,"raw"))
        self.processed_dir = Path(os.path.join(self.folder_path,"processed"))
        self.name = self.dataset_path.split("/")[-1].split(".")[0]
        self.use_node_degree = use_node_degree
        self.use_node_attrs = use_node_attrs
        self.use_one = use_one
        self.classification = classification
        self.KRON_REDUCTIONS = max_reductions  # will compute indices for 10 pooling layers --> approximately 1000 nodes
        self.precompute_kron_indices = precompute_kron_indices



        if not os.path.exists(self.raw_dir):
            os.mkdir(self.raw_dir)
        if not os.path.exists(self.processed_dir):
            os.mkdir(self.processed_dir)


    def df2files(self):
        print("csv->some txt files")
        raw_path = self.raw_dir
        if os.path.exists(raw_path):
            print("已存在{raw_path}目录,不再生成_A,_graph_indicator,_node_labels,.....txt文件")
            return
        fp_edge_index = open(f"{raw_path}/{self.name}_A.txt", "a")
        fp_graph_indicator = open(f"{raw_path}/{self.name}_graph_indicator.txt", "a")
        fp_graph_labels = open(f"{raw_path}/{self.name}_graph_labels.txt", "a")
        fp_node_labels = open(f"{raw_path}/{self.name}_node_labels.txt", "a")
        fp_node_labels = open(f"{raw_path}/{self.name}_node_labels.txt", "a")
        fp_node_attrs = open(f"{raw_path}/{self.name}_node_attributes.txt", "a")
        fp_edge_attrs = open(f"{raw_path}/{self.name}_edge_attributes.txt", "a")
        # r defer close

        df = pd.read_csv(self.dataset_path)
        cnt = 1
        for idx, (smiles, g_label) in enumerate(zip(df.iloc[:, 0], df.iloc[:, 1])):
            mol = Chem.MolFromSmiles(smiles)
            num_nodes = len(mol.GetAtoms())
            dict = {}
            for i, atom in enumerate(mol.GetAtoms()):
                dict[atom.GetIdx()] = cnt + i
                fp_node_labels.writelines(str(atom.GetAtomicNum())+"\n")
                fp_node_attrs.writelines(str(atom_features(atom))[1:-1]+"\n")

            bond_len = len(mol.GetBonds())
            fp_graph_indicator.write(f"{idx+1}\n"*num_nodes)  # node_i to graph id
            fp_graph_labels.write(str(g_label)+"\n")
            for bond in mol.GetBonds():
                node_1 = dict[bond.GetBeginAtomIdx()]
                node_2 = dict[bond.GetEndAtomIdx()]
                fp_edge_index.write(f"{node_1}, {node_2}\n{node_2}, {node_1}\n")
                fp_edge_attrs.write(
                    str([1 if i else 0 for i in bond_features(bond)])[1:-1]+"\n")
                fp_edge_attrs.write(
                    str([1 if i else 0 for i in bond_features(bond)])[1:-1]+"\n")

            cnt += num_nodes
            # print(f"mol:{smiles[:6]}... ok , idx:{idx+1}")
        # print(max_nodes)
        print("Create feats from df OK!")
        fp_graph_labels.close()
        fp_node_labels.close()
        fp_graph_indicator.close()
        fp_edge_index.close()
        fp_node_attrs.close()
        fp_edge_attrs.close()

    def process(self):  # y 处理成.pt 
        self.df2files()
        print("txt -> .pt")

        graphs_data, num_node_labels, num_edge_labels = parse_tu_data(
            self.name , self.raw_dir)
        targets = graphs_data.pop("graph_labels")  # y targets是graph labels

        # dynamically set maximum num nodes (useful if using dense batching, e.g. diffpool)
        max_num_nodes = max([len(v)
                             for (k, v) in graphs_data['graph_nodes'].items()])
        setattr(self, 'max_num_nodes', max_num_nodes)

        dataset = []
        for i, target in enumerate(targets, 1):  # y 下标从1开始取
            # y 取出第i个图的各个数据,如attrs ,v[i] 是list
            graph_data = {k: v[i] for (k, v) in graphs_data.items()}
            G = create_graph_from_tu_data(
                graph_data, target, num_node_labels, num_edge_labels)

            if self.precompute_kron_indices:
                laplacians, v_plus_list = self._precompute_kron_indices(G)
                G.laplacians = laplacians
                G.v_plus = v_plus_list

            if G.number_of_nodes() > 1 and G.number_of_edges() > 0:
                data = self._to_data(G)  # y G是networkx ,转成pyg.data
                dataset.append((data,data.y))

        torch.save(dataset, self.processed_dir / f"{self.name}.pt")

    def _to_data(self, G):
        datadict = {}

        node_features = G.get_x(self.use_node_attrs,
                                self.use_node_degree, self.use_one)
        datadict.update(x=node_features)  # y x是node_feats

        if G.laplacians is not None:
            datadict.update(laplacians=G.laplacians)
            datadict.update(v_plus=G.v_plus)

        edge_index = G.get_edge_index()
        datadict.update(edge_index=edge_index)

        if G.has_edge_attrs:
            edge_attr = G.get_edge_attr()
            datadict.update(edge_attr=edge_attr)

        target = G.get_target(classification=self.classification)  # y target是
        datadict.update(y=target)

        data = Data(**datadict)  # y Data()类调用PyG的data

        return data

    def _precompute_kron_indices(self, G):
        laplacians = []  # laplacian matrices (represented as 1D vectors)
        v_plus_list = []  # reduction matrices

        X = G.get_x(self.use_node_attrs, self.use_node_degree, self.use_one)
        lap = torch.Tensor(normalized_laplacian_matrix(
            G).todense())  # I - D^{-1/2}AD^{-1/2}
        # print(X.shape, lap.shape)

        laplacians.append(lap)

        for _ in range(self.KRON_REDUCTIONS):
            if lap.shape[0] == 1:  # Can't reduce further:
                v_plus, lap = torch.tensor([1]), torch.eye(1)
                # print(lap.shape)
            else:
                v_plus, lap = self._vertex_decimation(lap)
                # print(lap.shape)
                # print(lap)

            laplacians.append(lap.clone())
            v_plus_list.append(v_plus.clone().long())

        return laplacians, v_plus_list

    # For the Perron–Frobenius theorem, if A is > 0 for all ij then the leading eigenvector is > 0
    # A Laplacian matrix is symmetric (=> diagonalizable)
    # and dominant eigenvalue (true in most cases? can we enforce it?)
    # => we have sufficient conditions for power method to converge
    def _power_iteration(self, A, num_simulations=30):
        # Ideally choose a random vector
        # To decrease the chance that our vector
        # Is orthogonal to the eigenvector
        b_k = torch.rand(A.shape[1]).unsqueeze(dim=1) * 0.5 - 1

        for _ in range(num_simulations):
            # calculate the matrix-by-vector product Ab
            b_k1 = torch.mm(A, b_k)

            # calculate the norm
            b_k1_norm = torch.norm(b_k1)

            # re normalize the vector
            b_k = b_k1 / b_k1_norm

        return b_k

    def _vertex_decimation(self, L):

        max_eigenvec = self._power_iteration(L)
        v_plus, v_minus = (max_eigenvec >= 0).squeeze(
        ), (max_eigenvec < 0).squeeze()

        # print(v_plus, v_minus)

        # diagonal matrix, swap v_minus with v_plus not to incur in errors (does not change the matrix)
        if torch.sum(v_plus) == 0.:  # The matrix is diagonal, cannot reduce further
            if torch.sum(v_minus) == 0.:
                assert v_minus.shape[0] == L.shape[0], (v_minus.shape, L.shape)
                # I assumed v_minus should have ones, but this is not necessarily the case. So I added this if
                return torch.ones(v_minus.shape), L
            else:
                return v_minus, L

        L_plus_plus = L[v_plus][:, v_plus]
        L_plus_minus = L[v_plus][:, v_minus]
        L_minus_minus = L[v_minus][:, v_minus]
        L_minus_plus = L[v_minus][:, v_plus]

        L_new = L_plus_plus - \
            torch.mm(torch.mm(L_plus_minus, torch.inverse(
                L_minus_minus)), L_minus_plus)

        return v_plus, L_new


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", type=str,help="graph or MAT or ...")
    parser.add_argument("--dataset_path", type=str)
    return parser.parse_args()


def main():
    args = parse()
    if args.dataset_type == "graph":
        preparer = Graph2feats(args.dataset_path)
        preparer.process()


if __name__ == "__main__":
    main()