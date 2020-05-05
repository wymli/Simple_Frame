from collections import defaultdict
import numpy as np
import networkx as nx

# from .graph import Graph
# from utils.utils import one_hot
import networkx as nx

import torch
from torch_geometric.utils import dense_to_sparse


class Graph(nx.Graph):
    def __init__(self, target, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target = target
        self.laplacians = None
        self.v_plus = None

    def get_edge_index(self):
        adj = torch.Tensor(nx.to_numpy_array(self))
        edge_index, _ = dense_to_sparse(adj)
        return edge_index

    def get_edge_attr(self):
        features = []
        for _, _, edge_attrs in self.edges(data=True):
            data = []

            if edge_attrs["label"] is not None:
                data.extend(edge_attrs["label"])

            if edge_attrs["attrs"] is not None:
                data.extend(edge_attrs["attrs"])

            features.append(data)
        return torch.Tensor(features)

    def get_x(self, use_node_attrs=False, use_node_degree=False, use_one=False):
        features = []
        for node, node_attrs in self.nodes(data=True):
            data = []

            if node_attrs["label"] is not None:  # r attention !
                data.extend(node_attrs["label"])

            if use_node_attrs and node_attrs["attrs"] is not None:
                data.extend(node_attrs["attrs"])

            if use_node_degree:
                data.extend([self.degree(node)])

            if use_one:
                data.extend([1])

            features.append(data)

        return torch.Tensor(features)

    def get_target(self, classification=True):
        if classification:
            return torch.LongTensor([self.target])

        return torch.Tensor([self.target])

    @property
    def has_edge_attrs(self):
        _, _, edge_attrs = list(self.edges(data=True))[0]
        return edge_attrs["attrs"] is not None

    @property
    def has_edge_labels(self):
        _, _, edge_attrs = list(self.edges(data=True))[0]
        return edge_attrs["label"] is not None

    @property
    def has_node_attrs(self):
        _, node_attrs = list(self.node(data=True))[0]
        return node_attrs["attrs"] is not None

    @property
    def has_node_labels(self):
        _, node_attrs = list(self.node(data=True))[0]
        return node_attrs["label"] is not None


def one_hot(value, num_classes):
    vec = np.zeros(num_classes)
    vec[value - 1] = 1
    return vec


def parse_tu_data(name, raw_dir):
    # setup paths
    indicator_path = raw_dir / f'{name}_graph_indicator.txt'
    edges_path = raw_dir / f'{name}_A.txt'
    graph_labels_path = raw_dir / f'{name}_graph_labels.txt'
    node_labels_path = raw_dir / f'{name}_node_labels.txt'
    edge_labels_path = raw_dir / f'{name}_edge_labels.txt'
    node_attrs_path = raw_dir / f'{name}_node_attributes.txt'
    edge_attrs_path = raw_dir / f'{name}_edge_attributes.txt'

    unique_node_labels = set()
    unique_edge_labels = set()

    indicator, edge_indicator = [-1], [(-1, -1)]
    graph_nodes = defaultdict(list)
    graph_edges = defaultdict(list)
    node_labels = defaultdict(list)
    edge_labels = defaultdict(list)
    node_attrs = defaultdict(list)
    edge_attrs = defaultdict(list)

    with open(indicator_path, "r") as f:
        for i, line in enumerate(f.readlines(), 1):
            line = line.rstrip("\n")
            graph_id = int(line)
            indicator.append(graph_id)
            graph_nodes[graph_id].append(i)

    with open(edges_path, "r") as f:
        for i, line in enumerate(f.readlines(), 1):
            line = line.rstrip("\n")
            edge = [int(e) for e in line.split(',')]
            edge_indicator.append(edge)

            # edge[0] is a node id, and it is used to retrieve
            # the corresponding graph id to which it belongs to
            # (see README.txt)
            graph_id = indicator[edge[0]]  # y 每个结点id,即edge[0/1]对应的id都是唯一的!

            graph_edges[graph_id].append(edge)

    if node_labels_path.exists():
        with open(node_labels_path, "r") as f:
            for i, line in enumerate(f.readlines(), 1):
                line = line.rstrip("\n")
                node_label = int(line)
                unique_node_labels.add(node_label)
                graph_id = indicator[i]
                node_labels[graph_id].append(node_label)

    if edge_labels_path.exists():
        with open(edge_labels_path, "r") as f:
            for i, line in enumerate(f.readlines(), 1):
                line = line.rstrip("\n")
                edge_label = int(line)
                unique_edge_labels.add(edge_label)
                graph_id = indicator[edge_indicator[i][0]]
                edge_labels[graph_id].append(edge_label)

    if node_attrs_path.exists():
        with open(node_attrs_path, "r") as f:
            for i, line in enumerate(f.readlines(), 1):
                line = line.rstrip("\n")
                nums = line.split(",")
                node_attr = np.array([float(n) for n in nums])
                graph_id = indicator[i]
                node_attrs[graph_id].append(node_attr)

    if edge_attrs_path.exists():
        with open(edge_attrs_path, "r") as f:
            for i, line in enumerate(f.readlines(), 1):
                line = line.rstrip("\n")
                nums = line.split(",")
                edge_attr = np.array([float(n) for n in nums])
                graph_id = indicator[edge_indicator[i][0]]
                edge_attrs[graph_id].append(edge_attr)

    # get graph labels
    graph_labels = []
    with open(graph_labels_path, "r") as f:
        for i, line in enumerate(f.readlines(), 1):
            line = line.rstrip("\n")
            target = int(line)
            if target == -1:
                graph_labels.append(0)
            else:
                graph_labels.append(target)

        # Shift by one to the left. Apparently this is necessary for multiclass tasks.
        if min(graph_labels) == 1:
            graph_labels = [l - 1 for l in graph_labels]

    num_node_labels = max(
        unique_node_labels) if unique_node_labels != set() else 0
    # some datasets e.g. PROTEINS have labels with value 0
    if num_node_labels != 0 and min(unique_node_labels) == 0:
        num_node_labels += 1

    num_edge_labels = max(
        unique_edge_labels) if unique_edge_labels != set() else 0
    if num_edge_labels != 0 and min(unique_edge_labels) == 0:
        num_edge_labels += 1

    return {
        "graph_nodes": graph_nodes,
        "graph_edges": graph_edges,
        "graph_labels": graph_labels,
        "node_labels": node_labels,
        "node_attrs": node_attrs,
        "edge_labels": edge_labels,
        "edge_attrs": edge_attrs
    }, num_node_labels, num_edge_labels


# y G.add_node 和add_edge的过程 !
def create_graph_from_tu_data(graph_data, target, num_node_labels, num_edge_labels):
    nodes = graph_data["graph_nodes"]
    edges = graph_data["graph_edges"]  # y list

    G = Graph(target=target)

    for i, node in enumerate(nodes):
        label, attrs = None, None

        if graph_data["node_labels"] != []:
            label = one_hot(graph_data["node_labels"][i], num_node_labels)

        if graph_data["node_attrs"] != []:
            attrs = graph_data["node_attrs"][i]

        G.add_node(node, label=label, attrs=attrs)

    for i, edge in enumerate(edges):  # y 遍历某个图的所有边
        # print("here is ", i)
        n1, n2 = edge
        label, attrs = None, None

        if graph_data["edge_labels"] != []:
            label = one_hot(graph_data["edge_labels"][i], num_edge_labels)
        if graph_data["edge_attrs"] != []:
            attrs = graph_data["edge_attrs"][i]

        G.add_edge(n1, n2, label=label, attrs=attrs)

    return G


from torch_geometric import data
class Data(data.Data):
    def __init__(self,
                 x=None,
                 edge_index=None,
                 edge_attr=None,
                 y=None,
                 v_outs=None,
                 e_outs=None,
                 g_outs=None,
                 o_outs=None,
                 laplacians=None,
                 v_plus=None,
                 **kwargs):

        additional_fields = {
            'v_outs': v_outs,
            'e_outs': e_outs,
            'g_outs': g_outs,
            'o_outs': o_outs,
            'laplacians': laplacians,
            'v_plus': v_plus

        }
        super().__init__(x, edge_index, edge_attr, y, **additional_fields)
