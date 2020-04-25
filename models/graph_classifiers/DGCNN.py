import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing, global_sort_pool
from torch_geometric.utils import add_self_loops, degree



#b 整体架构就是先经过图卷积,得到输出batch,total_latent_dim,其输出是所有的中间层输出cat之后再global_sort_pool,所以feat数是所有feat之和,并取前n个结点 
#b 然后进入Conv1d,要先对输出unsqueeze,因为一维卷积的输入是N,C,L 
#b 然后进入fc 
class DGCNN(nn.Module):
    """
    Uses fixed architecture
    """

    def __init__(self, dim_features, dim_target, config):
        super(DGCNN, self).__init__()

        self.ks = {'NCI1': { '0.6': 30, '0.9': 46 },
                   'PROTEINS_full': { '0.6': 32, '0.9': 81 },
                   'DD': {'0.6': 291, '0.9': 503 },
                   'ENZYMES': { '0.6': 36, '0.9': 48 },
                   'IMDB-BINARY': { '0.6': 18, '0.9': 31 },
                   'IMDB-MULTI': { '0.6': 11, '0.9': 22 },
                   'REDDIT-BINARY': { '0.6': 370, '0.9': 1002 },
                   'REDDIT-MULTI-5K': { '0.6': 469, '0.9': 1081 },
                   'COLLAB': { '0.6': 61, '0.9': 130 },
                   "bbbp":{"0.6": 79 , "0.9": 120 }
                   }

        self.k = self.ks[config.dataset.name][str(config['k'])] #w k 是global_sort_pool的参数k,代表其sort得到的前k个node 
        self.embedding_dim = config['embedding_dim']
        self.num_layers = config['num_layers']

        self.convs = []
        for layer in range(self.num_layers):
            input_dim = dim_features if layer == 0 else self.embedding_dim
            self.convs.append(DGCNNConv(input_dim, self.embedding_dim))
        self.total_latent_dim = self.num_layers * self.embedding_dim

        # Add last embedding
        self.convs.append(DGCNNConv(self.embedding_dim, 1))
        self.total_latent_dim += 1

        self.convs = nn.ModuleList(self.convs)

        # should we leave this fixed?#g total_latent_dim就是每个node的feat数 ,conv1d_params1的输出就是batch,1,k 
        self.conv1d_params1 = nn.Conv1d(1, 16, self.total_latent_dim, self.total_latent_dim)
        self.maxpool1d = nn.MaxPool1d(2, 2)
        self.conv1d_params2 = nn.Conv1d(16, 32, 5, 1)

        dense_dim = int((self.k - 2) / 2 + 1)#y 经过maxpool,以2为k_size,stride
        self.input_dense_dim = (dense_dim - 5 + 1) * 32 #y dense_dim是 conv1d_params2的输入size 

        self.hidden_dense_dim = config['dense_dim']
        self.dense_layer = nn.Sequential(nn.Linear(self.input_dense_dim, self.hidden_dense_dim),
                                         nn.ReLU(),
                                         nn.Dropout(p=0.5),
                                         nn.Linear(self.hidden_dense_dim, dim_target))

    def forward(self, data):
        # Implement Equation 4.2 of the paper i.e. concat all layers' graph representations and apply linear model
        # note: this can be decomposed in one smaller linear model per layer
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # print("in DGCNN:",edge_index.shape)
        hidden_repres = [] # w 存储中间的输出 

        for conv in self.convs:
            x = torch.tanh(conv(x, edge_index))
            hidden_repres.append(x)

        # apply sortpool
        x_to_sortpool = torch.cat(hidden_repres, dim=1)
        x_1d = global_sort_pool(x_to_sortpool, batch, self.k)  # in the code the authors sort the last channel only
        #w x_id 的shape: b, k*f 
        # apply 1D convolutional layers
        x_1d = torch.unsqueeze(x_1d, dim=1) #w x_id 的shape: b,1, k*f ,nn.Conv1d的输入是N,Cin,signal,也就是Cin个平面,f=total_latent_dim 
        conv1d_res = F.relu(self.conv1d_params1(x_1d))
        conv1d_res = self.maxpool1d(conv1d_res)
        conv1d_res = F.relu(self.conv1d_params2(conv1d_res))
        conv1d_res = conv1d_res.reshape(conv1d_res.shape[0], -1) #w 把平面去点,得到b,32*...

        # apply dense layer
        out_dense = self.dense_layer(conv1d_res)
        return out_dense


class DGCNNConv(MessagePassing):
    """
    Extended from tuorial on GCNs of Pytorch Geometrics
    """

    def __init__(self, in_channels, out_channels):
        super(DGCNNConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # print("in DGCNN unit:",edge_index.shape)
        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3-5: Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x ,e=edge_index,s = (x.size(0), x.size(0)))

    def message(self, x_j, e, s):#r fixed  
        # x_j has shape [E, out_channels]
        edge_index = e
        size = s
        # Step 3: Normalize node features.
        src, dst = edge_index  # we assume source_to_target message passing
        deg = degree(src, size[0], dtype=x_j.dtype)
        deg = deg.pow(-1)
        norm = deg[dst]

        return norm.view(-1, 1) * x_j  # broadcasting the normalization term to all out_channels === hidden features

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]

        # Step 5: Return new node embeddings.
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
