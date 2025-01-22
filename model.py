# 这是模型文件，包含图神经网络（DL-GNN）的定义。在这里定义网络结构（例如GCN层）
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class DL_GNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DL_GNN, self).__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, out_channels)
        self.fc = nn.Linear(out_channels, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = torch.mean(x, dim=0)
        x = self.fc(x)
        return x


# https://blog.csdn.net/qq_44426403/article/details/134975686

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Module
from torch.nn import Linear
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import torch
import math


class GraphConvLayer(Module):
    def __init__(self, unit_num, alpha, beta, device) -> None:
        super(GraphConvLayer, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.device = device
        self.W = Parameter(torch.empty(size=(unit_num, unit_num)))
        self.init_param()
        pass

    def init_param(self):
        torch.nn.init.xavier_uniform_(self.W.data, gain=1.414)

    def forward(self, P, H, H0):
        # P : N*N
        # H : N*C
        # H0 : N *F
        # W : C*C
        initial_res_connect = (1 - self.alpha) * torch.mm(P, H) + self.alpha * H0
        I = torch.eye(self.W.shape[0]).to(self.device)
        identity_map = (1 - self.beta) * I + self.beta * self.W
        output = torch.mm(initial_res_connect, identity_map)
        return F.relu(output)


class GCNII(Module):
    def __init__(self, input_dim, hidden_dim, output_dim, k, alpha, lamda, dropout, device):
        super(GCNII, self).__init__()
        self.layer1 = Linear(input_dim, hidden_dim)
        self.layer2 = Linear(hidden_dim, output_dim)
        self.convs = nn.ModuleList()
        self.convs.append(self.layer1)
        self.k = k
        self.dropout = dropout
        # self.layers = []
        for i in range(k):
            beta = math.log(lamda / (i + 1) + 1)
            self.convs.append(GraphConvLayer(hidden_dim, alpha, beta, device))

        self.convs.append(self.layer2)

        self.reg_param = list(self.convs[1:-1].parameters())
        self.non_linear_param = list(self.convs[0:1].parameters()) + list(self.convs[-1:].parameters())
        # 直接在低维变换效果不好，高维变换再映射到低维空间
        # self.layers.append(GraphConvLayer(output_dim,alpha,beta,device))
        # for i,layer in enumerate(self.layers):
        #     self.add_module(f'{i}',layer)


def forward(self, features, adj):
    # H0 = self.layer1(features)
    H0 = F.dropout(features, self.dropout, training=self.training)
    H0 = F.relu(self.convs[0](H0))
    H = H0
    # for layer in self.layers:
    for layer in self.convs[1:-1]:
        H = F.dropout(H, self.dropout, training=self.training)
        H = layer(adj, H, H0)
    H = F.dropout(H, self.dropout, training=self.training)
    output = self.convs[-1](H)
    # output = self.layer2(H)
    return F.log_softmax(output, dim=1)


class GCNII_START(Module):
    def __init__(self, input_dim, hidden_dim, output_dim, k, alpha, lamda, dropout, device):
        super(GCNII_START, self).__init__()
        self.layer1 = Linear(input_dim, hidden_dim)
        self.layer2 = Linear(hidden_dim, output_dim)
        self.dropout = dropout
        self.layers = []

        for i in range(k):
            beta = lamda / (i + 1)
            self.layers.append(GraphConvLayer_START(hidden_dim, alpha, beta, device))
        for i, layer in enumerate(self.layers):
            self.add_module(f'{i}', layer)

    def forward(self, features, adj):
        H0 = F.dropout(features, self.dropout, training=self.training)
        H0 = F.relu(self.layer1(H0))
        H = H0
        for layer in self.layers:
            H = F.dropout(H, self.dropout, training=self.training)
            H = layer(adj, H, H0)
        H = F.dropout(H, self.dropout, training=self.training)
        output = self.layer2(H)
        return F.log_softmax(output, dim=1)
        pass


class GraphConvLayer_START(Module):
    def __init__(self, unit_num, alpha, beta, device):
        super(GraphConvLayer_START, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.device = device
        self.W1 = Parameter(torch.empty(size=(unit_num, unit_num)))
        self.W2 = Parameter(torch.empty(size=(unit_num, unit_num)))
        self.init_param()

    def init_param(self):
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        nn.init.xavier_uniform_(self.W2.data, gain=1.414)

    def forward(self, P, H, H0):
        I = torch.eye(self.W1.shape[0]).to(self.device)
        propagation = torch.mm(P, H)
        initial_res = H0

        identity_map1 = (1 - self.beta) * I + self.beta * self.W1
        identity_map2 = (1 - self.beta) * I + self.beta * self.W2

        output = (1 - self.alpha) * torch.mm(propagation, identity_map1) + self.alpha * torch.mm(initial_res,
                                                                                                 identity_map2)
        return F.relu(output)
