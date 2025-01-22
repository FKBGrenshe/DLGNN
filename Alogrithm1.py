import torch
from torch import nn
from torch.nn import Module
from torch.nn import Linear
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import math
from config import CONFIG


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
        # torch.mm(a, b) 是矩阵a和b矩阵相乘,只适用于二维矩阵
        # torch.matmul可以适用于高维  一维*二维；二维*一维；
        # torch.mul(a, b) 是矩阵 a 和 b 对应位相乘，a 和 b 的维度必须相等
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
            # self.layers.append(GraphConvLayer(hidden_dim,alpha,beta,device))
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


# if __name__ == '__main__':
    # model = GCNII(features.shape[1],
    #               CONFIG.hid_dim,
    #               max(labels).item() + 1,
    #               config.k,
    #               config.alpha,
    #               config.lamda,
    #               config.dropout,
    #               config.device
    #               )
    # model.to(config.device)