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
