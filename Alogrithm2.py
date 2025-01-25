import torch

from Alogrithm1 import *
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNIIGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_gcn_layers, alpha, lamda, dropout, device):
        """
        单层 GCNII-GRU 单元
        Args:
        - input_dim: 输入特征维度
        - hidden_dim: 隐藏层特征维度
        - num_gcn_layers: GCNII 中的层数
        - alpha: 初始残差强度
        - lamda: 自适应残差参数
        - dropout: Dropout 比例
        - device: 设备 (CPU/GPU)
        """
        super(GCNIIGRUCell, self).__init__()
        self.device = device

        # 更新门 z 的 GCNII 模块
        self.gcn_z_x = GCNII(input_dim, hidden_dim, hidden_dim, num_gcn_layers, alpha, lamda, dropout, device)
        self.gcn_z_h = GCNII(hidden_dim, hidden_dim, hidden_dim, num_gcn_layers, alpha, lamda, dropout, device)

        # 重置门 r 的 GCNII 模块
        self.gcn_r_x = GCNII(input_dim, hidden_dim, hidden_dim, num_gcn_layers, alpha, lamda, dropout, device)
        self.gcn_r_h = GCNII(hidden_dim, hidden_dim, hidden_dim, num_gcn_layers, alpha, lamda, dropout, device)

        # 候选隐藏状态 h~ 的 GCNII 模块
        self.gcn_h_x = GCNII(input_dim, hidden_dim, hidden_dim, num_gcn_layers, alpha, lamda, dropout, device)
        self.gcn_h_h = GCNII(hidden_dim, hidden_dim, hidden_dim, num_gcn_layers, alpha, lamda, dropout, device)

    def forward(self, X_f, A_f, H_prev):
        """
        Args:
        - X_f: 当前时间步的输入特征矩阵 (N, C)
        - A_f: 当前时间步的邻接矩阵 (N, N)
        - H_prev: 上一时间步的隐藏状态 (N, C)

        Returns:
        - H_new: 当前时间步的隐藏状态 (N, C)
        """
        # 更新门 z
        Z = torch.relu(self.gcn_z_x(X_f, A_f) + self.gcn_z_h(H_prev, A_f))

        # 重置门 r
        R = torch.relu(self.gcn_r_x(X_f, A_f) + self.gcn_r_h(H_prev, A_f))

        # 候选隐藏状态 h~
        H_tilde = torch.tanh(self.gcn_h_x(X_f, A_f) + self.gcn_h_h(R * H_prev, A_f))

        # 最终隐藏状态 H_f
        H_new = Z * H_prev + (1 - Z) * H_tilde
        return H_new



class GCNIIGRU(Module):
    def __init__(self, input_dim, hidden_dim, num_gcn_layers, alpha, lamda, dropout, device, num_gru_layers=2):
        """
        堆叠多层 GCNII-GRU 单元
        Args:
        - input_dim: 输入特征维度
        - hidden_dim: 隐藏层特征维度
        - num_gcn_layers: 每个 GCNII 单元中的层数
        - alpha: 初始残差强度
        - lamda: 自适应残差参数
        - dropout: Dropout 比例
        - device: 设备 (CPU/GPU)
        - num_gru_layers: 堆叠的 GRU 层数
        """
        super(GCNIIGRU, self).__init__()
        self.num_gru_layers = num_gru_layers

        # 堆叠多层 GCNII-GRU
        self.gru_cells = nn.ModuleList([
            # GCNIIGRUCell(input_dim if i == 0 else hidden_dim, hidden_dim, num_gcn_layers, alpha, lamda, dropout, device)
            GCNIIGRUCell( hidden_dim, hidden_dim, num_gcn_layers, alpha, lamda, dropout, device)
            for i in range(num_gru_layers)
        ])

    def forward(self, features_seq, adj_seq):
        """
        Args:
        - features_seq: 输入特征序列 (k, N, C)
        - adj_seq: 邻接矩阵序列 (k, N, N)

        Returns:
        - hidden_final: 最终隐藏状态 (N, C)
        """
        k, N, C = features_seq.shape
        hidden = torch.zeros((N, C), device=features_seq.device)  # 初始化隐藏状态

        # 保存所有时间步的隐藏状态
        H_all = []

        for t in range(k):
            features_t = features_seq[t]
            adj_t = adj_seq[t]

            # 通过每一层 GRU 单元更新隐藏状态
            for cell in self.gru_cells:
                hidden = cell(features_t, adj_t, hidden)

            # 保存当前时间步的隐藏状态
            H_all.append(hidden.unsqueeze(0))  # 在时间步维度插入一维

        # 将列表中的隐藏状态堆叠为 (K, N, C)
        H_all = torch.cat(H_all, dim=0)

        return H_all
