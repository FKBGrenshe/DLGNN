from Alogrithm1 import *
from Alogrithm2 import *
from ReadOutLayer import *
class GCNII_GRU(Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_gcn_layers, num_gru_layers, alpha, lamda, dropout, device, readout_type):
        """
        完整模型：GCNII + GCNII-GRU + Readout Layer
        Args:
        - input_dim: 输入特征维度
        - hidden_dim: 隐藏层特征维度
        - output_dim: 输出特征维度
        - num_gcn_layers: GCNII 层数
        - num_gru_layers: GCNII-GRU 层数
        - alpha: 初始残差强度
        - lamda: 自适应残差参数
        - dropout: Dropout 比例
        - device: 设备 (CPU/GPU)
        """
        super(GCNII_GRU, self).__init__()
        self.gcn = GCNII(input_dim, hidden_dim, hidden_dim, num_gcn_layers, alpha, lamda, dropout, device)
        self.gru = GCNIIGRU(hidden_dim, hidden_dim, num_gcn_layers, alpha, lamda, dropout, device, num_gru_layers)
        # self.readout = ReadoutLayer(readout_type)
        self.output_layer = Linear(hidden_dim, output_dim)

    def forward(self, features_seq, adj_seq):
        """
        Args:
        - features_seq: 输入特征序列 (k, N, C)
        - adj_seq: 邻接矩阵序列 (k, N, N)

        Returns:
        - logits: 输出特征 (N, output_dim)
        """
        k, N, _ = features_seq.shape
        k = features_seq.shape[0]
        # H_seq = torch.zeros_like(features_seq)  # 存储每个时间步的 GCNII 输出
        hidden_dim = self.gcn.convs[-1].out_features  # 从 GCNII 输出层获取隐藏特征维度
        H_seq = torch.zeros((k, N, hidden_dim), device=features_seq.device)

        # 空间特征提取 (GCNII)
        for t in range(k):
            H_seq[t] = self.gcn(features_seq[t], adj_seq[t])
            # 初始化 H_seq，维度为 (k, N, hidden_dim)
            # hidden_dim = self.gcn.convs[-1].out_features  # 从 GCNII 输出层获取隐藏特征维度
            # H_seq = torch.zeros((k, N, hidden_dim), device=features_seq.device)

        '# 时序特征提取 (GCNII-GRU)'
        hidden_final = self.gru(H_seq, adj_seq)

        # # 使用 Readout Layer 提取图级特征
        # graph_embedding = self.readout(hidden_final)  # 输出形状为 (1, hidden_dim)
        # # 输出层
        # logits = self.output_layer(graph_embedding)

        logits = self.output_layer(hidden_final)
        # logits = self.output_layer(H_seq)

        'crossEntropy  自动计算 log-softmax'
        # return F.softmax(logits, dim=1)
        return logits



# if __name__ == '__main__':
#     # 模拟数据
#     k, N, C = 5, 10, 16  # 时间步数、节点数、特征维度
#     features_seq = torch.rand((k, N, C))  # 输入特征序列
#     adj_seq = torch.stack([torch.eye(N) for _ in range(k)])  # 简单的对角邻接矩阵序列
#
#     # 初始化模型
#     model = GCNII_GRU(
#         input_dim=C, hidden_dim=32, output_dim=3,
#         num_gcn_layers=10, num_gru_layers=2,
#         alpha=0.5, lamda=1.0, dropout=0.3,
#         device="cuda" if torch.cuda.is_available() else "cpu"
#     )
#
#     # 前向传播
#     logits = model(features_seq, adj_seq)
#     print("Model Output Shape:", logits.shape)  # 输出形状应为 (N, output_dim)
