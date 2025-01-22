# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# import numpy as np
# from sklearn.model_selection import train_test_split
#
# from Algorithm3 import *
#
# # 假设 GCNIIGRU 类已经定义并导入
# # from model import GCNIIGRU
#
# # 数据集定义
# class GraphTimeSeriesDataset(Dataset):
#     def __init__(self, features, adj_matrices, labels):
#         """
#         Args:
#         - features: 图特征序列 (num_samples, k, N, input_dim)
#         - adj_matrices: 邻接矩阵序列 (num_samples, k, N, N)
#         - labels: 样本对应的标签 (num_samples,)
#         """
#         self.features = features
#         self.adj_matrices = adj_matrices
#         self.labels = labels
#
#     def __len__(self):
#         return len(self.labels)
#
#     def __getitem__(self, idx):
#         return (
#             torch.tensor(self.features[idx], dtype=torch.float32),
#             torch.tensor(self.adj_matrices[idx], dtype=torch.float32),
#             torch.tensor(self.labels[idx], dtype=torch.long),
#         )
#
# # 训练函数
# def train_model(model, train_loader, val_loader, epochs, criterion, optimizer, device):
#     """
#     训练模型
#     Args:
#     - model: GCNII-GRU 模型
#     - train_loader: 训练数据 DataLoader
#     - val_loader: 验证数据 DataLoader
#     - epochs: 训练轮数
#     - criterion: 损失函数
#     - optimizer: 优化器
#     - device: 设备 (CPU/GPU)
#     """
#     model.to(device)
#     best_val_loss = float("inf")
#     for epoch in range(epochs):
#         # 训练阶段
#         model.train()
#         train_loss = 0
#         correct = 0
#         total = 0
#         for features, adj_matrices, labels in train_loader:
#             features, adj_matrices, labels = features.squeeze(0).to(device), adj_matrices.squeeze(0).to(device), labels.to(device)
#
#             optimizer.zero_grad()
#             outputs = model(features, adj_matrices)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#             train_loss += loss.item()
#             _, predicted = outputs.max(1)
#             correct += (predicted == labels).sum().item()
#             total += labels.size(0)
#
#         train_acc = 100. * correct / total
#         print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
#
#         # 验证阶段
#         model.eval()
#         val_loss = 0
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for features, adj_matrices, labels in val_loader:
#                 features, adj_matrices, labels = features.to(device), adj_matrices.to(device), labels.to(device)
#
#                 outputs = model(features, adj_matrices)
#                 loss = criterion(outputs, labels)
#                 val_loss += loss.item()
#                 _, predicted = outputs.max(1)
#                 correct += (predicted == labels).sum().item()
#                 total += labels.size(0)
#
#         val_acc = 100. * correct / total
#         val_loss /= len(val_loader)
#         print(f"Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
#
#         # 保存最佳模型
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             torch.save(model.state_dict(), "best_model.pth")
#             print("Saved Best Model!")
#
# # 主程序
# def main():
#     # 模拟数据
#     num_samples = 100  # 样本数量
#     k, N, input_dim = 5, 10, 16  # 时间步数、节点数、输入特征维度
#     hidden_dim = 50  # 隐藏特征维度
#     output_dim = 2   # 输出类别数 -- 2分类
#     epochs = 50      # 训练轮数
#     batch_size = 1  # 批量大小
#
#     # 随机生成特征、邻接矩阵和标签
#     features = np.random.rand(num_samples, k, N, input_dim).astype(np.float32)
#     adj_matrices = np.stack([np.stack([np.eye(N) for _ in range(k)]) for _ in range(num_samples)]).astype(np.float32)
#     labels = np.random.randint(0, output_dim, size=num_samples).astype(np.int64)
#
#     # 划分训练集和验证集
#     X_train, X_val, A_train, A_val, y_train, y_val = train_test_split(
#         features, adj_matrices, labels, test_size=0.2, random_state=42
#     )
#
#     # 创建数据加载器
#     train_dataset = GraphTimeSeriesDataset(X_train, A_train, y_train)
#     val_dataset = GraphTimeSeriesDataset(X_val, A_val, y_val)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#
#     # 初始化模型
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = GCNII_GRU(
#         input_dim=input_dim,
#         hidden_dim=hidden_dim,
#         num_gcn_layers=10,
#         num_gru_layers=2,
#         alpha=0.5,
#         lamda=1.0,
#         dropout=0.3,
#         device=device,
#         output_dim=output_dim
#     )
#
#     # 定义损失函数和优化器
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
#
#     # 开始训练
#     train_model(model, train_loader, val_loader, epochs, criterion, optimizer, device)
#
# if __name__ == "__main__":
#     main()


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Module
from torch.optim import Adam
from Algorithm3 import *

if __name__ == '__main__':

    # 模拟数据
    k, N, input_dim = 5, 10, 16  # 时间步数、节点数、输入特征维度
    hidden_dim = 32  # 隐藏特征维度
    output_dim = 3   # 输出类别数
    features_seq = torch.rand((k, N, input_dim))  # 输入特征序列 (k, N, input_dim)
    adj_seq = torch.stack([torch.eye(N) for _ in range(k)])  # 邻接矩阵序列 (k, N, N)

    # 模拟标签 (每个节点有一个分类标签)
    labels = torch.randint(0, output_dim, (N,))  # 标签形状为 (N,)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GCNII_GRU(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_gcn_layers=10,  # 10 层 GCNII
        num_gru_layers=2,   # 2 层 GCNII-GRU
        alpha=0.5,
        lamda=1.0,
        dropout=0.3,
        device=device,
        readout_type="mean"
    ).to(device)

    # 定义优化器和损失函数
    criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=0.001)

    # 将数据移动到设备
    features_seq = features_seq.to(device)
    adj_seq = adj_seq.to(device)
    labels = labels.to(device)

    # 训练循环
    num_epochs = 100  # 训练轮数
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()  # 清空梯度

        # 前向传播
        logits = model(features_seq, adj_seq)  # 输出形状为 (N, output_dim)

        # 计算损失
        loss = criterion(logits, labels)  # 标签形状为 (N,)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 打印训练信息
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

    # 模型评估
    model.eval()
    with torch.no_grad():
        logits = model(features_seq, adj_seq)
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == labels).float().mean()
        print(f"Final Accuracy: {accuracy.item():.4f}")
