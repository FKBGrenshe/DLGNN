import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import train_test_split

from Algorithm3 import *
from dataProcess.dataPreprocess import *
from dataProcess.GraphBuild import *
from utils import *

# 假设 GCNIIGRU 类已经定义并导入
# from model import GCNIIGRU

# 数据集定义
class GraphTimeSeriesDataset(Dataset):
    def __init__(self, features, adj_matrices, labels):
        """
        Args:
        - features: 图特征序列 (num_samples, k, N, input_dim)
        - adj_matrices: 邻接矩阵序列 (num_samples, k, N, N)
        - labels: 样本对应的标签 (num_samples,)
        """
        self.features = features
        self.adj_matrices = adj_matrices
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.adj_matrices[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )

# 训练函数
def train_model(model, train_loader, val_loader, epochs, criterion, optimizer, device):
    """
    训练模型
    Args:
    - model: GCNII-GRU 模型
    - train_loader: 训练数据 DataLoader
    - val_loader: 验证数据 DataLoader
    - epochs: 训练轮数
    - criterion: 损失函数
    - optimizer: 优化器
    - device: 设备 (CPU/GPU)
    """
    model.to(device)
    best_val_loss = float("inf")
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for features, adj_matrices, labels in train_loader:
            adj_matrices = normalize_adj(adj_matrices)
            features, adj_matrices, labels = features.squeeze(0).to(device), adj_matrices.squeeze(0).to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(features, adj_matrices)
            # loss = criterion(logits, labels)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(
                        f"{name} - Mean: {param.data.mean().item()} - Grad: {param.grad.abs().mean().item() if param.grad is not None else None}")

            optimizer.step()

            train_loss += loss.item()
            predicted = torch.argmax(F.log_softmax(logits, dim=2),dim=2)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)*labels.size(1)

        train_acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")

        # 验证阶段
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for features, adj_matrices, labels in val_loader:
                adj_matrices = normalize_adj(adj_matrices)
                features, adj_matrices, labels = features.to(device), adj_matrices.to(device), labels.to(device)

                logits = model(features, adj_matrices)
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                val_loss += loss.item()
                predicted = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)*labels.size(1)

        val_acc = 100. * correct / total
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved Best Model!")

# 主程序
def main():
    # 模拟数据
    num_samples = 10  # 样本数量
    k, N, input_dim = 5, 181, 5  # 时间步数、节点数、输入特征维度
    hidden_dim = 50  # 隐藏特征维度
    output_dim = 2   # 输出类别数 -- 2分类
    epochs = 50      # 训练轮数
    batch_size = 5  # 批量大小

    # 初始化模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GCNII_GRU(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_gcn_layers=10,
        num_gru_layers=2,
        alpha=0.5,
        lamda=1.0,
        dropout=0.3,
        device=device,
        output_dim=output_dim,
        readout_type='mean'
    )
    print("模型初始化完成")

    ## NF-BoT-IoT.csv总共181个源IP目的IP组合，因此总共181条边
    totalData, totalLabel = Step1(100)
    snapshots = build_graph(totalData, 181)
    adjacency_matrices, feature_matrices = generate_line_graph_matrices(snapshots)
    # 划分训练集\验证集\测试集 -- 80：5：15
    train_size = 0.8
    validation_size = 0.05
    test_size = 0.15

    # X_train_data, X_rest_data, y_train, y_rest = train_test_split(
    #     totalData, totalLabel, test_size=1 - train_size, random_state=42
    # )
    # X_val_data, X_test_data, y_val, y_test = train_test_split(
    #     X_rest_data, y_rest, test_size=1 - train_size, random_state=42
    # )
    # 'train -- 生成图-线图邻接矩阵和特征矩阵'
    # A_train, X_train = generate_line_graph_matrices(build_graph(X_train_data, 181))
    # A_val, X_val = generate_line_graph_matrices(build_graph(X_val_data, 181))

    X_train, X_rest, A_train, A_rest, y_train, y_rest = train_test_split(
        feature_matrices, adjacency_matrices, torch.tensor(totalLabel).view(-1, 181), test_size=1 - train_size, random_state=42
    )
    X_val, X_test, A_val, A_test, y_val, y_test = train_test_split(
        X_rest, A_rest, y_rest, test_size= test_size / (test_size + validation_size), random_state=42
    )


    # 打印结果
    print(f"训练集大小: {len(X_train)}")
    print(f"验证集大小: {len(X_val)}")
    print(f"测试集大小: {len(X_test)}")

    # 创建数据加载器
    train_dataset = GraphTimeSeriesDataset(X_train, A_train, y_train)
    val_dataset = GraphTimeSeriesDataset(X_val, A_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)



    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

    # 开始训练
    print("开始训练")
    train_model(model, train_loader, val_loader, epochs, criterion, optimizer, device)

if __name__ == "__main__":
    main()

#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn import Linear, Module
# from torch.optim import Adam
# from Algorithm3 import *
#
# if __name__ == '__main__':
#
#     # 模拟数据
#     k, N, input_dim = 5, 181, 5  # 时间步数、节点数、输入特征维度
#     hidden_dim = 50  # 隐藏特征维度
#     output_dim = 2   # 输出类别数
#     features_seq = torch.rand((k, N, input_dim))  # 输入特征序列 (k, N, input_dim)
#     adj_seq = torch.stack([torch.eye(N) for _ in range(k)])  # 邻接矩阵序列 (k, N, N)
#
#     # 模拟标签 (每个节点有一个分类标签)
#     labels = torch.randint(0, output_dim, (N,))  # 标签形状为 (N,)
#
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = GCNII_GRU(
#         input_dim=input_dim,
#         hidden_dim=hidden_dim,
#         output_dim=output_dim,
#         num_gcn_layers=10,  # 10 层 GCNII
#         num_gru_layers=2,   # 2 层 GCNII-GRU
#         alpha=0.5,
#         lamda=1.0,
#         dropout=0.3,
#         device=device,
#         readout_type="mean"
#     ).to(device)
#
#     # 定义优化器和损失函数
#     criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
#     optimizer = Adam(model.parameters(), lr=0.01, weight_decay=0.001)
#
#     # 将数据移动到设备
#     features_seq = features_seq.to(device)
#     adj_seq = adj_seq.to(device)
#     labels = labels.to(device)
#
#     # 训练循环
#     num_epochs = 100  # 训练轮数
#     model.train()
#     for epoch in range(num_epochs):
#         optimizer.zero_grad()  # 清空梯度
#
#         # 前向传播
#         logits = model(features_seq, adj_seq)  # 输出形状为 (N, output_dim)
#
#         # 计算损失
#         loss = criterion(logits, labels)  # 标签形状为 (N,)
#
#         # 反向传播
#         loss.backward()
#         optimizer.step()
#
#         # 打印训练信息
#         if (epoch + 1) % 10 == 0:
#             print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")
#
#     # 模型评估
#     model.eval()
#     with torch.no_grad():
#         logits = model(features_seq, adj_seq)
#         predictions = torch.argmax(logits, dim=1)
#         accuracy = (predictions == labels).float().mean()
#         print(f"Final Accuracy: {accuracy.item():.4f}")
