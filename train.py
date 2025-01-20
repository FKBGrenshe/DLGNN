import torch
from torch.optim import Adam
from model import DL_GNN
from loss_functions import SemiSupervisedLoss
from graph_creation import create_dynamic_line_graph
from data_preprocessing import load_data, preprocess_data, split_data


def train_model(file_path, num_epochs=100):
    # 数据加载和处理
    data = load_data(file_path)
    features, labels = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(features, labels)

    # 创建图数据
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)  # 假设简单的边关系
    train_data = create_dynamic_line_graph(X_train, edge_index)

    # 模型初始化
    model = DL_GNN(in_channels=features.shape[1], out_channels=64)
    optimizer = Adam(model.parameters(), lr=0.01)
    loss_fn = SemiSupervisedLoss()

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(train_data)

        # 假设已经有mask
        loss = loss_fn(out, train_data.y, mask=(train_data.y != -1))

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')


if __name__ == '__main__':
    filepath = 'D:\\学习\\论文\\2024new-第二篇\\pmy\\dataset\\NF-BoT-IoT\\data\\NF-BoT-IoT.csv'
    train_model(filepath)
    train_model('network_traffic_data.csv')
