import torch
from model import DL_GNN
from graph_creation import create_dynamic_line_graph
from data_preprocessing import load_data, preprocess_data, split_data

def evaluate_model(file_path):
    # 数据加载和处理
    data = load_data(file_path)
    features, labels = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(features, labels)

    # 创建图数据
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)  # 假设简单的边关系
    test_data = create_dynamic_line_graph(X_test, edge_index)

    # 加载训练好的模型
    model = DL_GNN(in_channels=features.shape[1], out_channels=64)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()

    with torch.no_grad():
        out = model(test_data)
        _, pred = out.max(dim=1)
        correct = (pred == test_data.y).sum().item()
        accuracy = correct / len(test_data.y)
        print(f'Accuracy: {accuracy * 100:.2f}%')

if __name__ == '__main__':
    evaluate_model('network_traffic_data.csv')
