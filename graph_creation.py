# 这个文件用于构建动态线图（DLG），将数据转换为图数据格式（图节点、边及特征）
import torch
from torch_geometric.data import Data

def create_dynamic_line_graph(features, edge_index):
    """创建动态线图，返回图数据对象"""
    node_features = torch.tensor(features, dtype=torch.float)
    data = Data(x=node_features, edge_index=edge_index)
    return data
