import torch
from torch_geometric.data import Data


def construct_line_graph_pytorch(edge_index, num_nodes):
    """
    使用 PyTorch 构造线图 L(G_f)。

    参数:
        edge_index (torch.Tensor): 原始图 G_f 的边索引 (2, num_edges)。
        num_nodes (int): 原始图 G_f 的节点数。

    返回:
        line_edge_index (torch.Tensor): 线图 L(G_f) 的边索引 (2, num_line_edges)。
        line_node_features (torch.Tensor): 线图 L(G_f) 的节点特征。
    """
    # 初始化线图的边集合
    line_edges = []

    # 构造节点的边邻接字典
    edge_neighbors = {i: [] for i in range(num_nodes)}
    for idx, (u, v) in enumerate(edge_index.T):
        edge_neighbors[u.item()].append(idx)
        edge_neighbors[v.item()].append(idx)

    # 遍历每个节点的边邻接关系
    for edges in edge_neighbors.values():
        for i in range(len(edges)):
            for j in range(i + 1, len(edges)):
                line_edges.append([edges[i], edges[j]])

    # 转换为张量格式
    if len(line_edges) > 0:
        line_edge_index = torch.tensor(line_edges, dtype=torch.long).T
    else:
        line_edge_index = torch.empty((2, 0), dtype=torch.long)

    # 线图的节点特征：将原始图的边索引作为特征
    line_node_features = edge_index.T

    return line_edge_index, line_node_features


if __name__ == "__main__":
    # 示例：创建原始图 G_f 的边索引
    edge_index = torch.tensor([
        [0, 1, 2, 0],  # 边的起点
        [1, 2, 3, 3]  # 边的终点
    ], dtype=torch.long)

    num_nodes = 4  # 原始图的节点数

    # 构造线图
    line_edge_index, line_node_features = construct_line_graph_pytorch(edge_index, num_nodes)

    print("原始图的边索引:")
    print(edge_index)
    print("\n线图的边索引:")
    print(line_edge_index)
    print("\n线图的节点特征（对应原始图的边索引）:")
    print(line_node_features)

    # 构造 PyTorch Geometric 的数据对象
    line_graph = Data(edge_index=line_edge_index, x=line_node_features)
    print("\n线图数据对象:")
    print(line_graph)
