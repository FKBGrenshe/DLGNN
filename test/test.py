import pandas as pd
import networkx as nx
import numpy as np
import torch




# 构建时空图：Dynamic Spatiotemporal Graph Snapshots
def build_graph(data, max_edges_per_snapshot):
    """
    根据网络流的顺序生成动态图快照
    Args:
    - data: DataFrame，包含网络流量记录，字段包括：
        - IPV4_SRC_ADDR: 源IP地址
        - IPV4_DST_ADDR: 目标IP地址
        - PROTOCOL: 协议类型
        - FLOW_DURATION_MILLISECONDS: 流持续时间
        - IN_BYTES
        - OUT_BYTES
        - bytes_per_packet: 每个包的字节数 暂时设成 in+out / packetsNum
        - TCP_FLAGS: TCP标志
    - max_edges_per_snapshot: 每个快照中允许的最大边数（控制快照大小）

    Returns:
    - snapshots: 包含多个图快照的列表，每个快照是一个 NetworkX 图
    """
    snapshots = []
    G = nx.DiGraph()  # 初始化一个有向图
    edge_count = 0
    for _, row in data.iterrows():
        src, dest = row['IPV4_SRC_ADDR'], row['IPV4_DST_ADDR']

        # 添加节点
        if not G.has_node(src):
            G.add_node(src, type='IP')
        if not G.has_node(dest):
            G.add_node(dest, type='IP')

        # 添加边及其特征
        G.add_edge(
            src, dest,
            protocol=row['PROTOCOL'],
            flow_duration=row['FLOW_DURATION_MILLISECONDS'],
            incoming_flow_bytes=row['IN_BYTES'],
            bytes_per_packet=(row['IN_BYTES'] + row['OUT_BYTES']) / (row['IN_PKTS'] + row['OUT_PKTS']),
            tcp_flags=row['TCP_FLAGS']
        )
        edge_count += 1

        # 当达到最大边数时，保存当前快照，开始一个新图
        if edge_count >= max_edges_per_snapshot:
            snapshots.append(G)
            G = nx.DiGraph()  # 创建新图
            edge_count = 0

    # 如果最后一个图有剩余边，也需要保存
    if G.number_of_edges() > 0:
        snapshots.append(G)

    return snapshots




def generate_line_graph_snapshots(graph_snapshots):
    """
    基于时空图快照生成线图快照
    Args:
    - graph_snapshots: List[networkx.Graph]，原始时空图快照的列表

    Returns:
    - line_graph_snapshots: List[networkx.Graph]，线图快照的列表
    """
    line_graph_snapshots = []

    for i, G in enumerate(graph_snapshots):
        # 使用 NetworkX 的 line_graph 方法生成线图
        line_graph = nx.line_graph(G)

        # 传递原始图的边特征到线图的节点
        for edge in line_graph.nodes:
            # edge 是原始图中的一条边，例如 (u, v)
            if G.has_edge(*edge):
                # 将原始图边的属性复制到线图节点
                line_graph.nodes[edge].update(G.edges[edge])

        line_graph_snapshots.append(line_graph)

    return line_graph_snapshots

# 示例原始时空图快照（生成简单图作为示例）
def create_sample_graph_snapshots():
    G1 = nx.Graph()
    G1.add_edge('192.168.1.1', '192.168.1.2', protocol='TCP', bytes=100)
    G1.add_edge('192.168.1.2', '192.168.1.3', protocol='UDP', bytes=200)

    G2 = nx.Graph()
    G2.add_edge('192.168.1.1', '192.168.1.3', protocol='TCP', bytes=300)
    G2.add_edge('192.168.1.3', '192.168.1.4', protocol='UDP', bytes=400)

    return [G1, G2]

# # 生成原始时空图快照
# graph_snapshots = create_sample_graph_snapshots()
#
# # 生成线图快照
# line_graph_snapshots = generate_line_graph_snapshots(graph_snapshots)
#
# # 打印每个线图快照的节点和边
# for i, LG in enumerate(line_graph_snapshots):
#     print(f"Line Graph Snapshot {i}:")
#     print("Nodes:", LG.nodes(data=True))
#     print("Edges:", LG.edges())




def generate_line_graph_matrices(graph_snapshots):
    """
    生成线图的邻接矩阵和特征矩阵序列
    Args:
    - graph_snapshots: List[networkx.Graph]，原始图快照列表，每个图包含边特征

    Returns:
    - adjacency_matrices: List[torch.Tensor]，线图邻接矩阵列表 (A1, A2, ..., Ak)
    - feature_matrices: torch.Tensor，线图特征矩阵序列 (X1, X2, ..., Xk)，形状为 (k, n, d)
    """
    adjacency_matrices = []  # 存储线图的邻接矩阵
    feature_matrices = []    # 存储线图的特征矩阵

    for G in graph_snapshots:
        # Step 1: 创建线图
        line_graph = nx.line_graph(G)

        ### 如果线图没有边，跳过 ###
        if len(line_graph.edges) == 0:
            continue

        # Step 2: 获取线图邻接矩阵
        nodes = list(line_graph.nodes)  # 线图节点（对应原始图的边）
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        n = len(nodes)
        A = np.zeros((n, n), dtype=np.float32)
        for u, v in line_graph.edges:
            A[node_to_idx[u], node_to_idx[v]] = 1.0
            A[node_to_idx[v], node_to_idx[u]] = 1.0
        adjacency_matrices.append(torch.tensor(A))

        # Step 3: 获取线图特征矩阵
        d = len(G.edges[nodes[0]])  # 原始图边的特征维度
        X = np.zeros((n, d), dtype=np.float32)
        for edge, idx in node_to_idx.items():
            X[idx] = np.array([G.edges[edge][attr] for attr in G.edges[edge]])
        feature_matrices.append(torch.tensor(X))

    # 将特征矩阵堆叠为 (k, n, d)
    feature_matrices = torch.stack(feature_matrices)

    return adjacency_matrices, feature_matrices

# 示例原始图快照
def create_sample_graph_snapshots():
    G1 = nx.Graph()
    G1.add_edge('192.168.1.1', '192.168.1.2', protocol=1, flow_duration=5, bytes=100)
    G1.add_edge('192.168.1.2', '192.168.1.3', protocol=2, flow_duration=10, bytes=200)

    G2 = nx.Graph()
    G2.add_edge('192.168.1.1', '192.168.1.3', protocol=1, flow_duration=8, bytes=150)
    G2.add_edge('192.168.1.3', '192.168.1.4', protocol=3, flow_duration=12, bytes=300)

    return [G1, G2]



if __name__ == '__main__':
    data = pd.read_csv('D:\\学习\\论文\\2024new-第二篇\\pmy\\dataset\\NF-BoT-IoT\\data\\NF-BoT-IoT.csv')

    # 从 600,000 行中随机抽取 30,000 行
    data = data.sample(n=30000, random_state=42)

    '生成动态图快照'
    max_edges_per_snapshot = 50  # 每个快照最多包含50条边
    snapshots = build_graph(data, max_edges_per_snapshot)

    # 打印每个图快照的节点和边
    for i, G in enumerate(snapshots):
        print(f"Snapshot {i}:")
        print("Nodes:", G.nodes(data=True))
        print("Edges:", G.edges(data=True))
        if (i > 1):
            break
    '生成线图快照'
    line_graph_snapshots = generate_line_graph_snapshots(snapshots)

    '生成线图邻接矩阵和特征矩阵'
    adjacency_matrices, feature_matrices = generate_line_graph_matrices(line_graph_snapshots)


