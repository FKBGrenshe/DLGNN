import torch
# 邻接矩阵归一化
def normalize_adj(adj):
    rowsum = adj.sum(-1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
    return torch.matmul(d_mat_inv_sqrt, torch.matmul(adj, d_mat_inv_sqrt))


