from torch.nn import Module
import torch
class ReadoutLayer(Module):
    def __init__(self, readout_type="mean"):
        """
        Readout Layer 实现
        Args:
        - readout_type: 聚合方式 ("mean", "max", 或 "attention")
        """
        super(ReadoutLayer, self).__init__()
        self.readout_type = readout_type

    def forward(self, H):
        """
        Args:
        - H: 节点嵌入矩阵 (N, d)，N 是节点数，d 是特征维度

        Returns:
        - output: 图级嵌入 (1, d)
        """
        if self.readout_type == "mean":
            return torch.mean(H, dim=0, keepdim=True)  # 全局平均池化
        elif self.readout_type == "max":
            return torch.max(H, dim=0, keepdim=True)[0]  # 全局最大池化
        else:
            raise ValueError(f"Unsupported readout_type: {self.readout_type}")
