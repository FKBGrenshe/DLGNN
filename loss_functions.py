import torch
import torch.nn as nn

class SemiSupervisedLoss(nn.Module):
    def __init__(self):
        super(SemiSupervisedLoss, self).__init__()

    def forward(self, logits, labels, mask):
        loss = nn.CrossEntropyLoss()(logits[mask], labels[mask])
        return loss
