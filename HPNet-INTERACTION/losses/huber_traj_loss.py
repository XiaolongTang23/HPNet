import torch
import torch.nn as nn

class HuberTrajLoss(nn.Module):
    def __init__(self):
        super(HuberTrajLoss, self).__init__()
        self.huber = nn.SmoothL1Loss(reduction='none')

    def forward(self, predictions, targets):
        loss = self.huber(predictions, targets).sum(-1)
        return loss.mean()