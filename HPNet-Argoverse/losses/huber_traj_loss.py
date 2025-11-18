import torch
import torch.nn as nn


class HuberTrajLoss(nn.Module):

    def __init__(self, delta: float = 1.0) -> None:
        super(HuberTrajLoss, self).__init__()
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(pred - target)
        loss = torch.where(diff < self.delta, 0.5 * diff ** 2, self.delta * (diff - 0.5 * self.delta))
        return loss.mean()
