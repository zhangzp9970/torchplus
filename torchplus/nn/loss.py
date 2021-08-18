import torch
import torch.nn as nn
import torch.nn.functional as F


class MSEWithWeightLoss(nn.MSELoss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average=size_average, reduce=reduce, reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor, weight: torch.Tensor = None) -> torch.Tensor:
        inputw = input*weight
        targetw = target*weight
        return F.mse_loss(inputw, targetw, reduction=self.reduction)
