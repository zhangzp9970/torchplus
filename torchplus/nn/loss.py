import torch
import torch.nn as nn
from . import functional as F


class MSEWithWeightLoss(nn.MSELoss):
    def __init__(
        self,
        weight: torch.Tensor = None,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        super().__init__(size_average=size_average, reduce=reduce, reduction=reduction)
        self.weight = weight

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_with_weight_loss(
            input, target, self.weight, reduction=self.reduction
        )


class PixelLoss(nn.Module):
    def __init__(self, threshold: float = 0.0, reduction: str = "mean"):
        super().__init__()
        self.threshold = threshold
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.pixel_loss(input, target, self.threshold, self.reduction)
