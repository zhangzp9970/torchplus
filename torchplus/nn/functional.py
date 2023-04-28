import torch
import torch.nn.functional as F
from typing import Optional


def mse_with_weight_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    mse = F.mse_loss(input, target, reduction="none")
    if reduction == "none":
        return mse * weight
    elif reduction == "mean":
        return torch.mean(mse * weight)
    elif reduction == "sum":
        return torch.sum(mse * weight)
    else:
        raise NotImplementedError()


def pixel_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    threshold: Optional[float] = 0.0,
    reduction: str = "mean",
) -> torch.Tensor:
    assert input.shape == target.shape
    input = torch.round(input * 255.0)
    target = torch.round(target * 255.0)
    ToF = torch.abs(input - target) <= threshold
    ToF = ToF.to(torch.float)
    if reduction == "none":
        return ToF
    elif reduction == "sum":
        return torch.count_nonzero(ToF)
    elif reduction == "mean":
        return 1 - torch.count_nonzero(ToF) / torch.numel(ToF)
    else:
        raise NotImplementedError()
