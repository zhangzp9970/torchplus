import torch
import torch.nn.functional as F
from typing import Optional


def normalize(tensor: torch.Tensor, dim, keepdim: Optional[bool] = False, inplace: Optional[bool] = False) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            'Input tensor should be a torch tensor. Got {}.'.format(type(tensor)))
    if not inplace:
        tensor = tensor.clone()
    dtype = tensor.dtype
    mean = torch.mean(tensor, dim=dim, keepdim=keepdim)
    std = torch.std(tensor, dim=dim, keepdim=keepdim)
    if (std == 0).any():
        raise ValueError(
            'std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.sub_(mean).div_(std)
    return tensor


def mse_with_weight_loss(input: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, size_average: Optional[bool] = None, reduce: Optional[bool] = None, reduction: str = 'mean') -> torch.Tensor:
    mse = F.mse_loss(input, target, size_average=size_average,
                     reduce=reduce, reduction='none')
    if reduction == 'none':
        mse = mse*weight
    elif reduction == 'mean':
        mse = torch.mean(mse*weight)
    elif reduction == 'sum':
        mse = torch.sum(mse*weight)
    else:
        raise NotImplementedError()
    return mse
