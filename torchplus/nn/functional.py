import torch
import torch.nn.functional as F
from typing import Optional


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


def pixel_loss(im: torch.Tensor, im_gt: torch.Tensor, threshold: float = 0.0):
    assert im.shape == im_gt.shape
    bs, c, h, w = im.shape
    total = bs*c*h*w
    im = im.reshape(total)
    im_gt = im_gt.reshape(total)
    im *= 255.0
    im_gt *= 255.0
    im = im.round()
    im_gt = im_gt.round()
    ToF = (torch.abs(im - im_gt) <= threshold)
    ToF = ToF.to(torch.float)
    pos = torch.count_nonzero(ToF)
    acc = pos/total
    return acc
