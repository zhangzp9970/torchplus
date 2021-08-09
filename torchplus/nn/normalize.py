import torch
import torch.nn as nn
from typing import Optional
from . import functional as F


class Normalize(nn.Module):
    def __init__(self, dim, keepdim: Optional[bool] = False, inplace: Optional[bool] = False):
        super(Normalize, self).__init__()
        self.dim = dim
        self.keepdim = keepdim
        self.inplace = inplace

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return F.normalize(tensor, self.dim, self.keepdim, self.inplace)
