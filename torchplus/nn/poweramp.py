import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class PowerAmplification(nn.Module):
    def __init__(
        self, in_features: int, alpha: float = None, device=None, dtype=None
    ) -> None:
        super(PowerAmplification, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        if alpha is not None:
            self.alpha = Parameter(torch.tensor([alpha], **factory_kwargs))
        else:
            self.alpha = Parameter(torch.rand(1, **factory_kwargs))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.expand(self.in_features)
        return torch.pow(input, alpha)