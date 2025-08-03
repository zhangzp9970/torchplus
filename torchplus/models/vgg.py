import torch
import torch.nn as nn
from typing import Optional
from torchvision.models import (
    VGG,
    vgg11,
    VGG11_Weights,
    vgg13,
    VGG13_Weights,
    vgg16,
    VGG16_Weights,
    vgg19,
    VGG19_Weights,
)


class VGGFE(nn.Module):
    def __init__(self, vggmodel: VGG) -> None:
        super(VGGFE, self).__init__()
        self.features = vggmodel.features
        self.avgpool = vggmodel.avgpool

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def vgg11fe(weights: Optional[VGG11_Weights] = None, progress: bool = True) -> VGGFE:
    vggmodel = vgg11(weights, progress)
    return VGGFE(vggmodel)


def vgg13fe(weights: Optional[VGG13_Weights] = None, progress: bool = True) -> VGGFE:
    vggmodel = vgg13(weights, progress)
    return VGGFE(vggmodel)


def vgg16fe(weights: Optional[VGG16_Weights] = None, progress: bool = True) -> VGGFE:
    vggmodel = vgg16(weights, progress)
    return VGGFE(vggmodel)


def vgg19fe(weights: Optional[VGG19_Weights] = None, progress: bool = True) -> VGGFE:
    vggmodel = vgg19(weights, progress)
    return VGGFE(vggmodel)


VGGFE_Dim = 25088
