import torch
import torch.nn as nn
from typing import Optional
from torchvision.models import (
    ResNet,
    resnet18,
    ResNet18_Weights,
    resnet34,
    ResNet34_Weights,
    resnet50,
    ResNet50_Weights,
    resnet101,
    ResNet101_Weights,
    resnet152,
    ResNet152_Weights,
)


class ResNetFE(nn.Module):
    def __init__(self, resnetmodel: ResNet) -> None:
        super(ResNetFE, self).__init__()
        self.conv1 = resnetmodel.conv1
        self.bn1 = resnetmodel.bn1
        self.relu = resnetmodel.relu
        self.maxpool = resnetmodel.maxpool
        self.layer1 = resnetmodel.layer1
        self.layer2 = resnetmodel.layer2
        self.layer3 = resnetmodel.layer3
        self.layer4 = resnetmodel.layer4
        self.avgpool = resnetmodel.avgpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def resnet18fe(
    weights: Optional[ResNet18_Weights] = None, progress: bool = True
) -> ResNetFE:
    resnetmodel = resnet18(weights, progress)
    return ResNetFE(resnetmodel)


def resnet34fe(
    weights: Optional[ResNet34_Weights] = None, progress: bool = True
) -> ResNetFE:
    resnetmodel = resnet34(weights, progress)
    return ResNetFE(resnetmodel)


def resnet50fe(
    weights: Optional[ResNet50_Weights] = None, progress: bool = True
) -> ResNetFE:
    resnetmodel = resnet50(weights, progress)
    return ResNetFE(resnetmodel)


def resnet101fe(
    weights: Optional[ResNet101_Weights] = None, progress: bool = True
) -> ResNetFE:
    resnetmodel = resnet101(weights, progress)
    return ResNetFE(resnetmodel)


def resnet152fe(
    weights: Optional[ResNet152_Weights] = None, progress: bool = True
) -> ResNetFE:
    resnetmodel = resnet152(weights, progress)
    return ResNetFE(resnetmodel)
