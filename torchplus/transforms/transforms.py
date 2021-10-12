import torch
import torch.nn as nn
from torchvision.transforms.functional import *


class Crop(nn.Module):
    def __init__(self, top: int, left: int, height: int, width: int):
        super().__init__()
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def forward(self, img):
        return crop(img, self.top, self.left, self.height, self.width)

    def __repr__(self):
        return self.__class__.__name__ + '(top={0},left={1},height={2},width={3})'.format(self.top, self.left, self.height, self.width)
