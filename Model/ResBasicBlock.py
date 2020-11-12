# branch2
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
import torchvision.datasets
import numpy as np


class ResBasicBlock(nn.Module):
    # expansion = 1
    in_plane = 64

    def __init__(self, in_planes, planes, stride=1):
        super(ResBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        else:
            self.shortcut = lambda x: x

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


def make_basic_block_layer(block, planes, num_blocks, stride):
    strides = [stride] + [1]*(num_blocks-1) # the first layer should have stride 2 and others 1
    layers = []
    for stride in strides:
        layers.append(block(block.in_planes, planes, stride))
        block.in_planes = planes
    return nn.Sequential(*layers)