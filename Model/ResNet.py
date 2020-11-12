# branch2
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
import torchvision.datasets
import numpy as np
from ResBasicBlock import make_basic_block_layer
from ResBasicBlock import ResBasicBlock


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(3, 2, padding=1)
        self.layer1 = make_basic_block_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = make_basic_block_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = make_basic_block_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = make_basic_block_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.pool1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(ResBasicBlock, [2, 2, 2, 2])

