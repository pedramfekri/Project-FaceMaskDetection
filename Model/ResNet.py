# branch2
import torch.nn as nn
import torch.nn.functional as F
from Model.ResBasicBlock import make_basic_block_layer
from Model.ResBasicBlock import ResBasicBlock


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=3):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(3, 2, padding=1)
        self.layer1 = make_basic_block_layer(block, 64, 64, num_blocks[0], stride=1)
        self.layer2 = make_basic_block_layer(block, 64, 128, num_blocks[1], stride=2)
        self.layer3 = make_basic_block_layer(block, 128, 256, num_blocks[2], stride=2)
        self.layer4 = make_basic_block_layer(block, 256, 512, num_blocks[3], stride=2)

        self.linear = nn.Linear(512*block.expansion, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.pool1(out)
        # print(out.size())
        # print("first")
        out = self.layer1(out)
        # print(64)
        out = self.layer2(out)
        # print(128)
        out = self.layer3(out)
        # print(256)
        out = self.layer4(out)
        # print(512)
        out = F.avg_pool2d(out, 4)
        # print(out.size())
        out = out.view(out.size(0), -1) # flattening
        # print(out.size)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(ResBasicBlock, [2, 2, 2, 2])

# ResNet18()

