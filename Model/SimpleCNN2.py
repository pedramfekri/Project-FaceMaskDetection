import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) #3-channel image
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(6 * 110 * 110, 10)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) #224 -> 220 -> 110
        x = x.view(-1, 6 * 110 * 110)
        x = F.relu(self.fc1(x))
        return x

def SimpleCNN2():
    return Net()