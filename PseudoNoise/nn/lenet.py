import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    def __init__(self, relu, pool, bias, num_classes, mixup=False, bs=1):
        super(LeNet5, self).__init__()
        self.S = LeNet5S(relu=relu, pool=pool, bias=bias)
        self.U = LeNet5U(relu=relu, pool=pool, bias=bias, num_classes=num_classes)

    def forward(self, x):
        x = self.S(x)
        x = self.U(x)
        return x


class LeNet5S(nn.Module):
    def __init__(self, relu, pool, bias):
        super(LeNet5S, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, 1, 0, bias=bias)
        self.pool1 = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(6, 16, 5, 1, 0, bias=bias)
        self.pool2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU()
        self.full1 = nn.Linear(4*4*16, 120, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.pool1(x)
        # x = self.pool2(self.conv2(x))
        x = self.relu1(self.pool1(x))
        x = self.relu2(self.pool2(self.conv2(x)))
        x = x.view(-1, 4*4*16)
        x = self.full1(x)
        return x


class LeNet5U(nn.Module):
    def __init__(self, relu, pool, bias, num_classes=10):
        super(LeNet5U, self).__init__()
        self.relu3 = nn.ReLU()
        self.full2 = nn.Linear(120, num_classes, bias=bias)

    def forward(self, x):
        x = self.full2(self.relu3(x))
        # x = self.full2(x)
        return x


def lenet5(relu=nn.ReLU(), pool=nn.MaxPool2d(2), pretrained=False, bias=True, num_classes=10):
    model = LeNet5(relu=relu, pool=pool, bias=bias, num_classes=num_classes)
    if pretrained:
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '../pretrained/lenet5.pt')
        state_dict = torch.load(data_path)
        model.load_state_dict(state_dict)
    return model
