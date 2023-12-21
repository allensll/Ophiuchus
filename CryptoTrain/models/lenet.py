import os
import torch
import torch.nn as nn

from .layers import PConv2d, PLinear

torch.manual_seed(1)


class LeNet5(nn.Module):
    def __init__(self, relu, pool, bias, num_classes, n=1):
        super(LeNet5, self).__init__()
        self.conv1 = PConv2d(1, 6, 5, 1, 0, bias=bias, n=n)
        self.pool1 = pool
        self.relu1 = relu
        self.conv2 = PConv2d(6, 16, 5, 1, 0, bias=bias, n=n)
        self.pool2 = pool
        self.relu2 = relu

        self.full1 = PLinear(4*4*16, 120, bias=bias, n=n)
        self.relu3 = relu
        self.full2 = PLinear(120, num_classes, bias=bias, n=n)

        if bias:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(self.pool1(x))
        x = self.relu2(self.pool2(self.conv2(x)))
        x = x.view(-1, 4*4*16)
        x = self.full2(self.relu3(self.full1(x)))
        return x


class LeNet5S(nn.Module):
    def __init__(self, relu, pool, bias, num_classes, n=2):
        super(LeNet5S, self).__init__()
        self.conv1 = PConv2d(1, 6, 5, 1, 0, bias=bias, n=n)
        self.pool1 = pool
        self.relu1 = relu
        self.conv2 = PConv2d(6, 16, 5, 1, 0, bias=bias, n=n)
        self.pool2 = pool
        self.relu2 = relu
        self.full1 = PLinear(4*4*16, 120, bias=bias, n=n)
        # self.relu3 = relu
        # self.full2 = PLinear(120, num_classes, bias=bias, n=n)

        if bias:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(self.pool1(x))
        x = self.relu2(self.pool2(self.conv2(x)))
        x = x.view(-1, 4*4*16)
        x = self.full1(x)
        return x


class LeNet5U(nn.Module):
    def __init__(self, relu, pool, bias, num_classes=10, n=1):
        super(LeNet5U, self).__init__()
        self.conv1 = PConv2d(1, 6, 5, 1, 0, bias=bias, n=n)
        self.pool1 = pool
        self.relu1 = relu
        self.conv2 = PConv2d(6, 16, 5, 1, 0, bias=bias, n=n)
        self.pool2 = pool
        self.relu2 = relu
        self.full1 = PLinear(4*4*16, 120, bias=bias, n=n)

        self.relu3 = nn.ReLU()
        self.full2 = nn.Linear(120, num_classes, bias=bias)

        if bias:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.full2(self.relu3(x))
        return x


def lenet5(relu=nn.ReLU(), pool=nn.MaxPool2d(2), pretrained=False, bias=True, num_classes=10, n=1, role=None):
    if role:
        if role == 'S':
            model = LeNet5S(relu=relu, pool=pool, bias=bias, num_classes=num_classes)
        elif role == 'U':
            model = LeNet5U(relu=relu, pool=pool, bias=bias, num_classes=num_classes)
        else:
            ValueError('The role should S or U, but get {}'.format(role))
    else:
        model = LeNet5(relu=relu, pool=pool, bias=bias, num_classes=num_classes, n=n)
    if pretrained:
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pretrained/lenet5.pth')
        state_dict = torch.load(data_path)['state_dict']
        model.load_state_dict(state_dict)
    return model
