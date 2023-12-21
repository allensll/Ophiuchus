# https://github.com/akamaster/pytorch_resnet_cifar10

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

__all__ = ['resnet20', 'resnet32', 'resnet56', 'resnet110']


def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class Pre(nn.Module):
    def __init__(self, mixup=False, bs=1):
        super(Pre, self).__init__()
        self.mean = nn.Parameter(torch.Tensor([0, 0, 0]).view(-1, 1, 1))
        self.std = nn.Parameter(torch.Tensor([0.5, 0.5, 0.5]).view(-1, 1, 1))
        self.mixup = mixup
        if self.mixup:
            self.lam = nn.Parameter(torch.Tensor([0.5]*bs))

    def forward(self, x):
        x_ = (x - self.mean) / self.std
        return x_


class ResNet(nn.Module):
    def __init__(self, relu, pool, bias, num_classes, usepre, mixup=False, bs=1):
        super(ResNet, self).__init__()
        self.S = ResNetS(BasicBlock, [3, 3, 3])
        self.U = ResNetU(num_classes=num_classes)

        self.pre = Pre(mixup=mixup, bs=bs)

        self.usepre = usepre

    def forward(self, x):
        if self.usepre:
            x = self.pre(x)
        x = self.S(x)
        x = self.U(x)
        return x

class ResNetS(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNetS, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        return out


class ResNetU(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetU, self).__init__()
        self.linear = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.linear(x)
        return out


def resnet20(relu=nn.ReLU(), pool=nn.MaxPool2d(2), pre=True, pretrained=False, full=True, num_classes=10):
    model = ResNet(relu, BasicBlock, [3, 3, 3], num_classes=num_classes, usepre=pre)
    if pretrained:
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '../pretrained/resnet20_c10_e1.pt')
        state_dict = torch.load(data_path)
        model.S.load_state_dict(state_dict[0])
        if full:
            model.U.load_state_dict(state_dict[1])
    return model


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size()) > 1, net.parameters()))))