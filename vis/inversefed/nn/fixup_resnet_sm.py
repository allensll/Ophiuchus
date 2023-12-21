import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

__all__ = ['Bias', 'BasicBlock', 'fixup_resnet20', 'fixup_resnet32', 'fixup_resnet56', 'fixup_resnet110']


class Bias(nn.Module):
    def __init__(self):
        super(Bias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, input):
        return input + self.bias


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, relu, in_planes, planes, stride=1, bias=False):
        super(BasicBlock, self).__init__()
        self.relu = relu
        self.bias1a = Bias()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.bias1b = Bias()
        self.bias2a = Bias()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bias2b = Bias()
        self.scale = nn.Parameter(torch.ones(1))
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = LambdaLayer(lambda x:
                                        F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))

    def forward(self, x):
        out = self.conv1(self.bias1a(x))
        out = self.relu(self.bias1b(out))
        out = self.conv2(self.bias2a(out))
        out = self.bias2b(out * self.scale)
        out = self.relu(out + self.shortcut(x))
        return out


class ResNet(nn.Module):
    def __init__(self, relu, pool, bias, num_classes, mixup=False, bs=1, width=1):
        super(ResNet, self).__init__()
        self.S = ResNetS(relu, BasicBlock, [3, 3, 3], width=width)
        self.U = ResNetU(num_classes=num_classes, width=width)

    def forward(self, x):
        x = self.S(x)
        x = self.U(x)
        return x


class ResNetS(nn.Module):
    def __init__(self, relu, block, num_blocks, width=1):
        super(ResNetS, self).__init__()
        self.num_layers = sum(num_blocks)
        self.in_planes = 16 * width

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bias1 = Bias()
        self.relu = relu
        self.layer1 = self._make_layer(relu, block, 16*width, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(relu, block, 32*width, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(relu, block, 64*width, num_blocks[2], stride=2)
        self.bias2 = Bias()

        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.normal_(m.conv1.weight, mean=0, std=np.sqrt(2 / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * self.num_layers ** (-0.5))
                nn.init.constant_(m.conv2.weight, 0)

    def _make_layer(self, relu, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(relu, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(self.bias1(out))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.bias2(out)
        return out


class ResNetU(nn.Module):
    def __init__(self, num_classes=10, width=1):
        super(ResNetU, self).__init__()
        self.linear = nn.Linear(64*width, num_classes)

        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.constant_(m.weight, 0)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.linear(x)
        return out


def fixup_resnet20(relu=nn.ReLU(), pool=nn.MaxPool2d(2), pretrained=False, num_classes=10, width=1):
    model = ResNet(relu, BasicBlock, [3, 3, 3], num_classes=num_classes, width=width)
    if pretrained:
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '../pretrained/fixup_resnet20-{}_c10.pt'.format(width))
        state_dict = torch.load(data_path)
        model.load_state_dict(state_dict)
    return model


def fixup_resnet32(relu=nn.ReLU(), pool=nn.MaxPool2d(2), pretrained=False, num_classes=10, width=1):
    model = ResNet(relu, BasicBlock, [5, 5, 5], num_classes=num_classes, width=width)
    if pretrained:
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '../pretrained/fixup_resnet32-{}_c10.pt'.format(width))
        state_dict = torch.load(data_path)
        model.load_state_dict(state_dict)
    return model


def fixup_resnet56(relu=nn.ReLU(), pool=nn.MaxPool2d(2), pretrained=False, num_classes=10, width=1):
    model = ResNet(relu, BasicBlock, [9, 9, 9], num_classes=num_classes, width=width)
    if pretrained:
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '../pretrained/fixup_resnet56-{}_c10.pt'.format(width))
        state_dict = torch.load(data_path)
        model.load_state_dict(state_dict)
    return model


def fixup_resnet110(relu=nn.ReLU(), pool=nn.MaxPool2d(2), pretrained=False, num_classes=10, width=1):
    model = ResNet(relu, BasicBlock, [18, 18, 18], num_classes=num_classes, width=width)
    if pretrained:
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '../pretrained/fixup_resnet110-{}_c10.pt'.format(width))
        state_dict = torch.load(data_path)
        model.load_state_dict(state_dict)
    return model


def test(net):
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params {:2.3f}M".format(total_params/1e6))
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == '__main__':
    test(fixup_resnet20())
