import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

from .layers import PConv2d, PLinear, PBias

__all__ = ['fixup_resnet20', 'fixup_resnet32', 'fixup_resnet56', 'fixup_resnet110']

torch.manual_seed(1)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, relu, in_planes, planes, stride=1, n=1):
        super(BasicBlock, self).__init__()
        self.relu = relu
        self.bias1a = PBias(n)
        self.conv1 = PConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, n=n)
        self.bias1b = PBias(n)
        self.bias2a = PBias(n)
        self.conv2 = PConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, n=n)
        self.bias2b = PBias(n)
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
    def __init__(self, relu, block, num_blocks, num_classes=10, n=1):
        super(ResNet, self).__init__()
        self.num_layers = sum(num_blocks)
        self.in_planes = 16

        self.conv1 = PConv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False, n=n)
        self.bias1 = PBias(n)
        self.relu = relu
        self.layer1 = self._make_layer(relu, block, 16, num_blocks[0], stride=1, n=n)
        self.layer2 = self._make_layer(relu, block, 32, num_blocks[1], stride=2, n=n)
        self.layer3 = self._make_layer(relu, block, 64, num_blocks[2], stride=2, n=n)
        self.bias2 = PBias(n)
        self.linear = PLinear(64, num_classes, n=n)

        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.normal_(m.conv1.weight, mean=0, std=np.sqrt(2 / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * self.num_layers ** (-0.5))
                nn.init.constant_(m.conv2.weight, 0)
            elif isinstance(m, PLinear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, relu, block, planes, num_blocks, stride, n):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(relu, self.in_planes, planes, stride, n))
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
        out = self.linear(self.bias2(out))
        return out


class ResNetS(nn.Module):
    def __init__(self, relu, block, num_blocks, n=2):
        super(ResNetS, self).__init__()
        self.num_layers = sum(num_blocks)
        self.in_planes = 16

        self.conv1 = PConv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False, n=n)
        self.bias1 = PBias(n)
        self.relu = relu
        self.layer1 = self._make_layer(relu, block, 16, num_blocks[0], stride=1, n=n)
        self.layer2 = self._make_layer(relu, block, 32, num_blocks[1], stride=2, n=n)
        self.layer3 = self._make_layer(relu, block, 64, num_blocks[2], stride=2, n=n)
        self.bias2 = PBias(n)

        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.normal_(m.conv1.weight, mean=0, std=np.sqrt(2 / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * self.num_layers ** (-0.5))
                nn.init.constant_(m.conv2.weight, 0)

    def _make_layer(self, relu, block, planes, num_blocks, stride, n):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(relu, self.in_planes, planes, stride, n))
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
    def __init__(self, relu, block, num_blocks, num_classes=10, n=1):
        super(ResNetU, self).__init__()
        self.num_layers = sum(num_blocks)
        self.in_planes = 16

        self.conv1 = PConv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False, n=n)
        self.bias1 = PBias(n)
        self.relu = relu
        self.layer1 = self._make_layer(relu, block, 16, num_blocks[0], stride=1, n=n)
        self.layer2 = self._make_layer(relu, block, 32, num_blocks[1], stride=2, n=n)
        self.layer3 = self._make_layer(relu, block, 64, num_blocks[2], stride=2, n=n)
        self.bias2 = PBias(n)
        self.linear = PLinear(64, num_classes, n=n)

        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.normal_(m.conv1.weight, mean=0, std=np.sqrt(2 / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * self.num_layers ** (-0.5))
                nn.init.constant_(m.conv2.weight, 0)
            elif isinstance(m, PLinear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, relu, block, planes, num_blocks, stride, n):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(relu, self.in_planes, planes, stride, n))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.linear(x)
        return out


def fixup_resnet20(relu=nn.ReLU(), pool=nn.MaxPool2d(2), pretrained=False, num_classes=10, n=1, role=None):
    if role:
        if role == 'S':
            model = ResNetS(relu, BasicBlock, [3, 3, 3])
        elif role == 'U':
            model = ResNetU(relu, BasicBlock, [3, 3, 3], num_classes=num_classes)
        else:
            ValueError('The role should S or U, but get {}'.format(role))
    else:
        model = ResNet(relu, BasicBlock, [3, 3, 3], num_classes=num_classes, n=n)
    if pretrained:
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '../pretrained/fixup_resnet20_c10_e100.pt')
        state_dict = torch.load(data_path)
        model.load_state_dict(state_dict)
    return model


def fixup_resnet32(relu=nn.ReLU(), pool=nn.MaxPool2d(2), pretrained=False, num_classes=10, n=1, role=None):
    if role:
        if role == 'S':
            model = ResNetS(relu, BasicBlock, [5, 5, 5])
        elif role == 'U':
            model = ResNetU(relu, BasicBlock, [5, 5, 5], num_classes=num_classes)
        else:
            ValueError('The role should S or U, but get {}'.format(role))
    else:
        model = ResNet(relu, BasicBlock, [5, 5, 5], num_classes=num_classes, n=n)
    if pretrained:
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '../pretrained/fixup_resnet32_c10_e100.pt')
        state_dict = torch.load(data_path)
        model.load_state_dict(state_dict)
    return model


def fixup_resnet56(relu=nn.ReLU(), pool=nn.MaxPool2d(2), pretrained=False, num_classes=10, n=1, role=None):
    if role:
        if role == 'S':
            model = ResNetS(relu, BasicBlock, [9, 9, 9])
        elif role == 'U':
            model = ResNetU(relu, BasicBlock, [9, 9, 9], num_classes=num_classes)
        else:
            ValueError('The role should S or U, but get {}'.format(role))
    else:
        model = ResNet(relu, BasicBlock, [9, 9, 9], num_classes=num_classes, n=n)
    if pretrained:
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '../pretrained/fixup_resnet56_c10_e100.pt')
        state_dict = torch.load(data_path)
        model.load_state_dict(state_dict)
    return model



def fixup_resnet110(relu=nn.ReLU(), pool=nn.MaxPool2d(2), pretrained=False, num_classes=10, n=1, role=None):
    if role:
        if role == 'S':
            model = ResNetS(relu, BasicBlock, [18, 18, 18])
        elif role == 'U':
            model = ResNetU(relu, BasicBlock, [18, 18, 18], num_classes=num_classes)
        else:
            ValueError('The role should S or U, but get {}'.format(role))
    else:
        model = ResNet(relu, BasicBlock, [18, 18, 18], num_classes=num_classes, n=n)
    if pretrained:
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '../pretrained/fixup_resnet110_c10_e100.pt')
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
    test(resnet20())
