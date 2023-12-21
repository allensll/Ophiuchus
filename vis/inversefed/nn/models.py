"""Define basic models and translate some torchvision stuff."""
"""Stuff from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py."""
import torch
import torchvision
import torch.nn as nn

from torchvision.models.resnet import Bottleneck, BasicBlock
from .densenet import _DenseNet, _Bottleneck

from collections import OrderedDict
import numpy as np
from ..utils import set_random_seed

from .fixup_resnet_sm import fixup_resnet20, fixup_resnet32
from .resnet_sm import resnet20, resnet32
from .lenet import lenet5


def construct_model(model, num_classes=10, seed=None, modelkey=None):
    """Return various models."""
    if modelkey is None:
        if seed is None:
            model_init_seed = np.random.randint(0, 2**32 - 10)
        else:
            model_init_seed = seed
    else:
        model_init_seed = modelkey
    set_random_seed(model_init_seed)

    num_channels = 3
    if model == 'ResNet20':
        model = resnet20(pretrained=False, num_classes=num_classes, width=1)
    elif model == 'ResNet20-4':
        model = resnet20(pretrained=False, num_classes=num_classes, width=4)
    elif model == 'ResNet32':
        model = resnet32(pretrained=False, num_classes=num_classes, width=1)
    elif model == 'ResNet32-4':
        model = resnet32(pretrained=False, num_classes=num_classes, width=4)
    elif model == 'ResNet32-8':
        model = resnet32(pretrained=False, num_classes=num_classes, width=8)
    elif model == 'NF-ResNet20':
        model = fixup_resnet20(pretrained=False, num_classes=num_classes, width=1)
    elif model == 'NF-ResNet20-4':
        model = fixup_resnet20(pretrained=False, num_classes=num_classes, width=4)
    elif model == 'NF-ResNet32':
        model = fixup_resnet32(pretrained=False, num_classes=num_classes, width=1)
    elif model == 'NF-ResNet32-4':
        model = fixup_resnet32(pretrained=False, num_classes=num_classes, width=4)
    elif model == 'NF-ResNet32-8':
        model = fixup_resnet32(pretrained=False, num_classes=num_classes, width=8)
    elif model == 'LeNet5':
        model = lenet5(pretrained=False)
        num_channels = 1
    else:
        raise NotImplementedError('Model not implemented.')

    print(f'Model initialized with random key {model_init_seed}.')
    return model, Pre(num_channels=num_channels), model_init_seed


class Pre(nn.Module):
    def __init__(self, mixup=False, bs=1, num_channels=3):
        super(Pre, self).__init__()
        self.mean = nn.Parameter(torch.Tensor([0.5]*num_channels).view(-1, 1, 1))
        self.std = nn.Parameter(torch.Tensor([0.5]*num_channels).view(-1, 1, 1))
        self.mixup = mixup
        if self.mixup:
            self.lam = nn.Parameter(torch.Tensor([0.5]*bs))

    def forward(self, x):
        x_ = (x - self.mean) / self.std
        return x_