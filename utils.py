import os
import time
import random
from contextlib import contextmanager

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import Dataset, Sampler
from torchvision import datasets, transforms

data_path = os.path.join(os.path.dirname(__file__), 'dataset')


@contextmanager
def timer(text=''):
    """Helper for measuring runtime"""

    time_start = time.perf_counter()
    yield
    print('---{} time: {:.3f} s'.format(text, time.perf_counter()-time_start))


def contrastive_loss(output, target, cls=10):
    """compute contrastive loss."""

    target_ = target.tolist()
    n = len(target_)
    book = [[] for _ in range(cls)]
    for i in range(n):
        book[target_[i]].append(i)

    idx_p = [0 for _ in range(n)]
    for pag in book:
        # if len(pag) == 1:
        #     print('0')
        for j in range(len(pag)):
            idx_p[pag[j]] = pag[(j+1) % len(pag)]

    idx_n = list(range(1, n+1))
    idx_n[n-1] = 0

    anchor, positive, negative = output, output[idx_p], output[idx_n]
    loss = F.triplet_margin_with_distance_loss(anchor, positive, negative,
                                            distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y), margin=1)
    return loss


def contrastive_loss2(output, target, cls=10):
    """compute contrastive loss faster."""

    bs = output.shape[0]
    idx_p = torch.arange(bs, dtype=torch.long)
    idx_n = torch.arange(bs, dtype=torch.long)
    for i in range(bs):
        jn = jp = (i + 1) % bs
        while jn != i:
            if target[jn] != target[i]:
                idx_n[i] = jn
                break
            else:
                jn = (jn + 1) % bs
        while jp != i:
            if target[jp] == target[i]:
                idx_p[i] = jp
                break
            else:
                jp = (jp + 1) % bs

    anchor, positive, negative = output, output[idx_p], output[idx_n]
    loss = F.triplet_margin_with_distance_loss(anchor, positive, negative,
                                               distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y), margin=1)
    return loss


def partition(data, id, alpha=10, n=2):
    """secure split inpute images."""

    assert id == 1 or id == 2
    # To ensure generate the same random number
    # torch.random.manual_seed(147)
    # x1 = torch.rand_like(data) * alpha
    # global tmp_x
    xi = list()
    for i in range(n-1):
        # if tmp_x is None:
        #     tmp_x = torch.rand(data.shape)
        # x_r = tmp_x * alpha / (n - 1)
        x_r = torch.rand(data.shape) * alpha / (n-1)
        xi.append(x_r)

    # x1 = torch.rand(data.shape) * alpha
    x_r = data.clone().detach()
    for i in xi:
        x_r -= i
    xi.append(x_r)
    return xi[id-1]


def load_pretrained(model, name):
    path = os.path.join(os.path.dirname(__file__), 'pretrained', name)
    model.load_state_dict(torch.load(path))

    for n, l in model.named_parameters():
        if 'bias' in n:
            l.data = l.data * 0.5


class NewSampler(torch.utils.data.Sampler):
    """Sampler to generate triplet."""

    def __init__(self, data_source, cls_numbers):
        self.data_source = data_source
        self.num_samples = len(self.data_source)
        self.cls = cls_numbers

    def __iter__(self):
        n_cls = int(self.num_samples / self.cls)
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator()
        generator.manual_seed(seed)

        idx_by_cls = list()
        for i in range(self.cls):
            tmp_idx = (torch.randperm(n_cls, generator=generator) + (i*n_cls)).tolist()
            idx_by_cls.append(tmp_idx)
        res = list()
        last_cls = -1   # add
        for i in range(n_cls):
            tmp_cls = torch.randperm(self.cls, generator=generator).tolist()
            if tmp_cls[-1] == last_cls: # add
                tmp_cls[0], tmp_cls[1] = tmp_cls[1], tmp_cls[0] # add
            for j in tmp_cls:
                res.append(idx_by_cls[j][i])
            last_cls = tmp_cls[-1] # add
        return iter(res)

    def __len__(self):
        return self.num_samples


def load_MNIST(batch_size, test_batch_size=1000, **kwargs):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_path, train=True, download=True,
                       transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_path, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=test_batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader


def load_FashionMNIST(batch_size, test_batch_size=1000, new=False, **kwargs):
    dataset = datasets.ImageFolder(
        os.path.join(data_path, 'fashion-mnist-raw', 'train'),
        transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]))
    sampler = NewSampler(dataset, 10) if new else torch.utils.data.RandomSampler(dataset)
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, sampler=sampler, pin_memory=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        dataset=datasets.ImageFolder(
            os.path.join(data_path, 'fashion-mnist-raw', 'test'),
            transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])),
        batch_size=test_batch_size, shuffle=False, pin_memory=True, **kwargs)

    return train_loader, test_loader


def load_CIFAR10(batch_size, test_batch_size=1000, new=False, **kwargs):
    dataset = datasets.ImageFolder(
        os.path.join(data_path, 'cifar10-raw', 'train'),
        transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]))
    sampler = NewSampler(dataset, 10) if new else torch.utils.data.RandomSampler(dataset)
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, sampler=sampler, pin_memory=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        dataset=datasets.ImageFolder(
            os.path.join(data_path, 'cifar10-raw', 'test'),
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])),
        batch_size=test_batch_size, shuffle=False, pin_memory=True, **kwargs)

    return train_loader, test_loader


def load_CIFAR100(batch_size, test_batch_size=1000, new=False, **kwargs):
    dataset = datasets.ImageFolder(
        os.path.join(data_path, 'cifar100-raw', 'train'),
        transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4866, 0.4412), (0.2673, 0.2564, 0.2761)),
        ]))
    sampler = NewSampler(dataset, 100) if new else torch.utils.data.RandomSampler(dataset)
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, sampler=sampler, pin_memory=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        dataset=datasets.ImageFolder(
            os.path.join(data_path, 'cifar100-raw', 'test'),
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4866, 0.4412), (0.2673, 0.2564, 0.2761)),
            ])),
        batch_size=test_batch_size, shuffle=False, pin_memory=True, **kwargs)

    return train_loader, test_loader


def load_CelebA(batch_size, test_batch_size=1000, new=False, **kwargs):
    dataset = datasets.ImageFolder(
        os.path.join(data_path, 'CelebA', 'train'),
        transforms.Compose([
            transforms.Resize(80),
            transforms.RandomCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25)),
        ]))
    sampler = NewSampler(dataset, 2) if new else torch.utils.data.RandomSampler(dataset)
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, sampler=sampler, pin_memory=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        dataset=datasets.ImageFolder(
            os.path.join(data_path, 'CelebA', 'test'),
            transforms.Compose([
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25)),
            ])),
        batch_size=test_batch_size, shuffle=False, pin_memory=True, **kwargs)

    return train_loader, test_loader