import os
import sys
import argparse
absPath = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
sys.path.append(absPath)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import CryptoTrain.models as models
import CryptoTrain.mpc as mpc
import utils


def partition(data, alpha=10, n=2):
    xi = list()
    for i in range(n-1):
        x_r = torch.rand(data.shape) * alpha / (n-1)
        xi.append(x_r)

    x_r = data.clone().detach()
    for i in xi:
        x_r -= i
    xi.append(x_r)
    return xi


def main():
    parser = argparse.ArgumentParser(description='User')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10', 'cifar100', 'imagenet'])
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--wd', type=float, default=1e-3,
                        help='weight decay (default: 1e-3)')
    parser.add_argument('--noise', action='store_true', default=False,
                        help='add noise or not (default: false)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    batch_size = 1000
    if args.dataset == 'mnist':
        train_loader, test_loader = utils.load_MNIST(args.batch_size, test_batch_size=batch_size)
        n_cls = 10
    elif args.dataset == 'cifar10':
        train_loader, test_loader = utils.load_CIFAR10(args.batch_size, test_batch_size=batch_size)
        n_cls = 10
    elif args.dataset == 'cifar100':
        train_loader, test_loader = utils.load_CIFAR100(args.batch_size, test_batch_size=batch_size)
        n_cls = 100
    elif args.dataset == 'imagenet':
        train_loader, test_loader = utils.load_ImageNet(args.batch_size, test_batch_size=batch_size)
    else:
        ValueError('Not exist dataset:{}'.format(args.dataset))

    user = mpc.User()
    user.start('127.0.0.1', 14714) # 6006

    if args.dataset == 'mnist':
        net = models.lenet5(role='U').to(device)
        optimizer = optim.SGD(net.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()
    else:
        net = models.fixup_resnet20(role='U').to(device)
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs/2), int(3*args.epochs/4)], gamma=0.1)
        criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        with utils.timer('epoch'):
            net.train()
            correct = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                with utils.timer('----------------- batch all time -------- '):
                    optimizer.zero_grad()
                    img, target = data.to(device), target.to(device)
                    data = partition(img, alpha=10, n=2)
                    user.upload(data)
                    feature = user.get_feature()
                    feature.requires_grad = True
                    feature = feature.to(device)
                    output = net(feature)

                    loss = criterion(output, target)
                    # add pseudo noise
                    if args.noise:
                        loss2 = utils.contrastive_loss2(feature, target, n_cls)
                        alpha = 1 * (loss / loss2).detach()
                        loss = loss + alpha * loss2


                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    correct += pred.eq(target.view_as(pred)).sum().item()

                    loss.backward()
                    user.send_grad(feature.grad)

                    optimizer.step()

                if batch_idx % args.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(img), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), loss.item()))
        user.upload(['EpochEND', 'EpochEND'])
        print('Train Epoch: {} \tAcc: {:.2f}%'.format(epoch, 100. * correct / len(train_loader.dataset)))
        if args.dataset != 'mnist':
            lr_scheduler.step()

    user.close()


if __name__ == '__main__':
    main()
