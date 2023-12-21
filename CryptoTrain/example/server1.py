import os
import sys
import argparse
absPath = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
sys.path.append(absPath)

import torch
import torch.optim as optim

import CryptoTrain.mpc as mpc
import utils


def main():
    parser = argparse.ArgumentParser(description='Server s')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10', 'cifar100', 'imagenet'],
                        help='choose dataset (default: MNIST)')
    parser.add_argument('--agtip', type=str, default='127.0.0.1',
                        help='set ip of nl2pc agent (default: localhost)')
    parser.add_argument('--agtport', type=int, default=20202,
                        help='set port of nl2pc agent (default: 20202)')
    parser.add_argument('--ipu', type=str, default='127.0.0.1',
                        help='set ip of client (default: localhost)')
    parser.add_argument('--portu', type=int, default=14714,
                        help='set port of client (default: 14714)')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--wd', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--nthreads', type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    if args.dataset == 'mnist':
        model = 'lenet5'
    elif args.dataset == 'cifar10':
        model = 'fixup_resnet20'
    elif args.dataset == 'cifar100':
        model = 'fixup_resnet20'
    elif args.dataset == 'imagenet':
        model = 'fixup_resnet50'
    else:
        ValueError('Not exist dataset:{}'.format(args.dataset))

    srv = mpc.Server(args.dataset, model, relu='mpcrelu', pool='mpcmaxpool', t='s', scheme='ckks',
                     agtaddr=args.agtip, agtport=args.agtport, nthreads=args.nthreads)
    srv.connect(args.ipu, args.portu) # 58.213.25.18 XXXXX

    srv.model.train()

    if args.dataset == 'mnist':
        optimizer = optim.SGD(srv.model.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(srv.model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs/2), int(3*args.epochs/4)], gamma=0.1)

    for epoch in range(1, args.epochs + 1):
        data = srv.get_input()
        while data != 'EpochEND':
            print('------ get new input ------')
            optimizer.zero_grad()
            data = data.to(device)

            output = srv.model(data)
            srv.send_feature(output)
            grad = srv.get_grad()
            with utils.timer('---------------- back time -------- '):
                output.backward(grad)
                srv.update_grad()

                optimizer.step()

            data = srv.get_input()

        if args.dataset != 'mnist':
            lr_scheduler.step()

    srv.close()


if __name__ == '__main__':
    main()
