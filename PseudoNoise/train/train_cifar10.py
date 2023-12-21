import os
import sys

absPath = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(absPath)
sys.path.append('/root/github/Ophiuchus')

import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Laplace

import utils
from PseudoNoise.nn.resnet_sm import resnet20, resnet32
from PseudoNoise.nn.fixup_resnet_sm import fixup_resnet20, fixup_resnet32


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data, target = data.to(device), target.to(device)

        alpha = torch.Tensor([0])
        if args.noise is None:
            output_ = model.S(data)
            output = model.U(output_)
            loss = F.cross_entropy(output, target)
        elif args.noise == 'gauss':
            output_ = model.S(data)
            output = model.U(output_)
            def hook(grad):
                m = Normal(torch.tensor([0.0]), torch.tensor([args.sig]))
                n = m.sample(grad.size()).squeeze(-1)
                n = n.to(device)
                grad_ = grad + n
                return grad_
            output.register_hook(hook)
            loss = F.cross_entropy(output, target)
        elif args.noise == 'lap':
            output_ = model.S(data)
            output = model.U(output_)
            def hook(grad):
                m = Laplace(torch.tensor([0.0]), torch.tensor([args.sig]))
                n = m.sample(grad.size()).squeeze(-1)
                n = n.to(device)
                grad_ = grad + n
                return grad_
            output.register_hook(hook)
            loss = F.cross_entropy(output, target)
        elif args.noise == 'pseudo':
            output_ = model.S(data)
            output = model.U(output_)
            loss1 = F.cross_entropy(output, target)
            loss2 = utils.contrastive_loss2(output_, target, 10)
            alpha = args.lam * (loss1 / loss2).detach()
            loss = loss1 + alpha * loss2
        else:
            raise ValueError('noise should be gauss or pseudo!')

        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] \talpha: {:.3f}\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), alpha.item(), loss.item()))
            if args.dry_run:
                break
    print('Train Epoch: {} \tAcc: {:.2f}%'.format(epoch, 100. * correct / len(train_loader.dataset)))


def test(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:

            data, target = data.to(device), target.to(device)
            output = model(data)

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--width', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--wd', type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--gamma', type=float, default=1, metavar='M',
                        help='reduce the fixup learning rate (default: 1)')
    parser.add_argument('--lam', type=float, default=1.,
                        help='lambda used in Oph. (default: 1)')
    parser.add_argument('--sig', type=float, default=1e-3,
                        help='scale of Gaussian or Laplacian noise (default: 1e-3)')
    parser.add_argument('--noise', type=str, default=None,
                        help='noise type, gauss, lap, or oph (default: None)')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=144, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-pretrained', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = args.cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, test_loader = utils.load_CIFAR10(args.batch_size, test_batch_size=1000, new=True, num_workers=4)

    nf = True
    if nf:
        model = fixup_resnet20(pretrained=False, num_classes=10, width=args.width).to(device)
        parameters_bias = [p[1] for p in model.named_parameters() if 'bias' in p[0]]
        parameters_scale = [p[1] for p in model.named_parameters() if 'scale' in p[0]]
        parameters_others = [p[1] for p in model.named_parameters() if not ('bias' in p[0] or 'scale' in p[0])]
        optimizer = optim.SGD(
            [{'params': parameters_bias, 'lr': args.lr/args.gamma},
             {'params': parameters_scale, 'lr': args.lr/args.gamma},
             {'params': parameters_others}],
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.wd)
        path = '../pretrained/acc_nf_resnet20-{}-{}_c10.pkl'.format(args.width, args.lam)
    else:
        model = resnet20(pretrained=False, num_classes=10, width=args.width).to(device)
        optimizer = optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=args.wd)
        path = '../pretrained/acc_resnet20-{}_c10.pkl'.format(args.width)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs/2), int(3*args.epochs/4)], gamma=0.1)

    test_acc = list()
    for epoch in range(1, args.epochs + 1):
        with utils.timer('Epoch '):
            train(args, model, device, train_loader, optimizer, epoch)
        acc = test(model, device, test_loader)
        test_acc.append(acc)
        lr_scheduler.step()
        # break

    print(max(test_acc))
    # import pickle
    # with open(path, 'wb') as f:
    #     pickle.dump(test_acc, f)
    if args.save_pretrained:
        torch.save(model.state_dict(), '../pretrained/nf_resnet20-{}_c10.pt'.format(args.width))
        print('Saved model in pretrained.')


def acc():
    test_batch_size = 1000

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(147)
    device = torch.device('cuda' if use_cuda else 'cpu')
    # device = torch.device('cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader, test_loader = utils.load_CIFAR10(128, test_batch_size=test_batch_size, **kwargs)
    model = resnet20(pretrained=True)
    model = model.to(device)
    model.eval()

    test(model, device, test_loader)


if __name__ == '__main__':
    main()
    # acc()
