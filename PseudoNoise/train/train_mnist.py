import os
import sys

absPath = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(absPath)
sys.path.append('/root/github/Ophiuchus')

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Laplace
torch.set_printoptions(precision=10)

import utils
from PseudoNoise.nn.lenet import lenet5


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

            # target -= 1
            data, target = data.to(device), target.to(device)
            output = model(data)

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.9, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--lam', type=float, default=1.,
                        help='lambda used in Oph. (default: 1)')
    parser.add_argument('--sig', type=float, default=1e-3,
                        help='scale of Gaussian or Laplacian noise (default: 1e-3)')
    parser.add_argument('--noise', type=str, default=None,
                        help='noise type, gauss, lap, or oph (default: None)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=5, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-pretrained', action='store_true', default=False,
                        help='For saving the current Model')
    args = parser.parse_args()
    use_cuda = args.cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, test_loader = utils.load_MNIST(args.batch_size, test_batch_size=1000)
    model = lenet5().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        with utils.timer('epoch'):
            train(args, model, device, train_loader, optimizer, epoch)
            test(model, device, test_loader)

    if args.save_pretrained:
        torch.save(model.state_dict(), '../pretrained/lenet5.pt')
        print('Saved model in pretrained.')


if __name__ == '__main__':
    main()
