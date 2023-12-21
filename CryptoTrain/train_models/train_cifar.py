import os
import sys

absPath = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(absPath)

import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F

import utils
import CryptoTrain.models as models


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data, target = data.to(device), target.to(device)
        output = model(data)
        # output.register_hook(mask_grads)

        loss = F.cross_entropy(output, target)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
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


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--wd', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.9, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-pretrained', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = args.cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, test_loader = utils.load_CIFAR10(args.batch_size, test_batch_size=1000)

    model = models.resnet20(pre=False, num_classes=10).to(device)
    optimizer = optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=args.wd)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs/2), int(3*args.epochs/4)], gamma=0.1)

    for epoch in range(1, args.epochs + 1):
        with utils.timer('Epoch '):
            train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        lr_scheduler.step()
        break

    if args.save_pretrained:
        torch.save([model.S.state_dict(), model.U.state_dict()], '../../pretrained/resnet20_c10_e1.pt')
        print('Saved model in pretrained.')


def acc():
    test_batch_size = 1000

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(147)
    device = torch.device('cuda' if use_cuda else 'cpu')
    # device = torch.device('cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader, test_loader = utils.load_CIFAR10(128, test_batch_size=test_batch_size, **kwargs)
    model = models.resnet20(pre=False, pretrained=True)
    model = model.to(device)
    model.eval()

    test(model, device, test_loader)


if __name__ == '__main__':
    # main()
    acc()
