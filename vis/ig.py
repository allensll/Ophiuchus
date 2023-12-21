import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import lpips
from torchvision.transforms.functional import resize
from collections import defaultdict
from PIL import Image

import sys
sys.path.append('/root/github/Ophiuchus')

from inversefed.nn.lenet import lenet5
from inversefed.nn.fixup_resnet_sm import Bias, fixup_resnet20, fixup_resnet32, BasicBlock
from inversefed.nn.resnet_sm import resnet20, resnet32

import utils


def grid_plot(name, tensor, label):
    tensor = tensor.clone().detach()
    tensor.mul_(ds).add_(dm).clamp_(0, 1)
    if tensor.shape[0] == 1:
        plt.imshow(tensor[0].permute(1, 2, 0).cpu())
    else:
        fig, axes = plt.subplots(1, tensor.shape[0], figsize=(tensor.shape[0]*12, 12))
        for i, im in enumerate(tensor):
            axes[i].imshow(im.permute(1, 2, 0).cpu())
    plt.tight_layout()
    plt.axis('off')
    # plt.show()
    plt.savefig(name)


parser = argparse.ArgumentParser(description='vis')
parser.add_argument('--arch', type=str, default='ResNet20-4',
                    help='LeNet5, ResNet20, ResNet20-4, NF-ResNet20-4, NF-ResNet32')
parser.add_argument('--dataset', type=str, default='CIFAR10',
                    help='MNIST, CIFAR10, CIFAR100, CelebA')
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--gtype', type=str, default='w',
                    help='gradient type : f: feature gradient, w: weight gradient')
parser.add_argument('--ltype', type=str, default='ce',
                    help='loss type : ce, gauss, lap, oph')
parser.add_argument('--mtype', type=str, default='F',
                    help='model type : S: server model, F: full model')
parser.add_argument('--lam', type=float, default=0.5)
parser.add_argument('--sig', type=float, default=1e-3)
parser.add_argument('--id', type=int, default=1,
                    help='0 or 1')
parser.add_argument('--max_iter', type=int, default=8_000)
parser.add_argument('--restarts', type=int, default=1)
parser.add_argument('--tv', type=float, default=1e-6)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

arch = args.arch
dataset = args.dataset
num_classes = args.num_classes
gtype = args.gtype
ltype = args.ltype
mtype = args.mtype

num_images = 1
trained = False

config = dict(signed=True,
              boxed=True,
              cost_fn='sim',
              indices='def',
              weights='equal',
              lr=0.1,
              optim='adam',
              restarts=args.restarts,
              max_iterations=args.max_iter,
              total_variation=args.tv,
              init='randn',
              filter='none',
              lr_decay=True,
              scoring_choice='loss')

# ## System setup:
import inversefed
setup = inversefed.utils.system_startup()
defs = inversefed.training_strategy('conservative')
loss_fn, trainloader, validloader = inversefed.construct_dataloaders(dataset, defs, data_path='/root/data')
print('-- datasets finished ----------------')

model, pre, _, = inversefed.construct_model(arch, seed=args.seed, num_classes=num_classes)
model.to(**setup)
if mtype == 'F':
    pre.mean.data = torch.Tensor(inversefed.consts.d_m[dataset]).view(-1, 1, 1)
    pre.std.data = torch.Tensor(inversefed.consts.d_std[dataset]).view(-1, 1, 1)
pre.to(**setup)

# # # Choice image
# ground_truth, labels = [], []
ground_truth2, labels2 = [], []
# you can test the different ids
if dataset == 'MNIST':
    idx = [2, 4, 5, 6] # mnist labels [1 4 1 4]
elif dataset == 'Fashion-MNIST':
    idx = [3, 4, 5, 7]
elif dataset == 'CIFAR10':
    idx = [0, 6, 8, 9]
elif dataset == 'CIFAR100':
    idx = [4, 27, 10, 28]
elif dataset == 'CelebA':
    idx = [1, 6000, 2, 7000]
# print(validloader.dataset.targets[:30])
for i in idx:
    img, label = validloader.dataset[i]
    # img = validloader.dataset[idx][0] * 0.3 + validloader.dataset[idx+13][0] * 0.3 + validloader.dataset[idx+13][0] * 0.3
    labels2.append(torch.as_tensor((label,), device=setup['device']))
    ground_truth2.append(img.to(**setup))
ground_truth = torch.stack([ground_truth2[args.id]])
labels = torch.cat([labels2[args.id]])
ground_truth2 = torch.stack(ground_truth2)
labels2 = torch.cat(labels2)

# Show original image
dm = torch.as_tensor(inversefed.consts.d_m[dataset], **setup)[:, None, None]
ds = torch.as_tensor(inversefed.consts.d_std[dataset], **setup)[:, None, None]
name = 'res/{}_{}_original.png'.format(dataset, args.id)
grid_plot(name, ground_truth, [validloader.dataset.classes[l] for l in labels])
#
# model.zero_grad()
if arch == 'LeNet5':
    model_ful = lenet5(pretrained=trained)
elif arch == 'ResNet20':
    model_ful = resnet20(pretrained=trained, num_classes=num_classes, width=1)
elif arch == 'ResNet20-4':
    model_ful = resnet20(pretrained=trained, num_classes=num_classes, width=4)
elif arch == 'ResNet32':
    model_ful = resnet32(pretrained=trained, num_classes=num_classes, width=1)
elif arch == 'ResNet32-4':
    model_ful = resnet32(pretrained=trained, num_classes=num_classes, width=4)
elif arch == 'ResNet32-8':
    model_ful = resnet32(pretrained=trained, num_classes=num_classes, width=8)
elif arch == 'NF-ResNet20':
    model_ful = fixup_resnet20(pretrained=trained, num_classes=num_classes, width=1)
elif arch == 'NF-ResNet20-4':
    model_ful = fixup_resnet20(pretrained=trained, num_classes=num_classes, width=4)
elif arch == 'NF-ResNet32':
    model_ful = fixup_resnet32(pretrained=trained, num_classes=num_classes, width=1)
elif arch == 'NF-ResNet32-4':
    model_ful = fixup_resnet32(pretrained=trained, num_classes=num_classes, width=4)
elif arch == 'NF-ResNet32-8':
    model_ful = fixup_resnet32(pretrained=trained, num_classes=num_classes, width=8)
else:
    ValueError('no {}'.format(model))

model.S.load_state_dict(model_ful.S.state_dict())
if mtype == 'F':
    model.U.load_state_dict(model_ful.U.state_dict())
model_ful.to(**setup)

# ## Reconstruct
if ltype == 'ce':
    loss_fn = torch.nn.CrossEntropyLoss()
elif ltype == 'oph':
    def loss_fn(output, target):
        loss1 = F.cross_entropy(output, target)
        loss2 = utils.contrastive_loss2(output, target, num_classes)
        alpha = args.lam * (loss1 / loss2).detach()
        loss = loss1 + alpha * loss2
        return loss
elif ltype == 'gauss' or ltype == 'lap':
    pass
else:
    raise ValueError('ltype should be ce, gauss, lap, or oph, but get {}'.format(ltype))

if gtype == 'w':
    if ltype == 'ce':
        input_gradient = inversefed.ExtractWeGr(model_ful, loss_fn, ground_truth, labels, mtype)
    elif ltype == 'gauss' or ltype == 'lap':
        input_gradient = inversefed.ExtractWeGr_mod1(model_ful, ground_truth, labels, mtype, ltype, args.sig)
    elif ltype == 'oph':
        input_gradient = inversefed.ExtractWeGr_mod2(model_ful, ground_truth2, labels2, mtype, num_classes, args.lam, args.id)
    else:
        raise ValueError('ltype of w should be ce, noise, or oph')
    rec_machine = inversefed.WeGrReconstructor(model, pre, config, mtype, num_images=num_images)
elif gtype == 'f':
    if ltype == 'ce':
        input_gradient = inversefed.ExtractFeGr(model_ful, loss_fn, ground_truth, labels, mtype)
    elif ltype == 'oph':
        input_gradient = inversefed.ExtractFeGr(model_ful, loss_fn, ground_truth2, labels2, mtype)
    else:
        raise ValueError('ltype of f should be ce or oph')
    rec_machine = inversefed.FeGrReconstructor(model, pre, config, mtype, num_images=num_images)
else:
    raise ValueError('gtype should be w or f, but get {}'.format(gtype))

output, stats = rec_machine.reconstruct(input_gradient, labels, img_shape=inversefed.consts.d_size[dataset])

output.sub_(dm).div_(ds)
# show results
name = 'res/{}_{}_{}_{}_{}_{}_{}_recover.png'.format(args.dataset, args.arch, args.gtype, args.ltype, args.mtype, args.sig, args.id)
print(name)
grid_plot(name, output, [validloader.dataset.classes[l] for l in labels])

test_mse = (output.detach() - ground_truth).pow(2).mean()
feat_mse = (model(output.detach()) - model(ground_truth)).pow(2).mean()
test_psnr = inversefed.metrics.psnr(output, ground_truth, factor=1/ds)
mean_ssim, _ = inversefed.metrics.cw_ssim(output, ground_truth)

lpips_scorer = lpips.LPIPS(net="alex", verbose=False).to(**setup)
output_ = output.clamp_(0, 1)
ground_truth_ = ground_truth.clamp(0, 1)
if dataset == 'MNIST' or dataset == 'Fashion-MNIST':
    output_ = output_.repeat(1, 3, 1, 1)
    ground_truth_ = ground_truth.repeat(1, 3, 1, 1)
output_ = resize(output_, (64, 64), antialias=True)
ground_truth_ = resize(ground_truth_, (64, 64), antialias=True)
lpips_score = lpips_scorer(output_, ground_truth_, normalize=False).mean().item()

# SSIM 1 is best, LPIPS 0 is best
print('\r\n')
print(f"Rec. loss: {stats['opt']:2.4f} | MSE: {test_mse:2.4f} "
      f"| PSNR: {test_psnr:4.2f} | SSIM: {mean_ssim:2.4f} | LPIPS: {lpips_score:2.4f} |")
print('\r\n')
