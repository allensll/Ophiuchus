"""Mechanisms for image reconstruction from weight gradients."""
import itertools

import torch
import torch.nn.functional as F
from torch.distributions import Normal, Laplace
from collections import defaultdict, OrderedDict
from .nn import MetaMonkey
from .metrics import total_variation as TV
from .metrics import InceptionScore
from .medianfilt import MedianPool2d
from copy import deepcopy

import time
from itertools import chain

DEFAULT_CONFIG = dict(signed=False,
                      boxed=True,
                      cost_fn='sim',
                      indices='def',
                      weights='equal',
                      lr=0.1,
                      optim='adam',
                      restarts=1,
                      max_iterations=4800,
                      total_variation=1e-1,
                      init='randn',
                      filter='none',
                      lr_decay=True,
                      scoring_choice='loss')


def _validate_config(config):
    for key in DEFAULT_CONFIG.keys():
        if config.get(key) is None:
            config[key] = DEFAULT_CONFIG[key]
    for key in config.keys():
        if DEFAULT_CONFIG.get(key) is None:
            raise ValueError(f'Deprecated key in config dict: {key}!')
    return config


def ExtractWeGr(model, loss_fn, ground_truth, labels, mtype):
    """Extract the weights of plain training"""
    model.eval()
    model.zero_grad()
    target_loss = loss_fn(model(ground_truth), labels)
    print(target_loss)
    vaild_params = model.S.parameters() if mtype == 'S' else chain(model.S.parameters(), model.U.parameters())
    input_gradient = torch.autograd.grad(target_loss, vaild_params)
    input_gradient = [grad.detach() for grad in input_gradient]
    return input_gradient


def ExtractWeGr_mod1(model, ground_truth, labels, mtype, ltype, sig):
    """Extract the weights of Gauss. noise"""
    model.eval()
    model.zero_grad()
    output_ = model.S(ground_truth)
    output = model.U(output_)
    def hook(grad):
        print(sig)
        if ltype == 'gauss':
            # Gauss. noise
            m = Normal(torch.tensor([0.0]), torch.tensor([sig]))
        elif ltype == 'lap':
            # Lap. noise
            m = Laplace(torch.tensor([0.0]), torch.tensor([sig]))
        else:
            raise ValueError('noise should be gauss or lap, but get {}'.format(ltype))
        n = m.sample(grad.size()).squeeze(-1)
        grad_ = grad + n.to('cuda:0')
        return grad_
    output.register_hook(hook)

    loss = F.cross_entropy(output, labels)
    print(loss)
    vaild_params = model.S.parameters() if mtype == 'S' else chain(model.S.parameters(), model.U.parameters())
    input_gradient = torch.autograd.grad(loss, vaild_params)
    input_gradient = [grad.detach() for grad in input_gradient]
    return input_gradient


def ExtractWeGr_mod2(model, ground_truth, labels, mtype, num_classes, lam, id):
    """Extract the weights of Oph. pseudo-noise"""
    model.eval()
    model.zero_grad()
    output_ = model.S(ground_truth)
    output = model.U(output_)

    import utils
    # new

    loss1 = F.cross_entropy(torch.unsqueeze(output[id], 0), torch.unsqueeze(labels[id], 0))
    # loss1 = F.cross_entropy(output[id], labels[id])
    loss2 = utils.contrastive_loss2(output_, labels, num_classes)
    # print(loss1)
    # print(loss2)
    alpha = lam * (loss1 / loss2).detach()
    # print(alpha)
    loss = loss1 + alpha * loss2

    vaild_params = model.S.parameters() if mtype == 'S' else chain(model.S.parameters(), model.U.parameters())
    input_gradient = torch.autograd.grad(loss, vaild_params)
    input_gradient = [grad.detach() for grad in input_gradient]
    return input_gradient


class WeGrReconstructor:
    """Instantiate a reconstruction algorithm."""

    def __init__(self, model, pre, config=DEFAULT_CONFIG, mtype='F', num_images=1):
        """Initialize with algorithm setup."""
        self.config = _validate_config(config)
        self.model = model
        self.pre = pre
        self.setup = dict(device=next(model.parameters()).device, dtype=next(model.parameters()).dtype)
        self.mtype = mtype
        self.num_images = num_images

        if self.config['scoring_choice'] == 'inception':
            self.inception = InceptionScore(batch_size=1, setup=self.setup)

    def reconstruct(self, input_data, gt_labels, img_shape=(3, 32, 32), dryrun=False, eval=True, tol=None):
        """Reconstruct image from gradient."""
        start_time = time.time()
        if eval:
            self.model.S.eval()
            if self.mtype == 'F':
                self.model.eval()
        if self.mtype == 'F':
            self.pre.eval()

        stats = defaultdict(list)
        x = self._init_images(img_shape)
        scores = torch.zeros(self.config['restarts'])

        self.reconstruct_label = True

        def loss_fn(mtype):
            if mtype == 'S':
                def loss_fn_(pred, labels):
                    labels = torch.nn.functional.softmax(labels, dim=-1)
                    return torch.mean(torch.sum(- labels * torch.nn.functional.log_softmax(pred, dim=-1), 1))
            else:
                def loss_fn_(pred, labels):
                    return F.cross_entropy( pred, labels)
            return loss_fn_
        self.loss_fn = loss_fn(self.mtype)

        if self.mtype == 'S':
            init_pre_params = self.pre.state_dict()
            init_U_params = self.model.U.state_dict()
        try:
            for trial in range(self.config['restarts']):
                if self.mtype == 'S':
                    self.model.U.load_state_dict(init_U_params)
                    self.pre.load_state_dict(init_pre_params)
                x_trial, labels, params = self._run_trial(x[trial], input_data, gt_labels, dryrun=dryrun)
                # Finalize
                scores[trial] = self._score_trial(x_trial, input_data, labels, params)
                x[trial] = x_trial
                if tol is not None and scores[trial] <= tol:
                    break
                if dryrun:
                    break
        except KeyboardInterrupt:
            print('Trial procedure manually interruped.')
            pass

        # Choose optimal result:
        if self.config['scoring_choice'] in ['pixelmean', 'pixelmedian']:
            x_optimal, stats = self._average_trials(x, labels, input_data, stats)
        else:
            print('Choosing optimal result ...')
            scores = scores[torch.isfinite(scores)]  # guard against NaN/-Inf scores?
            optimal_index = torch.argmin(scores)
            print(f'Optimal result score: {scores[optimal_index]:2.8f}')
            stats['opt'] = scores[optimal_index].item()
            x_optimal = x[optimal_index]

        print(f'Total time: {time.time()-start_time}.')
        return x_optimal.detach(), stats

    def _init_images(self, img_shape):
        if self.config['init'] == 'randn':
            return torch.randn((self.config['restarts'], self.num_images, *img_shape), **self.setup)
        elif self.config['init'] == 'rand':
            return (torch.rand((self.config['restarts'], self.num_images, *img_shape), **self.setup) - 0.5) * 2
        elif self.config['init'] == 'zeros':
            return torch.zeros((self.config['restarts'], self.num_images, *img_shape), **self.setup)
        else:
            raise ValueError()

    def _run_trial(self, x_trial, input_data, gt_labels, dryrun=False):
        x_trial.requires_grad = True
        output_test = self.model(self.pre(x_trial))
        if self.mtype == 'S':
            labels = torch.randn((output_test.shape[0], output_test.shape[1])).to(**self.setup).requires_grad_(True)
        else:
            labels = gt_labels

        best_loss = float('inf')
        best_x_trial = x_trial
        best_labels = labels
        best_params = None

        if self.mtype == 'S':
            params = [
                {'params': x_trial},
                {'params': labels},
                {'params': self.pre.parameters()},
                {'params': self.model.U.parameters()}
            ]
        else:
            params = [
                {'params': x_trial},
            ]
        if self.config['optim'] == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.config['lr'])
        elif self.config['optim'] == 'sgd':  # actually gd
            optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, nesterov=True)
        elif self.config['optim'] == 'LBFGS':
            optimizer = torch.optim.LBFGS(params)
        else:
            raise ValueError()

        max_iterations = self.config['max_iterations']
        # dm, ds = self.mean_std
        if self.config['lr_decay']:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones=[max_iterations // 2.667, max_iterations // 1.6,
                                                                         max_iterations // 1.142], gamma=0.1)   # 3/8 5/8 7/8
        try:
            for iteration in range(max_iterations):
                closure = self._gradient_closure(optimizer, x_trial, input_data, labels)
                rec_loss = optimizer.step(closure)
                if self.config['lr_decay']:
                    scheduler.step()

                with torch.no_grad():
                    # Project into image space
                    if self.config['boxed']:
                        dm = dict(self.pre.named_parameters())['mean'].data
                        ds = dict(self.pre.named_parameters())['std'].data
                        # x_trial.data = torch.max(torch.min(x_trial, (1 - dm) / ds), -dm / ds)
                        x_trial.data = torch.max(torch.min(x_trial, torch.ones_like(dm)), torch.zeros_like(ds))

                    if (iteration + 1 == max_iterations) or iteration % 1000 == 0:
                        print(f'It: {iteration}. Rec. loss: {rec_loss.item():2.6f}.')

                    if (iteration + 1) % 500 == 0:
                        if self.config['filter'] == 'none':
                            pass
                        elif self.config['filter'] == 'median':
                            x_trial.data = MedianPool2d(kernel_size=3, stride=1, padding=1, same=False)(x_trial)
                        else:
                            raise ValueError()

                if rec_loss.item() < best_loss:
                    best_loss = rec_loss.item()
                    # print(best_loss)
                    best_x_trial = x_trial.detach()
                    best_labels = labels.detach()
                    if self.mtype == 'S':
                        best_params = [self.pre.state_dict(), self.model.U.state_dict()]

                if dryrun:
                    break
        except KeyboardInterrupt:
            print(f'Recovery interrupted manually in iteration {iteration}!')
            pass
        # return best_x_trial, best_labels, best_params
        return x_trial.detach(), labels, [self.pre.state_dict(), self.model.U.state_dict()]


    def _gradient_closure(self, optimizer, x_trial, input_gradient, label):

        def closure():
            optimizer.zero_grad()
            self.model.zero_grad()
            self.pre.zero_grad()
            x_trial_ = self.pre(x_trial)
            loss = self.loss_fn(self.model(x_trial_), label)

            vaild_params = self.model.S.parameters() if self.mtype == 'S' \
                else chain(self.model.S.parameters(), self.model.U.parameters())
            gradient = torch.autograd.grad(loss, vaild_params, create_graph=True)
            rec_loss = reconstruction_costs([gradient], input_gradient,
                                            cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                            weights=self.config['weights'])

            if self.config['total_variation'] > 0:
                rec_loss += self.config['total_variation'] * TV(x_trial)
            rec_loss.backward()
            if self.config['signed']:
                x_trial.grad.sign_()
            return rec_loss
        return closure

    def _score_trial(self, x_trial, input_gradient, label, params):
        if self.config['scoring_choice'] == 'loss':
            if self.mtype == 'S':
                self.pre.load_state_dict(params[0])
                self.model.U.load_state_dict(params[1])
            self.pre.zero_grad()
            self.model.zero_grad()
            x_trial.grad = None
            loss = self.loss_fn(self.model(self.pre(x_trial)), label)
            vaild_params = self.model.S.parameters() if self.mtype == 'S' \
                else chain(self.model.S.parameters(), self.model.U.parameters())
            gradient = torch.autograd.grad(loss, vaild_params, create_graph=False)
            return reconstruction_costs([gradient], input_gradient,
                                        cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                        weights=self.config['weights'])
        elif self.config['scoring_choice'] == 'tv':
            return TV(x_trial)
        elif self.config['scoring_choice'] == 'inception':
            # We do not care about diversity here!
            return self.inception(x_trial)
        elif self.config['scoring_choice'] in ['pixelmean', 'pixelmedian']:
            return 0.0
        else:
            raise ValueError()

    def _average_trials(self, x, labels, input_data, stats):
        print(f'Computing a combined result via {self.config["scoring_choice"]} ...')
        if self.config['scoring_choice'] == 'pixelmedian':
            x_optimal, _ = x.median(dim=0, keepdims=False)
        elif self.config['scoring_choice'] == 'pixelmean':
            x_optimal = x.mean(dim=0, keepdims=False)

        self.pre.zero_grad()
        self.model.zero_grad()
        if self.reconstruct_label:
            labels = self.model(self.pre(x_optimal)).softmax(dim=1)
        loss = self.loss_fn(self.model(self.pre(x_optimal)), labels)
        vaild_params = self.model.S.parameters() if self.mtype == 'S' \
            else chain(self.model.S.parameters(), self.model.U.parameters())
        gradient = torch.autograd.grad(loss, vaild_params, create_graph=False)
        stats['opt'] = reconstruction_costs([gradient], input_data,
                                            cost_fn=self.config['cost_fn'],
                                            indices=self.config['indices'],
                                            weights=self.config['weights'])
        print(f'Optimal result score: {stats["opt"]:2.4f}')
        return x_optimal, stats


def reconstruction_costs(gradients, input_gradient, cost_fn='l2', indices='def', weights='equal'):
    """Input gradient is given data."""
    if isinstance(indices, list):
        pass
    elif indices == 'def':
        indices = torch.arange(len(input_gradient))
    elif indices == 'batch':
        indices = torch.randperm(len(input_gradient))[:8]
    elif indices == 'topk-1':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 4)
    elif indices == 'top10':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 10)
    elif indices == 'top50':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 50)
    elif indices in ['first', 'first4']:
        indices = torch.arange(0, 4)
    elif indices == 'first5':
        indices = torch.arange(0, 5)
    elif indices == 'first10':
        indices = torch.arange(0, 10)
    elif indices == 'first50':
        indices = torch.arange(0, 50)
    elif indices == 'last5':
        indices = torch.arange(len(input_gradient))[-5:]
    elif indices == 'last10':
        indices = torch.arange(len(input_gradient))[-10:]
    elif indices == 'last50':
        indices = torch.arange(len(input_gradient))[-50:]
    else:
        raise ValueError()

    ex = input_gradient[0]
    if weights == 'linear':
        weights = torch.arange(len(input_gradient), 0, -1, dtype=ex.dtype, device=ex.device) / len(input_gradient)
    elif weights == 'exp':
        weights = torch.arange(len(input_gradient), 0, -1, dtype=ex.dtype, device=ex.device)
        weights = weights.softmax(dim=0)
        weights = weights / weights[0]
    else:
        weights = input_gradient[0].new_ones(len(input_gradient))

    total_costs = 0
    for trial_gradient in gradients:
        pnorm = [0, 0]
        costs = 0
        number_loss = 0
        if indices == 'topk-2':
            _, indices = torch.topk(torch.stack([p.norm().detach() for p in trial_gradient], dim=0), 4)
        for i in indices:
            if cost_fn == 'l2':
                costs += ((trial_gradient[i] - input_gradient[i]).pow(2)).sum() * weights[i]
            elif cost_fn == 'l1':
                costs += ((trial_gradient[i] - input_gradient[i]).abs()).sum() * weights[i]
            elif cost_fn == 'max':
                costs += ((trial_gradient[i] - input_gradient[i]).abs()).max() * weights[i]
            elif cost_fn == 'sim':
                costs -= (trial_gradient[i] * input_gradient[i]).sum() * weights[i]
                pnorm[0] += trial_gradient[i].pow(2).sum() * weights[i]
                pnorm[1] += input_gradient[i].pow(2).sum() * weights[i]
            elif cost_fn == 'sim2':
                def f(x):
                    res = 2*(x - torch.min(x)) / (torch.max(x) - torch.min(x) + 1e-8) - 1
                    return res

                if torch.max(trial_gradient[i].flatten()) != torch.min(trial_gradient[i].flatten())\
                        and torch.max(input_gradient[i].flatten()) != torch.min(input_gradient[i].flatten()):
                    a = f(trial_gradient[i]).flatten()
                    b = f(input_gradient[i]).flatten()
                    l = torch.nn.functional.cosine_similarity(a, b, 0)
                    costs += l
                    number_loss += 1
            elif cost_fn == 'simlocal':
                costs += 1 - torch.nn.functional.cosine_similarity(trial_gradient[i].flatten(),
                                                                   input_gradient[i].flatten(),
                                                                   0, 1e-10) * weights[i]
        if cost_fn == 'sim':
            costs = 1 + costs / pnorm[0].sqrt() / pnorm[1].sqrt()
        elif cost_fn == 'sim2':
            costs = costs / number_loss

        # Accumulate final costs
        total_costs += costs
    return total_costs / len(gradients)
