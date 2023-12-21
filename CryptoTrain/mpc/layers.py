import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# import cmpy
import utils


class MPCReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, mask):
        ctx.save_for_backward(mask)
        output = torch.mul(input, mask)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        grad_input = torch.mul(grad_output, mask.type_as(grad_output))
        return grad_input, None


class ReLU(nn.Module):
    def __init__(self, agt):
        super(ReLU, self).__init__()
        self.agt = agt
        self.scheme = agt.scheme

    def forward(self, x):
        print('--- ReLU ---')
        if self.agt and self.training:
            if self.scheme == 'emulate':
                mask = self.mpc_emulate(x)
            else:
                mask = self.mpc_ckks(x)
            x = MPCReLU.apply(x, mask)
        else:
            x = F.relu(x)

        return x

    def mpc_ckks(self, x):
        # with utils.timer('all'):
        shape = x.shape
        data = x.flatten().tolist()
        # print(len(data))
        # with utils.timer('cmp'):
        mask = self.agt.cmp(data)
        mask = torch.FloatTensor(mask).reshape(shape)
        return mask

    def mpc_emulate(self, x):
        # Server A
        if self.agt.asserver:
            # 1 encrypt
            self.agt.send(x)
            # 3 decrypt
            a = self.agt.recv()
            mask = a > 0
            mask = mask.astype(np.float32)
            mask = torch.from_numpy(mask)
            self.agt.send(mask)
            return mask
        # Server B
        else:
            # 2 compute
            x1 = self.agt.recv()
            k = np.random.randint(1, 100, size=tuple(x.size()))
            a = (x.detach().numpy() + x1.detach().numpy()) * k
            self.agt.send(a)
            # 4 get mask
            mask = self.agt.recv()
            return mask


class ReLU2(nn.Module):
    def __init__(self):
        super(ReLU2, self).__init__()

    def forward(self, x):
        mask = self.mpc_cmp(x)
        x = MPCReLU.apply(x, mask)

        return x

    def mpc_cmp(self, x):
        # with utils.timer('all'):
        shape = x.shape
        mask = x.detach().numpy() > 0
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)
        mask = torch.FloatTensor(mask).reshape(shape)
        return mask


class MPCMaxPool2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, indices, patches, k, s, p, op_shape, tmp_shape):
        ctx.k = k
        ctx.s = s
        ctx.p = p
        ctx.ip_shape = input.shape
        ctx.tmp_shape = tmp_shape
        ctx.save_for_backward(indices)
        output = torch.sum(patches * indices, dim=-1).reshape(op_shape)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        indices, = ctx.saved_tensors
        grad_ = grad_output.view(-1, grad_output.size(2), grad_output.size(3)).unsqueeze(1)
        patch_grad = nn.functional.unfold(grad_, kernel_size=1).permute(0, 2, 1) * indices
        grad_input = nn.functional.fold(patch_grad.permute(0, 2, 1), ctx.tmp_shape, kernel_size=ctx.k, stride=ctx.s, padding=ctx.p)
        grad_input = grad_input.reshape(ctx.ip_shape)
        return grad_input, None, None, None, None, None, None, None, None, None


class MaxPool2d(nn.Module):
    def __init__(self, agt, kernel_size, stride=None, padding=0):
        super(MaxPool2d, self).__init__()
        self.agt = agt
        self.scheme = agt.scheme
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size
        self.padding = padding

    def forward(self, x):
        print('--- MaxPool ---')
        if self.agt and self.training:
            if self.scheme == 'emulate':
                indices, patches, tmp_shape = self.mpc_emulate(x)
            else:
                indices, patches, tmp_shape = self.mpc_ckks(x)
            # print(indices[11][0:30])
            op_shape = F.max_pool2d(x, self.kernel_size, self.stride, self.padding).shape
            x = MPCMaxPool2d.apply(x, indices, patches, self.kernel_size, self.stride, self.padding, op_shape, tmp_shape)
        else:
            x = F.max_pool2d(x, self.kernel_size, self.stride, self.padding)

        return x

    def mpc_ckks(self, x):
        # Server A
        # 0 extract
        shape = x.shape
        s = self.kernel_size[0]*self.kernel_size[1] if type(self.kernel_size) is tuple else self.kernel_size**2
        x_ = x.view(-1, x.size(2), x.size(3)).unsqueeze(1)
        patches1 = nn.functional.unfold(x_, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        patches1_ = patches1.permute(0, 2, 1)

        data = patches1.permute(1, 0, 2).flatten().tolist()
        idx = self.agt.max(data, s)
        idx = torch.FloatTensor(idx).long().reshape(shape[0]*shape[1], -1)
        indices = nn.functional.one_hot(idx, num_classes=s)
        return indices, patches1_, x_.size()[2:]

    def mpc_emulate(self, x):
        # Server A
        # 0 extract
        s = self.kernel_size[0]*self.kernel_size[1] if type(self.kernel_size) is tuple else self.kernel_size**2
        x_ = x.view(-1, x.size(2), x.size(3)).unsqueeze(1)
        patches1 = nn.functional.unfold(x_, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        patches1 = patches1.permute(0, 2, 1)
        if self.agt.asserver:
            # 1 encrypt
            self.agt.send(patches1)
            # 3 decrypt
            patches = self.agt.recv()
            idx = patches.max(dim=-1)[1]
            indices = nn.functional.one_hot(idx, num_classes=s)
            self.agt.send(indices)
            return indices, patches1, x_.size()[2:]
        # Server B
        else:
            # 2 compute
            patches2 = self.agt.recv()
            patches = patches1 + patches2
            # y = k x + b
            self.agt.send(patches)
            # 4 get indices
            indices = self.agt.recv()
            return indices, patches1, x_.size()[2:]


class MaxPool2d2(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(MaxPool2d2, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size
        self.padding = padding

    def forward(self, x):
        indices, patches, tmp_shape = self.mpc_cmp(x)
        op_shape = F.max_pool2d(x, self.kernel_size, self.stride, self.padding).shape
        x = MPCMaxPool2d.apply(x, indices, patches, self.kernel_size, self.stride, self.padding, op_shape, tmp_shape)

        return x

    def mpc_cmp(self, x):
        shape = x.shape
        s = self.kernel_size[0]*self.kernel_size[1] if type(self.kernel_size) is tuple else self.kernel_size**2
        x_ = x.view(-1, x.size(2), x.size(3)).unsqueeze(1)
        patches = nn.functional.unfold(x_.view(shape[0]*shape[1], 1, shape[2], shape[3]), kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        patches = patches.permute(0, 2, 1)
        idx = patches.max(dim=-1)[1]
        indices = nn.functional.one_hot(idx, num_classes=s)
        return indices, patches, x_.size()[2:]