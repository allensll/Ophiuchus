import pickle
import socket
import time
from multiprocessing import cpu_count
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor
import torch.nn.functional as F

import torch
import torch.nn as nn

import nl2pc

from .utils import timer, recv_object, send_object
from .layers import ReLU, MaxPool2d
from CryptoTrain.models import *
import CryptoTrain.mpc as mpc


class Agent:
    def __init__(self, t, n_clt=1, max_nthreads=10, scheme='emulate'):
        self.n_clt = n_clt
        self.clients = list()
        self.max_nthreads = cpu_count() if max_nthreads == 0 else max_nthreads
        self.scheme = scheme
        self.rand_seed = None

        if t == 's':
            self.asserver = True
        elif t == 'c':
            self.asserver = False
        else:
            Exception('---(Error) Agent type should be "s" or "c"!')

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def start(self, ip, port):
        if self.asserver:
            self.socket.bind((ip, port))
            print('---Agent: started {}: {}'.format(ip, port))
            self.socket.listen(1)
            print('---Ageng: wait...')
            for i in range(self.n_clt):
                client, address = self.socket.accept()
                self.clients.append(client)
                print('---Agent: add client {} : {}'.format(i, address))
        else:
            print("---(Warning) Agent: not 's' type")

    def connect(self, ip, port):
        time.sleep(3)
        if not self.asserver:
            self.socket.connect((ip, port))
            print('---Agent: connected')
        else:
            print("---(Warning) Agent: not 'c' type")

    def close(self):
        for clt in self.clients:
            clt.close()
        self.socket.close()
        print("---Agent: close connect")

    def recv(self):
        if self.asserver:
            res = list()
            for i in range(self.n_clt):
                res.append(recv_object(self.clients[i]))
            # print("---Agent: receive")
            if len(res) == 1:
                return res[0]
            else:
                res = sum(res)
                return res
        else:
            return recv_object(self.socket)

    def send(self, x):
        if self.asserver:
            for i in range(self.n_clt):
                l = send_object(self.clients[i], x)
            print("---Agent: send {:.3f} MB".format(l/1024.0/1024.0))
        else:
            send_object(self.socket, x)


class Server:
    def __init__(self, dataset, model, relu='relu', pool='maxpool', t=None, agtaddr='127.0.0.1', agtport=20202, scheme='emulate', nthreads=10):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.dataset = dataset
        self.model_name = model
        self.relu = relu
        self.t = t
        if dataset == 'mnist' or dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        elif dataset == 'imagenet':
            num_classes = 1000
        else:
            raise ValueError('No dataset: {}'.format(dataset))

        if t:
            if scheme == 'emulate':
                if t == 's':
                    self.agt = Agent('s', max_nthreads=nthreads, scheme='emulate')
                    self.agt.start(agtaddr, agtport)
                elif t == 'c':
                    self.agt = Agent('c', max_nthreads=nthreads, scheme='emulate')
                    self.agt.connect(agtaddr, agtport)
                else:
                    raise ValueError('(s or c), but get {}'.format(t))
            elif scheme == 'ckks':
                if t == 's':
                    self.agt = nl2pc.Create(nl2pc.ckks_role.SERVER, address=agtaddr, port=agtport, nthreads=nthreads, verbose=True)
                elif t == 'c':
                    self.agt = nl2pc.Create(nl2pc.ckks_role.CLIENT, address=agtaddr, port=agtport, nthreads=nthreads, verbose=True)
                else:
                    raise ValueError('(s or c), but get {}'.format(t))
            else:
                raise ValueError('(emulate or ckks), but get {}'.format(t))

            # define agt2 used in update grad
            if t == 's':
                self.agt2 = Agent('s', max_nthreads=nthreads, scheme='grad')
                self.agt2.start(agtaddr, agtport+1)
            elif t == 'c':
                self.agt2 = Agent('c', max_nthreads=nthreads, scheme='grad')
                self.agt2.connect(agtaddr, agtport+1)
            else:
                raise ValueError('(s or c), but get {}'.format(t))

            self.scheme = scheme

            if relu == 'relu':
                relu = nn.ReLU()
            elif relu == 'mpcrelu':
                relu = mpc.ReLU(self.agt)
            else:
                raise ValueError('relu or mpcrelu, but get {}'.format(relu))

            if pool == 'avgpool':
                pool = nn.AvgPool2d(2) if model == 'lenet5' else nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            elif pool == 'maxpool':
                pool = nn.MaxPool2d(2) if model == 'lenet5' else nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            elif pool == 'mpcmaxpool':
                pool = mpc.MaxPool2d(self.agt, 2) if model == 'lenet5' else mpc.MaxPool2d(self.agt, kernel_size=3, stride=2, padding=1)
            else:
                raise ValueError('avgpool or maxpool or mpcmaxpool, but get {}'.format(pool))

            self.model = eval(model)(relu, pool, num_classes=num_classes, n=2, role='S')

    def connect(self, ip, port):
        self.socket.connect((ip, port))
        print('---Server: connected')

    def get_input(self):
        data = recv_object(self.socket)
        return data

    def send_feature(self, feature):
        sl = send_object(self.socket, feature)
        print("---Server: send feature {:.2f}MB".format(sl/1014.0/1024.0))

    def get_grad(self):
        return recv_object(self.socket)

    def update_grad(self):
        grad = list()
        for i, l in enumerate(self.model.parameters()):
            grad.append(l.grad.clone().detach())
        self.agt2.send(grad)
        grad2 = self.agt2.recv()
        for (n, l), g1, g2 in zip(self.model.named_parameters(), grad, grad2):
            l.grad = g1 + g2

    def close(self):
        if self.t == 'host':
            self.agt.close()
        self.socket.close()
        print("---Server: close connect")


class User:
    def __init__(self, n_srv=2):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.n_srv = n_srv
        self.servers = list()

    def start(self, ip, port):
        self.socket.bind((ip, port))
        print('---User: started {}: {}'.format(ip, port))
        self.socket.listen(1)
        print('---User: waiting...')
        for i in range(self.n_srv):
            server, address = self.socket.accept()
            self.servers.append(server)
            print('---User: add server {} : {}'.format(i, address))

    def upload(self, data):
        sl = 0
        for i in range(self.n_srv):
            sl += send_object(self.servers[i], data[i])
        # print("---User: finish upload send {:.2f}MB".format(sl/1014.0/1024.0))

    def get_feature(self):
        feature = list()
        for i in range(self.n_srv):
            feature.append(recv_object(self.servers[i]))
        # print("---User: get features")
        feature = sum(feature).detach()
        return feature

    def send_grad(self, grad):
        # Todo: add
        # mask = torch.rand(grad.shape[0])+0.5
        # grad = torch.mul(grad.T, mask).T
        sl = 0
        for i in range(self.n_srv):
            sl += send_object(self.servers[i], grad)
        # print("---User: finish upload send {:.2f}MB".format(sl/1014.0/1024.0))

    def close(self):
        for svr in self.servers:
            svr.close()
        self.socket.close()
        print("---User: close connect")

