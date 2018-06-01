import torch
import torch.nn as nn
import architecture
import pickle
import math
import time
import os
import sys
sys.path.append('../probtorch')
import probtorch
from probtorch.objectives.montecarlo import elbo
print('probtorch:', probtorch.__version__,
      'torch:', torch.__version__,
      'cuda:', torch.cuda.is_available())

CUDA = torch.cuda.is_available()

def cuda_tensors(obj):
    for attr in dir(obj):
        value = getattr(obj, attr)
        if isinstance(value, torch.Tensor):
            setattr(obj, attr, value.cuda())


class Model():
    def __init__(self, data,
                 num_samples,
                 num_pixels=784,
                 num_hidden=400,
                 num_style=10,
                 exp_name=None,
                 weights_path="weights"):

        self.data = data
        self.train_data = data.train_data
        self.test_data = data.test_data
        self.num_samples = num_samples
        self.num_pixels = num_pixels
        self.alpha = 0.1
        self.beta = 1.0
        self.EPS = 1e-9
        self.N_train = data.N
        self.N_test = len(self.test_data)*data.batch_size
        self.bias_test = (self.N_test - 1) / (data.batch_size - 1)

        self.enc = architecture.Encoder(num_pixels=num_pixels, num_hidden=num_hidden, num_style=num_style)
        self.dec = architecture.Decoder(num_pixels=num_pixels, num_hidden=num_hidden, num_style=num_style)

        if exp_name:
            self.enc.load_state_dict(torch.load('%s/%s-enc.rar'
                                        % (weights_path, exp_name)))
            self.dec.load_state_dict(torch.load('%s/%s-dec.rar'
                                        % (weights_path, exp_name)))
        if CUDA:
            self.enc.cuda()
            self.dec.cuda()
            cuda_tensors(self.enc)
            cuda_tensors(self.dec)

        self.optimizer = torch.optim.Adam(list(self.dec.parameters())+list(self.enc.parameters()),
                                          lr=1e-3)

    def loss(self, q, p):
        return -elbo(q=q, p=p, sample_dim=0, batch_dim=1,
                     alpha=self.alpha, beta=self.beta)

    def epoch(self):
        self.enc.train()
        self.dec.train()
        images = self.train_data.view(-1, self.num_pixels)
        batch_size = images.size(0)
        if CUDA:
            images = images.cuda()
        self.optimizer.zero_grad()
        q = self.enc(images, num_samples=self.num_samples)
        p = self.dec(images, q, self.num_samples, batch_size=batch_size)
        loss = self.loss(q=q, p=p)
        loss.backward()
        self.optimizer.step()
        return loss / self.N_train, \
               self.mutual_information(q=q, p=p, bias=1) / self.N_train

    def test(self):
        test_loss = 0
        self.enc.eval()
        self.dec.eval()
        I = 0
        for b, (images, labels) in enumerate(self.test_data):
            images = images.view(-1, self.num_pixels)
            if CUDA:
                images = images.cuda()
            q = self.enc(images, num_samples=self.num_samples)
            p = self.dec(images, q, num_samples=self.num_samples,
                                    batch_size=self.data.batch_size)
            loss = self.loss(q=q, p=p)
            test_loss += loss
            I += self.mutual_information(q=q, p=p, bias=self.bias_test)
        return test_loss / self.N_test, I / self.N_test

    def mutual_information(self, q, p, bias):
        z = [n for n in q.sampled() if n in p]
        with torch.no_grad():
            log_joint_avg_qz, _, _ = q.log_batch_marginal(sample_dim=0,
                                                          batch_dim=1,
                                                          nodes=z,
                                                          bias=bias)
            log_qz = q.log_joint(sample_dim=0, batch_dim=1, nodes=z)
            I = (log_qz - log_joint_avg_qz).sum()
            if CUDA:
                I = I.cpu()
            return I.numpy()


    def train(self, num_epochs=100, prefix=None, weights_path="weights"):
        for e in range(num_epochs):
            self.data.shuffle()
            self.train_data = self.data.train_data
            train_start = time.time()
            train_elbo, train_mi = self.epoch()
            train_end = time.time()
            test_start = time.time()
            test_elbo, test_mi = self.test()
            test_end = time.time()
            print('[Epoch %d] Train Loss: %.4e(%ds), Test Loss: %.4e(%ds), Train MI: %.4f/%.4f, Test MI: %.4f/%.4f' % (
                    e, train_elbo, train_end - train_start,
                    test_elbo, test_end - test_start,
                    train_mi, math.log(self.N_train),
                    test_mi, math.log(self.N_test)))
            if prefix is not None:
                if not os.path.isdir(weights_path):
                    os.mkdir(weights_path)
                torch.save(self.enc.state_dict(),
                           '%s/%s-enc.rar' % (weights_path, prefix))
                torch.save(self.dec.state_dict(),
                           '%s/%s-dec.rar' % (weights_path, prefix))
