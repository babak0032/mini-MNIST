import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
sys.path.append('../../probtorch')
import probtorch
from probtorch.util import partial_sum, log_mean_exp
CUDA = torch.cuda.is_available()



class Visualizer():

    def __init__(self, model):
        self.model = model
        self.train_data = model.train_data
        self.test_data = model.test_data
        self.num_pixels = model.num_pixels
        self.im_size = int(np.sqrt(self.model.num_pixels))

    def prior_samples(self):
        plt.figure(figsize=(20,20))
        with torch.no_grad():
            z = torch.randn(100, self.model.dec.num_style)
            null_images = torch.zeros(100, self.num_pixels)
            if CUDA:
                z = z.cuda()
                null_images = null_images.cpu()
            q = {'z': z}
            p = self.model.dec(null_images, q, 1, batch_size=100)
            images = p['images'].value
            if CUDA:
                images = images.cpu()
            images = images.numpy().reshape(100, self.im_size, self.im_size)
        ptr = 0
        for i in range(10):
            for j in range(10):
                plt.subplot(10, 10, ptr+1)
                plt.imshow(images[ptr], cmap='gray')
                ptr += 1
        plt.axis('off')

    def recons(self):
        plt.figure(figsize=(20,20))
        with torch.no_grad():
            images = self.model.data.sorted_data.view(-1, self.num_pixels)
            N = images.size(0)
            if CUDA:
                images = images.cuda()
            q = self.model.enc(images, num_samples=1)
            p = self.model.dec(images, q, 1, batch_size=N)
            images = p['images'].value
            if CUDA:
                images = images.cpu()
            images = images.numpy().reshape(N, self.im_size, self.im_size)
        num_column = 10
        num_row = int(self.model.data.N / num_column)
        ptr = 0
        for i in range(num_row):
            for j in range(num_column):
                plt.subplot(num_row, num_column, ptr+1)
                plt.imshow(images[ptr], cmap='gray')
                ptr += 1
        plt.axis('off')

    def recons_test(self):
        plt.figure(figsize=(20,20))
        with torch.no_grad():
            testdata = iter(self.model.test_data)
            images, _ = testdata.next()
            images = images.view(-1, self.num_pixels)[:100]
            N = images.size(0)
            if CUDA:
                images = images.cuda()
            q = self.model.enc(images, num_samples=1)
            p = self.model.dec(images, q, 1, batch_size=N)
            images = p['images'].value
            if CUDA:
                images = images.cpu()
            images = images.numpy().reshape(N, self.im_size, self.im_size)
        num_column = 10
        num_row = 10
        ptr = 0
        for i in range(num_row):
            for j in range(num_column):
                plt.subplot(num_row, num_column, ptr+1)
                plt.imshow(images[ptr], cmap='gray')
                ptr += 1
        plt.axis('off')

    def encode_traindata(self):
        with torch.no_grad():
            images = self.model.data.sorted_data.view(-1, self.num_pixels)
            if CUDA:
                images = images.cuda()
            q = self.model.enc(images, num_samples=1)
            mu = q['z'].dist.mean.squeeze()
            std = q['z'].dist.stddev.squeeze()
            if CUDA:
                mu= mu.cpu()
                std= std.cpu()
            mu = mu.numpy()
            std = std.numpy()

        return mu, std

    def z_histograms(self):
        mu, std = self.encode_traindata()
        fig, axs = plt.subplots(2, 5, figsize=(10,5), sharex=True, sharey=True)
        ptr = 0
        for i in range(2):
            for j in range(5):
                axs[i,j].hist(mu[:, ptr], bins=10, density=True)
                ptr += 1

        fig, axs = plt.subplots(2, 5, figsize=(10,5), sharex=True, sharey=True)
        ptr = 0
        for i in range(2):
            for j in range(5):
                axs[i,j].hist(std[:, ptr], bins=10, density=True)
                ptr += 1

    def z_2d_plot(self):
        mu, std = self.encode_traindata()
        plt.figure(figsize=(10,10))
        ax = plt.gca()
        for k in range(10):
            indk = k*10
            mu_k = mu[indk:indk+10]
            std_k = std[indk:indk+10]
            ax.scatter(mu_k[:,0], mu_k[:,1], label='digit=%d' % k, alpha=0.5)
            for i in range(10):
                ellipse = Ellipse(xy=(mu_k[i,0], mu_k[i,1]),
                                  width=2*std_k[i,0], height=2*std_k[i,1],
                                  edgecolor='r', fc='None', lw=2)
                ax.add_patch(ellipse)

        ax.legend()
