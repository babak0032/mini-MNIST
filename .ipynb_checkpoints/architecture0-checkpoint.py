import torch
import torch.nn as nn
import sys
sys.path.append('../../probtorch')
import probtorch
from probtorch.util import expand_inputs
CUDA = torch.cuda.is_available()

class Encoder(nn.Module):

    def __init__(self, num_pixels=784,
                       num_hidden1=400,
                       num_hidden2=200,
                       num_style=10,
                       num_digits=10,
                       digit_temp=0.66):

        super(self.__class__, self).__init__()
        self.num_pixels = num_pixels
        self.enc_hidden = nn.Sequential(
                            nn.Linear(num_pixels, num_hidden1),
                            nn.ReLU())
        self.style_mean = nn.Sequential(
                            nn.Linear(num_hidden1, num_hidden2),
                            nn.ReLU(),
                            nn.Linear(num_hidden2, num_style))
        self.style_log_std = nn.Sequential(
                                nn.Linear(num_hidden1, num_hidden2),
                                nn.ReLU(),
                                nn.Linear(num_hidden2, num_style))

    @expand_inputs
    def forward(self, images, labels=None, num_samples=None):
        q = probtorch.Trace()
        hidden = self.enc_hidden(images)
        styles_mean = self.style_mean(hidden)
        styles_std = torch.exp(self.style_log_std(hidden))
        q.normal(styles_mean, styles_std, name='z')
        return q

def binary_cross_entropy(x_mean, x, EPS=1e-9):
    return - (torch.log(x_mean + EPS) * x +
              torch.log(1 - x_mean + EPS) * (1 - x)).sum(-1)

class Decoder(nn.Module):

    def __init__(self, num_pixels=784,
                       num_hidden1=400,
                       num_hidden2=200,
                       num_style=10,
                       num_digits=10,
                       digit_temp=0.66):

        super(self.__class__, self).__init__()
        self.num_style = num_style
        self.dec_hidden = nn.Sequential(
                            nn.Linear(num_style, num_hidden2),
                            nn.ReLU(),
                            nn.Linear(num_hidden2, num_hidden1),
                            nn.ReLU())
        self.digit_temp = torch.tensor(digit_temp, requires_grad=False)
        self.dec_images = nn.Sequential(
                            nn.Linear(num_hidden1, num_pixels),
                            nn.Sigmoid())

    def forward(self, images, q, num_samples=None, batch_size=1):
        p = probtorch.Trace()
        style_mean = torch.zeros(num_samples,batch_size,self.num_style)
        style_std = torch.ones(num_samples,batch_size,self.num_style)
        if CUDA:
            style_mean = style_mean.cuda()
            style_std = style_std.cuda()
        styles = p.normal(style_mean,
                          style_std,
                          value=q['z'],
                          name='z')
        hiddens = self.dec_hidden(styles)
        images_mean = self.dec_images(hiddens)
        p.loss(binary_cross_entropy, images_mean, images, name='images')
        return p
