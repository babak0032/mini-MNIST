import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset


class mini_MNIST(Dataset):
    """
    use only N data points (uniformly over different digits)
    """
    def __init__(self, data_path, N, test_batch_size=100):

        self.train_data = datasets.MNIST(data_path, train=True, download=True,
                                    transform=transforms.ToTensor())
        self.test_data = torch.utils.data.DataLoader(
                                            datasets.MNIST(data_path,
                                            train=False, download=True,
                                            transform=transforms.ToTensor()),
                                            batch_size=test_batch_size,
                                            shuffle=True)
        self.data_path = data_path
        self.batch_size = test_batch_size
        self.N = N
        self.N_per_label = int(N/10)
        data = torch.tensor([]).float()

        for i in range(10):
            xi = self.train_data.train_labels.eq(i).nonzero().squeeze()
            num_i = xi.size(0)
            xi = xi[torch.randperm(num_i)]
            xi = self.train_data.train_data[xi][:self.N_per_label].float()/255.0
            data = torch.cat([data, xi], 0)

        #data = data[torch.randperm(N),:]
        self.sorted_data = data
        self.train_data = data

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return self.batches[index]

    def visualize(self, num_column=10):
        plt.figure(figsize=(20,20))
        num_row = int(self.N / num_column)
        ptr = 0
        for i in range(num_row):
            for j in range(num_column):
                plt.subplot(num_row, num_column, ptr+1)
                plt.imshow(self.train_data[ptr], cmap='gray')
                ptr +=1
        plt.axis('off')


    def shuffle(self):
        self.train_data = self.train_data[torch.randperm(self.N),:]
