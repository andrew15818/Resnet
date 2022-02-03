import os
import pickle
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from PIL import Image

import model.resnet as resnet

DATA_DIR = '/home/poncedeleon/usb/cifar-10-batches-py'
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Get each one and parse 
def get_batch(file:str):
    batch = {}
    with open(file, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')
    return batch

def reshape_row(row:np.array):
    r = row[0:1024].reshape(32, 32)
    g = row[1024:2048].reshape(32, 32)
    b = row[2048:3072].reshape(32, 32)
    img = np.array([r, g, b]).T
    return img 

class CIFAR10Dataset(Dataset):
    def __init__(self, img_dir, test=False):
        self.test = test
        self.batches = {}
        self.len = 0
        # TODO: Maybe add some data augmentations here
        self.trans = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ]
                )
        if self.test:
           self.batches = get_batch(os.path.join(img_dir, 'test_batch'))
           self.len = self.batches[i][b'data'].shape[0]
           return

        self.filesize = 10000
        
        # Dataset information is spread on 5 batches
        for i in range(1, 6):
            self.batches[i] = get_batch(os.path.join(img_dir, f'data_batch_{i}'))
            self.len += self.batches[i][b'data'].shape[0]

    def __len__(self):
       return self.len 

    def __getitem__(self, idx):

        # Index of dictionary which contains sample
        if self.test:
            x = self.batches[b'data'][idx]
            label = self.batches[b'labels'][idx]
        else:
            index = idx // self.filesize + 1
            offset = idx % self.filesize 
            x = self.batches[index][b'data'][offset]
            label = self.batches[index][b'labels'][offset]

        # Apply transformations
        img = reshape_row(x)

        x = Image.fromarray(img)
        x = self.trans(x)

        return x, label 

# Could use this method besides custom dataset class
''''
train_data = datasets.CIFAR10(
        root='data',
        train=True,
        transform=ToTensor(),
        download=True
        )
'''
if __name__=='__main__':
    # Get the dataset and dataloader ready
    dataset = CIFAR10Dataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for x, label in dataloader:
        print(x.shape, label)
        break
    #y = resnet.ResidualNet18()
    #print(y)
