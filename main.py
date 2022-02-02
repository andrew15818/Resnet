import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

import model.resnet as resnet

DATA_DIR = '/home/poncedeleon/usb/cifar-10-batches-py'
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Dataset is split across 5 files,
# Get each one and parse 
def get_batch(file:str):
    batch = {}
    with open(file, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')
    return batch

class CIFAR10Dataset(Dataset):
    def __init__(self, img_dir):
        batches = {}
        for i in range(1, 6):
            batches[i] = get_batch(os.path.join(img_dir, f'data_batch_{i}'))

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

class CIFAR10DataLoader(DataLoader):
    pass
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
    #y = resnet.ResidualNet18()
    #print(y)
