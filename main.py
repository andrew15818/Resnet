import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

import model.model_utils as utils
import model.resnet as resnet

DATA_DIR = '/home/poncedeleon/usb/cifar-10-batches-py'
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Could use this method besides custom dataset class
def train_loop():
    # TODO: Use cmd-line args later
    # Hyperparameters
    epochs = 10
    batch_size = 32
    learning_rate = 0.01

    # Get the dataset and dataloader ready
    dataset = utils.CIFAR10Dataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = resnet.ResidualNet18()
    loss = nn.Softmax()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    losses, counter = [], []
    for epoch in range(epochs):
        batch_loss = 0
        for idx, data in enumerate(dataloader):
            x, y = data
            optimizer.zero_grad()
            outputs = model(x)


def test_loop(data):
    pass
if __name__=='__main__':
    train_loop()
