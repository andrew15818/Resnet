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
def train_loop(dataloader, model, loss_fn):
    # TODO: Use cmd-line args later
    # Hyperparameters
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    epochs = 10 
    learning_rate = 0.01

    losses, counter = [], []
    for epoch in range(epochs):
        running_loss = 0
        for idx, data in enumerate(dataloader):
            x, y = data

            # reset the gradients
            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_fn(outputs, y)

            # backprop
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if idx % 2000 == 0:
                print(f'[epoch {epoch+1} batch {idx+1}]: {running_loss/2000}')
                running_loss = 0

                # save the mdoel
                torch.save(model.state_dict(), 'weights/model.pth')
    return model


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            preds = model(X)
            test_loss += loss_fn(preds, y).item()
            correct += (preds.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f'Test error: \nAccuracy {100*correct} Avg loss {test_loss:>8f}\n')

if __name__=='__main__':
    batch_size = 32
    # Get the dataset and dataloader ready
    dataset = utils.CIFAR10Dataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    test_dataset = utils.CIFAR10Dataset(DATA_DIR, test=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    model = resnet.ResidualNet18()

    # TODO: skip training for debugging atm
    skip_training = False
    try:
        model.load_state_dict(torch.load('weights/model.pth'))
        skip_training = True
    except:
        print('File not found')


    loss_fn = nn.CrossEntropyLoss()
   
    if not skip_training:
        train_loop(dataloader, model, loss_fn)

    test_loop(test_dataloader, model, loss_fn)
