import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset

import model.model_utils as utils
import model.resnet as resnet


DATA_DIR = '/home/poncedeleon/usb/cifar-10-batches-py'
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
parser = argparse.ArgumentParser(description="Set hyperparamters and other important options.")
parser.add_argument('-d', '--data', type=str, default=DATA_DIR,
                    help="Path to data folder.")
parser.add_argument('-b', '--batch_size', type=int, default=32,
                    help="Batch size.")
parser.add_argument('--lr',  type=float, nargs=1, default=0.001,
                    help='Base learning rate.')
parser.add_argument('--checkp', type=str, nargs=1,
                    help="Path to checkpoint file.")
parser.add_argument('--epochs', type=int, default=10,
                    help='Times we loop through entire dataset.')
parser.add_argument('--use_default', action='store_true',
                    help='Use default PyTorch implementation of Resnet18.')

# Train the model one epoch
def train_loop(dataloader, model, loss_fn, losses):

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
        
        if idx % 200 == 0:
            print(f'[batch {idx+1}]: {running_loss/2000}')
            losses.append(running_loss)
            running_loss = 0

    # save the model 
    torch.save(model.state_dict(), 'weights/model.pt')
    return model


def test_loop(dataloader, model, loss_fn) -> (float, float):
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
    return test_loss, correct

# Plot the loss/accuracy values
def plot(loss:list, accuracies=None, title=None):
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(loss, color='red')
    if accuracies:
        plt.ylabel('Accuracy')
        plt.plot(accuracies, color='green')
    plt.show()
    plt.close()
    
if __name__=='__main__':

    args = parser.parse_args() 
     # Hyperparameters
    
    
    # Get the dataset and dataloader ready
    dataset = utils.CIFAR10Dataset(args.data)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = utils.CIFAR10Dataset(args.data, test=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
     
    # Use default Pytorch model or our own
    if args.use_default:
        model = models.resnet18(pretrained=False)
    else:
        model = resnet.ResidualNet18() 

    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    skipTraining = False
    try:
        model.load_state_dict(torch.load('weights/model.pth'))
        skipTraining = True
    except:
        print('Weight file not found')
    
    losses, test_losses, test_accs  = [], [], []
    for i in range(args.epochs):

   
        train_loop(dataloader, model, loss_fn, losses)

        test_loss, acc = test_loop(test_dataloader, model, loss_fn)
        test_losses.append(test_loss)
        test_accs.append(acc)

    plot(losses, None, 'training_loss')
    plot(test_losses, test_accs, 'testing_loss')
    torch.save(model.state_dict(), 'weights/model.pth')
