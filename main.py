import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

import model.model_utils as utils
import model.resnet as resnet

DATA_DIR = '/home/poncedeleon/usb/cifar-10-batches-py'
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

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
    torch.save(model.state_dict(), 'weights/model.pth')
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
        plt.plot(accuracies, color='green')
    plt.show()
    plt.close()
    
if __name__=='__main__':
    
     # Hyperparameters
    batch_size = 32
    learning_rate = 0.01
    
    epochs = 10 
    
    
    # Get the dataset and dataloader ready
    dataset = utils.CIFAR10Dataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    test_dataset = utils.CIFAR10Dataset(DATA_DIR, test=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    model = resnet.ResidualNet18()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    skipTraining = False
    try:
        model.load_state_dict(torch.load('weights/model.pth'))
        skipTraining = True
    except:
        print('Weight file not found')
    
    losses, test_losses, test_accs  = [], [], []
    for i in range(epochs):

   
        train_loop(dataloader, model, loss_fn, losses)

        test_loss, acc = test_loop(test_dataloader, model, loss_fn)
        test_losses.append(test_loss)
        test_accs.append(acc)

    plot(losses, None, 'training_loss')
    plot(test_losses, test_accs, 'testing_loss')
    torch.save(model.state_dict(), 'weights/model.pth')
