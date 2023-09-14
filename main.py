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
parser.add_argument('-d', '--data_dir', type=str, default=DATA_DIR,
                    help="Path to data folder.")
parser.add_argument('-b', '--batch_size', type=int, default=32,
                    help="Batch size.")
parser.add_argument('--lr',  type=float, nargs=1, default=0.5,
                    help='Base learning rate.')
parser.add_argument('--checkpoint', type=str, nargs=1, default='',
                    help="Path to checkpoint file.")
parser.add_argument('--epochs', type=int, default=90,
                    help='Training epochs')
parser.add_argument('--use_default', action='store_true', default=False,
                    help='Use default PyTorch implementation of Resnet18.')


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# Train the model one epoch
def train(dataloader, model, loss_fn, optimizer, scheduler, args):

    model.train()
    running_loss = 0
    for idx, data in enumerate(dataloader):
        x = data[0].to(args.device)
        y = data[1].to(args.device)

        # reset the gradients
        optimizer.zero_grad()
        outputs = model(x)
        loss = loss_fn(outputs, y)

        # backprop
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() 
        
        if idx % 200 == 0:
            print(f'  Batch {idx:3}/{len(dataloader)}: Loss {(running_loss / (idx+1)):.4f}')

        #running_loss += (loss.item() / x.shape[0])

    loss_avg = running_loss / len(dataloader)

    return loss_avg


def test(dataloader, model, loss_fn, args) -> (float, float):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    acc = utils.AverageMeter('Test Accuracies')

    with torch.no_grad():
        model.eval()
        for x, y in dataloader:
            x = x.to(args.device)
            y = y.to(args.device)
            
            preds = model(x)
            test_loss += loss_fn(preds, y).item()
            acc.update(accuracy(preds, y)[0].item())
    test_loss /= num_batches

    print(f'Accuracy {acc.avg} Avg loss {test_loss:>8f}\n')
    return test_loss, acc.avg

# Plot the loss/accuracy values
def plot(train_val=None, test_val=None, title=None):
    plt.title(title)
    if train_val:
        n = len(train_val)
        plt.plot(range(n), train_val, label='Train')
    
    testn = len(test_val)
    plt.plot(range(testn), test_val, label='Test')
    plt.legend()
    plt.show()


        
def main():
    args = parser.parse_args() 
     # Hyperparameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    args.device = device

    train_transforms = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.4),
                    transforms.RandomRotation(degrees=70),
                    transforms.ToTensor(),
                    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]
                )

    # Get the dataset and dataloader ready
    dataset = utils.CIFAR10Dataset(args.data_dir, 
                                   train_transforms)
    train_loader = DataLoader(dataset, 
                            batch_size=args.batch_size, 
                            shuffle=True
                        )

    test_dataset = utils.CIFAR10Dataset(args.data_dir, 
                                        transform=transforms.ToTensor(),
                                        train=False
                                    )
    test_loader = DataLoader(test_dataset, 
                             batch_size=args.batch_size, 
                             shuffle=True
                        )

    # Use default Pytorch model or our own
    if args.use_default:
        model = models.resnet18(pretrained=False)
    else:
        model = resnet.ResNet18() 

    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                     T_max=len(dataset)
                                                )
    loss_fn = nn.CrossEntropyLoss()

    start_epoch = 0 

    # Load model from checkpoint
    if args.checkpoint != '':
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])
    
    losses, test_losses, test_accs  = [], [], []

    for i in range(start_epoch, args.epochs):

        print(f'Epoch {i}') 
        train_loss = train(train_loader, 
                           model, 
                           loss_fn, 
                           optimizer, 
                           scheduler, 
                           args
                        )

        test_loss, acc = test(test_loader, model, loss_fn, args)

        losses.append(train_loss)
        test_losses.append(test_loss)
        test_accs.append(acc)

        plot(losses, test_losses, 'losses')
        plot(None, test_accs, 'accuracies')

    state = {'model': model.state_dict(),
             'optimizer': optimimer.state_dict(),
             'epoch': epoch,
             }
    torch.save(state, 'weights/model.pt')

if __name__=='__main__':
    main()
    
