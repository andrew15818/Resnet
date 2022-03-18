import os
import pickle
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset

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
                    transforms.RandomHorizontalFlip(p=0.4),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    #transforms.RandomRotation(degrees=70)
                    ]
                )
        if self.test:
           self.batches = get_batch(os.path.join(img_dir, 'test_batch'))
           self.len = self.batches[b'data'].shape[0]
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


