import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import model.resnet as resnet

def build_model():
    pass
def main():
    pass

if __name__=='__main__':
    #x = resnet.res_block(1, 64, 64, 3, 2)
    y = resnet.ResidualNet18()
    print(y)
    pass
