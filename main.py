import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import model.resnet as resnet


if __name__=='__main__':
    #x = resnet.res_block(1, 64, 64, 3, 2)
    y = resnet.ResidualNet18()
    print(y)
    print(y.conv2_x)
    pass
