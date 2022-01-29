import torch
import torch.nn as nn

class res_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(res_block, self).__init__()
        self.layers = nn.Sequential(    
                        nn.Conv2d(in_channels, out_channels, kernel_size, stride),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(),
                        nn.Conv2d(in_channels, out_channels, kernel_size, stride),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU()
                    )

    def forward(self, x):
        x += self.layers(x)
        return x

class ResidualNet18(nn.Module):
    def __init__(self):
        super(ResidualNet18, self).__init__()
        self.conv_1 = nn.Conv2d(
                            in_channels=3, 
                            out_channels=64, 
                            kernel_size=7, 
                            stride=2
                        )
        self.conv2_x = []
        self.conv2_x.append(nn.Conv2d(
                            in_channels=64,
                            out_channels=64,
                            kernel_size=3,
                            stride=2
                        )
        )
        for i in range(2):
            self.conv2_x.append(res_block(
                in_channels=64, 
                out_channels=64, 
                kernel_size=3, 
                stride=2)
            )
        

        self.conv3_x = []
        for i in range(2):
            self.conv3_x.append(res_block(
                    in_channels=64,
                    out_channels=128,
                    kernel_size=3,
                    stride=2
                )
            )

        self.conv4_x = []
        for i in range(2):
            self.conv4_x.append(res_block(
                    in_channels=128,
                    out_channels=256,
                    kernel_size=3,
                    stride=2
                )
            )

        self.conv5_x = []
        for i in range(2):
            self.conv5_x.append(res_block(
                    in_channels=256,
                    out_channels=512,
                    kernel_size=3,
                    stride=2
                )
            )
        self.avg_pooling = nn.AvgPool2d(3, stride=2)
        # nn.flatten() ?
        # What is the output shape?
        self.fc = nn.Linear(,1000)
        self.softmax = nn.Softmax()

    def forward(self, x):
        pass

