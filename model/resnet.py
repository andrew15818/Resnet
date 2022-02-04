import torch
import torch.nn as nn

class res_block(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=2):
        super(res_block, self).__init__()
        self.layers = nn.Sequential(    
                        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1),
                        nn.BatchNorm2d(out_channels),
                        #nn.ReLU(),
                        nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=1),
                        nn.BatchNorm2d(out_channels),
                        #nn.ReLU()
                    )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None

        # Downsample the input for the residual
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 
                    kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )


    def forward(self, x):
        identity = x
        out = self.layers(x) 
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out
'''
According to the paper, if the layer has a stride of 2, it will downsample the 
inputs, so in the residual we have to change the dimensions of the input.
'''
class ResidualNet18(nn.Module):
    def __init__(self):
        super(ResidualNet18, self).__init__()
        self.conv_1 = nn.Conv2d(
                            in_channels=3, 
                            out_channels=32, 
                            kernel_size=3, # maybe change to 5
                            stride=1,      # maybe change to 2
                            padding=2
                        )
        self.bn = nn.BatchNorm2d(num_features=32)
        self.relu = nn.ReLU()

        
        
        self.conv2_x = nn.Sequential(
                res_block(in_channels=32, out_channels=32), 
                res_block(in_channels=32, out_channels=32)
            )
        self.conv3_x = nn.Sequential(
                res_block(in_channels=32, out_channels=64),
                res_block(in_channels=64, out_channels=64)
                )

        self.conv4_x = nn.Sequential(
                res_block(in_channels=64, out_channels=128),
                res_block(in_channels=128, out_channels=128)
                )
        self.conv5_x = nn.Sequential(
                res_block(in_channels=128, out_channels=256),
                res_block(in_channels=256, out_channels=256)
                )
        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
         
        self.fc = nn.Linear(1000,10)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv_1(x)
        
        for layer in self.conv2_x:
            x = layer(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pooling(x)
        print(x.shape)
        x = self.fc(x)
        x = self.softmax(x)
        return x

