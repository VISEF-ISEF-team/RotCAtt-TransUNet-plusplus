import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x
    
class Downsampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(2,2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.conv(self.pool(x))
        return x

class SkipConnection(nn.Module):
    def __init__(self, in_channels, out_channels, layer):
        super().__init__()
        self.layer = layer
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ConvBlock(in_channels[0]*layer+in_channels[1], out_channels)
        
    def forward(self, d, e1=None, e2=None, e3=None, e4=None):
        d = self.up(d)
        if self.layer==1: 
            f = torch.cat([e1,d], dim=1)
        elif self.layer==2:
            f = torch.cat([e1,e2,d], dim=1)
        elif self.layer==3: 
            f = torch.cat([e1,e2,e3,d], dim=1)
            
        o = self.conv(f)
        return o

class Dense(nn.Module):
    def __init__(self, config):
        super().__init__()
        filters = config.dense_filters
        self.down1 = ConvBlock(filters[0], filters[1])
        
        # Downsampling
        self.down2 = Downsampling(filters[1], filters[2])
        self.down3 = Downsampling(filters[2], filters[3])
        self.down4 = Downsampling(filters[3], filters[3])
        
        self.skip1_2 = SkipConnection([filters[1], filters[2]], filters[1], layer=1)
        self.skip2_2 = SkipConnection([filters[2], filters[3]], filters[2], layer=1)
        
        self.skip1_3 = SkipConnection([filters[1], filters[2]], filters[1], layer=2)
        
    def forward(self, input):
        x1_1 = self.down1(input)
        
        x2_1 = self.down2(x1_1)
        x1_2 = self.skip1_2(x2_1, x1_1)
        
        x3_1 = self.down3(x2_1)
        x2_2 = self.skip2_2(x3_1, x2_1)
        x1_3 = self.skip1_3(x2_2, x1_1, x1_2)
        
        x4_1 = self.down4(x3_1)
        
        return x1_3, x2_2, x3_1, x4_1
