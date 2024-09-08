import numpy as np
import torch.nn as nn

class ReconBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super().__init__()
        if kernel_size == 3: padding = 1
        else: padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        B, n_patch, hidden = x.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        
        x = nn.Upsample(scale_factor=self.scale_factor)(x)
        out = self.conv(x)
        out = self.batch_norm(out)
        out = self.act(out)
        return out

class Reconstruction(nn.Module):
    def __init__(self, config):
        super().__init__()
        df = config.df
        p = config.p
        self.reconstruct_1 = ReconBlock(df[0], df[0], kernel_size=1, scale_factor=(p[0], p[0]))
        self.reconstruct_2 = ReconBlock(df[1], df[1], kernel_size=1, scale_factor=(p[1], p[1]))
        self.reconstruct_3 = ReconBlock(df[2], df[2], kernel_size=1, scale_factor=(p[2], p[2]))
        
    def forward(self, f1, f2, f3):
        o1 = self.reconstruct_1(f1)
        o2 = self.reconstruct_2(f2)
        o3 = self.reconstruct_3(f3)
        return o1, o2, o3
