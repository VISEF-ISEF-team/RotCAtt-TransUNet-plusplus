import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair

class Embeddings(nn.Module):
    def __init__(self, config, img_size, patch_size, in_channels):
        super().__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patch_size)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        
        self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
                                          out_channels=in_channels,
                                          kernel_size=patch_size,
                                          stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, in_channels))
        self.dropout = nn.Dropout(config.dropout_rate)
        
    def forward(self, x):
        x = self.patch_embeddings(x)
        x = x.flatten(2).transpose(-1, -2)
        z = x + self.position_embeddings
        z = self.dropout(z)
        return z

class LinearEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        img_size = config.img_size
        df = config.df
        p = config.p
        
        self.embed_1 = Embeddings(config, img_size=img_size,      patch_size=p[0], in_channels=df[0])
        self.embed_2 = Embeddings(config, img_size=img_size // 2, patch_size=p[1], in_channels=df[1])
        self.embed_3 = Embeddings(config, img_size=img_size // 4, patch_size=p[2], in_channels=df[2])
        
    def forward(self, x1, x2, x3):
        z1 = self.embed_1(x1)
        z2 = self.embed_2(x2)
        z3 = self.embed_3(x3)

        return z1, z2, z3