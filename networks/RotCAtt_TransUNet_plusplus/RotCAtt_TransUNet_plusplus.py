import torch.nn as nn
from .dense_feature_extraction import Dense
from .linear_embedding import LinearEmbedding
from .transformer import Transformer
from .rotatory_attention import RotatoryAttention
from .reconstruction import Reconstruction
from .uct_decoder import UCTDecoder

class RotCAtt_TransUNet_plusplus(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vis = config.vis
        self.vis = True
        self.dense = Dense(config)
        self.linear_embedding = LinearEmbedding(config)
        self.transformer = Transformer(config)
        self.rotatory_attention = RotatoryAttention(config)
        self.reconstruct = Reconstruction(config)
        self.decoder = UCTDecoder(config)
        self.out = nn.Conv2d(config.df[0], config.num_classes, kernel_size=(1,1), stride=(1,1))
        
    def forward(self, x):
        x1, x2, x3, x4 = self.dense(x)
        z1, z2, z3 = self.linear_embedding(x1, x2, x3)
        e1, e2, e3 = self.transformer(z1, z2, z3)
        r1, r2, r3 = self.rotatory_attention(z1, z2, z3)

        f1 = e1 + r1
        f2 = e2 + r2
        f3 = e3 + r3
        
        o1, o2, o3 = self.reconstruct(f1, f2, f3)
        y = self.decoder(o1, o2, o3, x4)
        return self.out(y)