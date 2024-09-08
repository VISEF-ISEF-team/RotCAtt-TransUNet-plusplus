from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn


class RotCAtt_TransUNet_plusplus_GradCam(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.dense = model.dense
        self.linear_embedding = model.linear_embedding
        self.transformer = model.transformer
        self.rotatory_attention = model.rotatory_attention
        self.reconstruct = model.reconstruct
        self.decoder = model.decoder

        self.out = model.out

        self.gradients = []

    def activations_hook(self, grad):
        self.gradients.append(grad)

    def get_activations_gradient(self):
        return self.gradients

    def clear_activations_gradient(self):
        self.gradients.clear()

    def get_activations(self, x):
        x1, x2, x3, x4 = self.dense(x)
        emb1, emb2, emb3 = self.linear_embedding(x1, x2, x3)
        enc1, enc2, enc3 = self.transformer(emb1, emb2, emb3)
        r1, r2, r3 = self.rotatory_attention(emb1, emb2, emb3)

        f1 = enc1 + r1
        f2 = enc2 + r2
        f3 = enc3 + r3

        o1, o2, o3 = self.reconstruct(f1, f2, f3)
        y = self.decoder(o1, o2, o3, x4)
        return y

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
        y.register_hook(self.activations_hook)
        return self.out(y)
