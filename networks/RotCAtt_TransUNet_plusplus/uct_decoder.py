import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
            
    def forward(self, x):
        return self.act(self.batch_norm(self.conv(x)))

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
 
class CCA(nn.Module):
    def __init__(self, F_d, F_e):
        super().__init__()
        self.mlp_e = nn.Sequential(
            Flatten(),
            nn.Linear(F_d, F_e)
        )
        self.mlp_d = nn.Sequential(
            Flatten(),
            nn.Linear(F_d, F_e)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, d, e):
        # channel-wise attention
        avg_pool_e = F.avg_pool2d(e, (e.size(2), e.size(3)), stride=(e.size(2), e.size(3)))
        channel_att_e = self.mlp_e(avg_pool_e)
        
        avg_pool_d = F.avg_pool2d(d, (d.size(2), d.size(3)), stride=(d.size(2), d.size(3)))
        channel_att_d = self.mlp_d(avg_pool_d)
        
        channel_att_sum = (channel_att_e + channel_att_d) / 2.0
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(e)
        e_after_channel = e * scale
        
        out = self.relu(e_after_channel)
        return out

class UpBlockAtt(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.coatt = CCA(F_d=in_channels//2, F_e=in_channels//2)
                
        layers = []
        layers.append(ConvBatchNorm(in_channels, out_channels))
        for _ in range(nb_Conv - 1): 
            layers.append(ConvBatchNorm(out_channels, out_channels))
        self.nConvs = nn.Sequential(*layers)
        

    def forward(self, d, skip_e):
        d = self.up(d)
        skip_att = self.coatt(d=d, e=skip_e)
        x = torch.cat([skip_att, d], dim=1)
        return self.nConvs(x)

class UCTDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        df = config.df
        self.up3 = UpBlockAtt(df[2]*2, df[1], nb_Conv=2)
        self.up2 = UpBlockAtt(df[1]*2, df[0], nb_Conv=2)
        self.up1 = UpBlockAtt(df[1]  , df[0], nb_Conv=2)
    
    def forward(self, o1, o2, o3, d3):
        d2 = self.up3(d3, o3)
        d1 = self.up2(d2, o2)
        d0 = self.up1(d1, o1)
        return d0