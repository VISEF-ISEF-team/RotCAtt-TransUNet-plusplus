import torch
import torch.nn as nn

class ExtractKeyVal(nn.Module):
    def __init__(self, config, level):
        super().__init__()
        self.W_K = nn.Linear(config.df[level], config.dk[level])
        self.W_V = nn.Linear(config.df[level], config.dv[level])
        
    def forward(self, x):
        K = self.W_K(x)
        V = self.W_V(x)
        return K, V
        
class ActivatedGeneral(nn.Module):
    def __init__(self, config, level):
        super().__init__()
        self.W = nn.Linear(config.dk[level], config.df[level])
        self.act = nn.Tanh()
        self.align = nn.Softmax(dim=0)
    
    def forward(self, r, K):
        e = self.act(self.W(K) @ r)
        a = self.align(e)
        return a
                    
class SingleAttention(nn.Module):
    def __init__(self, config, level):
        super().__init__()
        self.extract_key_val = ExtractKeyVal(config, level)
        self.activated_general = ActivatedGeneral(config, level)
        
    def forward(self, r, x):
        K, V = self.extract_key_val(x)
        a = self.activated_general(r, K)
        r_x = torch.matmul(V.transpose(-1,-2), a)
        return r_x
    
class AttentionBLock(nn.Module):
    def __init__(self, config, level):
        super().__init__()
        self.single_attention_target_left  = SingleAttention(config, level)
        self.single_attention_target_right = SingleAttention(config, level)
        self.single_attention_left_target  = SingleAttention(config, level)
        self.single_attention_right_target = SingleAttention(config, level)
        
    def forward(self, left, target, right):        
        r_t = torch.mean(target, dim=0)
        r_l = self.single_attention_target_left(r_t, left)
        r_r = self.single_attention_target_right(r_t, right)
            
        r_l_t = self.single_attention_left_target(r_l, target)
        r_r_t = self.single_attention_right_target(r_r, target)
        
        r = torch.concat([r_l, r_r, r_l_t, r_r_t])
        return r
    
class DenseBlock(nn.Module):
    def __init__(self, config, level):
        super().__init__()
        self.attention_block = AttentionBLock(config, level)
    
    def forward(self, x):
        N = x.size(0)
        r = []
        for index in range(1, N-1):
            left, target, right = x[index-1], x[index], x[index+1]
            r.append(self.attention_block(left, target, right))
        
        r = torch.stack(r, dim=0)
        r = torch.mean(r,  dim=0) 
        return r
    
class RotatoryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        df = config.df
        self.df = df
        self.dense_block1 = DenseBlock(config, level=0)
        self.dense_block2 = DenseBlock(config, level=1)
        self.dense_block3 = DenseBlock(config, level=2)
        
        self.W_r_1 = nn.Linear(df[0]*4, df[0])
        self.W_r_2 = nn.Linear(df[1]*4, df[1])
        self.W_r_3 = nn.Linear(df[2]*4, df[2])
                
    def forward(self, emb1, emb2, emb3):
        r1 = self.dense_block1(emb1)
        r2 = self.dense_block2(emb2)
        r3 = self.dense_block3(emb3)
        
        # Linear transformation
        r1 = self.W_r_1(r1).view(1, 1, self.df[0])
        r2 = self.W_r_2(r2).view(1, 1, self.df[1])
        r3 = self.W_r_3(r3).view(1, 1, self.df[2])
        
        return r1, r2, r3