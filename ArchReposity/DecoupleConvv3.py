import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class DeCoupleConvv3(nn.Module):
    def __init__(self, dim, ffn_expansion_factor = 2.66, bias=False):
        super(DeCoupleConvv3, self).__init__()
        self.dim = dim
        hidden_features = int(dim * ffn_expansion_factor)
        self.hidden_features = hidden_features
        self.avgconv = nn.Conv2d(4*hidden_features, 4*hidden_features, 3, 1, 'same', padding_mode='replicate', bias=bias)
        self.avgconv.weight.data = torch.ones_like(self.avgconv.weight.data)/9
        self.avgconv.weight.requires_grad = False
        self.inconv = nn.Conv2d(dim, 2*hidden_features, 1, 1, 0, bias=bias)
        self.mainconv = nn.Conv2d(2*hidden_features, 4*hidden_features, 3, 1, 1, groups=2*hidden_features, bias=bias)
        self.mainrelu = nn.ReLU()
        self.l_pointconv = nn.Conv2d(4*hidden_features, 4*hidden_features, 3, 1, 1, groups=4*hidden_features, bias=bias)
        self.h_pointconv = nn.Conv2d(4*hidden_features, 4*hidden_features, 3, 1, 1, groups=4*hidden_features, bias=bias)
        self.l_depthconv = nn.Conv2d(4*hidden_features, hidden_features, 3, 1, 1, groups=hidden_features, bias=bias)
        self.h_depthconv = nn.Conv2d(4*hidden_features, hidden_features, 3, 1, 1, groups=hidden_features, bias=bias)
        self.lrelu1 = nn.ReLU()
        self.lrelu2 = nn.ReLU()
        self.hrelu1 = nn.ReLU()
        self.hrelu2 = nn.ReLU()
        self.outconv = nn.Conv2d(2*hidden_features, dim, 1, 1, 0, bias=bias)
    
    def forward(self, x):
        x = self.inconv(x)          # dim -> 2*hidden_features
        x = self.mainconv(x)        # 2*hidden_features -> 4*hidden_features
        x = self.mainrelu(x)        # 4*hidden_features -> 4*hidden_features
        l = self.avgconv(x)         # 4*hidden_features -> 4*hidden_features
        h = x - l                   # 4*hidden_features -> 4*hidden_features
        l_tol, l_toh = torch.split(self.l_pointconv(l), self.hidden_features, dim=1)    # 4*hidden_features -> 4*hidden_features
        h_toh, h_tol = torch.split(self.h_pointconv(h), self.hidden_features, dim=1)    # 4*hidden_features -> 4*hidden_features
        l = self.lrelu1(l_tol + h_toh)  # 4*hidden_features -> 4*hidden_features
        h = self.hrelu1(h_tol + l_toh)  # 4*hidden_features -> 4*hidden_features
        l = self.l_depthconv(torch.cat([l_tol, h_tol], dim=1))                          # 4*hidden_features -> hidden_features
        h = self.h_depthconv(torch.cat([h_toh, l_toh], dim=1))                          # 4*hidden_features -> hidden_features
        l = self.lrelu(l)           # hidden_features -> hidden_features
        h = self.hrelu(h)           # hidden_features -> hidden_features
        x = torch.cat([l, h], dim=1)    # 2*hidden_features -> 2*hidden_features
        x = self.outconv(x)         # 2*hidden_features -> dim
        return x

if __name__ == "__main__":
    x = torch.randn(1, 32, 32, 32)
    model = DeCoupleConvv3(32, 2.66, False)
    y = model(x)
    print(y.shape)
    def getModelSize(model):
        param_size = 0
        param_sum = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
            param_sum += param.nelement()
        buffer_size = 0
        buffer_sum = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            buffer_sum += buffer.nelement()
        all_size = (param_size + buffer_size) / 1024 / 1024
        print(f"Total size of the {model.__class__.__name__} :{all_size:.3f} MB")
        return (param_size, param_sum, buffer_size, buffer_sum, all_size)

    getModelSize(model)


        