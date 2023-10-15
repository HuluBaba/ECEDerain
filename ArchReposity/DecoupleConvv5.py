import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class DeCoupleConvv5(nn.Module):
    def __init__(self, dim, ffn_expansion_factor = 2.66, bias=False):
        super(DeCoupleConvv5, self).__init__()
        self.dim = dim
        hidden_features = int(dim * ffn_expansion_factor)
        self.hidden_features = hidden_features
        self.avgconv = nn.Conv2d(2*hidden_features, 2*hidden_features, 3, 1, 'same', padding_mode='replicate', bias=bias)
        self.avgconv.weight.data = torch.ones_like(self.avgconv.weight.data)/9
        self.avgconv.weight.requires_grad = False
        self.inconv = nn.Conv2d(dim, 2*hidden_features, 1, 1, 0, bias=bias)
        self.hconv1 = nn.Conv2d(2*hidden_features, 2*hidden_features, 3, 1, 1, groups=2*hidden_features, bias=bias)
        self.hconv2 = nn.Conv2d(2*hidden_features, 2*hidden_features, 3, 1, 1, groups=2*hidden_features, bias=bias)
        self.lthconv = nn.Conv2d(2*hidden_features, 2*hidden_features, 3, 1, 1, groups=2*hidden_features, bias=bias)
        self.hthconv = nn.Conv2d(2*hidden_features, 2*hidden_features, 3, 1, 1, groups=2*hidden_features, bias=bias)
        self.lconv1 = nn.Conv2d(2*hidden_features, 2*hidden_features, 3, 1, 2, dilation=2 ,groups=2*hidden_features, bias=bias)
        self.outconv = nn.Conv2d(2*hidden_features, dim, 1, 1, 0, bias=bias)

    def forward(self, x):
        x = self.inconv(x)          # dim -> 2*hidden_features
        l = self.avgconv(x)         # 2*hidden_features -> 2*hidden_features
        h = x - l                   # 2*hidden_features -> 2*hidden_features
        h = self.hconv1(h)          # 2*hidden_features -> 2*hidden_features
        h = h + l
        h = self.hconv2(h)          # 2*hidden_features -> 2*hidden_features
        l = self.lconv1(l)          # 2*hidden_features -> 2*hidden_features
        lth = self.lthconv(l)       # 2*hidden_features -> 2*hidden_features
        hth = self.hthconv(h)       # 2*hidden_features -> 2*hidden_features
        h = hth + lth               # 2*hidden_features -> 2*hidden_features
        x = h + l                   # 2*hidden_features -> 2*hidden_features
        x = self.outconv(x)         # 2*hidden_features -> dim
        return x





if __name__ == "__main__":
    x = torch.randn(1, 32, 32, 32)
    model = DeCoupleConvv5(32, 2.66, False)
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


        