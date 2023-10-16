import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class DeCoupleConvv6(nn.Module):
    def __init__(self, dim, ffn_expansion_factor = 2.66, bias=False):
        super(DeCoupleConvv6, self).__init__()
        self.dim = dim
        hidden_features = int(dim * ffn_expansion_factor)
        self.hidden_features = hidden_features
        self.avgconv = nn.Conv2d(4*hidden_features, 4*hidden_features, 3, 1, 'same', padding_mode='replicate', groups=4*hidden_features, bias=bias)
        self.avgconv.weight.data = torch.ones_like(self.avgconv.weight.data)/9
        self.avgconv.weight.requires_grad = False
        self.inprocess = nn.Sequential(
            nn.Conv2d(dim, 2*hidden_features, 1, 1, 0, bias=bias),
            nn.Conv2d(2*hidden_features, 4*hidden_features, 3, 1, 1, groups=2*hidden_features, bias=bias),
            nn.ReLU(inplace=True),
        )
        self.hconv1 = nn.Conv2d(4*hidden_features, 2*hidden_features, 3, 1, 1, groups=2*hidden_features, bias=bias)
        self.hconv2 = nn.Conv2d(2*hidden_features,   hidden_features, 3, 1, 1, groups=  hidden_features, bias=bias)
        self.lconv1 = nn.Conv2d(4*hidden_features, 2*hidden_features, 3, 1, 2, dilation=2 ,groups=2*hidden_features, bias=bias)
        self.lconv2 = nn.Conv2d(2*hidden_features,   hidden_features, 3, 1, 2, dilation=2 ,groups=  hidden_features, bias=bias)
        self.hrelu1 = nn.ReLU()
        self.hrelu2 = nn.ReLU(inplace=True)
        self.lrelu1 = nn.ReLU()
        self.lrelu2 = nn.ReLU(inplace=True)
        self.outconv = nn.Conv2d(2*hidden_features, dim, 1, 1, 0, bias=bias)
    
    def forward(self, x):
        x = self.inprocess(x)       # dim -> 4*hidden_features
        l = self.avgconv(x)         # 4*hidden_features -> 4*hidden_features
        h = x - l                   # 4*hidden_features -> 4*hidden_features
        h2h, h2l = torch.split(self.hrelu1(self.hconv1(h)), self.hidden_features, dim=1)          # 4*hidden_features -> 2*hidden_features
        l2l, l2h = torch.split(self.lrelu1(self.lconv1(l)), self.hidden_features, dim=1)          # 4*hidden_features -> 2*hidden_features
        h = torch.cat([h2h, l2h], dim=1)    # 2*hidden_features -> 2*hidden_features
        l = torch.cat([l2l, h2l], dim=1)    # 2*hidden_features -> 2*hidden_features
        h = self.hrelu2(self.hconv2(h))     # 2*hidden_features -> hidden_features
        l = self.lrelu2(self.lconv2(l))     # 2*hidden_features -> hidden_features
        x = torch.cat([l, h], dim=1)        # 2*hidden_features -> 2*hidden_features
        x = self.outconv(x)                 # 2*hidden_features -> dim
        return x




if __name__ == "__main__":
    x = torch.randn(1, 32, 32, 32)
    model = DeCoupleConvv6(32, 2.66, False)
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


        