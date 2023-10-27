import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class DecoupleConvv7(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(DecoupleConvv7, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, 2*hidden_features, 1, 1, 0, bias=bias)

        self.dconv1 = nn.Conv2d(2*hidden_features, 4*hidden_features, 5, 1, 2, groups=hidden_features, bias=bias)
        self.dconv2 = nn.Conv2d(4*hidden_features, 2*hidden_features, 5, 1, 2, groups=hidden_features, bias=bias)

        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.project_out = nn.Conv2d(2*hidden_features, dim, 1, 1, 0, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.relu1(self.dconv1(x))
        x = self.relu2(self.dconv2(x))
        x = self.project_out(x)
        return x



if __name__ == "__main__":
    x = torch.randn(1, 8, 32, 32)
    model = DecoupleConvv7(8, 2.66, False)
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


        