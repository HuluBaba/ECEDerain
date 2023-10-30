import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

class SideBranch(nn.Module):
    def __init__(self,channel) -> None:
        super().__init__()
        # Linear layer for channel with bias
        self.linear = nn.Conv2d(channel, 1, kernel_size=1, bias=True)
        # init weight to make the output of linear layer to be 1
        self.linear.weight.data.fill_(0)
        self.linear.bias.data.fill_(1)

    def forward(self, x):
        # box filter 3x3 by conv2d
        x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        # Linear layer for channel with bias
        x = self.linear(x)
        x = F.relu(x)
        return x


class DecoupleConvv9(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(DecoupleConvv9, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, 2*hidden_features, 1, 1, 0, bias=bias)
        self.sidebranch = SideBranch(2*hidden_features)

        self.dconv1 = nn.Conv2d(2*hidden_features, 4*hidden_features, 5, 1, 2, groups=hidden_features, bias=bias)
        self.dconv2 = nn.Conv2d(4*hidden_features, 2*hidden_features, 5, 1, 2, groups=hidden_features, bias=bias)

        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.project_out = nn.Conv2d(2*hidden_features, dim, 1, 1, 0, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        lp_guide = self.sidebranch(x)
        x = self.relu1(self.dconv1(x))
        x = self.relu2(self.dconv2(x))
        x = self.project_out(x)
        x = x*lp_guide
        return x




if __name__ == "__main__":
    x = torch.randn(1, 8, 32, 32)
    model = DecoupleConvv9(8, 2.66, False)
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


        