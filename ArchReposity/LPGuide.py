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


if __name__ == "__main__":
    x = torch.randn(1, 32, 5, 5)
    model = SideBranch(32)
    out = model(x)
    z = x*out
    print(z.shape)
    print(torch.equal(z, x))
    print(out.shape)
    print(out)