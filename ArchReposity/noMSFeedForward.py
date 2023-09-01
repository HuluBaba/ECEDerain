import torch
import torch.nn as nn
import torch.nn.functional as F

class noMSFeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(noMSFeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv1 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.relu1 = nn.ReLU()

        self.dwconv2 = nn.Conv2d(hidden_features * 2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2 , bias=bias)

        self.relu2 = nn.ReLU()

        self.project_out = nn.Conv2d(hidden_features * 2, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv1(x)
        x = self.relu1(x)
        x = self.dwconv2(x)
        x = self.relu2(x)
        x = self.project_out(x)

        return x


if __name__ == '__main__':
    x = torch.randn(1, 64, 32, 32)
    model = noMSFeedForward(64, 2, False)
    y = model(x)
    print(y.shape)