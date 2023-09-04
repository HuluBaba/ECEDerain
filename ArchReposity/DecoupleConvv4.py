import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, mid_channels, *, ksize, stride):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp

        branch_main = [
            # pw
            nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            # pw-linear
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
        ]
        self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:
            branch_proj = [
                # dw
                nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
            ]
            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            self.branch_proj = None

    def forward(self, old_x):
        if self.stride==1:
            x_proj, x = self.channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride==2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert (num_channels % 4 == 0)
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]


class ShuffleV1Block(nn.Module):
    def __init__(self, inp, oup, *, group, first_group, mid_channels, ksize, stride):
        super(ShuffleV1Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp
        self.group = group

        if stride == 2:
            outputs = oup - inp
        else:
            outputs = oup

        branch_main_1 = [
            # pw
            nn.Conv2d(inp, mid_channels, 1, 1, 0, groups=1 if first_group else group, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
        ]
        branch_main_2 = [
            # pw-linear
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, groups=group, bias=False),
            nn.BatchNorm2d(outputs),
        ]
        self.branch_main_1 = nn.Sequential(*branch_main_1)
        self.branch_main_2 = nn.Sequential(*branch_main_2)

        if stride == 2:
            self.branch_proj = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, old_x):
        x = old_x
        x_proj = old_x
        x = self.branch_main_1(x)
        if self.group > 1:
            x = self.channel_shuffle(x)
        x = self.branch_main_2(x)
        if self.stride == 1:
            return F.relu(x + x_proj)
        elif self.stride == 2:
            return torch.cat((self.branch_proj(x_proj), F.relu(x)), 1)

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert num_channels % self.group == 0
        group_channels = num_channels // self.group
        
        x = x.reshape(batchsize, group_channels, self.group, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batchsize, num_channels, height, width)

        return x


class DeCoupleConvv4(nn.Module):
    def __init__(self, dim, ffn_expansion_factor = 2.66, bias=False):
        super(DeCoupleConvv4, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.hidden_features = hidden_features
        self.inPconv = nn.Conv2d(dim, 2*hidden_features, 1, 1, 0, bias=bias)
        self.inDconv = nn.Conv2d(2*hidden_features,4*hidden_features, 3, 1, 1, groups=2*hidden_features, bias=bias)
        self.inrelu = nn.ReLU()
        self.avgconv = nn.Conv2d(4*hidden_features, 4*hidden_features, 3, 1, 'same', padding_mode='replicate', bias=bias)
        self.avgconv.weight.data = torch.ones_like(self.avgconv.weight.data)/9
        self.avgconv.weight.requires_grad = False
        self.l_pointconv = nn.Conv2d(4*hidden_features, 4*hidden_features, 1, 1, 0, groups=hidden_features, bias=bias)
        self.h_pointconv = nn.Conv2d(4*hidden_features, 4*hidden_features, 1, 1, 0, groups=hidden_features, bias=bias)
        self.lrelu1 = nn.ReLU()
        self.hrelu1 = nn.ReLU()
        self.l_pointconv2 = nn.Conv2d(4*hidden_features, 4*hidden_features, 1, 1, 0, groups=hidden_features, bias=bias)
        self.h_pointconv2 = nn.Conv2d(4*hidden_features, 4*hidden_features, 1, 1, 0, groups=hidden_features, bias=bias)
        self.l_depthconv = nn.Conv2d(4*hidden_features, hidden_features, 3, 1, 1, groups=hidden_features, bias=bias)
        self.h_depthconv = nn.Conv2d(4*hidden_features, hidden_features, 3, 1, 1, groups=hidden_features, bias=bias)
        self.lrelu2 = nn.ReLU()
        self.hrelu2 = nn.ReLU()
        self.outconv = nn.Conv2d(2*hidden_features, dim, 1, 1, 0, bias=bias)

    def forward(self, old_x):
        residual = old_x
        x = self.inPconv(old_x)
        x = self.inDconv(x)
        x = self.inrelu(x)
        l = self.avgconv(x)
        h = x - l
        l = self.l_pointconv(l)
        h = self.h_pointconv(h)
        l = self.lrelu1(l)
        h = self.hrelu1(h)
        l = self.shuffle(l)
        h = self.shuffle(h)
        l_tol, l_toh = torch.split(self.l_pointconv2(l), 2*self.hidden_features, dim=1)
        h_toh, h_tol = torch.split(self.h_pointconv2(h), 2*self.hidden_features, dim=1)
        l = self.l_depthconv(torch.cat((l_tol, h_tol), dim=1))
        h = self.h_depthconv(torch.cat((h_toh, l_toh), dim=1))
        l = self.lrelu2(l)
        h = self.hrelu2(h)
        out = self.outconv(torch.cat((l, h), dim=1))
        return out + residual

    def shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        group_channels = num_channels // 4
        x = x.reshape(batchsize, group_channels, 4, height*width).permute(0, 2, 1, 3).reshape(batchsize, num_channels, height, width)
        return x



if __name__ == "__main__":

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

