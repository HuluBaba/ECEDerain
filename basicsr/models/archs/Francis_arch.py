'''
Bill:
It makes us so frightened that we find that Jeremy.out is not used at all.
And BatchNorm is not suitable for low batch size.
1 Transformer style Intrans -> Linear Projection
2 Out and DownRes with BN -> DownRes without BN
Francis:
New style EDC Module
'''

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from torchsummary import summary

from einops import rearrange


import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from torchsummary import summary

from einops import rearrange


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##  Sparse Threshold Sparse Attention (STS A)
class STSAttnGen(nn.Module):
    def __init__(self, in_ch):
        super(STSAttnGen, self).__init__()
        self.in_ch = in_ch
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.ffn = nn.Sequential(nn.Linear(in_ch,in_ch),
                                #  nn.BatchNorm1d(in_ch),
                                 nn.ReLU(),
                                 nn.Linear(in_ch, in_ch),
                                 nn.Sigmoid(),
                                 )

    def forward(self, x):
        x_raw = x
        x = torch.abs(x)
        x_abs = x
        x = self.gap(x)
        x = torch.flatten(x, 1)
        average = x
        x = self.ffn(x)
        x = torch.mul(average, x)
        x = x.unsqueeze(2).unsqueeze(2)
        # soft thresholding
        sub = x_abs - x
        zeros = torch.zeros_like(sub)
        n_sub = torch.max(sub, zeros)
        x = torch.mul(torch.sign(x_raw), n_sub)
        return x

class STSAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(STSAttention, self).__init__()
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.attn1 = STSAttnGen(num_heads)
        self.attn2 = STSAttnGen(num_heads)
        self.attn3 = STSAttnGen(num_heads)
        self.attn4 = STSAttnGen(num_heads)

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(0.)

        self.thresholdweight = nn.Parameter(0.2*torch.ones(4, 1, 1, 1))

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, C, _ = q.shape


        attn = (q @ k.transpose(-2, -1)) * self.temperature

        attn1 = self.attn1(attn)
        attn2 = self.attn2(attn)
        attn3 = self.attn3(attn)
        attn4 = self.attn4(attn)

        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)
        attn4 = attn4.softmax(dim=-1)

        composed_attn = torch.stack([attn1, attn2, attn3, attn4], dim=1)
        composed_attn = torch.sum(torch.mul(composed_attn, self.thresholdweight),dim=1)

        out = composed_attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##  LP Guide FeedForward (LPG)
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

class LPGuideFeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(LPGuideFeedForward, self).__init__()

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


##  Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, lp_guide=True):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = STSAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = LPGuideFeedForward(dim, ffn_expansion_factor, bias) if lp_guide else FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

class InTrans(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, lp_guide=False):
        super(InTrans, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = NormalAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = LPGuideFeedForward(dim, ffn_expansion_factor, bias) if lp_guide else FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##  Normal Attention
class NormalAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(NormalAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

##  FeedForward Size 5
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

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

##  Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##  DownChannel Res
class ResUnit(nn.Module):
    def __init__(self, channels):
        super(ResUnit, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out += residual
        out = self.relu2(out)
        return out        

class DownRes(nn.Module):
    def __init__(self,in_ch=48*2):
        super(DownRes, self).__init__()
        self.resunit1 = ResUnit(in_ch)
        self.updim1 = nn.Conv2d(in_ch,32,3,1,1)
        self.resunit2 = ResUnit(32)
        self.updim2 = nn.Conv2d(32,8,3,1,1)
        self.resunit3 = ResUnit(8)
        self.updim3 = nn.Conv2d(8,3,3,1,1)

    def forward(self, x):
        x = self.resunit1(x)
        x = self.updim1(x)
        x = self.resunit2(x)
        x = self.updim2(x)
        x = self.resunit3(x)
        x = self.updim3(x)
        return x


## Error Compensation Module
NUM_OPS_A = 5
NUM_OPS_B = 5

class SESideBranch(nn.Module):
    def __init__(self, input_dim, mid_ch, output_dim):
        super(SESideBranch, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, mid_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(mid_ch, output_dim)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y)
        return y

class Experts_Layer_A(torch.nn.Module):
    def __init__(self, innerch=64):
        super(Experts_Layer_A, self).__init__()
        self.weight_gen = SESideBranch(innerch,2*innerch,NUM_OPS_A)
        self.dialated1 = nn.Sequential(nn.Conv2d(innerch,innerch,3,1,2,dilation=2,groups=innerch),
                                       nn.Conv2d(innerch,innerch,1,1,0))
        self.dialated2 = nn.Sequential(nn.Conv2d(innerch,innerch,5,1,4,dilation=2,groups=innerch),
                                       nn.Conv2d(innerch,innerch,1,1,0))
        self.dialated3 = nn.Sequential(nn.Conv2d(innerch,innerch,3,1,3,dilation=3,groups=innerch),
                                        nn.Conv2d(innerch,innerch,1,1,0))
        self.avgpool = nn.AvgPool2d(3,1,1,count_include_pad=False)
        self.maxpool = nn.MaxPool2d(3,1,1)
        self.postprocess = nn.Sequential(nn.Conv2d(innerch,innerch,1,1,0),nn.ReLU())

    def forward(self, x):
        weight = self.weight_gen(x)     # weight: [batch, NUM_OPS]
        x1 = self.dialated1(x)
        x2 = self.dialated2(x)
        x3 = self.dialated3(x)
        x4 = self.avgpool(x)
        x5 = self.maxpool(x)        # x1~x5: [batch, innerch, h, w]
        x = torch.stack([x1,x2,x3,x4,x5], dim=1)       # x: [batch, NUM_OPS, innerch, h, w], weights: [batch, NUM_OPS]
        y = torch.sum(x*weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), dim=1)                         # y: [batch, innerch, h, w]
        y = self.postprocess(y)                                 # y: [batch, innerch, h, w]
        return y

class Experts_Layer_B(torch.nn.Module):
    def __init__(self, innerch=64):
        super(Experts_Layer_B, self).__init__()
        self.weight_gen = SESideBranch(innerch,2*innerch,NUM_OPS_B)
        self.separable1 = nn.Sequential(nn.Conv2d(innerch,innerch,3,1,1,groups=innerch),
                                        nn.Conv2d(innerch,innerch,1,1,0),
                                        nn.ReLU(),
                                        nn.Conv2d(innerch,innerch,3,1,1,groups=innerch),
                                        nn.Conv2d(innerch,innerch,1,1,0))
        self.separable2 = nn.Sequential(nn.Conv2d(innerch,innerch,5,1,2,groups=innerch),
                                        nn.Conv2d(innerch,innerch,1,1,0),
                                        nn.ReLU(),
                                        nn.Conv2d(innerch,innerch,5,1,2,groups=innerch),
                                        nn.Conv2d(innerch,innerch,1,1,0))
        self.separable3 = nn.Sequential(nn.Conv2d(innerch,innerch,7,1,3,groups=innerch),
                                        nn.Conv2d(innerch,innerch,1,1,0),
                                        nn.ReLU(),
                                        nn.Conv2d(innerch,innerch,7,1,3,groups=innerch),
                                        nn.Conv2d(innerch,innerch,1,1,0))
        self.avgpool = nn.AvgPool2d(3,1,1,count_include_pad=False)
        self.maxpool = nn.MaxPool2d(3,1,1)
        self.postprocess = nn.Sequential(nn.Conv2d(innerch,innerch,1,1,0),nn.ReLU())

    def forward(self, x):
        weight = self.weight_gen(x)     # weight: [batch, NUM_OPS]
        x1 = self.separable1(x)
        x2 = self.separable2(x)
        x3 = self.separable3(x)
        x4 = self.avgpool(x)
        x5 = self.maxpool(x)        # x1~x5: [batch, innerch, h, w]
        x = torch.stack([x1,x2,x3,x4,x5], dim=1)       # x: [batch, NUM_OPS, innerch, h, w], weights: [batch, NUM_OPS]
        y = torch.sum(x*weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), dim=1)                         # y: [batch, innerch, h, w]
        y = self.postprocess(y)                                 # y: [batch, innerch, h, w]
        return y

class Expert_Extraction(torch.nn.Module):
    def __init__(self, num_layers=3, innerch=64, in_channel=3):
        super(Expert_Extraction, self).__init__()
        self.preconv = nn.Conv2d(in_channel, innerch, 3, 1, 1)
        self.prerelu = nn.ReLU()
        self.experts = torch.nn.ModuleList()
        for i in range(num_layers):
            self.experts.append(Experts_Layer_B(innerch=innerch))
            self.experts.append(Experts_Layer_A(innerch=innerch))

    def forward(self, x):
        x = self.preconv(x)
        x = self.prerelu(x)
        for i in range(len(self.experts)):
            res = x
            x = self.experts[i](x)
            x = x + res
            x = F.relu(x)
        return x

class SEBlock(nn.Module):
    def __init__(self, input_dim, reduction):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, reduction),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(reduction, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        mid_planes = int(out_planes/4)
        self.bn1 = nn.GroupNorm(num_groups=out_planes, num_channels=inter_planes)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.bn2 = nn.GroupNorm(num_groups=mid_planes, num_channels=out_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=True)
        
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return torch.cat([x, out], 1)

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.GroupNorm(num_groups=4, num_channels=out_planes)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.se = SEBlock(out_planes, out_planes//4)
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.se(out)
        return out

class ConvModule(nn.Module):
    def __init__(self):
        super(ConvModule, self).__init__()
        self.conv1 = nn.Conv2d(6,32,3,1,1)
        self.dense_block1=BottleneckBlock(32,32)
        self.trans_block1=TransitionBlock(64,32)

        ############# Block2-down 32-32  ##############
        self.dense_block2=BottleneckBlock(32,32)
        self.trans_block2=TransitionBlock(64,32)

        ############# Block3-down  16-16 ##############
        self.dense_block3=BottleneckBlock(32,32)
        self.trans_block3=TransitionBlock(64,32)

        ############# Block4-up  8-8  ##############
        self.dense_block4=BottleneckBlock(32,32)
        self.trans_block4=TransitionBlock(64,32)

        ############# Block5-up  16-16 ##############
        self.dense_block5=BottleneckBlock(64,32)
        self.trans_block5=TransitionBlock(96,32)

        self.dense_block6=BottleneckBlock(64,32)
        self.trans_block6=TransitionBlock(96,32)
        self.dense_block7=BottleneckBlock(64,32)
        self.trans_block7=TransitionBlock(96,32)
        self.dense_block8=BottleneckBlock(32,32)
        self.trans_block8=TransitionBlock(64,32)
        self.dense_block9=BottleneckBlock(32,32)
        self.trans_block9=TransitionBlock(64,32)
        self.dense_block10=BottleneckBlock(32,32)
        self.trans_block10=TransitionBlock(64,32)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=32)
        self.refine3 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x1=self.relu(self.norm(self.conv1(x)))
        x1=self.dense_block1(x1)
        x1=self.trans_block1(x1)
        x_1=F.avg_pool2d(x1, 2)
        ###  32x32
        x2=(self.dense_block2(x_1))
        x2=self.trans_block2(x2)
        x_2=F.avg_pool2d(x2, 2)
        ### 16 X 16
        x3=(self.dense_block3(x_2))
        x3=self.trans_block3(x3)
        x_3=F.avg_pool2d(x3, 2)
        ## Classifier  ##
        
        x4=(self.dense_block4(x_3))
        x4=self.trans_block4(x4)
        x_4=F.upsample_nearest(x4, scale_factor=2)
        x_4=torch.cat([x_4,x3],1)

        x5=(self.dense_block5(x_4))
        x5=self.trans_block5(x5)
        x_5=F.upsample_nearest(x5, scale_factor=2)
        x_5=torch.cat([x_5,x2],1)

        x6=(self.dense_block6(x_5))
        x6=(self.trans_block6(x6))
        x_6=F.upsample_nearest(x6, scale_factor=2)
        x_6=torch.cat([x_6,x1],1)
        x_6=(self.dense_block7(x_6))
        x_6=(self.trans_block7(x_6))
        x_6=(self.dense_block8(x_6))
        x_6=(self.trans_block8(x_6))
        x_6=(self.dense_block9(x_6))
        x_6=(self.trans_block9(x_6))
        x_6=(self.dense_block10(x_6))
        x_6=(self.trans_block10(x_6))
        residual = torch.sigmoid(self.refine3(x_6))

        return residual

class Error_estimator(nn.Module):
    def __init__(self):
        super(Error_estimator, self).__init__()
        self.expert_extraction = Expert_Extraction(innerch=48, in_channel=3)
        self.convblock2 = nn.Sequential(nn.Conv2d(48, 48, 3, 1, 1), nn.ReLU(), nn.Conv2d(48,3,3,1,1), nn.Sigmoid())
    def forward(self, pred_b):
        x = self.expert_extraction(pred_b)
        pred_err = self.convblock2(x)
        return pred_err

class Confidence_estimator(nn.Module):
    def __init__(self):
        super(Confidence_estimator, self).__init__()
        self.convblock_mix = nn.Sequential(nn.Conv2d(6, 12, 3, 1, 1), nn.ReLU(), nn.Conv2d(12, 6, 3, 1, 1), nn.ReLU())
        self.expert_extraction = Expert_Extraction(innerch=48, in_channel=6)
        self.convblock2 = nn.Sequential(nn.Conv2d(96, 48, 3, 1, 1, groups=48), nn.ReLU(), nn.Conv2d(48,48,3,1,1), nn.ReLU(), nn.Conv2d(48,3,3,1,1), nn.Sigmoid())
    def forward(self, o, pred_r, latent):
        _, _, h, w = o.size()
        # mix o and pred_r one channel by one one channel
        mix = torch.cat([o, pred_r], dim=2).reshape(-1,6,h,w)
        mix = self.convblock_mix(mix)
        mix = self.expert_extraction(mix)
        x = torch.cat([latent, mix], dim=2).reshape(-1,96,h,w)
        x = self.convblock2(x)
        return x
        

## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


## Main Model
class Francis(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 num_ertrans=3,
                 ):

        super(Francis, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim, bias=bias)
        
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, lp_guide=False) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, lp_guide=False) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, lp_guide=False) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, lp_guide=False) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, lp_guide=False) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0]+num_ertrans)])

        self.output = DownRes(in_ch=dim * 2)

        self.err_predictor = Error_estimator()
        self.com_predictor = Confidence_estimator()


    def forward(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)  

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        pred_r = self.output(out_dec_level1)

        pred_b = inp_img - pred_r

        pred_err = self.err_predictor(pred_b)
        latent = self.up2_1(self.up3_2(self.up4_3(latent)))
        pred_con = self.com_predictor(inp_img, pred_r, latent)
        output = pred_b - pred_err*pred_con


        return output

if __name__=="__main__":
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


    model = Francis()
    model.to('cuda')
    summary(model,(3,128,128))
    getModelSize(model)
    # Count the size of each submodule
    for name, module in model.named_children():
        print(f"Total size of the {name} :{getModelSize(module)[-1]:.3f} MB")
    input_tensor = torch.rand((2,3,64,64)).to('cuda')
    print(input_tensor.shape)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)
