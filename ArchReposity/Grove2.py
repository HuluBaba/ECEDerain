'''
--------------------------------------------
Other variants of Plain should modify this pharagraph.
--------------------------------------------
Grove2, STS+FGF +refinement, no EDC
its in-embedding is a Expert Extraction.
its MSA module is Soft Threshold Sparse attention.
its FFA module is full FDF module.
3 FDFtrans refinement.
its outprojection is a 3x3 conv layer.
no EDC.
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

    def forward(self, x, f):
        x = self.project_in(x)
        lp_guide = self.sidebranch(x)
        x = self.relu1(self.dconv1(x))
        x = self.relu2(self.dconv2(x))
        x = self.project_out(x)
        x = x*lp_guide
        return x


##  Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, lp_guide=True, sb_guide=True):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = STSAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        if not lp_guide and not sb_guide:
            self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        elif lp_guide and not sb_guide:
            self.ffn = LPGuideFeedForward(dim, ffn_expansion_factor, bias)
        elif not lp_guide and sb_guide:
            self.ffn = SBFeedForward(dim, ffn_expansion_factor, bias)
        else:
            self.ffn = LPSBFeedForward(dim, ffn_expansion_factor, bias)
        # self.ffn = LPGuideFeedForward(dim, ffn_expansion_factor, bias) if lp_guide else FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, p, f):
        

        p = p + self.attn(self.norm1(p))
        p = p + self.ffn(self.norm2(p), f)

        return p, f

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

    def forward(self, x, f):
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
    def __init__(self, num_layers=3, innerch=64):
        super(Expert_Extraction, self).__init__()
        self.preconv = nn.Conv2d(3, innerch, 3, 1, 1)
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


## Frequency Side Branch Series
class Genf(nn.Module):
    def __init__(self):
        super(Genf, self).__init__()
    
    def forward(self, x):
        return x

class Freq_Fusion(nn.Module):
    def __init__(
            self,
            dim,
    ):
        super(Freq_Fusion, self).__init__()
        self.dim = dim
        self.conv_init_1 = nn.Sequential(  # PW
            nn.Conv2d(dim, dim, 1),
            nn.GELU()
        )
        self.conv_init_2 = nn.Sequential(  # DW
            nn.Conv2d(dim, dim, 1),
            nn.GELU()
        )
        self.conv_mid = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1),
            nn.GELU()
        )
        self.FFC = FourierUnit(self.dim*2, self.dim*2)

        self.bn = torch.nn.BatchNorm2d(dim*2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x_1, x_2 = torch.split(x, self.dim, dim=1)
        x_1 = self.conv_init_1(x_1)
        x_2 = self.conv_init_2(x_2)
        x0 = torch.cat([x_1, x_2], dim=1)
        x = self.FFC(x0) + x0
        x = self.relu(self.bn(x))

        return x


class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1):
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        batch, c, h, w = x.size()

        # (batch, c, h, w/2+1, 2)
        ffted = torch.fft.rfft2(x, norm='ortho')
        x_fft_real = torch.unsqueeze(torch.real(ffted), dim=-1)
        x_fft_imag = torch.unsqueeze(torch.imag(ffted), dim=-1)
        ffted = torch.cat((x_fft_real, x_fft_imag), dim=-1)
        # (batch, c, 2, h, w/2+1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.view_as_complex(ffted)

        output = torch.fft.irfft2(ffted, s=(h, w), norm='ortho')

        return output


class IDI(nn.Module):
    def __init__(self, dim=3):
        super(IDI, self).__init__()
        self.freq_fusion = Freq_Fusion(dim)
        self.se = SESideBranch(dim, dim//3, dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.freq_fusion(x)
        se_k = self.sigmoid(self.se(x))
        x = x * se_k
        return x

class fDownsample(nn.Module):
    def __init__(self, input_dim=4, output_dim=1, kernel_size=4, stride=2):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = output_dim

        self.proj = nn.Sequential(nn.PixelUnshuffle(2),
                                  nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1, bias=False),
                                  )


    def forward(self, x):
        x = self.proj(x)
        return x

class fUpsample(nn.Module):
    def __init__(self, input_dim=4, output_dim=1, kernel_size=4, stride=2):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = output_dim

        self.proj = nn.Sequential(nn.PixelShuffle(2),
                                  nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1, bias=False),
                                  )

    def forward(self, x):
        x = self.proj(x)
        return x

class SBFeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(SBFeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, 2*hidden_features, 1, 1, 0, bias=bias)

        self.dconv1 = nn.Conv2d(2*hidden_features, 4*hidden_features, 5, 1, 2, groups=hidden_features, bias=bias)
        self.dconv2 = nn.Conv2d(4*hidden_features, 2*hidden_features, 5, 1, 2, groups=hidden_features, bias=bias)

        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.project_out = nn.Conv2d(2*hidden_features, dim, 1, 1, 0, bias=bias)

    def forward(self, x, f):
        x = self.project_in(x)
        x = self.relu1(self.dconv1(x))
        x = self.relu2(self.dconv2(x))
        x = self.project_out(x)
        x = x * f
        return x

class LPSBFeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(LPSBFeedForward, self).__init__()

        self.gamma = nn.Parameter(torch.zeros(1))
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, 2*hidden_features, 1, 1, 0, bias=bias)
        self.sidebranch = SideBranch(2*hidden_features)

        self.dconv1 = nn.Conv2d(2*hidden_features, 4*hidden_features, 5, 1, 2, groups=hidden_features, bias=bias)
        self.dconv2 = nn.Conv2d(4*hidden_features, 2*hidden_features, 5, 1, 2, groups=hidden_features, bias=bias)

        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.project_out = nn.Conv2d(2*hidden_features, dim, 1, 1, 0, bias=bias)

    def forward(self, x, f):
        x = self.project_in(x)
        lp_guide = self.sidebranch(x)
        x = self.relu1(self.dconv1(x))
        x = self.relu2(self.dconv2(x))
        x = self.project_out(x)
        x = x*(self.gamma*f + (1-self.gamma)*lp_guide)
        return x

## Main Model
class Grove2(nn.Module):
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

        super(Grove2, self).__init__()

        
        self.patch_embed = Expert_Extraction(num_layers=3, innerch=dim)
        self.genf = Genf()
        
        self.fencoder_level1 = IDI()
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type, lp_guide=True, sb_guide=True) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.fdown1_2 = fDownsample()

        self.fencoder_level2 = IDI()
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, lp_guide=False, sb_guide=True) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.fdown2_3 = fDownsample()

        self.fencoder_level3 = IDI()
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, lp_guide=False, sb_guide=True) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.fdown3_4 = fDownsample()

        self.flatent = IDI()
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, lp_guide=False, sb_guide=True) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.fup4_3 = fUpsample()
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)

        self.fdecoder_level3 = IDI()
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, lp_guide=False, sb_guide=True) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.fup3_2 = fUpsample()
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)

        self.fdecoder_level2 = IDI()
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, lp_guide=False, sb_guide=True) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.fup2_1 = fUpsample()

        self.fdecoder_level1 = IDI()
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, lp_guide=True, sb_guide=True) for i in range(num_blocks[0]+num_ertrans)])

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)


    def forward(self, input):
        p_e1_i = self.patch_embed(input)
        f_e1_i = self.genf(input)

        f_e1_i = self.fencoder_level1(f_e1_i)
        p_e1_o, f_e1_o = self.encoder_level1(p_e1_i, f_e1_i)

        p_e2_i = self.down1_2(p_e1_o)
        f_e2_i = self.fdown1_2(f_e1_o)

        f_e2_i = self.fencoder_level2(f_e2_i)
        p_e2_o, f_e2_o = self.encoder_level2(p_e2_i, f_e2_i)

        p_e3_i = self.down2_3(p_e2_o)
        f_e3_i = self.fdown2_3(f_e2_o)

        f_e3_i = self.fencoder_level3(f_e3_i)
        p_e3_o, f_e3_o = self.encoder_level3(p_e3_i, f_e3_i)

        p_latent_i = self.down3_4(p_e3_o)
        f_latent_i = self.fdown3_4(f_e3_o)

        f_latent_i = self.flatent(f_latent_i)
        p_latent_o, f_latent_o = self.latent(p_latent_i, f_latent_i)

        p_d3_i = self.up4_3(p_latent_o)
        p_d3_i = torch.cat([p_d3_i, p_e3_o], 1)
        p_d3_i = self.reduce_chan_level3(p_d3_i)
        f_d3_i = self.fup4_3(f_latent_o)

        f_d3_i = self.fdecoder_level3(f_d3_i)
        p_d3_o, f_d3_o = self.decoder_level3(p_d3_i, f_d3_i)

        p_d2_i = self.up3_2(p_d3_o)
        p_d2_i = torch.cat([p_d2_i, p_e2_o], 1)
        p_d2_i = self.reduce_chan_level2(p_d2_i)
        f_d2_i = self.fup3_2(f_d3_o)

        f_d2_i = self.fdecoder_level2(f_d2_i)
        p_d2_o, f_d2_o = self.decoder_level2(p_d2_i, f_d2_i)

        p_d1_i = self.up2_1(p_d2_o)
        p_d1_i = torch.cat([p_d1_i, p_e1_o], 1)
        f_d1_i = self.fup2_1(f_d2_o)

        f_d1_i = self.fdecoder_level1(f_d1_i)
        p_d1_o, f_d1_o = self.decoder_level1(p_d1_i, f_d1_i)

        pred_b = self.output(p_d1_o)
        output = pred_b + input

        return output







    # def forward(self, inp_img):

    #     inp_enc_level1 = self.patch_embed(inp_img)
    #     # fi_1 = self.genf(inp_img)

    #     out_enc_level1 = self.encoder_level1(inp_enc_level1)  

    #     inp_enc_level2 = self.down1_2(out_enc_level1)
    #     out_enc_level2 = self.encoder_level2(inp_enc_level2)

    #     inp_enc_level3 = self.down2_3(out_enc_level2)
    #     out_enc_level3 = self.encoder_level3(inp_enc_level3)

    #     inp_enc_level4 = self.down3_4(out_enc_level3)
    #     latent = self.latent(inp_enc_level4)

    #     inp_dec_level3 = self.up4_3(latent)
    #     inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
    #     inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
    #     out_dec_level3 = self.decoder_level3(inp_dec_level3)

    #     inp_dec_level2 = self.up3_2(out_dec_level3)
    #     inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
    #     inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
    #     out_dec_level2 = self.decoder_level2(inp_dec_level2)

    #     inp_dec_level1 = self.up2_1(out_dec_level2)
    #     inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
    #     out_dec_level1 = self.decoder_level1(inp_dec_level1)
    #     pred_b = self.output(out_dec_level1)

    #     output = pred_b + inp_img

    #     return output



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


    model = Grove2()
    model.to('cuda')
    summary(model,(3,128,128))
    getModelSize(model)
    # Count the size of each submodule
    for name, module in model.named_children():
        print(f"Total size of the {name} :{getModelSize(module)[-1]:.3f} MB")
    input_tensor = torch.rand((2,3,128,128)).to('cuda')
    print(input_tensor.shape)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)
