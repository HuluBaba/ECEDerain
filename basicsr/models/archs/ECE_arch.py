import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

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

##  Top-K Sparse Attention (TKSA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(0.)

        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

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

        mask1 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask2 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask3 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask4 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        index = torch.topk(attn, k=int(C/2), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*2/3), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*3/4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*4/5), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))

        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)
        attn4 = attn4.softmax(dim=-1)

        out1 = (attn1 @ v)
        out2 = (attn2 @ v)
        out3 = (attn3 @ v)
        out4 = (attn4 @ v)

        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

##  Sparse Transformer Block (STB) 
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = DecoupleConv(dim, dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine))

    def forward(self, x):
        return self.op(x)

class ReLUConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.ReLU(inplace=False))

    def forward(self, x):
        return self.op(x)

class ResBlock(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False)
        self.conv2 = nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False)
        self.relu  = nn.ReLU(inplace=False)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + residual
        out = self.relu(out)
        return out


## Resizing modules
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

class ECE(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=32,
                 num_blocks=[4, 6, 6, 8],
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias'  ## Other option 'BiasFree'
                 ):

        super(ECE, self).__init__()

        # self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        
        self.encoder_level0 = UpRes(out_ch = dim)
        # self.encoder_level0 = subnet(dim)  ## We do not use MEFC for training Rain200L and SPA-Data

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.outconv = DownRes(in_ch = 2*dim)
        self.err_predictor = Error_Predictor()
        self.com_predictor = Compensator_predicter()
        self.dehazer = Dehazer(kernel_size=15, top_candidates_ratio=0.0001,omega=0.95,radius=40,eps=1e-3,open_threshold=True,depth_est=True)
        # self.refinement = subnet(dim=int(dim*2**1)) ## We do not use MEFC for training Rain200L and SPA-Data

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):

        # inp_enc_level1 = self.patch_embed(inp_img)
        inp_enc_level0 = self.encoder_level0(inp_img) ## We do not use MEFC for training Rain200L and SPA-Data
        out_enc_level1 = self.encoder_level1(inp_enc_level0)  

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
        pred_b = self.outconv(out_dec_level1)

        pred_err = self.err_predictor(pred_b, inp_img)
        pred_com = self.com_predictor(pred_b)
        derain_output = pred_b - pred_err*pred_com
        dehaze_output = self.dehazer(derain_output)
        output = dehaze_output + inp_img
        # out_dec_level1 = self.refinement(out_dec_level1) ## We do not use MEFC for training Rain200L and SPA-Data

        # out_dec_level1 = self.output(out_dec_level1) + inp_img

        return output


## Frequence Decouple Conv
class LowThresholdDC_test_3(nn.Module):
    '''
     f-space decouple with step size by conv
     recombine with depth-wise conv and point-wise conv
    '''
    def __init__(self, inchannel, patch_size=2):
        super(LowThresholdDC_test_3,self).__init__()
        self.channel=inchannel
        self.conv = nn.Conv2d(inchannel,inchannel,patch_size,1,'same',groups=inchannel)
        self.lhandle = nn.Sequential(nn.Conv2d(inchannel,inchannel,1,1,0),
                                    nn.ReLU(),
                                    nn.Conv2d(inchannel,inchannel,3,1,1))
        self.hhandle = nn.Sequential(nn.Conv2d(inchannel,inchannel,1,1,0),
                                    nn.ReLU(),
                                    nn.Conv2d(inchannel,inchannel,3,1,1))
        self.comprehensive = nn.Sequential(nn.Sigmoid(),
                                            nn.Conv2d(2*inchannel,inchannel,1,1,0),
                                            nn.ReLU())
LowThresholdDC = LowThresholdDC_test_3

class HighThresholdDC(nn.Module):
    def __init__(self, in_channel) -> None:
        super(HighThresholdDC , self).__init__()

        self.fscale_d = nn.Parameter(torch.zeros(in_channel), requires_grad=True)
        self.fscale_h = nn.Parameter(torch.zeros(in_channel), requires_grad=True)
        self.gap = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        x_d = self.gap(x)
        x_h = (x - x_d) * (self.fscale_h[None, :, None, None] + 1.)
        x_d = x_d  * self.fscale_d[None, :, None, None]
        return x_d + x_h

class DecoupleConv(nn.Module):
    def __init__(self, in_ch, out_ch, wave_vector_threshold=8):
        super(DecoupleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 2*out_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(2*out_ch, out_ch, 3, 1, 1)
        self.ltdc = LowThresholdDC(out_ch, patch_size=wave_vector_threshold)
        self.htdc = HighThresholdDC(out_ch)
    
    def forward(self, x):
        input = x
        x = self.conv1(x)
        ltdc_input , htdc_input = torch.chunk(x, 2, dim=1)
        ltdc_output = self.ltdc(ltdc_input)
        htdc_output = self.htdc(htdc_input)
        x = torch.cat([ltdc_output, htdc_output], dim=1)
        x = self.conv2(x)
        x = x + input
        return x


## Error Compensator
# 8 experts are dialated(pointwise+depthwise) 3(2),5(2),3(3), separable(3),(5),(7), avgpool(3), maxpool(3)
NUM_OPS = 8

class Experts_Layer(torch.nn.Module):
    def __init__(self, innerch=64):
        super(Experts_Layer, self).__init__()
        self.dialated1 = nn.Sequential(nn.Conv2d(innerch,innerch,3,1,2,dilation=2,groups=innerch),
                                       nn.Conv2d(innerch,innerch,1,1,0))
        self.dialated2 = nn.Sequential(nn.Conv2d(innerch,innerch,5,1,4,dilation=2,groups=innerch),
                                       nn.Conv2d(innerch,innerch,1,1,0))
        self.dialated3 = nn.Sequential(nn.Conv2d(innerch,innerch,3,1,3,dilation=3,groups=innerch),
                                        nn.Conv2d(innerch,innerch,1,1,0))
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

    def forward(self, x, weights):
        x1 = self.dialated1(x)
        x2 = self.dialated2(x)
        x3 = self.dialated3(x)
        x4 = self.separable1(x)
        x5 = self.separable2(x)
        x6 = self.separable3(x)
        x7 = self.avgpool(x)
        x8 = self.maxpool(x)        # x1~x8: [batch, innerch, h, w]
        x = torch.stack([x1,x2,x3,x4,x5,x6,x7,x8], dim=1)       # x: [batch, NUM_OPS, innerch, h, w], weights: [batch, NUM_OPS]
        y = torch.sum(x*weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), dim=1)                         # y: [batch, innerch, h, w]
        y = self.postprocess(y)                                 # y: [batch, innerch, h, w]
        return y

class Expert_Extraction(torch.nn.Module):
    def __init__(self, num_layers=3):
        super(Expert_Extraction, self).__init__()
        self.preconv = nn.Conv2d(3, 64, 3, 1, 1)
        self.prerelu = nn.ReLU()
        self.weight_gen = Expert_Weight_Gen(num_layers=num_layers)
        self.experts = torch.nn.ModuleList()
        for i in range(num_layers):
            self.experts.append(Experts_Layer())

    def forward(self, x):
        x = self.preconv(x)
        weights = self.weight_gen(x)
        x = self.prerelu(x)
        for i in range(len(self.experts)):
            res = x
            x = self.experts[i](x, weights[:,i,:])
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
        self.se = SEBlock(out_planes, 6)
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

class Expert_Weight_Gen(torch.nn.Module):
    def __init__(self, num_layers=3, inputch=64):
        super(Expert_Weight_Gen, self).__init__()
        self.num_layers = num_layers
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.lrl = nn.Sequential(
            nn.Linear(inputch, num_layers*NUM_OPS*2),
            nn.ReLU(),
            nn.Linear(num_layers*NUM_OPS*2, num_layers*NUM_OPS),
        )
    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.lrl(x)
        x = x.view(x.size(0), self.num_layers, -1)
        return x

class Error_Predictor(torch.nn.Module):
    def __init__(self, innerch=64):
        super(Error_Predictor, self).__init__()
        self.error_detector = Expert_Extraction(num_layers=3)   #3->64
        self.convblock1 = nn.Sequential(nn.Conv2d(innerch, innerch, 3, 1, 1), nn.ReLU(), nn.Conv2d(innerch,3,1,1))    #64->3
        self.convblock2 = nn.Sequential(nn.Conv2d(innerch, innerch, 3, 1, 1), nn.ReLU(), nn.Conv2d(innerch,3,1,1), nn.Sigmoid())    #64->3
        self.summaryblock = ConvModule()                        #6->3
        self.extractor = Expert_Extraction(num_layers=3)        #3->64
    def forward(self, pred_b, o):
        pred_b = self.convblock1(self.extractor(pred_b))        #3
        x = self.summaryblock(torch.cat([o,pred_b],dim=1))      #3
        x = self.error_detector(x)                              #64
        pred_err = self.convblock2(x)                                  #3
        return pred_err
        
class Compensator_predicter(torch.nn.Module):
    def __init__(self,innerch=64):
        super(Compensator_predicter, self).__init__()
        self.extractor = Expert_Extraction(num_layers=3)
        self.postprocess = nn.Sequential(nn.Conv2d(innerch, innerch, 3, 1, 1), nn.ReLU(), nn.Conv2d(innerch, 3, 3, 1, 1), nn.Sigmoid())
    def forward(self, pred_b):
        x = self.extractor(pred_b)
        x = self.postprocess(x)
        return x


## Dehaze
class Dehazer(nn.Module):
    def __init__(self,kernel_size,top_candidates_ratio,omega,
                 radius, eps,
                 open_threshold=True,
                 depth_est=False):
        super(Dehazer,self).__init__()
        
        # dark channel piror
        self.kernel_size = kernel_size
        self.pad = nn.ReflectionPad2d(padding=kernel_size//2)
        self.unfold = nn.Unfold(kernel_size=(self.kernel_size,self.kernel_size),padding=0)
        
        # airlight estimation.
        self.top_candidates_ratio = top_candidates_ratio
        self.open_threshold = open_threshold
        
        # raw transmission estimation 
        self.omega = omega
        
        # image guided filtering
        self.radius = radius
        self.eps = eps
        self.guide_filter = GuidedFilter2d(radius=self.radius,eps= self.eps)
        
        self.depth_est = depth_est
        
    def forward(self,image):
        
        # compute the dark channel piror of given image.
        b,c,h,w = image.shape
        image_pad = self.pad(image)
        local_patches = self.unfold(image_pad)
        dc,dc_index = torch.min(local_patches,dim=1,keepdim=True)
        dc = dc.view(b,1,h,w)
        dc_vis = dc
        # airlight estimation.
        top_candidates_nums = int(h*w*self.top_candidates_ratio)
        dc = dc.view(b,1,-1) # dark channels
        searchidx = torch.argsort(-dc,dim=-1)[:,:,:top_candidates_nums]
        searchidx = searchidx.repeat(1,3,1)
        image_ravel = image.view(b,3,-1)
        value = torch.gather(image_ravel,dim=2,index=searchidx)
        airlight,image_index = torch.max(value,dim =-1,keepdim=True)
        airlight = airlight.squeeze(-1)
        if self.open_threshold:
            airlight = torch.clamp(airlight,max=220)
        
        # get the raw transmission
        airlight = airlight.unsqueeze(-1).unsqueeze(-1)
        processed = image/airlight
        
        processed_pad = self.pad(processed)
        local_patches_processed = self.unfold(processed_pad)
        dc_processed, dc_index_processed = torch.min(local_patches_processed,dim=1,keepdim=True)
        dc_processed = dc_processed.view(b,1,h,w)
        
        raw_t = 1.0 - self.omega * dc_processed
        if self.open_threshold:
            raw_t = torch.clamp(raw_t,min=0.2)
            
        # raw transmission guided filtering.
        # refined_tranmission = soft_matting(image_data_tensor,raw_transmission,r=40,eps=1e-3)
        normalized_img = simple_image_normalization(image)
        refined_transmission = self.guide_filter(raw_t,normalized_img)
        
        
        # recover image: get radiance.
        image = image.float()
        tiledt = refined_transmission.repeat(1,3,1,1)
        
        dehaze_images = (image - airlight)*1.0/tiledt + airlight
        
        # recover scaled depth or not
        if self.depth_est:
            depth = recover_depth(refined_transmission)
            # return dehaze_images, dc_vis,airlight,raw_t,refined_transmission,depth
            return dehaze_images
        
        return dehaze_images
        # return dehaze_images, dc_vis,airlight,raw_t,refined_transmission

def simple_image_normalization(tensor):
    b,c,h,w = tensor.shape
    tensor_ravel = tensor.view(b,3,-1)
    image_min,_ = torch.min(tensor_ravel,dim=-1,keepdim=True)
    image_max,_ = torch.max(tensor_ravel,dim=-1,keepdim=True)
    image_min = image_min.unsqueeze(-1)
    image_max = image_max.unsqueeze(-1)
    
    normalized_image = (tensor - image_min) /(image_max-image_min)
    return normalized_image

def recover_depth(transmission,beta=0.001):
    negative_depth = torch.log(transmission)
    return (-negative_depth)/beta

class GuidedFilter2d(nn.Module):
    def __init__(self, radius: int, eps: float):
        super().__init__()
        self.r = radius
        self.eps = eps

    def forward(self, x, guide):
        if guide.shape[1] == 3:
            return guidedfilter2d_color(guide, x, self.r, self.eps)
        elif guide.shape[1] == 1:
            return guidedfilter2d_gray(guide, x, self.r, self.eps)
        else:
            raise NotImplementedError

def guidedfilter2d_color(guide, src, radius, eps, scale=None):
    """guided filter for a color guide image
    
    Parameters
    -----
    guide: (B, 3, H, W)-dim torch.Tensor
        guide image
    src: (B, C, H, W)-dim torch.Tensor
        filtering image
    radius: int
        filter radius
    eps: float
        regularization coefficient
    """
    assert guide.shape[1] == 3
    if src.ndim == 3:
        src = src[:, None]
    if scale is not None:
        guide_sub = guide.clone()
        src = F.interpolate(src, scale_factor=1./scale, mode="nearest")
        guide = F.interpolate(guide, scale_factor=1./scale, mode="nearest")
        radius = radius // scale

    guide_r, guide_g, guide_b = torch.chunk(guide, 3, 1) # b x 1 x H x W
    ones = torch.ones_like(guide_r)
    N = boxfilter2d(ones, radius)

    mean_I = boxfilter2d(guide, radius) / N # b x 3 x H x W
    mean_I_r, mean_I_g, mean_I_b = torch.chunk(mean_I, 3, 1) # b x 1 x H x W

    mean_p = boxfilter2d(src, radius) / N # b x C x H x W

    mean_Ip_r = boxfilter2d(guide_r * src, radius) / N # b x C x H x W
    mean_Ip_g = boxfilter2d(guide_g * src, radius) / N # b x C x H x W
    mean_Ip_b = boxfilter2d(guide_b * src, radius) / N # b x C x H x W

    cov_Ip_r = mean_Ip_r - mean_I_r * mean_p # b x C x H x W
    cov_Ip_g = mean_Ip_g - mean_I_g * mean_p # b x C x H x W
    cov_Ip_b = mean_Ip_b - mean_I_b * mean_p # b x C x H x W

    var_I_rr = boxfilter2d(guide_r * guide_r, radius) / N - mean_I_r * mean_I_r + eps # b x 1 x H x W
    var_I_rg = boxfilter2d(guide_r * guide_g, radius) / N - mean_I_r * mean_I_g # b x 1 x H x W
    var_I_rb = boxfilter2d(guide_r * guide_b, radius) / N - mean_I_r * mean_I_b # b x 1 x H x W
    var_I_gg = boxfilter2d(guide_g * guide_g, radius) / N - mean_I_g * mean_I_g + eps # b x 1 x H x W
    var_I_gb = boxfilter2d(guide_g * guide_b, radius) / N - mean_I_g * mean_I_b # b x 1 x H x W
    var_I_bb = boxfilter2d(guide_b * guide_b, radius) / N - mean_I_b * mean_I_b + eps # b x 1 x H x W

    # determinant
    cov_det = var_I_rr * var_I_gg * var_I_bb \
        + var_I_rg * var_I_gb * var_I_rb \
            + var_I_rb * var_I_rg * var_I_gb \
                - var_I_rb * var_I_gg * var_I_rb \
                    - var_I_rg * var_I_rg * var_I_bb \
                        - var_I_rr * var_I_gb * var_I_gb # b x 1 x H x W

    # inverse
    inv_var_I_rr = (var_I_gg * var_I_bb - var_I_gb * var_I_gb) / cov_det # b x 1 x H x W
    inv_var_I_rg = - (var_I_rg * var_I_bb - var_I_rb * var_I_gb) / cov_det # b x 1 x H x W
    inv_var_I_rb = (var_I_rg * var_I_gb - var_I_rb * var_I_gg) / cov_det # b x 1 x H x W
    inv_var_I_gg = (var_I_rr * var_I_bb - var_I_rb * var_I_rb) / cov_det # b x 1 x H x W
    inv_var_I_gb = - (var_I_rr * var_I_gb - var_I_rb * var_I_rg) / cov_det # b x 1 x H x W
    inv_var_I_bb = (var_I_rr * var_I_gg - var_I_rg * var_I_rg) / cov_det # b x 1 x H x W

    inv_sigma = torch.stack([
        torch.stack([inv_var_I_rr, inv_var_I_rg, inv_var_I_rb], 1),
        torch.stack([inv_var_I_rg, inv_var_I_gg, inv_var_I_gb], 1),
        torch.stack([inv_var_I_rb, inv_var_I_gb, inv_var_I_bb], 1)
    ], 1).squeeze(-3) # b x 3 x 3 x H x W

    cov_Ip = torch.stack([cov_Ip_r, cov_Ip_g, cov_Ip_b], 1) # b x 3 x C x H x W

    a = torch.einsum("bichw,bijhw->bjchw", (cov_Ip, inv_sigma))
    b = mean_p - a[:, 0] * mean_I_r - a[:, 1] * mean_I_g - a[:, 2] * mean_I_b # b x C x H x W

    mean_a = torch.stack([boxfilter2d(a[:, i], radius) / N for i in range(3)], 1)
    mean_b = boxfilter2d(b, radius) / N

    if scale is not None:
        guide = guide_sub
        mean_a = torch.stack([F.interpolate(mean_a[:, i], guide.shape[-2:], mode='bilinear') for i in range(3)], 1)
        mean_b = F.interpolate(mean_b, guide.shape[-2:], mode='bilinear')

    q = torch.einsum("bichw,bihw->bchw", (mean_a, guide)) + mean_b

    return q

def guidedfilter2d_gray(guide, src, radius, eps, scale=None):
    """guided filter for a gray scale guide image
    
    Parameters
    -----
    guide: (B, 1, H, W)-dim torch.Tensor
        guide image
    src: (B, C, H, W)-dim torch.Tensor
        filtering image
    radius: int
        filter radius
    eps: float
        regularization coefficient
    """
    if guide.ndim == 3:
        guide = guide[:, None]
    if src.ndim == 3:
        src = src[:, None]

    if scale is not None:
        guide_sub = guide.clone()
        src = F.interpolate(src, scale_factor=1./scale, mode="nearest")
        guide = F.interpolate(guide, scale_factor=1./scale, mode="nearest")
        radius = radius // scale

    ones = torch.ones_like(guide)
    N = boxfilter2d(ones, radius)

    mean_I = boxfilter2d(guide, radius) / N
    mean_p = boxfilter2d(src, radius) / N
    mean_Ip = boxfilter2d(guide*src, radius) / N
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = boxfilter2d(guide*guide, radius) / N
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = boxfilter2d(a, radius) / N
    mean_b = boxfilter2d(b, radius) / N

    if scale is not None:
        guide = guide_sub
        mean_a = F.interpolate(mean_a, guide.shape[-2:], mode='bilinear')
        mean_b = F.interpolate(mean_b, guide.shape[-2:], mode='bilinear')

    q = mean_a * guide + mean_b
    return q

def _diff_x(src, r):
    cum_src = src.cumsum(-2)

    left = cum_src[..., r:2*r + 1, :]
    middle = cum_src[..., 2*r + 1:, :] - cum_src[..., :-2*r - 1, :]
    right = cum_src[..., -1:, :] - cum_src[..., -2*r - 1:-r - 1, :]

    output = torch.cat([left, middle, right], -2)

    return output

def _diff_y(src, r):
    cum_src = src.cumsum(-1)

    left = cum_src[..., r:2*r + 1]
    middle = cum_src[..., 2*r + 1:] - cum_src[..., :-2*r - 1]
    right = cum_src[..., -1:] - cum_src[..., -2*r - 1:-r - 1]

    output = torch.cat([left, middle, right], -1)

    return output

def boxfilter2d(src, radius):
    return _diff_y(_diff_x(src, radius), radius)

## NewRes
class ResUnit(nn.Module):
    def __init__(self, channels):
        super(ResUnit, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu2(out)
        return out        

class UpRes(nn.Module):
    def __init__(self, out_ch=48):
        super(UpRes, self).__init__()
        self.updim1 = nn.Conv2d(3,8,3,1,1)
        self.resunit1 = ResUnit(8)
        self.updim2 = nn.Conv2d(8,32,3,1,1)
        self.resunit2 = ResUnit(32)
        self.updim3 = nn.Conv2d(32,out_ch,3,1,1)
        self.resunit3 = ResUnit(out_ch)

    def forward(self, x):
        x = self.updim1(x)
        x = self.resunit1(x)
        x = self.updim2(x)
        x = self.resunit2(x)
        x = self.updim3(x)
        x = self.resunit3(x)
        return x

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


if __name__=="__main__":
    # input_tensor = torch.rand((4,3,256,256))
    # print(input_tensor.shape)
    # model = ECE()
    
    # output_tensor = model(input_tensor)
    # print(output_tensor.shape)
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

    model=ECE()
    getModelSize(model)
    # model=Attention(dim,1,False)
    # getModelSize(model)
    # model=Attention(dim*8,8,False)
    # getModelSize(model)
    # model=TransformerBlock(dim=dim, num_heads=1, ffn_expansion_factor=2.66, bias=False,
    #                          LayerNorm_type='WithBias')
    # getModelSize(model)
    # model=TransformerBlock(dim=dim*8, num_heads=8, ffn_expansion_factor=2.66, bias=False,
    #                          LayerNorm_type='WithBias')
    # getModelSize(model)
    # model=DecoupleConv(dim, dim, wave_vector_threshold=2)
    # getModelSize(model)
    # model=DecoupleConv(dim*4, dim*4, wave_vector_threshold=4)
    # getModelSize(model)
    # model=DecoupleConv(dim*8, dim*8, wave_vector_threshold=4)
    # getModelSize(model)
    # model=Error_Predictor()
    # getModelSize(model)
    # model=Compensator_predicter()
    # getModelSize(model)
    # model=UpRes()
    # getModelSize(model)
    # model=ConvModule()
    # getModelSize(model)
    # model=Dehazer(kernel_size=15, top_candidates_ratio=0.0001,omega=0.95,radius=40,eps=1e-3,open_threshold=True,depth_est=True)
    # getModelSize(model)

# model=ECE()
# model.to('cuda')
# from torchsummary import summary
# summary(model, input_size=(3, 128, 128), batch_size=1)