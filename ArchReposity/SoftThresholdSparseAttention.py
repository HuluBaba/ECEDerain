import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

class STSAttnGen(nn.Module):
    def __init__(self, in_ch):
        super(STSAttnGen, self).__init__()
        self.in_ch = in_ch
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.ffn = nn.Sequential(nn.Linear(in_ch,in_ch),
                                 nn.BatchNorm1d(in_ch),
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

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
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

        self.thresholdweight = nn.Parameter(0.2*torch.ones(4, 1, 1))

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

if __name__ == '__main__':
    x = torch.randn(2, 8, 12, 12)
    attn = Attention(8, 4, True)
    y = attn(x)
    print(y.shape)
    # layer = nn.BatchNorm1d(3)
    # input_tensor = torch.tensor([[1,1,1],[2,2,3]], dtype=torch.float32)
    # output = layer(input_tensor)
    # print(output)
