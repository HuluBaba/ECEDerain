import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class LowThresholdDC(nn.Module):
    def __init__(self, inchannel, patch_size=2):
        super(LowThresholdDC, self).__init__()

        self.ap = nn.AdaptiveAvgPool2d((1,1))

        self.patch_size = patch_size
        self.channel = inchannel * patch_size**2
        self.h = nn.Parameter(torch.zeros(self.channel))
        self.l = nn.Parameter(torch.zeros(self.channel))

    def forward(self, x):

        patch_x = rearrange(x, 'b c (p1 w1) (p2 w2) -> b c p1 w1 p2 w2', p1=self.patch_size, p2=self.patch_size)
        patch_x = rearrange(patch_x, ' b c p1 w1 p2 w2 -> b (c p1 p2) w1 w2', p1=self.patch_size, p2=self.patch_size)

        low = self.ap(patch_x)
        high = (patch_x - low) * self.h[None, :, None, None]
        out = high + low * self.l[None, :, None, None]
        out = rearrange(out, 'b (c p1 p2) w1 w2 -> b c (p1 w1) (p2 w2)', p1=self.patch_size, p2=self.patch_size)

        return out

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
    def __init__(self, in_ch, out_ch, wave_vector_threshold=2):
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


if __name__=="__main__":
    img_path = "D:/Code/Dehaze1/fogg.png"
    img = cv2.imread(img_path)
    cv2.imshow("img",img)
    cv2.waitKey(0)
    img_batch = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float()
    decoupleconv = DecoupleConv(3,3)
    output_batch = decoupleconv(img_batch)
    catted_img_batch = torch.cat([img_batch,output_batch],dim=3)
    catted_img = catted_img_batch.detach().squeeze(0).permute(1,2,0).numpy().astype(np.uint8)
    cv2.imshow("catted_img",catted_img)
    cv2.waitKey(0)
    pass
