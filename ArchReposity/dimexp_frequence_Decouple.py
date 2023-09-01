import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

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
                                    nn.Conv2d(inchannel,inchannel,3,1,1,groups=inchannel))
        self.hhandle = nn.Sequential(nn.Conv2d(inchannel,inchannel,1,1,0),
                                    nn.ReLU(),
                                    nn.Conv2d(inchannel,inchannel,3,1,1,groups=inchannel))
        self.comprehensive = nn.Sequential(nn.Sigmoid(),
                                            nn.Conv2d(2*inchannel,inchannel,1,1,0),
                                            nn.ReLU())
        
    def forward(self,x):
        low_signal = self.conv(x)
        high_signal = x - low_signal
        low_signal = self.lhandle(low_signal)
        high_signal = self.hhandle(high_signal)
        out = self.comprehensive(torch.cat([low_signal,high_signal],dim=1))
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

class DecoupleConvv2(nn.Module):
    def __init__(self, in_ch, out_ch, inner_ch , wave_vector_threshold=2, aggregate=True):
        super(DecoupleConvv2, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.inner_ch = inner_ch
        if aggregate:
            self.inconv = nn.Conv2d(in_ch, 2*inner_ch, 3, 1, 1)
            self.outconv = nn.Conv2d(inner_ch*2, out_ch, 3, 1, 1)
        else:
            self.inconv = nn.Conv2d(in_ch, 2*inner_ch, 1, 1, 0)
            self.outconv = nn.Conv2d(inner_ch*2, out_ch, 1, 1, 0)
        self.ltdc = LowThresholdDC_test_3(inner_ch, wave_vector_threshold)
        self.htdc = HighThresholdDC(inner_ch)
    def forward(self, x):
        x = self.inconv(x)
        x = F.relu(x)
        x1, x2 = torch.split(x, self.inner_ch, dim=1)
        x1 = self.ltdc(x1)
        x2 = self.htdc(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.outconv(x)
        return x



if __name__=="__main__":
    img_path = "D:/Code/Dehaze1/fogg.png"
    img = cv2.imread(img_path)
    cv2.imshow("img",img)
    cv2.waitKey(0)
    img_batch = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float()
    decoupleconv = DecoupleConvv2(3,3,9, aggregate=False)
    output_batch = decoupleconv(img_batch)
    catted_img_batch = torch.cat([img_batch,output_batch],dim=3)
    catted_img = catted_img_batch.detach().squeeze(0).permute(1,2,0).numpy().astype(np.uint8)
    cv2.imshow("catted_img",catted_img)
    cv2.waitKey(0)
    pass
