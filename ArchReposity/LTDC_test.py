from math import ceil
import torch
from torch import nn
from torch.nn import functional as F
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

class LowThresholdDC_test_0(nn.Module):
    '''
    f-space decouple with patch nums by pool
    recombine with weight parameters in channel
    '''
    def __init__(self, inchannel, patch_size=2):
        super(LowThresholdDC_test_0, self).__init__()
        self.patch_size = patch_size
        self.channel = inchannel
        self.h = nn.Parameter(torch.zeros(self.channel))
        self.l = nn.Parameter(torch.zeros(self.channel))
    
    def forward(self,x):
        _1, _2, h, w = x.shape
        com_patch_size = (ceil(h/self.patch_size),ceil(w/self.patch_size))
        x_pool = F.avg_pool2d(x,com_patch_size,com_patch_size,ceil_mode=True)
        x_pool_flatten = x_pool.repeat_interleave(self.patch_size,dim=3).repeat_interleave(self.patch_size,dim=2)
        low_signal = x_pool_flatten[:,:,0:h,0:w]
        high = (x-low_signal) * self.h[None, :, None, None]
        out = high + low_signal * self.l[None, :, None, None]
        return out

class LowThresholdDC_test_1(nn.Module):
    '''
     f-space decouple with step size by pool
     recombine with weight parameter in channel
    '''
    def __init__(self, inchannel, patch_size=2):
        super(LowThresholdDC_test_1,self).__init__()
        self.patch_size = patch_size
        self.channel = inchannel
        self.h = nn.Parameter(torch.zeros(self.channel))
        self.l = nn.Parameter(torch.ones(self.channel))
        self.pool = nn.AvgPool2d(patch_size,patch_size,ceil_mode=True)

    def forward(self,x):
        _1, _2, h, w = x.shape
        x_pool = self.pool(x)
        x_pool_flatten = x_pool.repeat_interleave(2,dim=3).repeat_interleave(2,dim=2)
        low_signal = x_pool_flatten[:,:,0:h,0:w]
        high_signal = x-low_signal
        out = low_signal*self.l[None, :, None, None] + high_signal*self.h[None, :, None, None]
        return out

class LowThresholdDC_test_2(nn.Module):
    '''
     f-space decouple with step size by conv
     recombine with weight parameter in channel
    '''
    def __init__(self, inchannel, patch_size=2):
        super(LowThresholdDC_test_2,self).__init__()
        self.channel=inchannel
        self.conv = nn.Conv2d(inchannel,inchannel,patch_size,1,'same',groups=inchannel)
        self.l = nn.Parameter(torch.ones(self.channel))
        self.h = nn.Parameter(torch.zeros(self.channel))


    def forward(self,x):
        low_signal = self.conv(x)
        high_signal = x - low_signal
        out = low_signal*self.l[None, :, None, None] + high_signal*self.h[None, :, None, None]
        return out
        
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
        
    def forward(self,x):
        low_signal = self.conv(x)
        high_signal = x - low_signal
        low_signal = self.lhandle(low_signal)
        high_signal = self.hhandle(high_signal)
        out = self.comprehensive(torch.cat([low_signal,high_signal],dim=1))
        return out

        


if __name__=="__main__":
    input_tensor=torch.rand(1,3,3,4)
    # ltdc_ori = LowThresholdDC(3,2)
    ltdc_test0 = LowThresholdDC_test_0(3,2)
    ltdc_test1 = LowThresholdDC_test_1(3,2)
    ltdc_test2 = LowThresholdDC_test_2(3,2)
    ltdc_test3 = LowThresholdDC_test_3(3,2)
    # output_ori = ltdc_ori(input_tensor)
    # print(output_ori.shape)
    output_test0 = ltdc_test0(input_tensor)
    print(output_test0.shape)
    output_test1 = ltdc_test1(input_tensor)
    print(output_test1.shape)
    output_test2 = ltdc_test2(input_tensor)
    print(output_test2.shape)
    output_test3 = ltdc_test3(input_tensor)
    print(output_test3.shape)
