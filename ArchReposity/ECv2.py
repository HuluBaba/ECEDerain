import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

'''
ECv2: 主要目标是改变ECv1中的专家提取结构，次要目标是改变瓶颈结构对信息传递能力的损失、去掉大量的Norm层、改变SEBlock的固定中间通道数。
'''



# 5 experts are dialated(pointwise+depthwise) 3(2),5(2),3(3), avgpool(3), maxpool(3)
NUM_OPS = 5

class Experts_Layer(torch.nn.Module):
    def __init__(self, innerch=64):
        super(Experts_Layer, self).__init__()
        self.weight_gen = SESideBranch(innerch,innerch//4,NUM_OPS)
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
        weight = self.weight_gen(x)
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
    def __init__(self, num_layers=3):
        super(Expert_Extraction, self).__init__()
        self.preconv = nn.Conv2d(3, 64, 3, 1, 1)
        self.prerelu = nn.ReLU()
        self.experts = torch.nn.ModuleList()
        for i in range(num_layers):
            self.experts.append(Experts_Layer())

    def forward(self, x):
        x = self.preconv(x)
        x = self.prerelu(x)
        for i in range(len(self.experts)):
            res = x
            x = self.experts[i](x)
            x = x + res
            x = F.relu(x)
        return x

class SESideBranch(nn.Module):
    def __init__(self, input_dim, reduction, output_dim):
        super(SESideBranch, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, reduction),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(reduction, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y)
        return y

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
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=True)
        
    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        return torch.cat([x, out], 1)

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.GroupNorm(num_groups=4, num_channels=out_planes)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.se = SEBlock(out_planes, in_planes//4)
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.se(out)
        return out

class ConvModule(nn.Module):
    def __init__(self, basic_dim = 16):
        super(ConvModule, self).__init__()
        self.conv1 = nn.Conv2d(6,basic_dim,3,1,1)
        self.dense_block1=BottleneckBlock(basic_dim,basic_dim)
        self.trans_block1=TransitionBlock(2*basic_dim,basic_dim)
        self.downsample1 = Downsample(basic_dim)

        ############# Block2-down 32-32  ##############
        self.dense_block2=BottleneckBlock(2*basic_dim,2*basic_dim)
        self.trans_block2=TransitionBlock(4*basic_dim,2*basic_dim)
        self.downsample2 = Downsample(2*basic_dim)

        ############# Block3-down  16-16 ##############
        self.dense_block3=BottleneckBlock(4*basic_dim,4*basic_dim)
        self.trans_block3=TransitionBlock(8*basic_dim,4*basic_dim)
        self.downsample3 = Downsample(4*basic_dim)

        ############# Block4-up  8-8  ##############
        self.dense_block4=BottleneckBlock(8*basic_dim,8*basic_dim)
        self.trans_block4=TransitionBlock(16*basic_dim,8*basic_dim)
        self.upsample4 = Upsample(8*basic_dim)

        ############# Block5-up  16-16 ##############
        self.dense_block5=BottleneckBlock(8*basic_dim,8*basic_dim)
        self.trans_block5=TransitionBlock(16*basic_dim,4*basic_dim)
        self.upsample5 = Upsample(4*basic_dim)

        self.dense_block6=BottleneckBlock(4*basic_dim,4*basic_dim)
        self.trans_block6=TransitionBlock(8*basic_dim,2*basic_dim)
        self.upsample6 = Upsample(2*basic_dim)

        self.dense_block7=BottleneckBlock(2*basic_dim,2*basic_dim)
        self.trans_block7=TransitionBlock(4*basic_dim,2*basic_dim)
        self.dense_block8=BottleneckBlock(2*basic_dim,2*basic_dim)
        self.trans_block8=TransitionBlock(4*basic_dim,2*basic_dim)
        self.dense_block9=BottleneckBlock(2*basic_dim,2*basic_dim)
        self.trans_block9=TransitionBlock(4*basic_dim,2*basic_dim)
        self.dense_block10=BottleneckBlock(2*basic_dim,2*basic_dim)
        self.trans_block10=TransitionBlock(4*basic_dim,basic_dim)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.refine3 = nn.Conv2d(basic_dim, 3, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x1=self.relu(self.conv1(x))
        x1=self.dense_block1(x1)
        x1=self.trans_block1(x1)
        x_1=self.downsample1(x1)
        ###  32x32
        x2=(self.dense_block2(x_1))
        x2=self.trans_block2(x2)
        x_2=self.downsample2(x2)
        ### 16 X 16
        x3=(self.dense_block3(x_2))
        x3=self.trans_block3(x3)
        x_3=self.downsample3(x3)
        ## Classifier  ##
        
        x4=(self.dense_block4(x_3))
        x4=self.trans_block4(x4)
        x_4=self.upsample4(x4)
        x_4=torch.cat([x_4,x3],1)

        x5=(self.dense_block5(x_4))
        x5=self.trans_block5(x5)
        x_5=self.upsample5(x5)
        x_5=torch.cat([x_5,x2],1)

        x6=(self.dense_block6(x_5))
        x6=(self.trans_block6(x6))
        x_6=self.upsample6(x6)
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
        
class Confidence_Predictor(torch.nn.Module):
    def __init__(self,innerch=64):
        super(Confidence_Predictor, self).__init__()
        self.extractor = Expert_Extraction(num_layers=3)
        self.postprocess = nn.Sequential(nn.Conv2d(innerch, innerch, 3, 1, 1), nn.ReLU(), nn.Conv2d(innerch, 3, 3, 1, 1), nn.Sigmoid())
    def forward(self, pred_b):
        x = self.extractor(pred_b)
        x = self.postprocess(x)
        return x

class Error_Compensator(nn.Module):
    def __init__(self, innerch=64):
        super(Error_Compensator, self).__init__()
        self.error_predictor = Error_Predictor(innerch)
        self.confidence_predictor = Confidence_Predictor(innerch)

    def forward(self, pred_b, o):
        pred_err = self.error_predictor(pred_b, o)
        pred_conf = self.confidence_predictor(pred_b)
        compensated_pred_b = pred_b - pred_err * pred_conf
        return compensated_pred_b

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



if __name__ == '__main__':
    lq = torch.randn(1,3,128,128).to('cuda')
    pred_hq = torch.randn(1,3,128,128).to('cuda')
    error_compensator = Error_Compensator()
    error_compensator.to('cuda')
    compensated_pred_hq = error_compensator(pred_hq, lq)
    print(compensated_pred_hq.shape)
    summary(error_compensator,[(3,128,128),(3,128,128)],batch_size=1,device='cuda')

