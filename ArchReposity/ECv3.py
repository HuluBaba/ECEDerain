import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import numbers

'''
Expert extraction module two kinds of expert layer.
 A:5 experts[dialated 3(2),dialated 5(2),dialated 5(3), maxpool, avgpool] 
 B:5 experts[separable 3(2),separable 5(2),separable 5(3), maxpool, avgpool]] 

 改变Error Predictor不对称结构
 改变SEBlock的固定中间通道数
 Expert层的提取层数量改为3A+3B
增加少数用于Refine的Transformer层
考察ConvModule的结构，考虑直接去掉
 即时生成Expert权重
'''



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
    def __init__(self, num_layers=3):
        super(Expert_Extraction, self).__init__()
        self.preconv = nn.Conv2d(3, 64, 3, 1, 1)
        self.prerelu = nn.ReLU()
        self.experts = torch.nn.ModuleList()
        for i in range(num_layers):
            self.experts.append(Experts_Layer_B())
            self.experts.append(Experts_Layer_A())

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



class Error_esitimator(torch.nn.Module):
    def __init__(self, innerch=64):
        super(Error_esitimator, self).__init__()
        self.error_detector = Expert_Extraction(num_layers=3)   #3->64
        self.convblock2 = nn.Sequential(nn.Conv2d(innerch, innerch, 3, 1, 1), nn.ReLU(), nn.Conv2d(innerch,3,1,1), nn.Sigmoid())    #64->3
        self.summaryblock = ConvModule()                        #6->3
        self.extractor = Expert_Extraction(num_layers=3)        #3->64
    def forward(self, pred_b, o):
        x = self.summaryblock(torch.cat([o,pred_b],dim=1))      #3
        x = self.error_detector(x)                              #64
        pred_err = self.convblock2(x)                                  #3
        return pred_err
        
class Confidence_estimator(torch.nn.Module):
    def __init__(self,innerch=64):
        super(Confidence_estimator, self).__init__()
        self.extractor = Expert_Extraction(num_layers=3)
        self.postprocess = nn.Sequential(nn.Conv2d(innerch, innerch, 3, 1, 1), nn.ReLU(), nn.Conv2d(innerch, 3, 3, 1, 1), nn.Sigmoid())
    def forward(self, pred_b):
        x = self.extractor(pred_b)
        x = self.postprocess(x)
        return x

class Error_Compensator(nn.Module):
    def __init__(self, innerch=64):
        super(Error_Compensator, self).__init__()
        self.error_predictor = Error_esitimator(innerch)
        self.confidence_predictor = Confidence_estimator(innerch)

    def forward(self, pred_b, o):
        pred_err = self.error_predictor(pred_b, o)
        pred_conf = self.confidence_predictor(pred_b)
        compensated_pred_b = pred_b - pred_err * pred_conf
        return compensated_pred_b

if __name__=="__main__":

    model = Error_Compensator()
    model.cuda()
    x = torch.randn(1,3,8,8).cuda()
    o = torch.randn(1,3,8,8).cuda()
    y = model(x,o)
    print(y.shape)