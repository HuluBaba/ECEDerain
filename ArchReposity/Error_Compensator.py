import torch
from torch import nn
import torch.nn.functional as F
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
        y = torch.sum(x*weights, dim=1)                         # y: [batch, innerch, h, w]
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

class Error_compensator(torch.nn.Module):
    def __init__(self, innerch=64):
        super(Error_compensator, self).__init__()
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
        self.postprocess = nn.Sequential(nn.Conv2d(innerch, innerch, 3, 1, 1), nn.Sigmoid())
    def forward(self, pred_b):
        x = self.extractor(pred_b)
        x = self.postprocess(x)
        return x

if __name__=="__main__":

    model = Error_compensator()
    model.cuda()
    x = torch.randn(1,3,8,8).cuda()
    o = torch.randn(1,3,8,8).cuda()
    y = model(x,o)
    print(y.shape)