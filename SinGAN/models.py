import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock,self).__init__()
        self.add_module('conv',nn.Conv2d(in_channel ,out_channel,kernel_size=ker_size,stride=stride,padding=padd)),
        self.add_module('norm',nn.BatchNorm2d(out_channel)),
        self.add_module('LeakyRelu',nn.LeakyReLU(0.2, inplace=True))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)



class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        return x * y


class ConvBlockWithSE(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, reduction=16):
        super(ConvBlockWithSE, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(out_channels, reduction)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        out = self.se(out)
        return out

class ConvBlockWithSE_atten(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, reduction=16,spatial_attention=True):
        super(ConvBlockWithSE_atten, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(out_channels, reduction)
        self.spatial_attention = spatial_attention
        
        if self.spatial_attention:
            self.s_attention = SpatialGate()  

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        out = self.se(out)
        out = self.s_attention(out)

        return out

class WDiscriminator(nn.Module):
    def __init__(self, opt):
        super(WDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_im,N,opt.ker_size,opt.padd_size,1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2,(i+1)))
            block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1)
            self.body.add_module('block%d'%(i+1),block)
        self.tail = nn.Conv2d(max(N,opt.min_nfc),1,kernel_size=opt.ker_size,stride=1,padding=opt.padd_size)

    def forward(self,x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x


class GeneratorConcatSkip2CleanAdd(nn.Module):
    def __init__(self, opt):
        super(GeneratorConcatSkip2CleanAdd, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlockWithSE(opt.nc_im, N, opt.ker_size, opt.padd_size, 1)  # Head에 ConvBlockWithSE 적용
        self.body = nn.Sequential()

        N = int(opt.nfc / pow(2, (1)))
        block = ConvBlockWithSE(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
        self.body.add_module('block%d' % (1), block)

        N = int(opt.nfc / pow(2, (2)))
        block = ConvBlockWithSE_atten(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
        self.body.add_module('block%d' % (2), block)               

        N = int(opt.nfc / pow(2, (3)))
        block = ConvBlockWithSE(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, 1)
        self.body.add_module('block%d' % (3), block)

        self.tail = nn.Sequential(
            nn.Conv2d(max(N, opt.min_nfc), opt.nc_im, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size),
            nn.Tanh()
        )


    def forward(self,x,y):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        ind = int((y.shape[2]-x.shape[2])/2)
        y = y[:,:,ind:(y.shape[2]-ind),ind:(y.shape[3]-ind)]
        return x+y



class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
            )
    
    def forward(self,x):
        x = x.to(torch.float32)
        x_avg_pool = torch.mean(x,1).unsqueeze(1)
        x_max_pool = torch.max(x,1)[0].unsqueeze(1)
        attention = torch.cat((x_avg_pool,x_max_pool),dim=1)
        attention = self.conv(attention)
        return x * attention
