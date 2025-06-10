import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from . import model_utils

class FeatExtractor(nn.Module):
    def __init__(self, batchNorm=False, c_in=3, other={}):
        super(FeatExtractor, self).__init__()
        self.other = other
        self.conv1 = model_utils.conv(batchNorm, c_in, 64,  k=3, stride=1, pad=1)
        self.conv2 = model_utils.conv(batchNorm, 64,   128, k=3, stride=2, pad=1)
        self.conv3 = model_utils.conv(batchNorm, 128,  128, k=3, stride=1, pad=1)
        self.conv4 = model_utils.conv(batchNorm, 128,  256, k=3, stride=2, pad=1)
        self.conv5 = model_utils.conv(batchNorm, 256,  256, k=3, stride=1, pad=1)
        self.conv6 = model_utils.deconv(256, 128)
        self.conv7 = model_utils.conv(batchNorm, 128, 128, k=3, stride=1, pad=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        n, c, h, w = out.data.shape
        out   = out.view(n,-1).max(dim=0)[0]
        return out, [1, c, h, w]

class Regressor(nn.Module):
    def __init__(self, batchNorm=False, other={}): 
        super(Regressor, self).__init__()
        self.other   = other
        self.deconv1 = model_utils.conv(batchNorm, 128, 128,  k=3, stride=1, pad=1)
        self.deconv2 = model_utils.conv(batchNorm, 128, 128,  k=3, stride=1, pad=1)
        self.deconv3 = model_utils.deconv(128, 64)
        self.est_normal= self._make_output(64, 3, k=3, stride=1, pad=1)
        self.other   = other

    def _make_output(self, cin, cout, k=3, stride=1, pad=1):
        return nn.Sequential(
               nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False))

    def forward(self, x, shape):
        x      = x.view(shape[0], shape[1], shape[2], shape[3])
        out    = self.deconv1(x)
        out    = self.deconv2(out)
        out    = self.deconv3(out)
        normal = self.est_normal(out)
        normal = torch.nn.functional.normalize(normal, 2, 1)
        return normal

class PS_FCN(nn.Module):
    def __init__(self, fuse_type='max', batchNorm=False, c_in=3, other={}):
        super(PS_FCN, self).__init__()
        self.extractor = FeatExtractor(batchNorm, c_in, other)
        self.regressor = Regressor(batchNorm, other)
        self.c_in      = c_in
        self.fuse_type = fuse_type
        self.other = other

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        img   = x[0]
        img_split = torch.split(img, 3, 1)
        if len(x) > 1: # Have lighting
            light = x[1]
            light_split = torch.split(light, 3, 1)

        feats = torch.Tensor()
        
        batch = 12
        x1 = len(img_split)//batch
        y1 = len(img_split)%batch
        
        for j in range(x1):
            nets_in=[]
            for k in range(j*batch,(j+1)*batch):
                net_in = img_split[k] if len(x) == 1 else torch.cat([img_split[k], light_split[k]], 1)
                nets_in.append(net_in)
            net_in = torch.cat(nets_in,0)  #torch.Size([batchsize, 6, 256, 256])
            
            feat, shape = self.extractor(net_in)   #torch.Size([batchsize,-1])
            
            if j==0:
                feats = feat

            else:
                feats= torch.stack([feats, feat], 1).max(1)[0]
     
                del feat


        if y1!=0:
            nets_in = []
            for k in range(x1*batch,len(img_split)):
                net_in = img_split[k] if len(x) == 1 else torch.cat([img_split[k], light_split[k]], 1)
                nets_in.append(net_in)
            net_in = torch.cat(nets_in,0)
            feat, shape= self.extractor(net_in) #torch.Size([batchsize,-1])

            if x1!=0:
                feats= torch.stack([feats, feat], 1).max(1)[0]

            else:
                feats = feat

        
        normal = self.regressor(feats, shape)
        return normal
