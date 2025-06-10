import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from relation import model_utils
from torch.nn.init import kaiming_normal_




class FeatExtractor1(nn.Module):
    def __init__(self, batchNorm=False, c_in=3, other={}):
        super(FeatExtractor1, self).__init__()
        self.other = other
        self.conv1 = model_utils.conv(batchNorm, c_in, 16, k=3, stride=1, pad=1)
        self.conv2 = model_utils.conv(batchNorm, 16, 16, k=3, stride=1, pad=1)
        self.conv3 = model_utils.conv(batchNorm, 16, 16, k=3, stride=1, pad=1)
        self.conv4 = model_utils.conv(batchNorm, 16, 16, k=3, stride=1, pad=1)
        self.conv5 = model_utils.conv(batchNorm, 16, 16, k=3, stride=1, pad=1)
        self.conv6 = model_utils.conv(batchNorm, 16, 16, k=3, stride=1, pad=1)
        self.conv7 = model_utils.conv(batchNorm, 16, 32, k=3, stride=2, pad=1)  # from c2
        self.conv8 = model_utils.conv(batchNorm, 32, 32, k=3, stride=1, pad=1)
        self.conv9 = model_utils.conv(batchNorm, 32, 32, k=3, stride=1, pad=1)
        self.conv10 = model_utils.conv(batchNorm, 32, 32, k=3, stride=1, pad=1)
        self.conv11 = model_utils.conv(batchNorm, 32, 64, k=3, stride=2, pad=1)  # from c8
        self.conv12 = model_utils.conv(batchNorm, 64, 64, k=3, stride=1, pad=1)
        self.conv13 = model_utils.conv(batchNorm, 64, 64, k=3, stride=1, pad=1)
        self.conv3down1 = model_utils.conv(batchNorm, 16, 32, k=3, stride=2, pad=1)  # to c8
        self.conv4down1 = model_utils.conv(batchNorm, 16, 32, k=3, stride=2, pad=1)  # to c9
        self.conv4down2 = model_utils.conv(batchNorm, 16, 64, k=3, stride=4, pad=1)  # to c11
        self.conv5down1 = model_utils.conv(batchNorm, 16, 32, k=3, stride=2, pad=1)  # to c10
        self.conv5down2 = model_utils.conv(batchNorm, 16, 64, k=3, stride=4, pad=1)  # to c12
        self.conv10down1 = model_utils.conv(batchNorm, 32, 64, k=3, stride=2, pad=1)  # to c13
        self.convnor8 = model_utils.conv(batchNorm, 32, 16, k=1, stride=1, pad=0)
        self.convnor122 = model_utils.conv(batchNorm, 64, 32, k=1, stride=1, pad=0)
        self.convnor124 = model_utils.conv(batchNorm, 64,16, k=1, stride=1, pad=0)
        self.convnor10 = model_utils.conv(batchNorm, 32, 16, k=1, stride=1, pad=0)

    def forward(self, x):
        out1 = self.conv1(x) #torch.Size([1, 64, 128, 128])
        out2 = self.conv2(out1) #torch.Size([1, 64, 128, 128])
        out3 = self.conv3(out2) #torch.Size([1, 64, 128, 128])
        out3d = self.conv3down1(out3) #torch.Size([1, 128, 64, 64])
        out7 = self.conv7(out2) #torch.Size([1, 128, 64, 64])
        out8 = self.conv8(out7) #torch.Size([1, 128, 64, 64])
        out8up = torch.nn.functional.upsample(out8, scale_factor=2, mode='bilinear', align_corners=True) #torch.Size([1, 128, 128, 128])
        out8upnor = self.convnor8(out8up) #torch.Size([1, 64, 128, 128])
        out8 = torch.add(out8, out3d) #torch.Size([1, 128, 64, 64])
        out4 = self.conv4(out3) #torch.Size([1, 64, 128, 128])
        out4 = torch.add(out4, out8upnor) #torch.Size([1, 64, 128, 128])
        out4d1 = self.conv4down1(out4) #torch.Size([1, 128, 64, 64])
        out4d2 = self.conv4down2(out4) #torch.Size([1, 256, 32, 32])
        out9 = self.conv9(out8) #torch.Size([1, 128, 64, 64])
        out9 = torch.add(out9, out4d1) #torch.Size([1, 128, 64, 64])
        out5 = self.conv5(out4) #torch.Size([1, 64, 128, 128])
        out5d1 = self.conv5down1(out5) #torch.Size([1, 128, 64, 64])
        out5d2 = self.conv5down2(out5) #torch.Size([1, 256, 32, 32])
        out11 = self.conv11(out8) #torch.Size([1, 256, 32, 32])
        out11 = torch.add(out11, out4d2) #torch.Size([1, 256, 32, 32])
        out12 = self.conv12(out11) #torch.Size([1, 256, 32, 32])
        out12 = torch.add(out12, out5d2) #torch.Size([1, 256, 32, 32])
        out12up2 = torch.nn.functional.upsample(out12, scale_factor=2, mode='bilinear', align_corners=True) #torch.Size([1, 256, 64, 64])
        out12up2nor = self.convnor122(out12up2) #torch.Size([1, 128, 64, 64])
        out12up4 = torch.nn.functional.upsample(out12, scale_factor=4, mode='bilinear', align_corners=True) #torch.Size([1, 256, 128, 128])
        out12up4nor = self.convnor124(out12up4) #torch.Size([1, 64, 128, 128])
        out10 = self.conv10(out9) #torch.Size([1, 128, 64, 64])
        out10 = torch.add(out10, out5d1) #torch.Size([1, 128, 64, 64])
        out10 = torch.add(out10, out12up2nor) #torch.Size([1, 128, 64, 64])
        out10d = self.conv10down1(out10) #torch.Size([1, 256, 32, 32])
        out10up = torch.nn.functional.upsample(out10, scale_factor=2, mode='bilinear', align_corners=True) #torch.Size([1, 128, 128, 128])
        out10upnor = self.convnor10(out10up) #torch.Size([1, 64, 128, 128])
        out13 = self.conv13(out12) #torch.Size([1, 256, 32, 32])
        out13 = torch.add(out13, out10d) #torch.Size([1, 256, 32, 32])
        out6 = self.conv6(out5) #torch.Size([1, 64, 128, 128])
        out6 = torch.add(out6, out12up4nor) #torch.Size([1, 64, 128, 128])
        out6 = torch.add(out6, out10upnor) #torch.Size([1, 64, 128, 128])
        out6m = out6 #torch.Size([1, 16, 128, 128])
        out10m = out10 #torch.Size([1, 32, 64, 64])
        out13m = out13 #torch.Size([1, 64, 32, 32])
        # n6, c6, h6, w6 = out6m.data.shape
        # out6m = out6m.view(-1)
        # n10, c10, h10, w10 = out10m.data.shape
        # out10m = out10m.view(-1)
        # n13, c13, h13, w13 = out13m.data.shape
        # out13m = out13m.view(-1)
        return out6m, out10m, out13m
        # return out6m, [n6, c6, h6, w6], out10m, [n10, c10, h10, w10], out13m, [n13, c13, h13, w13]
    


class FeatExtractor2(nn.Module):
    def __init__(self, batchNorm=False, c_in=3, other={}):
        super(FeatExtractor2, self).__init__()
        self.other = other
        self.conv1 = model_utils.conv(batchNorm, c_in, 16, k=3, stride=1, pad=1)
        self.conv2 = model_utils.conv(batchNorm, 16, 16, k=3, stride=1, pad=1)
        self.conv3 = model_utils.conv(batchNorm, 16, 16, k=3, stride=1, pad=1)
        self.conv4 = model_utils.conv(batchNorm, 16, 16, k=3, stride=1, pad=1)
        self.conv5 = model_utils.conv(batchNorm, 16, 16, k=3, stride=1, pad=1)
        self.conv6 = model_utils.conv(batchNorm, 16, 16, k=3, stride=1, pad=1)
        self.conv7 = model_utils.conv(batchNorm, 16, 32, k=3, stride=2, pad=1)  # from c2
        self.conv8 = model_utils.conv(batchNorm, 32, 32, k=3, stride=1, pad=1)
        self.conv9 = model_utils.conv(batchNorm, 32, 32, k=3, stride=1, pad=1)
        self.conv10 = model_utils.conv(batchNorm, 32, 32, k=3, stride=1, pad=1)
        self.conv11 = model_utils.conv(batchNorm, 32, 64, k=3, stride=2, pad=1)  # from c8
        self.conv12 = model_utils.conv(batchNorm, 64, 64, k=3, stride=1, pad=1)
        self.conv13 = model_utils.conv(batchNorm, 64, 64, k=3, stride=1, pad=1)
        self.conv3down1 = model_utils.conv(batchNorm, 16, 32, k=3, stride=2, pad=1)  # to c8
        self.conv4down1 = model_utils.conv(batchNorm, 16, 32, k=3, stride=2, pad=1)  # to c9
        self.conv4down2 = model_utils.conv(batchNorm, 16, 64, k=3, stride=4, pad=1)  # to c11
        self.conv5down1 = model_utils.conv(batchNorm, 16, 32, k=3, stride=2, pad=1)  # to c10
        self.conv5down2 = model_utils.conv(batchNorm, 16, 64, k=3, stride=4, pad=1)  # to c12
        self.conv10down1 = model_utils.conv(batchNorm, 32, 64, k=3, stride=2, pad=1)  # to c13
        self.convnor8 = model_utils.conv(batchNorm, 32, 16, k=1, stride=1, pad=0)
        self.convnor122 = model_utils.conv(batchNorm, 64, 32, k=1, stride=1, pad=0)
        self.convnor124 = model_utils.conv(batchNorm, 64,16, k=1, stride=1, pad=0)
        self.convnor10 = model_utils.conv(batchNorm, 32, 16, k=1, stride=1, pad=0)

    def forward(self, x):
        out1 = self.conv1(x) #torch.Size([1, 64, 128, 128])
        out2 = self.conv2(out1) #torch.Size([1, 64, 128, 128])
        out3 = self.conv3(out2) #torch.Size([1, 64, 128, 128])
        out3d = self.conv3down1(out3) #torch.Size([1, 128, 64, 64])
        out7 = self.conv7(out2) #torch.Size([1, 128, 64, 64])
        out8 = self.conv8(out7) #torch.Size([1, 128, 64, 64])
        out8up = torch.nn.functional.upsample(out8, scale_factor=2, mode='bilinear', align_corners=True) #torch.Size([1, 128, 128, 128])
        out8upnor = self.convnor8(out8up) #torch.Size([1, 64, 128, 128])
        out8 = torch.add(out8, out3d) #torch.Size([1, 128, 64, 64])
        out4 = self.conv4(out3) #torch.Size([1, 64, 128, 128])
        out4 = torch.add(out4, out8upnor) #torch.Size([1, 64, 128, 128])
        out4d1 = self.conv4down1(out4) #torch.Size([1, 128, 64, 64])
        out4d2 = self.conv4down2(out4) #torch.Size([1, 256, 32, 32])
        out9 = self.conv9(out8) #torch.Size([1, 128, 64, 64])
        out9 = torch.add(out9, out4d1) #torch.Size([1, 128, 64, 64])
        out5 = self.conv5(out4) #torch.Size([1, 64, 128, 128])
        out5d1 = self.conv5down1(out5) #torch.Size([1, 128, 64, 64])
        out5d2 = self.conv5down2(out5) #torch.Size([1, 256, 32, 32])
        out11 = self.conv11(out8) #torch.Size([1, 256, 32, 32])
        out11 = torch.add(out11, out4d2) #torch.Size([1, 256, 32, 32])
        out12 = self.conv12(out11) #torch.Size([1, 256, 32, 32])
        out12 = torch.add(out12, out5d2) #torch.Size([1, 256, 32, 32])
        out12up2 = torch.nn.functional.upsample(out12, scale_factor=2, mode='bilinear', align_corners=True) #torch.Size([1, 256, 64, 64])
        out12up2nor = self.convnor122(out12up2) #torch.Size([1, 128, 64, 64])
        out12up4 = torch.nn.functional.upsample(out12, scale_factor=4, mode='bilinear', align_corners=True) #torch.Size([1, 256, 128, 128])
        out12up4nor = self.convnor124(out12up4) #torch.Size([1, 64, 128, 128])
        out10 = self.conv10(out9) #torch.Size([1, 128, 64, 64])
        out10 = torch.add(out10, out5d1) #torch.Size([1, 128, 64, 64])
        out10 = torch.add(out10, out12up2nor) #torch.Size([1, 128, 64, 64])
        out10d = self.conv10down1(out10) #torch.Size([1, 256, 32, 32])
        out10up = torch.nn.functional.upsample(out10, scale_factor=2, mode='bilinear', align_corners=True) #torch.Size([1, 128, 128, 128])
        out10upnor = self.convnor10(out10up) #torch.Size([1, 64, 128, 128])
        out13 = self.conv13(out12) #torch.Size([1, 256, 32, 32])
        out13 = torch.add(out13, out10d) #torch.Size([1, 256, 32, 32])
        out6 = self.conv6(out5) #torch.Size([1, 64, 128, 128])
        out6 = torch.add(out6, out12up4nor) #torch.Size([1, 64, 128, 128])
        out6 = torch.add(out6, out10upnor) #torch.Size([1, 64, 128, 128])
        out6m = out6 #torch.Size([1, 16, 128, 128])
        out10m = out10 #torch.Size([1, 32, 64, 64])
        out13m = out13 #torch.Size([1, 64, 32, 32])
        # n6, c6, h6, w6 = out6m.data.shape
        # out6m = out6m.view(-1)
        # n10, c10, h10, w10 = out10m.data.shape
        # out10m = out10m.view(-1)
        # n13, c13, h13, w13 = out13m.data.shape
        # out13m = out13m.view(-1)
        return out6m, out10m, out13m
        # return out6m, [n6, c6, h6, w6], out10m, [n10, c10, h10, w10], out13m, [n13, c13, h13, w13]


class FeatExtractor3(nn.Module):
    def __init__(self, batchNorm=False, c_in=3, other={}):
        super(FeatExtractor3, self).__init__()
        self.other = other
        self.conv1 = model_utils.conv(batchNorm, c_in, 16, k=3, stride=1, pad=1)
        self.conv2 = model_utils.conv(batchNorm, 16, 16, k=3, stride=1, pad=1)
        self.conv3 = model_utils.conv(batchNorm, 16, 16, k=3, stride=1, pad=1)
        self.conv4 = model_utils.conv(batchNorm, 16, 16, k=3, stride=1, pad=1)
        self.conv5 = model_utils.conv(batchNorm, 16, 16, k=3, stride=1, pad=1)
        self.conv6 = model_utils.conv(batchNorm, 16, 16, k=3, stride=1, pad=1)
        self.conv7 = model_utils.conv(batchNorm, 16, 32, k=3, stride=2, pad=1)  # from c2
        self.conv8 = model_utils.conv(batchNorm, 32, 32, k=3, stride=1, pad=1)
        self.conv9 = model_utils.conv(batchNorm, 32, 32, k=3, stride=1, pad=1)
        self.conv10 = model_utils.conv(batchNorm, 32, 32, k=3, stride=1, pad=1)
        self.conv11 = model_utils.conv(batchNorm, 32, 64, k=3, stride=2, pad=1)  # from c8
        self.conv12 = model_utils.conv(batchNorm, 64, 64, k=3, stride=1, pad=1)
        self.conv13 = model_utils.conv(batchNorm, 64, 64, k=3, stride=1, pad=1)
        self.conv3down1 = model_utils.conv(batchNorm, 16, 32, k=3, stride=2, pad=1)  # to c8
        self.conv4down1 = model_utils.conv(batchNorm, 16, 32, k=3, stride=2, pad=1)  # to c9
        self.conv4down2 = model_utils.conv(batchNorm, 16, 64, k=3, stride=4, pad=1)  # to c11
        self.conv5down1 = model_utils.conv(batchNorm, 16, 32, k=3, stride=2, pad=1)  # to c10
        self.conv5down2 = model_utils.conv(batchNorm, 16, 64, k=3, stride=4, pad=1)  # to c12
        self.conv10down1 = model_utils.conv(batchNorm, 32, 64, k=3, stride=2, pad=1)  # to c13
        self.convnor8 = model_utils.conv(batchNorm, 32, 16, k=1, stride=1, pad=0)
        self.convnor122 = model_utils.conv(batchNorm, 64, 32, k=1, stride=1, pad=0)
        self.convnor124 = model_utils.conv(batchNorm, 64,16, k=1, stride=1, pad=0)
        self.convnor10 = model_utils.conv(batchNorm, 32, 16, k=1, stride=1, pad=0)

    def forward(self, x):
        out1 = self.conv1(x) #torch.Size([1, 64, 128, 128])
        out2 = self.conv2(out1) #torch.Size([1, 64, 128, 128])
        out3 = self.conv3(out2) #torch.Size([1, 64, 128, 128])
        out3d = self.conv3down1(out3) #torch.Size([1, 128, 64, 64])
        out7 = self.conv7(out2) #torch.Size([1, 128, 64, 64])
        out8 = self.conv8(out7) #torch.Size([1, 128, 64, 64])
        out8up = torch.nn.functional.upsample(out8, scale_factor=2, mode='bilinear', align_corners=True) #torch.Size([1, 128, 128, 128])
        out8upnor = self.convnor8(out8up) #torch.Size([1, 64, 128, 128])
        out8 = torch.add(out8, out3d) #torch.Size([1, 128, 64, 64])
        out4 = self.conv4(out3) #torch.Size([1, 64, 128, 128])
        out4 = torch.add(out4, out8upnor) #torch.Size([1, 64, 128, 128])
        out4d1 = self.conv4down1(out4) #torch.Size([1, 128, 64, 64])
        out4d2 = self.conv4down2(out4) #torch.Size([1, 256, 32, 32])
        out9 = self.conv9(out8) #torch.Size([1, 128, 64, 64])
        out9 = torch.add(out9, out4d1) #torch.Size([1, 128, 64, 64])
        out5 = self.conv5(out4) #torch.Size([1, 64, 128, 128])
        out5d1 = self.conv5down1(out5) #torch.Size([1, 128, 64, 64])
        out5d2 = self.conv5down2(out5) #torch.Size([1, 256, 32, 32])
        out11 = self.conv11(out8) #torch.Size([1, 256, 32, 32])
        out11 = torch.add(out11, out4d2) #torch.Size([1, 256, 32, 32])
        out12 = self.conv12(out11) #torch.Size([1, 256, 32, 32])
        out12 = torch.add(out12, out5d2) #torch.Size([1, 256, 32, 32])
        out12up2 = torch.nn.functional.upsample(out12, scale_factor=2, mode='bilinear', align_corners=True) #torch.Size([1, 256, 64, 64])
        out12up2nor = self.convnor122(out12up2) #torch.Size([1, 128, 64, 64])
        out12up4 = torch.nn.functional.upsample(out12, scale_factor=4, mode='bilinear', align_corners=True) #torch.Size([1, 256, 128, 128])
        out12up4nor = self.convnor124(out12up4) #torch.Size([1, 64, 128, 128])
        out10 = self.conv10(out9) #torch.Size([1, 128, 64, 64])
        out10 = torch.add(out10, out5d1) #torch.Size([1, 128, 64, 64])
        out10 = torch.add(out10, out12up2nor) #torch.Size([1, 128, 64, 64])
        out10d = self.conv10down1(out10) #torch.Size([1, 256, 32, 32])
        out10up = torch.nn.functional.upsample(out10, scale_factor=2, mode='bilinear', align_corners=True) #torch.Size([1, 128, 128, 128])
        out10upnor = self.convnor10(out10up) #torch.Size([1, 64, 128, 128])
        out13 = self.conv13(out12) #torch.Size([1, 256, 32, 32])
        out13 = torch.add(out13, out10d) #torch.Size([1, 256, 32, 32])
        out6 = self.conv6(out5) #torch.Size([1, 64, 128, 128])
        out6 = torch.add(out6, out12up4nor) #torch.Size([1, 64, 128, 128])
        out6 = torch.add(out6, out10upnor) #torch.Size([1, 64, 128, 128])
        out6m = out6 #torch.Size([1, 16, 128, 128])
        out10m = out10 #torch.Size([1, 32, 64, 64])
        out13m = out13 #torch.Size([1, 64, 32, 32])
        # n6, c6, h6, w6 = out6m.data.shape
        # out6m = out6m.view(-1)
        # n10, c10, h10, w10 = out10m.data.shape
        # out10m = out10m.view(-1)
        # n13, c13, h13, w13 = out13m.data.shape
        # out13m = out13m.view(-1)
        return out6m, out10m, out13m
        # return out6m, [n6, c6, h6, w6], out10m, [n10, c10, h10, w10], out13m, [n13, c13, h13, w13]
        

class FeatExtractor4(nn.Module):
    def __init__(self, batchNorm=False, c_in=3, other={}):
        super(FeatExtractor4, self).__init__()
        self.other = other
        self.conv1 = model_utils.conv(batchNorm, c_in, 16, k=3, stride=1, pad=1)
        self.conv2 = model_utils.conv(batchNorm, 16, 16, k=3, stride=1, pad=1)
        self.conv3 = model_utils.conv(batchNorm, 16, 16, k=3, stride=1, pad=1)
        self.conv4 = model_utils.conv(batchNorm, 16, 16, k=3, stride=1, pad=1)
        self.conv5 = model_utils.conv(batchNorm, 16, 16, k=3, stride=1, pad=1)
        self.conv6 = model_utils.conv(batchNorm, 16, 16, k=3, stride=1, pad=1)
        self.conv7 = model_utils.conv(batchNorm, 16, 32, k=3, stride=2, pad=1)  # from c2
        self.conv8 = model_utils.conv(batchNorm, 32, 32, k=3, stride=1, pad=1)
        self.conv9 = model_utils.conv(batchNorm, 32, 32, k=3, stride=1, pad=1)
        self.conv10 = model_utils.conv(batchNorm, 32, 32, k=3, stride=1, pad=1)
        self.conv11 = model_utils.conv(batchNorm, 32, 64, k=3, stride=2, pad=1)  # from c8
        self.conv12 = model_utils.conv(batchNorm, 64, 64, k=3, stride=1, pad=1)
        self.conv13 = model_utils.conv(batchNorm, 64, 64, k=3, stride=1, pad=1)
        self.conv3down1 = model_utils.conv(batchNorm, 16, 32, k=3, stride=2, pad=1)  # to c8
        self.conv4down1 = model_utils.conv(batchNorm, 16, 32, k=3, stride=2, pad=1)  # to c9
        self.conv4down2 = model_utils.conv(batchNorm, 16, 64, k=3, stride=4, pad=1)  # to c11
        self.conv5down1 = model_utils.conv(batchNorm, 16, 32, k=3, stride=2, pad=1)  # to c10
        self.conv5down2 = model_utils.conv(batchNorm, 16, 64, k=3, stride=4, pad=1)  # to c12
        self.conv10down1 = model_utils.conv(batchNorm, 32, 64, k=3, stride=2, pad=1)  # to c13
        self.convnor8 = model_utils.conv(batchNorm, 32, 16, k=1, stride=1, pad=0)
        self.convnor122 = model_utils.conv(batchNorm, 64, 32, k=1, stride=1, pad=0)
        self.convnor124 = model_utils.conv(batchNorm, 64,16, k=1, stride=1, pad=0)
        self.convnor10 = model_utils.conv(batchNorm, 32, 16, k=1, stride=1, pad=0)

    def forward(self, x):
        out1 = self.conv1(x) #torch.Size([1, 64, 128, 128])
        out2 = self.conv2(out1) #torch.Size([1, 64, 128, 128])
        out3 = self.conv3(out2) #torch.Size([1, 64, 128, 128])
        out3d = self.conv3down1(out3) #torch.Size([1, 128, 64, 64])
        out7 = self.conv7(out2) #torch.Size([1, 128, 64, 64])
        out8 = self.conv8(out7) #torch.Size([1, 128, 64, 64])
        out8up = torch.nn.functional.upsample(out8, scale_factor=2, mode='bilinear', align_corners=True) #torch.Size([1, 128, 128, 128])
        out8upnor = self.convnor8(out8up) #torch.Size([1, 64, 128, 128])
        out8 = torch.add(out8, out3d) #torch.Size([1, 128, 64, 64])
        out4 = self.conv4(out3) #torch.Size([1, 64, 128, 128])
        out4 = torch.add(out4, out8upnor) #torch.Size([1, 64, 128, 128])
        out4d1 = self.conv4down1(out4) #torch.Size([1, 128, 64, 64])
        out4d2 = self.conv4down2(out4) #torch.Size([1, 256, 32, 32])
        out9 = self.conv9(out8) #torch.Size([1, 128, 64, 64])
        out9 = torch.add(out9, out4d1) #torch.Size([1, 128, 64, 64])
        out5 = self.conv5(out4) #torch.Size([1, 64, 128, 128])
        out5d1 = self.conv5down1(out5) #torch.Size([1, 128, 64, 64])
        out5d2 = self.conv5down2(out5) #torch.Size([1, 256, 32, 32])
        out11 = self.conv11(out8) #torch.Size([1, 256, 32, 32])
        out11 = torch.add(out11, out4d2) #torch.Size([1, 256, 32, 32])
        out12 = self.conv12(out11) #torch.Size([1, 256, 32, 32])
        out12 = torch.add(out12, out5d2) #torch.Size([1, 256, 32, 32])
        out12up2 = torch.nn.functional.upsample(out12, scale_factor=2, mode='bilinear', align_corners=True) #torch.Size([1, 256, 64, 64])
        out12up2nor = self.convnor122(out12up2) #torch.Size([1, 128, 64, 64])
        out12up4 = torch.nn.functional.upsample(out12, scale_factor=4, mode='bilinear', align_corners=True) #torch.Size([1, 256, 128, 128])
        out12up4nor = self.convnor124(out12up4) #torch.Size([1, 64, 128, 128])
        out10 = self.conv10(out9) #torch.Size([1, 128, 64, 64])
        out10 = torch.add(out10, out5d1) #torch.Size([1, 128, 64, 64])
        out10 = torch.add(out10, out12up2nor) #torch.Size([1, 128, 64, 64])
        out10d = self.conv10down1(out10) #torch.Size([1, 256, 32, 32])
        out10up = torch.nn.functional.upsample(out10, scale_factor=2, mode='bilinear', align_corners=True) #torch.Size([1, 128, 128, 128])
        out10upnor = self.convnor10(out10up) #torch.Size([1, 64, 128, 128])
        out13 = self.conv13(out12) #torch.Size([1, 256, 32, 32])
        out13 = torch.add(out13, out10d) #torch.Size([1, 256, 32, 32])
        out6 = self.conv6(out5) #torch.Size([1, 64, 128, 128])
        out6 = torch.add(out6, out12up4nor) #torch.Size([1, 64, 128, 128])
        out6 = torch.add(out6, out10upnor) #torch.Size([1, 64, 128, 128])
        out6m = out6 #torch.Size([1, 16, 128, 128])
        out10m = out10 #torch.Size([1, 32, 64, 64])
        out13m = out13 #torch.Size([1, 64, 32, 32])
        # n6, c6, h6, w6 = out6m.data.shape
        # out6m = out6m.view(-1)
        # n10, c10, h10, w10 = out10m.data.shape
        # out10m = out10m.view(-1)
        # n13, c13, h13, w13 = out13m.data.shape
        # out13m = out13m.view(-1)
        return out6m, out10m, out13m
        # return out6m, [n6, c6, h6, w6], out10m, [n10, c10, h10, w10], out13m, [n13, c13, h13, w13]
    

class Regressor(nn.Module):
    def __init__(self, batchNorm=False, other={}):
        super(Regressor, self).__init__()
        self.other = other
        self.deconv1 = model_utils.deconv(192, 64)
        self.deconv2 = model_utils.conv(batchNorm, 64, 64, k=3, stride=1, pad=1)
        self.deconv3 = model_utils.deconv(64, 32)
        self.deconv4 = model_utils.deconv(96, 32)
        self.deconv5 = model_utils.conv(batchNorm, 48, 32, k=3, stride=1, pad=1)
        self.deconv6 = model_utils.conv(batchNorm, 96, 48, k=3, stride=1, pad=1)
        self.est_normal = self._make_output(48, 1, k=3, stride=1, pad=1)
        self.other = other

    def _make_output(self, cin, cout, k=3, stride=1, pad=1):
        return nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False))

    def forward(self, x6, x10, x13):

        out1 = self.deconv1(x13) #torch.Size([1, 128, 64, 64])
        out2 = self.deconv2(out1) #torch.Size([1, 128, 64, 64])
        out3 = self.deconv3(out2)  #torch.Size([1, 64, 128, 128])
        out4 = self.deconv4(x10) #torch.Size([1, 64, 128, 128])
        out5 = self.deconv5(x6) #torch.Size([1, 64, 128, 128])
        outcat = torch.cat((out3, out4, out5), 1) #torch.Size([1, 192, 128, 128])
        out6 = self.deconv6(outcat) #torch.Size([1, 64, 128, 128])
        pre_angular_map = self.est_normal(out6) #torch.Size([1, 3, 128, 128])
        # normal = torch.nn.functional.normalize(normal, 2, 1)
        return pre_angular_map


class Regressor2(nn.Module):
    def __init__(self, batchNorm=False, other={}):
        super(Regressor2, self).__init__()
        self.other = other
        self.deconv1 = model_utils.deconv(128, 64)
        self.deconv2 = model_utils.conv(batchNorm, 64, 64, k=3, stride=1, pad=1)
        self.deconv3 = model_utils.deconv(64, 32)
        self.deconv4 = model_utils.deconv(64, 32)
        self.deconv5 = model_utils.conv(batchNorm, 32, 32, k=3, stride=1, pad=1)
        self.deconv6 = model_utils.conv(batchNorm, 96, 48, k=3, stride=1, pad=1)
        self.est_normal = self._make_output(48, 3, k=3, stride=1, pad=1)
        self.other = other

    def _make_output(self, cin, cout, k=3, stride=1, pad=1):
        return nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False))

    def forward(self, x6, x10, x13):

        out1 = self.deconv1(x13) #torch.Size([1, 128, 64, 64])
        out2 = self.deconv2(out1) #torch.Size([1, 128, 64, 64])
        out3 = self.deconv3(out2)  #torch.Size([1, 64, 128, 128])
        out4 = self.deconv4(x10) #torch.Size([1, 64, 128, 128])
        out5 = self.deconv5(x6) #torch.Size([1, 64, 128, 128])
        outcat = torch.cat((out3, out4, out5), 1) #torch.Size([1, 192, 128, 128])
        out6 = self.deconv6(outcat) #torch.Size([1, 64, 128, 128])
        rel_img = self.est_normal(out6) #torch.Size([1, 3, 128, 128])
        # normal = torch.nn.functional.normalize(normal, 2, 1)
        return rel_img


    
class Errormap_FCN(nn.Module):
    def __init__(self,batchNorm):
        super(Errormap_FCN, self).__init__()
        
        # self.init_encoder = init_encoder(c_in=7)
        # self.relightNet = relightNet()
        # self.decoder = decoder()
        self.featExtractor1 = FeatExtractor1(batchNorm=batchNorm, c_in=3, other={})
        self.featExtractor2 = FeatExtractor2(batchNorm=batchNorm, c_in=1, other={})
        self.featExtractor3 = FeatExtractor3(batchNorm=batchNorm, c_in=3, other={})
        self.featExtractor4 = FeatExtractor4(batchNorm=batchNorm, c_in=3, other={})
        self.regressor = Regressor(batchNorm=False, other={})
        self.regressor2 = Regressor2(batchNorm=False, other={})
        
        # self.featExtractor1 = FeatExtractor1(batchNorm=False, c_in=3, other={})
        # self.featExtractor2 = FeatExtractor2(batchNorm=True, c_in=1, other={})
        # self.featExtractor3 = FeatExtractor3(batchNorm=False, c_in=3, other={})
        # self.featExtractor4 = FeatExtractor4(batchNorm=False, c_in=3, other={})
        # self.regressor = Regressor(batchNorm=False, other={})
        # self.regressor2 = Regressor2(batchNorm=False, other={})
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        #         kaiming_normal_(m.weight.data)
        #         if m.bias is not None:
        #             m.bias.data.zero_()
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
                
    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file,
                   _use_new_zipfile_serialization=False)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))

    def forward(self, predict_normal,angular_map,img,l_dir):
        # l2_normal  = self.featExtractor1(l2_normal)
        predict_normal_out6m, predict_normal_out10m ,predict_normal_out13m= self.featExtractor1(predict_normal)
        angular_map_out6m, angular_map_out10m ,angular_map_out13m = self.featExtractor2(angular_map)
        img_out6m,  img_out10m, img_out13m= self.featExtractor3(img)
        l_dir_out6m,l_dir_out10m,l_dir_out13m = self.featExtractor4(l_dir)
        
        feats1_6 = torch.cat((predict_normal_out6m, angular_map_out6m, img_out6m),1)
        feats1_10 = torch.cat((predict_normal_out10m, angular_map_out10m, img_out10m),1)
        feats1_13 = torch.cat((predict_normal_out13m, angular_map_out13m, img_out13m),1)
        # feats1 = torch.cat((predict_normal,angular_map,img),1)
        predict_angular_map = self.regressor(feats1_6 ,feats1_10 ,feats1_13)
        
        feats2_6 = torch.cat((img_out6m,l_dir_out6m), dim=1)
        feats2_10 = torch.cat((img_out10m,l_dir_out10m), dim=1)
        feats2_13 = torch.cat((img_out13m,l_dir_out13m), dim=1)
       
        rel_img = self.regressor2(feats2_6,feats2_10,feats2_13)


        return predict_angular_map, rel_img
    
    

