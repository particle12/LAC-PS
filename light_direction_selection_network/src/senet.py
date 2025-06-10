import numpy as np
import torch
from torch import nn
from torch.nn import init
import math


'''-------------一、SE模块-----------------------------'''
#全局平均池化+1*1卷积核+ReLu+1*1卷积核+Sigmoid
class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16, init_weight=True):
        super(SE_Block, self).__init__()
        # 全局平均池化(Fsq操作)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # 两个全连接层(Fex操作)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )
        
        
        if init_weight:
            for m in self.modules():  # 对于模型的每一层
                if isinstance(m, nn.Conv2d): # 如果是卷积层
                    # 使用初始化
                    nn.init.orthogonal_(m.weight)
                    # 如果bias不为空，固定为0
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                    # 应用权重归一化
                    m = nn.utils.weight_norm(m)

                        
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):# 如果是线性层
                    # 正态初始化
                    nn.init.normal_(m.weight, 0, 0.01)
                    # bias则固定为0
                    # nn.init.constant_(m.bias, 0)
                    
                    # 应用权重归一化
                    m = nn.utils.weight_norm(m)
 
    def forward(self, x):
            # 读取批数据图片数量及通道数
            b, c, h, w = x.size()
            # Fsq操作：经池化后输出b*c的矩阵
            y = self.gap(x).view(b, c)
            # Fex操作：经全连接层输出（b，c，1，1）矩阵
            y = self.fc(y).view(b, c, 1, 1)
            # Fscale操作：将得到的权重乘以原来的特征图x
            return x * y.expand_as(x)
        
        
        
if __name__ == '__main__':
    # input=torch.randn(50,512,7,7)
    # input=torch.randn(8,128,16,16)
    # input = torch.randn(512, 64, 24, 24)
    input = torch.randn(8, 512, 8, 8)
    kernel_size=input.shape[2]
    se = SE_Block(512)
    output=se(input)
    print(output.shape)