import os
import torch
import torch.nn as nn
import torch.nn.functional as F


from cbam import CBAMBlock
from senet import SE_Block
_n_light_axix = 24
_activation = nn.ReLU()
_activation = nn.LeakyReLU()
_pool = nn.MaxPool2d(2)
_padding_mode = "reflect"

def _weight_norm(module):
  nn.init.orthogonal_(module.weight)
  module.bias.data.zero_()
  return nn.utils.weight_norm(module)


class ImageConv(nn.Module):
    def __init__(self,init_weight=True):
        super(ImageConv, self).__init__()
        # 特征提取层
        self.features = nn.Sequential(
            _weight_norm(nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=1)),
            _weight_norm(nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=1)),
            nn.MaxPool2d(kernel_size=2,stride=2),
            _weight_norm(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)),
            _weight_norm(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            _weight_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)),
            _weight_norm(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)),
            _weight_norm(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)),
            nn.MaxPool2d(kernel_size=2,stride=2),
            _weight_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)),
            _weight_norm(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)),
            _weight_norm(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)),
            nn.MaxPool2d(kernel_size=2, stride=2),

        )
        self.cbam = CBAMBlock(channel=256,reduction=8,kernel_size=3)
 
 
    def forward(self,x):
        n_batch = x.shape[0] 
        n_max_images = x.shape[1] 
        x = x.view(n_batch * n_max_images, x.shape[2], x.shape[3], x.shape[4]) 

        x = self.features(x)

        
        img_size = x.shape[2]  
        img_channel = x.shape[1] 
        x = x.view(n_batch, n_max_images, img_channel, img_size, img_size) 
        return x


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None,init_weight=True):
        super(Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = (nn.Conv2d(inplanes, planes, kernel_size=1, bias=True))
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = (nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True))
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = (nn.Conv2d(planes, planes * 2, kernel_size=1, bias=True))
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


        if init_weight:
            for m in self.modules():  
                if isinstance(m, nn.Conv2d): 
                 
                    nn.init.orthogonal_(m.weight)
                 
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                
                    m = nn.utils.weight_norm(m)

                        
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
              
                    nn.init.normal_(m.weight, 0, 0.01)
                
                    nn.init.constant_(m.bias, 0)

                    m = nn.utils.weight_norm(m)
                    
    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes*block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n-1][0](x)  
        low1 = F.max_pool2d(x, 2, stride=2) 
        low1 = self.hg[n-1][1](low1)  

        if n > 1:
            low2 = self._hour_glass_forward(n-1, low1) 
        else:
            low2 = self.hg[n-1][3](low1)
        low3 = self.hg[n-1][2](low2) 
        up2 = F.interpolate(low3, scale_factor=2) 
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)

class ANet(nn.Module):
  def __init__(self):
    super().__init__()

    self.conv_trans_b3 = _weight_norm(nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1))
    self.conv_c5 = _weight_norm(nn.Conv2d(64, 64, 3, padding=1, padding_mode=_padding_mode))
    self.conv_c6 = _weight_norm(nn.Conv2d(64, 1, 3, padding=1, padding_mode=_padding_mode))
    
    self.hourglass = Hourglass(block=Bottleneck, num_blocks=4, planes=32, depth=2)
    self.cbam = CBAMBlock(channel=64,reduction=8,kernel_size=3)
    self.senet = SE_Block(64)

  def forward(self, images, light_dirs, debug=False):
    images = images.permute(0, 1, 3, 4, 2) 
    if debug:
      print("images", images.shape)
    n_batch, n_max_images, img_size, _, _ = images.shape
    img_channel = 64
    x = torch.zeros(images.shape[:-1] + (img_channel, (_n_light_axix//2)**2), device=images.device) 
    if debug:
      print("light_dirs", light_dirs)
    for i_image in range(n_max_images):
      x[:, i_image, :, :, 0, :] = 1.0 - 0.9 ** (i_image + 1)
    for i_batch in range(n_batch):
      for i_image in range(n_max_images):
        x[i_batch, i_image:, :, :, :, light_dirs[i_batch, i_image]] += \
          images[None, i_batch, i_image, :, :, :img_channel]
    if debug:
      print("x", x.shape, x.min(), x.max())

    x = x.view(n_batch * n_max_images * img_size * img_size, img_channel, _n_light_axix//2, _n_light_axix//2)
    x= _activation(self.hourglass(x))

    x = _activation(self.conv_trans_b3(x)) 
    if debug:
      print("x", x.shape)
    x = _activation(self.conv_c5(x)) 
    
    x= _activation(self.cbam(x))

    a_f = x
    x = self.conv_c6(x) 
    if debug:
      print("x", x.shape)

    x = x.view(n_batch, n_max_images, img_size, img_size, _n_light_axix, _n_light_axix) 
    a_f = a_f.mean(dim=(1), keepdim=True).view(n_batch, n_max_images, img_size, img_size, _n_light_axix, _n_light_axix) 
    if debug:
      print("x", x.shape)
    return x,a_f

class VNet(nn.Module):
  def __init__(self):
    super().__init__()

    self.conv_a4 = _weight_norm(nn.Conv2d(512, 1, 3, padding=1, padding_mode=_padding_mode))
    self.hourglass = Hourglass(block=Bottleneck, num_blocks=1, planes=256, depth=1)
    self.cbam = CBAMBlock(channel=512,reduction=8,kernel_size=3)
    self.senet = SE_Block(512)

  def forward(self, images, debug=False):
    n_batch, n_max_images, img_channel, img_size, _ = images.shape 
    if debug:
      print("images", images.shape)
    x = torch.zeros([n_batch, n_max_images, img_channel*2, img_size, img_size], device=images.device) 
    if debug:
      print("x", x.shape)
    for i_image in range(n_max_images):
      x[:, i_image, :img_channel, :, :] = images[:, :(i_image+1), :, :, :].mean(dim=1)
      x[:, i_image, img_channel:, :, :] = images[:, :(i_image+1), :, :, :].max(dim=1)[0]
      x[:, i_image, 0, :, :] = 1.0 - 0.9 ** (i_image + 1)
    if debug:
      print("x", x.shape) 
    x = x.view(n_batch * n_max_images, img_channel*2, img_size, img_size) 

    x = _activation(self.hourglass(x))
   
    x = _activation(self.cbam(x))

    v_f = x
    x = self.conv_a4(x) 
    if debug:
      print("x", x.shape)
    x = x.view(n_batch, n_max_images, img_size, img_size) 
    v_f = v_f.mean(dim=(1), keepdim=True).view(n_batch, n_max_images, img_size, img_size) 
    if debug:
      print("x", x.shape)
    return x,v_f
class QNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.image_conv = ImageConv()
    self.anet = ANet()
    self.vnet = VNet()

  def forward(self, images, light_dirs, debug=False):
    images = self.image_conv(images) 
    if debug:
      print("images", images.shape)
    x_a ,a_f= self.anet(images, light_dirs) 
    if debug:
      print("x_a", x_a.shape)
    x_v ,v_f= self.vnet(images) 
    if debug:
      print("x_v", x_v.shape)
    x = x_a - x_a.mean(dim=(-2, -1), keepdim=True) + x_v[..., None, None]
    
    return x,a_f,v_f



class QNetEnhance(nn.Module):
  def __init__(self):
    super().__init__()

    self.qnet = QNet()



  def forward(self, images_orig, light_dirs_orig, debug=False):  
    
    q_vals,a_f,v_f= self.qnet(images_orig, light_dirs_orig)

    return q_vals,a_f,v_f
