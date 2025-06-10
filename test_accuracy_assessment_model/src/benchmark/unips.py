import os
import py_compile

import numpy as np
import importlib
import torch
import torch.nn.functional as F

import cv2
import glob
canonical_resolution = 256
def load_models( model, dirpath):
    pytmodel = "".join(glob.glob(f'{dirpath}/*.pytmodel'))
    print(pytmodel)
    model = loadmodel(model, pytmodel, strict=False)
    return model

def loadmodel(model, filename, strict=True):
    if os.path.exists(filename):
        params = torch.load('%s' % filename, map_location=torch.device('cpu'))
        model.load_state_dict(params,strict=strict)
        print('Loading pretrained model... %s ' % filename)
    else:
        print('Pretrained model not Found')
    return model

from pathlib import Path

# 获取当前文件的绝对路径
current_file = Path(__file__).resolve()

# 获取当前文件所在的目录
current_dir = current_file.parent


device = torch.device("cuda:0")

unips= importlib.import_module(".sdm_unips.modules.model.model", __package__).Net
net_nml = unips(pixel_samples=10000, output='normal', device = device).to(device)
net_nml = torch.nn.DataParallel(net_nml)
model_dir = os.path.join(current_dir,"sdm_unips/checkpoint/normal")
net_nml = load_models(net_nml, model_dir)
net_nml.eval()


def benchmark(images, light_dirs, object_mask, normal_gt=None, dtype=np.float32, debug=False):
  images = images.astype(dtype).copy()[:, :, :, ::-1] #bgr_rgb  #(n_images , 128, 128, 3)
  images = images.transpose(0, 2, 1, 3)[:, ::-1, :,  :] #

  n_images = images.shape[0]

  h = 256
  w = h

  resized_images = []
  for img in images:  
    resized_img = cv2.resize(img, dsize=(h, w), interpolation=cv2.INTER_CUBIC) 
    resized_images.append(resized_img)
  images = np.array(resized_images)  # shape=(n_images, h, w, 3)
  del resized_images
   
  images = np.reshape(images, (-1, h * w, 3)) #(96, 262144, 3)
  
  mask = object_mask.astype(np.float32) 
  mask = mask.transpose(1, 0)[::-1, :]
  mask =cv2.resize(mask, dsize=(h, w),interpolation=cv2.INTER_CUBIC)#(512, 512)
  

  """Data Normalization"""
  temp = np.mean(images[:, mask.flatten()==1.0,:], axis=2)
  mean = np.mean(temp, axis=1)
  mx = np.max(temp, axis=1)
  scale = np.random.rand(images.shape[0],)
  temp = (1-scale) * mean + scale * mx
  temp = mx
  images /= (temp.reshape(-1,1,1) + 1.0e-6)   #(96, 262144, 3)     

  images = np.transpose(images, (1, 2, 0)) #(262144, 3, 96)
  images = images.reshape(h, w, 3, n_images) #(512, 512, 3, 96)
  

  
  images= images.transpose(2,0,1,3) # c, h, w, N    #(3, 512, 512, 96)
  images = np.expand_dims(images,axis=0)
  I = torch.tensor(images, dtype=torch.float32, device=device)
  del images
           
  nml = normal_gt.transpose(2,0,1) # 3, h, w   #(3, 512, 612)
 
  roi = np.array([512, 612, 185, 342, 229, 386])   #array([512, 612, 185, 342, 229, 386])
  
  
  
  N = cv2.resize(normal_gt, dsize=(h, w),interpolation=cv2.INTER_CUBIC) #(512, 512, 3)
  nml = N.transpose(2,0,1)
#   mask = np.float32(cv2.resize(object_mask, dsize=(h, w),interpolation=cv2.INTER_CUBIC)> 0.5) #(512, 512)
#   mask = mask.transpose(1, 0)[::-1, :]
  mask = (mask.reshape(h, w, 1)).astype(np.float32) # (512, 512, 1)
  mask = np.transpose(mask, (2,0,1)) # 1, h, w  (1, 512, 512)
  mask = np.expand_dims(mask, axis=0) # 1,1, h, w  (1,1, 512, 512)
  M = torch.tensor(mask, dtype=torch.float32, device=device)
  


  nImgArray = torch.tensor(n_images, dtype=torch.int64, device=device).unsqueeze(0)
  

  with torch.no_grad():
   
    
    nout, _, _, _  = net_nml(I, M, nImgArray.reshape(-1,1), decoder_resolution= h * torch.ones(I.shape[0],1),canonical_resolution=canonical_resolution* torch.ones(I.shape[0],1))
    normal_pred = (nout * M).squeeze().permute(1,2,0).cpu().detach().numpy()


  torch.cuda.empty_cache()

  normal_pred = normal_pred.transpose(1, 0, 2)[:, ::-1, :]

  normal_pred = cv2.resize(normal_pred, (128, 128), interpolation=cv2.INTER_AREA)


  return normal_pred
