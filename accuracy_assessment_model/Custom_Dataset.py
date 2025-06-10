from __future__ import division
import os,cv2
import numpy as np
#from scipy.ndimage import imread
from imageio import imread
import scipy.io as sio
from itertools import islice
import re

import torch
import torch.utils.data as data

import pms_transforms
import util
import time
import imageio

class Custom_Dataset(data.Dataset):
    def __init__(self, args,dataset_dir):
        self.root           = os.path.join(dataset_dir)
        self.args           = args

        self.alldata_dir = util.readList(os.path.join("1.txt")) 
    def __getitem__(self, index):

        data_dir = self.alldata_dir[index]

        data_dir = os.path.join(self.root,data_dir)


       
        normal_map = cv2.imread(os.path.join(data_dir, self.args.backbone+"_normal.tif"),cv2.IMREAD_UNCHANGED)
        normal_map = cv2.cvtColor(normal_map, cv2.COLOR_BGR2RGB)
        
        pre_image = cv2.imread(os.path.join(data_dir,'pre_image.tif'),cv2.IMREAD_UNCHANGED)#(128, 128, 3)
        pre_image = cv2.cvtColor(pre_image, cv2.COLOR_BGR2RGB)
        
        next_image = cv2.imread(os.path.join(data_dir,'next_image.tif'),cv2.IMREAD_UNCHANGED)
        next_image = cv2.cvtColor(next_image, cv2.COLOR_BGR2RGB)
        
        angular_map12 = cv2.imread(os.path.join(data_dir,"angular_map12.tif"),cv2.IMREAD_UNCHANGED) #the angular map between two normal map(128, 128)
        angular_map2 = cv2.imread(os.path.join(data_dir,"angular_map2.tif"),cv2.IMREAD_UNCHANGED) #the t+1angular map between normalmap and groundtruth normalmap
    
        
        mask = imread(os.path.join(data_dir,"mask.png"))
        mask = mask[:,:,0:1]
        mask = mask.astype(np.float32) / 255.0 #( 128, 128 ,1)

        item={}

        angular_map12 = np.expand_dims(angular_map12, axis=2) #(128,128,1)
        angular_map2 = np.expand_dims(angular_map2, axis=2)#(128,128,1)


        with open(os.path.join(data_dir,"lights.txt"), 'r') as f:
            data = f.read()

        # 将字符串转换为数组
        lights = np.loadtxt(data.splitlines(), delimiter=' ')
        pre_light = lights[0]
        next_light = lights[1]

        next_light= np.array(next_light,dtype=float)
        next_light = torch.from_numpy(next_light).view(-1, 1, 1).float()

        

        item["normal_map"] = normal_map
        item["angular_map12"] = angular_map12
        item["angular_map2"] = angular_map2

        item["pre_image"] = pre_image
        item["next_image"] = next_image
        item["mask"] = mask
        
        
        for k in item.keys(): 
            item[k] = pms_transforms.arrayToTensor(item[k])

        item["next_light"] = next_light
        
        return item

    def __len__(self):

        return len(self.alldata_dir)


