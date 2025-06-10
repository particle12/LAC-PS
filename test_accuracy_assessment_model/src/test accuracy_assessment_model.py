import random
random.seed(100)
import util
import torch
import os
import numpy as np
import glob
from light_dirs import light_dirs

import importlib
from collections import deque
import os
import sys
import glob
from benchmark import  benchmark_from_ids
from scipy.ndimage import minimum_filter
device = torch.device("cuda:1")
import cv2
import imageio



batch_size = 1
down_sample = 2

algo = "psfcn"

for input_path in ["dataset/test_blobs","dataset/test_sculpture","dataset/diligent_test"]:

    
    print(input_path)

    input_files = []
    for input_file in sorted(glob.glob(os.path.join(input_path, "*.npz"))):
        input_files.append(input_file)
        
    if input_path == "dataset/diligent_test":
        dataset = "diligent"
        
    if input_path == "dataset/test_blobs":
        dataset = "blobby"    

    if input_path == "dataset/test_sculpture":
        dataset = "sculpture"    
    
    Errormap_FCN = importlib.import_module("relation.network", __package__).Errormap_FCN
    

    if dataset=="diligent" :
        net = Errormap_FCN(True).to(device)
        checkpoint_path = os.path.join(rf"relation_ckpt/{algo}/checkp.pth.tar")
        
    if dataset=="blobby" or dataset=="sculpture" or algo=="unips":
        net = Errormap_FCN(False).to(device)
        checkpoint_path = os.path.join(f"relation_ckpt/{algo}/bn_false/checkp.pth.tar")
        
    checkpoint = torch.load(os.path.join(os.path.dirname(__file__), checkpoint_path),map_location=device)
    net.load_state_dict(checkpoint['state_dict'])
    
    for i,input_file in enumerate(input_files) :  
        count = 0
        normal_queue = deque(maxlen=2)     
        data_loader = np.load(input_file, allow_pickle=True)
        all_light_dirs = light_dirs(data_loader["meta_info"].item()["light_dirs_mode"]) 
        
        if dataset=="diligent":
            filename =data_loader["meta_info"].item()["filename"] 
            
        if dataset=="blobby" or dataset=="sculpture":
            filename = str(i)

        numbers = all_light_dirs.shape[0] 
        numbers = list(range(numbers)) 
        numbers = random.sample(numbers,32)
        

        selected_ids = random.sample(numbers,1)
        error, error_mean, N_psfcn,normal_gt,object_mask,images,lights= benchmark_from_ids(data_loader, selected_ids, algo, down_sample)
        
        
        normal_gt =normal_gt.transpose(1, 0, 2)[::-1, :, [2, 1, 0]]
        normal_mask = object_mask.transpose(1, 0)[::-1, :]
        N_psfcn_first = N_psfcn.transpose(1, 0, 2)[::-1, :, :]

        
        normal_queue.append(N_psfcn_first)

        unselected_ids = [num for num in numbers if num not in selected_ids]
        for idx, k in enumerate(unselected_ids) :
            count+=1
            next_select_id = random.sample(unselected_ids,1)[0]
            selected_ids.append(next_select_id)
            
            error, error_mean_psfcn,N_psfcn, normal_gt, object_mask,images,lights= benchmark_from_ids(data_loader, selected_ids, algo, down_sample)

            N_psfcn = N_psfcn.transpose(1, 0, 2)[::-1, :, :]

            normal_mask = object_mask.transpose(1, 0)[::-1, :]
            normal_queue.append(N_psfcn)

            error =(normal_queue[0] *normal_queue[1]).sum(axis=-1) #(128, 128)
            error = np.clip(error,-1,1)
            error[~normal_mask] = 0.0
            error_degree = np.arccos(error) / np.pi #(128, 128)
        
            error_degree[~normal_mask] = 0.0
            
            error_first =(N_psfcn*N_psfcn_first).sum(axis=-1)
            error_first = np.clip(error_first,-1,1)
            error_first[~normal_mask] = 0.0
            error_first = np.arccos(error_first) / np.pi
            error_first[~normal_mask] = 0.0
            
            pre_image = images[1][0]
            pre_image[~normal_mask,:] = 0.0
            
            next_image = images[1][1]
            next_image[~normal_mask,:] = 0.0

            PSFCN_normal = N_psfcn*0.5+0.5 #0-1

            PSFCN_normal[~normal_mask,:] = 0.0

            
            PSFCN_normal = PSFCN_normal.transpose(2, 0, 1)[np.newaxis, ...]
            PSFCN_normal = torch.tensor(PSFCN_normal, dtype=torch.float32, device=device)
            
            error_degree = np.expand_dims(error_degree,axis=2)#(128,128,1)
            error_degree = error_degree.transpose(2, 0, 1)[np.newaxis, ...]
            error_degree =  torch.tensor(error_degree, dtype=torch.float32, device=device)
            
            error_first = np.expand_dims(error_first,axis=2)#(128,128,1)
            error_first = error_first.transpose(2, 0, 1)[np.newaxis, ...]
            error_first =  torch.tensor(error_first, dtype=torch.float32, device=device)
            
            pre_image = pre_image.copy().transpose(2,0,1)[np.newaxis, ...]
            pre_image = torch.tensor(pre_image,dtype=torch.float32,device=device)
            
            next_image = next_image.copy().transpose(2,0,1)[np.newaxis, ...]
            next_image = torch.tensor(next_image,dtype=torch.float32,device=device)
            
            next_light = lights[1][1]
            next_light = torch.from_numpy(next_light).view(-1, 1, 1).float().unsqueeze(dim=0)
            next_light = next_light.expand_as(next_image)
            next_light = next_light.cuda(device=device)
        
            
            with torch.no_grad():
                angular_map ,rel_img= net(PSFCN_normal,error_degree,pre_image,next_light)
            
            angular_map = angular_map*180

            angular_map = angular_map.squeeze().cpu().numpy()
        

            angular_map[~normal_mask]=0

            error_mean_pre = angular_map[normal_mask].mean()

            unselected_ids = [num for num in numbers if num not in selected_ids]

            
            save_dir = os.path.join(f"results_{algo}_{dataset}", filename)
            os.makedirs(save_dir,exist_ok=True)
            
            print(filename,error_mean_pre,error_mean_psfcn)
            util.write_to_txt3(os.path.join(save_dir,"pre_err.txt"),error_mean_pre)
            util.write_to_txt3(os.path.join(save_dir,"actual_err.txt"),error_mean_psfcn)

    

