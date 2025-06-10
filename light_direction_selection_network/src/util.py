
import os
import torch
import math
import numpy as np

def clear_txt(txt_dir):
    open(txt_dir, 'w').close()
    
def write_to_txt(txt_dir,content):
    dir=txt_dir
    with open(dir,"a",encoding='utf-8') as file:
        if os.path.getsize(dir):
            file.writelines("\n"+str(content))

        else:
            file.writelines(str(content))

            
def write_to_txt_bm(txt_dir,content):
    with open(txt_dir,"w",encoding='utf-8') as file:

        file.writelines(str(content))

def record(file_path,numbers):
    with open(file_path, 'w') as file:
        for idx,item in enumerate(numbers) :
            file.write(str(item) + "\n")  # 写

def get_tensors(obs):
    if obs.ndim == 1:
        obs = np.expand_dims(obs, 0)
    elif obs.ndim == 3:
        obs = np.transpose(obs, (2, 0, 1))
        obs = np.expand_dims(obs, 0)
    elif obs.ndim == 4:
        obs = np.transpose(obs, (0, 3, 1, 2))
    obs = torch.tensor(obs, dtype=torch.float32)
    # if self.args.cuda:
    #     obs = obs.cuda()
    return obs


def write_to_txt3(dir,number):
    
    with open(dir,"a",encoding='utf-8') as file:
        if os.path.getsize(dir):
            file.writelines("\n"+str(number))
            
        else:
            file.writelines(str(number))   


def calNormalAcc(gt_n, pred_n, mask=None):
    """Tensor Dim: NxCxHxW"""
    dot_product = (gt_n * pred_n).sum(1).clamp(-1,1)
    error_map   = torch.acos(dot_product) # [-pi, pi]
    angular_map = error_map * 180.0 / math.pi
    angular_map = angular_map * mask.narrow(1, 0, 1).squeeze(1)
    
    # error_map = error_map* mask.narrow(1, 0, 1).squeeze(1)
    
    valid = mask.narrow(1, 0, 1).sum()
    ang_valid   = angular_map[mask.narrow(1, 0, 1).squeeze(1).byte()]
    n_err_mean  = ang_valid.sum() / valid
    # value = {'n_err_mean': n_err_mean.item()}
    
    return n_err_mean.item(), angular_map
    # return round(n_err_mean.item(),4), angular_map
    
    
import math
import scipy.io

def save_mat(save_dir,obj,out_var):
    # out_var = out_var.squeeze(0)
    # out_var = out_var.cpu().numpy()
    # out_var = np.transpose(out_var,(1,2,0))
    mat_dict = {'Normal_est': out_var}

    os.makedirs(os.path.join(save_dir),exist_ok=True)
    # 保存为.mat文件
    scipy.io.savemat(os.path.join(save_dir,obj+'.mat'), mat_dict)