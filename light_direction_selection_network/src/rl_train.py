import os
import sys
import glob

import torch
import cv2 as cv
import numpy as np
import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F
from light_dirs import light_dirs
from rl_env import Env, mesh_action_ids
from rl_model import QNetEnhance, _n_light_axix
from benchmark import imload_downsample_from_ids

device = torch.device("cuda:0")

batch_size = 1
down_sample = 2


class Buffer:
  def __init__(self, size=256,prioritized=True):
    self.size = size
    self.items = 0
    self.input_file = [None for i in range(size)]
    self.reward = [None for i in range(size)]
    self.ids = [None for i in range(size)]
    
    
    self.prioritized = prioritized
    if prioritized:
        self.priority = [None for i in range(size)]
        self.max_priority = 1
    
  def insert(self, input_file, ids, reward):
    pos = self.items % self.size
    self.items += 1
    self.input_file[pos] = input_file
    self.reward[pos] = reward
    self.ids[pos] = ids
    
    if self.prioritized:
        self.priority[pos] = self.max_priority

  def state(self, pos):
    data_loader = np.load(self.input_file[pos], allow_pickle=True)
    ids = self.ids[pos]
    _, images, _ = imload_downsample_from_ids(data_loader, ids, down_sample=down_sample)
    images = images.transpose(0, 3, 1, 2)
    reward = self.reward[pos]
    light_dirs_mode = data_loader["meta_info"].item()["light_dirs_mode"]
    state_dirs = mesh_action_ids(light_dirs(light_dirs_mode), _n_light_axix//2)[ids]
    action_ids = mesh_action_ids(light_dirs(light_dirs_mode), _n_light_axix)
    return images[:-1], state_dirs[:-1], action_ids[ids[1:]], reward, images[1:], state_dirs[1:], action_ids

  def sample(self, size, rng):
    if self.prioritized:
        csum =np.cumsum(self.priority[:self.items], 0)
        val = rng.uniform(0,1,size=size)*csum[-1]
        self.sample_ids = np.searchsorted(csum, val)
    else:
        replace = self.items < size
        self.sample_ids = rng.choice(np.arange(min(self.items, self.size)), size=size, replace=replace)
    state = [self.state(pos) for pos in self.sample_ids]
    state = [np.stack(curr_state, axis=0) for curr_state in zip(*state)]
    return state

  def update_priority(self, priority):

        self.priority[self.sample_ids[0]] = priority
        self.max_priority = max(float(priority.max()), self.max_priority)


  def reset_max_priority(self):
    self.max_priority = float(max(self.priority[:self.items]))

def main():

  n_max_images = 20
   
  algo = "cnnps"
  
  rng = np.random.default_rng(0) 
  torch.manual_seed(0)
  input_files = []

  for input_path in "dataset/train_blobs/+dataset/train_sculpture/".split("+"):
    
    for input_file in sorted(glob.glob(os.path.join(input_path, "*.npz"))):
      input_files.append(input_file)

  env = Env(input_files, rng, algo=algo, down_sample=down_sample)
  env.training = True
  state, info = env.reset() 
  buffer = Buffer(size=16384)

  qnet = QNetEnhance().to(device)
  qnet_target = QNetEnhance().to(device)
  qnet_target.eval()
  for target_param, param in zip(qnet_target.parameters(), qnet.parameters()):
    target_param.data.copy_(param.data)
  optimizer = optim.AdamW(qnet.parameters(), lr=1e-4, weight_decay=1e-8)

  train_downsample = 2
  batch_start = batch_size * n_max_images ** 2 // train_downsample 
  it = 0
  errors = []
  error_min = sys.float_info.max  

  loss_bellman = 0.0
  loss_smooth = 0.0
  peer_loss = 0.0

  for it_total in range(200000):

    eps = min(1.0, 2000 / (it_total + 2000 - batch_start))

    with torch.no_grad():
      qnet.eval()
      x_images = state[0] 
  
      x_images = torch.tensor(x_images, dtype=torch.float32, device=device)[None, ...] 
 
      x_dirs = state[1]  
  
      x_dirs = env.mesh_action_ids(_n_light_axix//2)[x_dirs] 
      x_dirs = torch.tensor(x_dirs, dtype=torch.long, device=device)[None, ...] 
  
      q ,_,_= qnet(x_images, x_dirs)
      q_vals =q[0, -1, ...].cpu().numpy()
     
      action = np.argmax(q_vals.mean(axis=(0, 1)).flatten()[env.mesh_action_ids(_n_light_axix)]) 
    sel = rng.random()
    if sel < eps:
      action = env.action_greedy() 
      
    action = env.action_around(action)
    next_state, reward, info = env.step(action) 

    it += 1
    mask = info > 0.0
    print("action % 4d" % action, \
          "reward % 10.3f" % reward[mask].mean(), \
          "loss % 12.6f % 12.6f" % (loss_bellman, peer_loss), \
          "q_vals % 8.3f" % q_vals.min(), \
          "it_total % 6d" % it_total, "it % 4d" % it, "eps % 8.5f" % eps, "info % 8.3f" % info[mask].mean())
    state = next_state
    # env.render()
    if it == n_max_images - 1:
      errors.append(info[mask].mean())
      input_file, ids, reward = env.state_compressed()
      reward = reward.reshape(n_max_images - 1, 8, 16, 8, 16).mean(axis=(2, 4))
      buffer.insert(input_file, ids, reward)
      state, info = env.reset()
      it = 0
    if it_total < batch_start or it_total % train_downsample:
      continue
    batch_state_images, batch_state_dirs, batch_action, batch_reward, \
      batch_next_images, batch_next_dirs, batch_action_ids = buffer.sample(batch_size, rng)  
    batch_state_images = torch.tensor(batch_state_images, dtype=torch.float32, device=device) 
    batch_state_dirs = torch.tensor(batch_state_dirs, dtype=torch.long, device=device)  
    batch_next_images = torch.tensor(batch_next_images, dtype=torch.float32, device=device) 
    batch_next_dirs = torch.tensor(batch_next_dirs, dtype=torch.long, device=device)
    with torch.no_grad():
      q_vals , current_feature_a,current_feature_v= qnet(batch_next_images, batch_next_dirs)  
      q_vals = q_vals.view(batch_size, q_vals.shape[1], q_vals.shape[2], q_vals.shape[3], -1)
 
      q_vals_target ,target_feature_a,target_feature_v = qnet_target(batch_next_images, batch_next_dirs)  
      q_vals_target = q_vals_target.view(batch_size, q_vals.shape[1], q_vals.shape[2], q_vals.shape[3], -1) 
  
      target = torch.zeros([batch_size, n_max_images - 1, 8, 8], dtype=torch.float32, device=device)
      for i_batch in range(batch_size):
        for i_image in range(n_max_images - 1):
          action_ids = torch.tensor(batch_action_ids[i_batch], dtype=torch.long, device=device)  
       
          target_index = torch.argmax(q_vals[i_batch, i_image].mean(dim=(0, 1))[action_ids]) 
        
          target[i_batch, i_image] = q_vals_target[i_batch, i_image, :, :, action_ids[target_index]]
      batch_reward = torch.tensor(batch_reward, dtype=torch.float32, device=device)

      target = 0.5 * target + batch_reward
    qnet.train()
    q_vals,_,_ = qnet(batch_state_images, batch_state_dirs)
    loss_smooth = 1e-8 * batch_size * n_max_images * (
      (q_vals[..., :, :-1] - q_vals[..., :, :1]).abs().mean() +
      (q_vals[..., :-1, :] - q_vals[..., 1:, :]).abs().mean())
    q_vals = q_vals.view(batch_size, n_max_images - 1, 8, 8, -1)
    loss_bellman = 0.0
    peer_loss1 = 0.0
    peer_loss2 = 0.0
    criterion_SML1=nn.SmoothL1Loss()
    for i_batch in range(batch_size):
      for i_image in range(n_max_images - 1):
        action_ids = torch.tensor(batch_action_ids[i_batch], dtype=torch.long, device=device)
        weight = (1.0 - 0.9 ** (i_image + 1))
        diff = q_vals[i_batch, i_image, :, :, batch_action[i_batch, i_image]] - target[i_batch, i_image]
        
        
        loss_bellman += (weight * diff).pow(2).mean()
        peer_loss1 += weight*torch.einsum('ij,ij->i', [current_feature_a[i_batch, i_image, :,:,:,:].view(-1,1), target_feature_a[i_batch, i_image,:,:,:,:].view(1,-1)]).mean()
        peer_loss2 +=  weight*torch.einsum('ij,ij->i', [current_feature_v[i_batch, i_image, :,:].view(-1,1), target_feature_v[i_batch, i_image, :,:].view(1,-1)]).mean()
   
    priority = loss_bellman.clamp(min=0.1).pow(0.4).detach().cpu().numpy()
    buffer.update_priority(priority)

    peer_loss = peer_loss1+peer_loss2
    loss = loss_bellman + loss_smooth+peer_loss*0.0001
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    

    if it_total // train_downsample % 1000 == 0:
      error_curr = np.mean(errors[-100:])
      if it_total >= 10000 and error_min > error_curr:
        print("save model at iter % 4d, error: % 10.6f" % (it_total, error_curr))
        error_min = error_curr
 
        ckpt_dir = f"models/{algo}/{it_total}"
        os.makedirs(ckpt_dir,  exist_ok=True)
        torch.save({"qnet": qnet.state_dict()}, os.path.join(ckpt_dir,f"{algo}.bin"))
   
      for target_param, param in zip(qnet_target.parameters(), qnet.parameters()):
        target_param.data.copy_(param.data)
       
      buffer.reset_max_priority()
      
    if it_total >= 10000 and (it_total // train_downsample % 5000 ==0 or it_total==199998):
        ckpt_dir1 = f"models/{algo}/{it_total}"
        os.makedirs(ckpt_dir1,  exist_ok=True)
        torch.save({"qnet": qnet.state_dict()}, os.path.join(ckpt_dir1,f"{algo}.bin"))

if __name__ == "__main__":

  main()
  sys.exit(0)
