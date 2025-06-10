import os
import math
import cv2 as cv
import numpy as np
from light_dirs import light_dirs
from benchmark import action_best,imload_downsample_from_ids, benchmark_from_ids, action_greedy

def mesh_action_ids(all_light_dirs, mesh_size, dtype=np.float32): 
  mesh_x = np.linspace(-1.0, 1.0, num=mesh_size, endpoint=True, dtype=dtype) 
  mesh_y = np.linspace(-1.0, 1.0, num=mesh_size, endpoint=True, dtype=dtype) 
  mesh_x, mesh_y = np.meshgrid(mesh_x, mesh_y, indexing="ij")
  meshgrid = np.stack([mesh_x.flatten(), mesh_y.flatten()], axis=1) 
  dist = np.linalg.norm(all_light_dirs[:, None, :2] - meshgrid[None, :, :], ord=2, axis=2) 
  ids = dist.argmin(axis=1) 
  return ids

class Env:
  def __init__(self, input_files, rng, algo, down_sample=2):
    self.input_files = input_files
    self.down_sample = down_sample 
    self.rng = rng
    self.algo = algo
    self.images = []
    self.normal = []
    self.reward = []
    self.training = False
    self.input_file = None
    self.data_loader = None
    self.all_light_dirs = None
    self.max_step = 20

  def action_space(self):
    return len(self.all_light_dirs)

  def mesh_action_ids(self, mesh_size=24, dtype=np.float32):
    return mesh_action_ids(self.all_light_dirs, mesh_size, dtype)

  def action_greedy(self):
    return action_greedy(self.data_loader, self.ids, start=self.rng.integers(7), step=7)

  def get_virtual_expert_action(self):
        return action_best(self.data_loader, self.ids, self.algo,start=self.rng.integers(1), step=1)

  def action_around(self, action):
    dist = np.sum(self.all_light_dirs * self.all_light_dirs[None, action, :], axis=1)
    p = np.power(dist.clip(min=0.0), 32)
    p/= p.sum()

    return self.rng.choice(len(self.all_light_dirs), p=p)

  def benchmark(self):
    error, error_mean, normal = benchmark_from_ids(self.data_loader, self.ids, self.algo, self.down_sample)
    return error, error_mean, normal 

  def state_compressed(self):

    return self.input_file, self.ids, np.array(self.reward)

  def state(self, dtype=np.float32):
    images = self.images.astype(dtype).transpose(0, 3, 1, 2) 

    return images, self.ids

  def reset(self, scene_id=None):
    self.episode_step=0
    if scene_id is None:
      self.input_file = self.rng.choice(self.input_files) 
    else:
      self.input_file = self.input_files[scene_id]
    self.data_loader = np.load(self.input_file, allow_pickle=True)
    self.all_light_dirs = light_dirs(self.data_loader["meta_info"].item()["light_dirs_mode"]) 
    self.ids = [np.argmax(self.all_light_dirs[:, 2])]  
    self.normal_gt, self.images, _ = imload_downsample_from_ids(self.data_loader, self.ids, self.down_sample) 
    self.error, _, self.normal_pred = self.benchmark() 
    _, self.error_all, _ = benchmark_from_ids(self.data_loader, range(len(self.all_light_dirs)), self.algo, self.down_sample)

    self.normal = [self.normal_pred]
    self.reward = []

    print("input_file", self.input_file, "error_all", self.error_all)
    return self.state(), self.error 

  def step(self, action):
    self.episode_step+=1
    self.ids.append(action)
    _, self.images, _ = imload_downsample_from_ids(self.data_loader, self.ids, self.down_sample)
    new_error, error_mean, self.normal_pred = self.benchmark()  


    weight = (1.0 - 0.9 ** (len(self.ids) - 1)) * 10.0 / (self.error_all + 10.0)
    reward = weight * ( 0.5 * self.error - new_error)

    self.error = new_error
    self.normal.append(self.normal_pred)
    self.reward.append(reward)

    return self.state(), reward, error_mean

