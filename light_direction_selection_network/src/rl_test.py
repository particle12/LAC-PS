import io
import os
import sys
import glob
import math
import time
import zipfile
from functools import partial
import torch
import cv2 as cv
import numpy as np
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "true"
import torch.optim as optim
import torch.nn.functional as F
from rl_env import Env
from benchmark import _all_algos, display_normal, display_normal_err
from rl_model import QNetEnhance, _n_light_axix

device = torch.device("cuda:0")


def rl_test(qnet, input_files, n_max_images, algo, down_sample=2):
  rng = np.random.default_rng(0)
  env = Env(input_files, rng, algo=algo, down_sample=down_sample)
  results = []
  it_total = 0
  for scene_id in range(len(input_files)):
    state, info = env.reset(scene_id)
    results_iter = []
    results_q_vals = []
    results_normal_pred = []
    for it in range(n_max_images - 1):
      if it == 0:
        init_image = env.images[0]
        normal_gt = env.normal_gt
        # print("init_image", init_image.shape, "init_normal", init_normal.shape)
      with torch.no_grad():
        x_images = torch.tensor(state[0], dtype=torch.float32, device=device)[None, ...]
        x_dirs = env.mesh_action_ids(_n_light_axix//2)[state[1]]
        x_dirs = torch.tensor(x_dirs, dtype=torch.long, device=device)[None, ...]
        q_vals = qnet(x_images, x_dirs)[0][0, -1, ...].cpu().numpy()
        q_vals_clip = q_vals.mean(axis=(0, 1)).flatten()[env.mesh_action_ids(_n_light_axix)]
        q_vals_clip[env.ids] = -1e9
        print(env.ids)
        action = np.argmax(q_vals_clip)

      next_state, reward, info = env.step(action)

      print(f"{time.time():.3f}", \
            "action % 4d" % action, \
            "reward % 10.3f" % reward.mean(), \
            "loss % 10.3f" % 0.0, \
            "q_vals % 8.2f" % q_vals.max(), \
            "it_total % 6d" % it_total, "it % 4d" % (it + 1), "eps % 8.5f" % 0.0, "info % 8.3f" % info)
      it_total += 1
      state = next_state

      if it >= 1:
        results_iter.append(info)
   
    results.append(results_iter)
    print(f"Result {algo} ({env.input_file }): {results_iter}")

  print(f"Result {algo} (mean): {np.mean(results, axis=0).tolist()}")

def main():

  n_max_images =20
  input_files = []

  for input_path in ["dataset/test_blobs","dataset/test_sculpture","dataset/diligent_test"]:
    print(input_path)
    input_files = []
    for input_file in sorted(glob.glob(os.path.join(input_path, "*.npz"))):
        input_files.append(input_file)
    qnet = QNetEnhance().to(device)
    algo = "psfcn"

    checkpoint = torch.load(os.path.join(f"rl_models_mine/{algo}/{algo}.bin"),map_location=device)
    qnet.load_state_dict(checkpoint["qnet"])

    qnet.eval()
    rl_test(qnet, input_files, n_max_images, algo)
    print("\n")
    print("\n")


if __name__ == "__main__":
  main()
