import os
import sys
import cProfile
import multiprocessing
import numpy as np
# from . import benchmark_all_pool, benchmark_circle_pool, benchmark_okabe_pool, benchmark_greedy_pool, \
#   benchmark_random_pool, _all_algos, _algo_processes
  
  
import os
import sys
import glob
import zipfile
import importlib
import itertools
from functools import partial
import cv2 as cv
import numpy as np
from scipy.ndimage import minimum_filter
from light_dirs import light_dirs
# from imload import imload_downsample_from_ids

# _all_algos = ["LS"]
# _algo_processes = [32]
# _all_algos = ["CNNPS"]
# _algo_processes = [12]
# _all_algos = ["PSFCN"]
# _algo_processes = [1]
# _all_algos = ["SPLINENET"]
# _algo_processes = [1]
_all_algos = ["LS", "CNNPS", "PSFCN", "SPLINENET"]
_algo_processes = [32, 12, 12, 1]

import os
import time
import threading
import cv2 as cv
import numpy as np
from light_dirs import light_dirs

_image_cache = None

def _image_cache_clean():
  global _image_cache
  while True:
    time.sleep(10)
    # old_len = len(_image_cache)
    _image_cache = dict(filter(lambda val: val[1][1], _image_cache.items()))
    _image_cache = dict(map(lambda val: [val[0], [val[1][0], False]], _image_cache.items()))
    # print("_image_cache_clean: before", old_len, "after", len(_image_cache))

def imload_downsample_from_ids(data_loader, ids, down_sample=2):
  global _image_cache
  if _image_cache is None:
    _image_cache = {}
    _image_cache_clean_thread = threading.Thread(target=_image_cache_clean, daemon=True)
    _image_cache_clean_thread.start()
  normal_key = (data_loader.zip.filename, down_sample)
  if normal_key in _image_cache:
    _image_cache[normal_key][1] = True
    normal = _image_cache[normal_key][0]
  else:
    normal = data_loader["normal"]
    res_x = normal.shape[0] // down_sample
    res_y = normal.shape[1] // down_sample
    normal = normal.reshape(res_x, down_sample, res_y, down_sample, 3)
    normal = np.mean(normal, axis=(1, 3))
    _image_cache[normal_key] = [normal, True]
  selected_light_dirs = light_dirs(data_loader["meta_info"].item()["light_dirs_mode"])[ids, :]
  selected_images = []
  for curr_id, curr_light_dir in zip(ids, selected_light_dirs):
    curr_image_key = (data_loader.zip.filename, curr_id, down_sample)
    if curr_image_key in _image_cache:
      _image_cache[curr_image_key][1] = True
      curr_image = _image_cache[curr_image_key][0]
    else:
      if os.environ.get("RLPS_ABLATION") == "1":
        curr_image = (normal * curr_light_dir[np.newaxis, np.newaxis, :]).sum(axis=-1, keepdims=True)
        curr_image = curr_image.clip(0.0, 1.0).repeat(3, -1) * 0.9
        curr_image += np.random.rand(*curr_image.shape) * 0.1
        # print("curr_image", curr_image.shape, curr_image.min(), curr_image.max())
      else:
        curr_image = cv.imdecode(data_loader["image_%06d" % curr_id], cv.IMREAD_UNCHANGED).astype(np.float32)
        curr_image = np.power(curr_image / np.float32(255.0), np.float32(2.2))
        res_x = curr_image.shape[0] // down_sample
        res_y = curr_image.shape[1] // down_sample
        curr_image = curr_image.reshape(res_x, down_sample, res_y, down_sample, 3)
        curr_image = np.mean(curr_image, axis=(1, 3))
      _image_cache[curr_image_key] = [curr_image, True]
    # print("curr_image", curr_image.shape, curr_image.dtype)
    selected_images.append(curr_image)
  selected_images = np.stack(selected_images, axis=0)
  return normal, selected_images, selected_light_dirs

def display_normal(normal, normal_mask):
  normal_mask = minimum_filter(normal_mask, size=3, mode="constant") > 0.0
  normal_mask = normal_mask.transpose(1, 0)[::-1, :]
  normal_alpha = (normal_mask * 255).astype(np.uint8)
  normal = normal.transpose(1, 0, 2)[::-1, :, [2, 1, 0]]
  normal = np.clip((normal * 0.5 + 0.5) * 255.0, 0.0, 255.0).astype(np.uint8)
  normal = np.concatenate([normal, normal_alpha[:, :, np.newaxis]], axis=-1)
  return normal

def display_normal_err(normal_err, normal_mask):
  normal_mask = minimum_filter(normal_mask, size=3, mode="constant") > 0.0
  normal_mask = normal_mask.transpose(1, 0)[::-1, :]
  normal_alpha = (normal_mask * 255).astype(np.uint8)
  normal_err = normal_err.transpose(1, 0)[::-1, :]
  normal_err = np.clip(normal_err, 0.0, 90) / 90
  normal_err = cv.applyColorMap((normal_err * 255).astype(np.uint8), cv.COLORMAP_TURBO)
  normal_err = np.concatenate([normal_err, normal_alpha[:, :, np.newaxis]], axis=-1)
  return normal_err

def benchmark(images, light_dirs, object_mask, normal_gt, algo, dtype=np.float32, debug=False):
  images = images.astype(dtype).copy()
  light_dirs = light_dirs.astype(dtype).copy()
  normal_gt = normal_gt.astype(dtype).copy()
  if debug:
    print("images", images.shape, images.dtype)
    print("light_dirs", light_dirs.shape, light_dirs.dtype)
    print("normal_gt", normal_gt.shape, normal_gt.dtype)
  normal_gt /= (np.linalg.norm(normal_gt, ord=2, axis=2, keepdims=True) + 1e-6)
  normal_gt[~object_mask, :] = 0.0
  benchmark_algo = importlib.import_module(f"{algo.lower()}", __package__).benchmark
  normal_pred = benchmark_algo(images, light_dirs, object_mask, normal_gt, dtype)
  normal_pred /= (np.linalg.norm(normal_pred, ord=2, axis=2, keepdims=True) + 1e-6)
  normal_pred[~object_mask, :] = 0.0
  error = 1.0 - (normal_pred * normal_gt).sum(axis=-1)
  error[~object_mask] = 0.0
  if debug:
    print("error", error.shape, error.dtype, error.min(), error.max())
  error_degree = np.arccos(1.0 - error) / np.pi * 180
  return error_degree, normal_pred

def benchmark_from_ids(data_loader, ids, algo, down_sample=2):
  ids = list(set(ids))
  normal_gt, selected_images, selected_light_dirs = imload_downsample_from_ids(data_loader, ids, down_sample)
  object_mask = minimum_filter(normal_gt[..., -1], size=3, mode="constant") > 0.0
  error, N = benchmark(selected_images, selected_light_dirs, object_mask, normal_gt, algo)
  error_mean = error[object_mask].mean()
  return error, error_mean, N

def benchmark_random_file(test_id, input_file, n_lights, algo, down_sample=2):
  print("benchmark_random_file", test_id, input_file, algo)
  rng = np.random.default_rng(test_id)
  data_loader = np.load(input_file, allow_pickle=True)
  all_light_dirs = light_dirs(data_loader["meta_info"].item()["light_dirs_mode"])
  num_light_dirs = len(all_light_dirs)
  ids = rng.choice(np.arange(num_light_dirs), size=n_lights, replace=False)
  ids[0] = np.argmax(all_light_dirs[:, 2])
  results = []
  for j in range(2, n_lights):
    _, result, _ = benchmark_from_ids(data_loader, ids[:(j+1)], algo, down_sample=down_sample)
    results.append(result)
  return results

def benchmark_random_pool(pool, algo, input_path, n_lights, n_samples, rng):
  input_files = sorted(glob.glob(os.path.join(input_path, "*.npz")))
  input_files = [input_files[i % len(input_files)] for i in range(n_samples)]
  benchmark_random_file_partial = partial(benchmark_random_file, n_lights=n_lights, algo=algo)
  results = pool.starmap(benchmark_random_file_partial, enumerate(input_files))
  for input_file, result in zip(input_files, results):
    print(f"Result {algo} ({input_file}): {result}")
  print(f"Result {algo} (mean): {np.mean(results, axis=0).tolist()}")

def benchmark_all_file(test_id, input_file, algo, down_sample=2):
  print("benchmark_all_file", test_id, input_file, algo)
  data_loader = np.load(input_file, allow_pickle=True)
  num_light_dirs = len(light_dirs(data_loader["meta_info"].item()["light_dirs_mode"]))
  _, result, _ = benchmark_from_ids(data_loader, range(num_light_dirs), algo, down_sample=down_sample)
  return result

def benchmark_all_pool(pool, algo, input_path):
  input_files = sorted(glob.glob(os.path.join(input_path, "*.npz")))
  benchmark_all_file_partial = partial(benchmark_all_file, algo=algo)
  results = pool.starmap(benchmark_all_file_partial, enumerate(input_files))
  for input_file, result in zip(input_files, results):
    print("Result " + algo + " (" + input_file + "):", result)
  print("Result " + algo + " (mean):", np.mean(results).item())

def action_greedy(data_loader, ids, start=0, step=1, down_sample=4):
  all_light_dirs = light_dirs(data_loader["meta_info"].item()["light_dirs_mode"])
  num_light_dirs = len(all_light_dirs)
  ids = ids + [0]
  id_best = 0
  error_best = sys.float_info.max
  for k in range(start, num_light_dirs, step):
    if k in ids[:-1]:
      continue
    ids[-1] = k
    _, result, _ = benchmark_from_ids(data_loader, ids, algo="LS", down_sample=down_sample)
    if result < error_best:
      error_best = result
      id_best = k
  return id_best


_all_algos = ["LS", "CNNPS", "PSFCN", "SPLINENET"]
_algo_processes = [32, 12, 12, 1]

def main(*args):
  multiprocessing.set_start_method("spawn")
  # for algo, processes in zip(_all_algos, _algo_processes):
  algo = "LS"
  processes = 32

  with multiprocessing.Pool(processes=processes) as pool:
    if args[0] == "all":
      benchmark_all_pool(pool, algo, args[1])
      # elif sys.argv[1] == "circle":
      #   benchmark_circle_pool(pool, algo, sys.argv[2], int(sys.argv[3]), sys.argv[4].lower() == "true")
      # elif sys.argv[1] == "okabe":
      #   rng = np.random.default_rng(0)
      #   benchmark_okabe_pool(pool, algo, sys.argv[2], int(sys.argv[3]), rng)
      # elif sys.argv[1] == "greedy":
      #   benchmark_greedy_pool(pool, algo, sys.argv[2], int(sys.argv[3]))
    elif args[0] == "random":
      rng = np.random.default_rng(0)
      benchmark_random_pool(pool, algo, args[1], int(args[2]), int(args[3]), rng)
    else:
      print("Unknown operation:", args[0])
      sys.exit(-1)

if __name__ == "__main__":
  # cProfile.run("main()")
  # main("all","data/diligent_test")  #diligent 可换成blobs or sculpture
  main("random","data/diligent_test",20,100)
