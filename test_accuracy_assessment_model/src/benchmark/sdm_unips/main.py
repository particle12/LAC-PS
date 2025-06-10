"""
Scalable, Detailed and Mask-free Universal Photometric Stereo Network (CVPR2023)
# Copyright (c) 2023 Satoshi Ikehata
# All rights reserved.
"""

from __future__ import print_function, division
import torch
if torch.cuda.is_available():
    print(1111)
    
print(2222)
# from modules.model.model_utils import *
from modules.builder import builder
from modules.io import dataio
import sys
import argparse
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] ="0" 
sys.path.append('..')  # add parent directly for importing

# Argument parser
parser = argparse.ArgumentParser()

# Properties
parser.add_argument('--session_name', default='sdm_unips')
parser.add_argument('--target', default='normal', choices=['normal', 'brdf', 'normal_and_brdf'])
parser.add_argument('--checkpoint', default='checkpoint')

# Data Configuration
parser.add_argument('--max_image_res', type=int, default=4096)
parser.add_argument('--max_image_num', type=int, default=96)
parser.add_argument('--test_ext', default='PNG')
parser.add_argument('--test_dir', default='/home/mengwenjia/hhm/DiLiGenT/pmsData')
parser.add_argument('--test_prefix', default='0*')
parser.add_argument('--mask_margin', type=int, default=8)

# Network Configuration
parser.add_argument('--canonical_resolution', type=int, default=256)
parser.add_argument('--pixel_samples', type=int, default=10000)
parser.add_argument('--scalable', action='store_true')


def main():
    args = parser.parse_args()
    print(f'\nStarting a session: {args.session_name}')
    print(f'target: {args.target}\n')
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device("cuda:{1,2}".format(1) if torch.cuda.is_available() else torch.device("cpu"))
    # device = torch.device("cuda:1") 
    sdf_unips = builder.builder(args, device)
    test_data = dataio.dataio('Test', args)

    start_time = time.time()
    sdf_unips.run(testdata=test_data,
                  max_image_resolution=args.max_image_res,
                  canonical_resolution=args.canonical_resolution,
                  )
    end_time = time.time()
    print(f"Prediction finished (Elapsed time is {end_time - start_time:.3f} sec)")
    print("\nExecute the following script to render a video under new lighting conditions based on the generated BRDF and normal map.\n")
    print(f"        python sdm_unips/relighting.py --datadir ./{args.session_name}/results/{test_data.data.objname}\n")
         


if __name__ == '__main__':
    main()
