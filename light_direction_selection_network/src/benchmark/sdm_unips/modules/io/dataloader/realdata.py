"""
Scalable, Detailed and Mask-free Universal Photometric Stereo Network (CVPR2023)
# Copyright (c) 2023 Satoshi Ikehata
# All rights reserved.
"""

import glob
import os
import numpy as np
import cv2
import re

class dataloader():
    def __init__(self, numberOfImages = None, outdir = '.', mask_margin=16, ctype='ORTHO'):
        self.mask_margin=mask_margin
        self.numberOfImages = numberOfImages
        self.outdir = outdir
        self.use_mask = True
        self.ctype = ctype

    def img_tile(self, imgs, rows, cols, outdir): # [N, h, w, c]
        n, h, w, c = np.shape(imgs)
        if rows * cols <= n:
            img_tiled = []
            for i in range(cols):
                temp = np.reshape(imgs[rows*i:rows*i+rows,:,:,:], (-1, w, 3))
                img_tiled.append(temp)
            img_tiled = np.concatenate(img_tiled, axis = 1)            
            os.makedirs(outdir, exist_ok=True)
            cv2.imwrite(f'{outdir}/tiled.png', (255 * img_tiled[:,:,::-1]).astype(np.uint8))
    
    def load(self, objdir, prefix, margin = 0, max_image_resolution = 2048, aug=[]):

        self.objname = re.split(r'\\|/',objdir)[-1]
        self.data_workspace = f'{self.outdir}/results/{self.objname}'
        os.makedirs(self.data_workspace, exist_ok=True)

        print(f'Testing on {self.objname}')


        directlist = []
        [directlist.append(p) for p in glob.glob(objdir + '/%s[!.txt]' % prefix, recursive=True) if os.path.isfile(p)]
        directlist = sorted(directlist)

        if len(directlist) == 0:
            return False
        if os.name == 'posix':
            temp = directlist[0].split("/")
        if os.name == 'nt':
            temp = directlist[0].split("\\")
        img_dir = "/".join(temp[:-1])

        if self.numberOfImages is not None:
            indexset = np.random.permutation(len(directlist))[:self.numberOfImages]
        else:
            indexset = range(len(directlist))
        numberOfImages = np.min([len(indexset), self.numberOfImages])
        print(f"image index: {indexset}")

        for i, indexofimage in enumerate(indexset):
            img_path = directlist[indexofimage]
            mask_path = img_dir + '/mask.png'
            img = cv2.cvtColor(cv2.imread(img_path, flags = cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH), cv2.COLOR_BGR2RGB) #(512, 612, 3)

            if i == 0:
                h0 = img.shape[0]
                w0 = img.shape[1] 
                margin = self.mask_margin #8
                
                nml_path_diligent = img_dir + '/Normal_gt.png'
                nml_path_others = img_dir + '/normal.tif'
                if os.path.isfile(nml_path_diligent):
                    nml_path = nml_path_diligent
                elif os.path.isfile(nml_path_others):
                    nml_path = nml_path_others
                else:
                    nml_path = "no_normal" 

                # if ground truth normal map is avelable, generate normal-based mask               
                mask_flag = False
                if os.path.isfile(nml_path):
                    N = cv2.cvtColor(cv2.imread(nml_path, flags = cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH), cv2.COLOR_BGR2RGB) #(512, 612, 3)
                    if N.dtype == 'uint8':
                        bit_depth = 255.0
                    if N.dtype == 'uint16':
                        bit_depth = 65535.0
                    N = np.float32(N)/bit_depth #max 1 min 0
                    N = 2 * N - 1 #max 1 min -1
                    mask = np.float32(np.abs(1 - np.sqrt(np.sum(N * N, axis=2))) < 0.5)  #(512, 612) max 1 min 0
                    N = N/np.sqrt(np.sum(N * N, axis=2, keepdims=True))
                    N = N * mask[:, :, np.newaxis] #(512, 612, 3)
                    mask_flag = True
                else:
                    N = np.zeros((h0, w0, 3), np.float32)               

                if os.path.isfile(mask_path) and i == 0:
                    mask = (cv2.imread(mask_path, flags = cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) > 0).astype(np.float32)#(512, 612, 3)
                    if len(mask.shape) == 3:
                        mask = mask[:, :, 0] #(512, 612)
                    mask_flag = True

                # Keep mask and normal of the original resolution
                n_true = N

                # Based on object mask, crop boudning rectangular area 
                if  mask_flag == True:
                    rows, cols = np.nonzero(mask)
                    rowmin = np.min(rows)
                    rowmax = np.max(rows)
                    row = rowmax - rowmin
                    colmin = np.min(cols)
                    colmax = np.max(cols)
                    col = colmax - colmin
                    if rowmin - margin <= 0 or rowmax + margin > img.shape[0] or colmin - margin <= 0 or colmax + margin > img.shape[1]:
                        flag = False
                    else:
                        flag = True

                    if row > col and flag:
                        r_s = rowmin-margin
                        r_e = rowmax+margin
                        c_s = np.max([colmin- int(0.5 * (row-col))-margin,0])
                        c_e = np.min([colmax+int(0.5 * (row-col))+margin,img.shape[1]])
                    elif col >= row and flag:
                        r_s = np.max([rowmin-int(0.5*(col-row))-margin,0])
                        r_e = np.min([rowmax+int(0.5*(col-row))+margin, img.shape[0]])
                        c_s = colmin-margin
                        c_e = colmax+margin  
                    if flag == True:
                        mask = mask[r_s:r_e,c_s:c_e]                        
                    else:
                        r_s = 0
                        r_e = h0
                        c_s = 0
                        c_e = w0
                else:
                    mask = np.ones((h0, w0), np.float32)
                    rows, cols = np.nonzero(mask)
                    rowmin = np.min(rows)
                    rowmax = np.max(rows)
                    row = rowmax - rowmin
                    colmin = np.min(cols)
                    colmax = np.max(cols)
                    col = colmax - colmin
                    margin = 0

                    flag = True 
                    if row <= col and flag:
                        r_s = rowmin-margin
                        r_e = rowmax+margin
                        c_s = int(0.5 * col) - int(0.5 * row)
                        c_e = int(0.5 * col) + int(0.5 * row)
                    elif row > col and flag:
                        r_s = int(0.5 * row) - int(0.5 * col) #185
                        r_e = int(0.5 * row) + int(0.5 * col) #342
                        c_s = colmin-margin   #229
                        c_e = colmax+margin   #386                
                    mask = mask[r_s:r_e,c_s:c_e]     #(157, 157)        

            if flag:
                img  = img[r_s:r_e, c_s:c_e, :] #(157, 157, 3)
                if i == 0:
                    N = N[r_s:r_e, c_s:c_e, :]  #(157, 157, 3)


            h = int(np.floor(np.max([img.shape[0], img.shape[1]]) / 512) * 512) #157
            if h > max_image_resolution:
                h = max_image_resolution
            if h < 512:
                h = 512
            if i == 0:
                print(f"original crop size: {img.shape[0]} x {img.shape[1]}\nresized crop size: {h} x {h}") #original crop size: 157 x 157 resized crop size: 512 x 512

                
            w = h
            img = cv2.resize(img, dsize=(h, w),interpolation=cv2.INTER_CUBIC) #(512, 512, 3)
            N = cv2.resize(N, dsize=(h, w),interpolation=cv2.INTER_CUBIC) #(512, 512, 3)
            mask = np.float32(cv2.resize(mask, dsize=(h, w),interpolation=cv2.INTER_CUBIC)> 0.5) #(512, 512)

            if img.dtype == 'uint8':
                bit_depth = 255.0
            if img.dtype == 'uint16':
                bit_depth = 65535.0

            img = np.float32(img) / bit_depth

            if i == 0:
                I = np.zeros((len(indexset), h, w, 3), np.float32) # [N, h, w, c] #(96, 512, 512, 3)
            I[i, :, :, :] = img

        self.img_tile(I, 3, 3, self.data_workspace)
        I = np.reshape(I, (-1, h * w, 3)) #(96, 262144, 3)

        """Data Normalization"""
        temp = np.mean(I[:, mask.flatten()==1,:], axis=2)
        mean = np.mean(temp, axis=1)
        mx = np.max(temp, axis=1)
        scale = np.random.rand(I.shape[0],)
        temp = (1-scale) * mean + scale * mx
        temp = mx
        I /= (temp.reshape(-1,1,1) + 1.0e-6)   #(96, 262144, 3)     

        I = np.transpose(I, (1, 2, 0)) #(262144, 3, 96)
        I = I.reshape(h, w, 3, numberOfImages) #(512, 512, 3, 96)
        mask = (mask.reshape(h, w, 1)).astype(np.float32) # 1, h, w #(512, 512, 1)

        h = h0
        w = w0
        h = mask.shape[0] #512
        w = mask.shape[1]
        
        self.h = h
        self.w = w
        self.I = I      #(512, 512, 3, 96)  (0,1)
        self.N = n_true #(512, 612, 3)      (-1,1)

        self.roi = np.array([h0, w0, r_s, r_e, c_s, c_e]) #array([512, 612, 185, 342, 229, 386])
        if self.use_mask == True:
            self.mask = mask
        else:
            self.mask = np.ones(mask.shape, np.float32)
        
        print(f'number of images: {I.shape[3]} / {self.numberOfImages} (max)\n') #number of images: 96 / 96 (max)