import torch
import random
import numpy as np
# from skimage.transform import resize
random.seed(0)
np.random.seed(0)

def arrayToTensor(array):
    if array is None:
        return array
    array = np.transpose(array, (2, 0, 1))
    tensor = torch.from_numpy(array)
    # tensor = tensor.float().cuda()
    return tensor.float()

def rgbToGray(img):
    h, w, c = img.shape
    img = img[:,:,0] * 0.229 + img[:,:,1] * 0.587 + img[:,:,2] * 0.114
    return img.reshape(h, w, 1)

def normalToMask(normal, thres=0.01):
    """
    Due to the numerical precision of uint8, [0, 0, 0] will save as [127, 127, 127] in gt normal,
    When we load the data and rescale normal by N / 255 * 2 - 1, the [127, 127, 127] becomes 
    [-0.003927, -0.003927, -0.003927]
    """
    mask = (np.square(normal).sum(2, keepdims=True) > thres).astype(np.float32)
    return mask

def randomCrop(pre_normal,next_normal, angular_map12, angular_map2, mask, size):
    # if not __debug__: print('RandomCrop: input, target', inputs.shape, target.shape, size)
    h, w, _ = pre_normal.shape
    c_h, c_w = size
    if h == c_h and w == c_w:
        return pre_normal,next_normal, angular_map12,angular_map2, mask
    x1 = random.randint(0, w - c_w)
    y1 = random.randint(0, h - c_h)
    pre_normal = pre_normal[y1: y1 + c_h, x1: x1 + c_w]
    next_normal = next_normal[y1: y1 + c_h, x1: x1 + c_w]
    angular_map12 = angular_map12[y1: y1 + c_h, x1: x1 + c_w]
    angular_map2 = angular_map2[y1: y1 + c_h, x1: x1 + c_w]
    mask = mask[y1: y1 + c_h, x1: x1 + c_w]
    return pre_normal,next_normal, angular_map12, angular_map2, mask



def randomNoiseAug(pre_normal,next_normal, angular_map12, noise_level=0.05):
    # if not __debug__: print('RandomNoiseAug: input, noise level', inputs.shape, noise_level)
    noise = np.random.random(pre_normal.shape)
    noise = (noise - 0.5) * noise_level
    pre_normal += noise
    next_normal+= noise
    return pre_normal,next_normal,angular_map12



def normalize(imgs):
    h, w, c = imgs[0].shape
    imgs = [img.reshape(-1, 1) for img in imgs]
    img = np.hstack(imgs)
    norm = np.sqrt((img * img).clip(0.0).sum(1))
    img = img / (norm.reshape(-1,1) + 1e-10)
    imgs = np.split(img, img.shape[1], axis=1)
    imgs = [img.reshape(h, w, -1) for img in imgs]
    return imgs

