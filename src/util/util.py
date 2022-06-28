import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as functional
import torchvision
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import pickle

from skimage.metrics import structural_similarity

get_mse = lambda x, y: torch.mean((x - y) ** 2)

    
def get_psnr(x, y):
    if torch.max(x) == 0 or torch.max(y) == 0:
        return torch.zeros(1)
    else:
        x_norm = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
        y_norm = (y - torch.min(y)) / (torch.max(y) - torch.min(y))
        mse = get_mse(x_norm, y_norm)
        psnr = -10. * torch.log(mse) / torch.log(torch.Tensor([10.]).to(x.device))
    return psnr


def get_psnr_3d(arr1, arr2, size_average=True, PIXEL_MAX=1.0):
    '''
    :param arr1:
        Format-[NDHW], OriImage [0,1]
    :param arr2:
        Format-[NDHW], ComparedImage [0,1]
    :return:
        Format-None if size_average else [N]
    '''
    if torch.is_tensor(arr1):
        arr1 = arr1.cpu().detach().numpy()
    if torch.is_tensor(arr2):
        arr2 = arr2.cpu().detach().numpy()
    arr1 = arr1[np.newaxis, ...]
    arr2 = arr2[np.newaxis, ...]
    arr1 = arr1.astype(np.float64)
    arr2 = arr2.astype(np.float64)
    eps = 1e-10
    se = np.power(arr1 - arr2, 2)
    mse = se.mean(axis=1).mean(axis=1).mean(axis=1)
    zero_mse = np.where(mse == 0)
    mse[zero_mse] = eps
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    # #zero mse, return 100
    psnr[zero_mse] = 100

    if size_average:
        return psnr.mean()
    else:
        return psnr


def get_ssim_3d(arr1, arr2, size_average=True, PIXEL_MAX=1.0):
    '''
    :param arr1:
        Format-[NDHW], OriImage [0,1]
    :param arr2:
        Format-[NDHW], ComparedImage [0,1]
    :return:
        Format-None if size_average else [N]
    '''
    if torch.is_tensor(arr1):
        arr1 = arr1.cpu().detach().numpy()
    if torch.is_tensor(arr2):
        arr2 = arr2.cpu().detach().numpy()
    arr1 = arr1[np.newaxis, ...]
    arr2 = arr2[np.newaxis, ...]
    assert (arr1.ndim == 4) and (arr2.ndim == 4)
    arr1 = arr1.astype(np.float64)
    arr2 = arr2.astype(np.float64)

    N = arr1.shape[0]
    # Depth
    arr1_d = np.transpose(arr1, (0, 2, 3, 1))
    arr2_d = np.transpose(arr2, (0, 2, 3, 1))
    ssim_d = []
    for i in range(N):
        ssim = structural_similarity(arr1_d[i], arr2_d[i])
        ssim_d.append(ssim)
    ssim_d = np.asarray(ssim_d, dtype=np.float64)

    # Height
    arr1_h = np.transpose(arr1, (0, 1, 3, 2))
    arr2_h = np.transpose(arr2, (0, 1, 3, 2))
    ssim_h = []
    for i in range(N):
        ssim = structural_similarity(arr1_h[i], arr2_h[i])
        ssim_h.append(ssim)
    ssim_h = np.asarray(ssim_h, dtype=np.float64)

    # Width
    # arr1_w = np.transpose(arr1, (0, 1, 2, 3))
    # arr2_w = np.transpose(arr2, (0, 1, 2, 3))
    ssim_w = []
    for i in range(N):
        ssim = structural_similarity(arr1[i], arr2[i])
        ssim_w.append(ssim)
    ssim_w = np.asarray(ssim_w, dtype=np.float64)

    ssim_avg = (ssim_d + ssim_h + ssim_w) / 3

    if size_average:
        return ssim_avg.mean()
    else:
        return ssim_avg


def cast_to_image(tensor, normalize=True):
    # tensor range: [0, 1]
    # Conver to PIL Image and then np.array (output shape: (H, W))
    if torch.is_tensor(tensor):
        img = tensor.cpu().detach().numpy()
    else:
        img = tensor
    if normalize:
        img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)
    return img[..., np.newaxis]
