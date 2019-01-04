import math
import torch
import torch.nn as nn
import numpy as np
from skimage.measure.simple_metrics import compare_psnr



def batch_PSNR(img, imclean, data_range):
    Img = img.cpu().detach().numpy().astype(np.float32)
    Iclean = imclean.cpu().detach().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])

def batch_SNR(img, imclean, data_range):
    Img = img.detach()
    Iclean = imclean.detach()
    SNR = 0
    for i in range(Img.size(0)):
        norm = (Iclean[i,:,:,:] * Iclean[i, :, :, :]).mean()
        mse = ((Iclean[i,:,:,:] - Img[i,:,:,:]) ** 2).mean()
        SNR += 10*torch.log10(norm/mse)
    return (SNR/Img.size(0))    

def data_augmentation(image, mode):
    out = np.transpose(image, (1,2,0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2,0,1))

