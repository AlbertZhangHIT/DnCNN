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



