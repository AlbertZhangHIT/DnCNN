import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import exp
# SSIM calculation
# Reference:
# https://github.com/tensorflow/models/blob/master/research/compression/image_encoder/msssim.py
# https://github.com/jorge-pessoa/pytorch-msssim/blob/master/pytorch_msssim/__init__.py

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1, sigma=1.5):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssimFunction(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret

class SSIMLoss(nn.Module):
	#  A loss layer that calculates SSIM loss.
	def __init__(self, radius=5., win_size=11, size_average=True, mode=''):
		super(SSIMLoss, self).__init__()
		self.size_average = size_average
		self.C1 = 0.01 ** 2
		self.C2 = 0.03 ** 2
		self.radius = radius
		# initialize thte gaussian filter
		if mode == 'matlab':
			self._matlab_gaussian_filter()
		else:
			self._gaussian_filter(filtSize=win_size)


	def forward(self, input, target):
		if input.size() != target.size():
			raise RuntimeError('Input images must have the same shape (%s vs. %s).',
						input.size(), target.size())
		if input.dim() != 4:
			raise RuntimeError('Input images must have 4 dimensions, not %d.',
							input.dim())

		padding = self.win_size // 2
		groups = input.size(1)
		window = self.filt.expand([input.size(1), 1, self.filt.size(0), self.filt.size(1)]).type_as(input)
		mux = F.conv2d(input, window, padding=padding, groups=groups)
		muy = F.conv2d(target, window, padding=padding, groups=groups)
		sigmax2 = F.conv2d(input*input, window, padding=padding, groups=groups) - mux ** 2
		sigmay2 = F.conv2d(target*target, window, padding=padding, groups=groups) - muy ** 2
		sigmaxy = F.conv2d(target*input, window, padding=padding, groups=groups) - mux * muy
		l = (2 * mux * muy + self.C1) / (mux ** 2 + muy ** 2 + self.C1)
		cs = (2 * sigmaxy + self.C2) / (sigmax2 + sigmay2 + self.C2)

		ssim_map = l * cs

		if self.size_average:
			return 1. - ssim_map.mean()
		else:
			return (1. - ssim_map.mean(1).mean(1).mean(1)).sum()

	def _gaussian_filter(self, filtSize):
		filtRadius = filtSize // 2
		offset = 0.0
		start, stop = -filtRadius, filtRadius+1
		sigma = self.radius
		if filtSize%2 == 0:
			offset = 0.5
			stop -= 1
		x1, x2 = np.mgrid[offset+start:stop, offset+start:stop].astype('float32')
		assert len(x1) == filtSize
		gaussFilt = np.exp(-(x1**2 + x2**2)/(2.0 * sigma**2))
		gaussFilt = gaussFilt / np.sum(gaussFilt)
		self.filt = torch.from_numpy(gaussFilt)
		self.win_size = filtSize

	def _matlab_gaussian_filter(self):
		# Reference: ssim function in matlab
		eps = 1e-16
		filtRadius = math.ceil(self.radius * 3)
		filtSize = 2*filtRadius + 1
		x1, x2 = np.mgrid[-filtRadius:filtRadius+1, -filtRadius:filtRadius+1].astype('float32')
		arg = - (x1 * x1 + x2 * x2) / (2*self.radius*self.radius)
		gaussFilt = np.exp(arg)
		gaussFilt[gaussFilt < eps*np.max(gaussFilt)] = 0.
		sumFilt = np.sum(gaussFilt)
		if sumFilt != 0.:
			gaussFilt /= sumFilt

		self.filt = torch.from_numpy(gaussFilt)
		self.win_size = filtSize

