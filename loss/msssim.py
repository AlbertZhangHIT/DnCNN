import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import exp
from .ssim import ssimFunction, create_window
# Reference:
# Hang Zhao et al., Loss Functions for Image Restoration With Neural Networks. IEEE TCI, 2017.
# https://github.com/jorge-pessoa/pytorch-msssim/blob/master/pytorch_msssim/__init__.py


def msssimFunction(img1, img2, window_size=11, window=None, size_average=True, val_range=None, normalize=False):
	weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(img1.device)
	levels = weights.size()[0]
	mssim = []
	mcs = []
	for _ in range(levels):
		sim, cs = ssimFunction(img1, img2, window_size=window_size, window=window, size_average=size_average, full=True, val_range=val_range)
		mssim.append(sim)
		mcs.append(cs)
		img1 = F.avg_pool2d(img1, (2, 2))
		img2 = F.avg_pool2d(img2, (2, 2))

	mssim = torch.stack(mssim)
	mcs = torch.stack(mcs)

	# Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
	if normalize:
		mssim = (mssim + 1) / 2
		mcs = (mcs + 1) / 2

	pow1 = mcs ** weights
	pow2 = mssim ** weights
	output = torch.prod(pow1[:-1] * pow2[-1])
	return output

class OMSSSIMLoss(nn.Module):
	def __init__(self, channel=1, win_size=11, size_average=True, val_range=None):
		super(OMSSSIMLoss, self).__init__()
		self.size_average = size_average
		self.win_size = win_size
		self.val_range = val_range
		self.window = create_window(window_size=win_size, channel=channel, sigma=1.5)

	def forward(self, input, target):
		self.window = self.window.to(input.device)
		return msssimFunction(input, target, window_size=self.win_size, window=self.window, size_average=self.size_average)

class OMSSSIML1Loss(nn.Module):
	def __init__(self, channel=1, alpha=0.84, size_average=True):
		super(OMSSSIML1Loss, self).__init__()
		self.size_average = size_average
		self.alpha = alpha
		self.channel = channel
		self.msssimloss = OMSSSIMLoss(channel=channel, size_average=size_average)
		self.l1loss = nn.L1Loss(size_average=size_average)

	def forward(self, input, target):
		msssim_loss = self.msssimloss(input, target)
		l1_loss = self.l1loss(input, target)
		return self.alpha * msssim_loss + (1. - self.alpha) * l1_loss

class OMSSSIML2Loss(nn.Module):
	def __init__(self, channel=1, alpha=0.84, size_average=True):
		super(OMSSSIML2Loss, self).__init__()
		self.size_average = size_average
		self.alpha = alpha
		self.channel = channel
		self.msssimloss = OMSSSIMLoss(channel=channel, size_average=size_average)
		self.l2loss = nn.MSELoss(size_average=size_average)

	def forward(self, input, target):
		msssim_loss = self.msssimloss(input, target)
		l2_loss = self.L2Loss(input, target)
		return self.alpha * msssim_loss + (1. - self.alpha) * l2_loss

class MSSSIMLoss(nn.Module):
	# A loss layer that calculates (1 - MS_SSIM) loss
	def __init__(self, sigma=[.5, 1., 2., 3., 4.], win_size=11, size_average=True, val_range=None):
		super(MSSSIMLoss, self).__init__()
		self.size_average = size_average
		self.sigma = sigma
		self.win_size = win_size
		self.val_range = val_range

		self._gaussian_filter(win_size=win_size)

	def forward(self, input, target):
		if input.size() != target.size():
			raise RuntimeError('Input images must have the same shape (%s vs. %s).',
						input.size(), target.size())
		if input.dim() != 4:
			raise RuntimeError('Input images must have 4 dimensions, not %d.',
							input.dim())
		(_, channel, _, _) = input.size()
		if self.val_range is None:
			if torch.max(input) > 128:
				max_val = 255
			else:
				max_val = 1
			if torch.min(input) < -0.5:
				min_val = -1
			else:
				min_val = 0
			L = max_val - min_val
		else:
			L = self.val_range
		C1 = (0.01 * L) ** 2
		C2 = (0.03 * L) ** 2

		padding = 0
		window = self.filt.type_as(input)
		mux = F.conv2d(input, window, padding=padding, groups=channel)
		muy = F.conv2d(target, window, padding=padding, groups=channel)
		sigmax2 = F.conv2d(input * input, window, padding=padding, groups=channel) - mux ** 2
		sigmay2 = F.conv2d(target* target, window, padding=padding, groups=channel) - muy ** 2
		sigmaxy = F.conv2d(target * input, window, padding=padding, groups=channel) - mux * muy
		l = (2 * mux * muy + C1) / (mux ** 2 + muy ** 2 + C1)
		cs = (2 * sigmaxy + C2) / (sigmax2 + sigmay2 + C2)		

		msssim_map = l[:,-1,:,:] * torch.prod(cs, dim=1)

		if self.size_average:
			return 1. - msssim_map.mean()
		else:
			return (1. - msssim_map.mean(1).mean(1)).sum()

	def _gaussian_filter(self, win_size=11):
		radius = win_size // 2
		offset = 0.0
		start, stop = -radius, radius+1
		sigma = self.sigma
		if win_size%2 == 0:
			offset = 0.5
			stop -= 1
		x1, x2 = np.mgrid[offset+start:stop, offset+start:stop].astype('float32')
		assert len(x1) == win_size
		ws = np.empty((len(sigma), win_size, win_size))
		for i in range(len(sigma)):
			w = np.exp(-(x1**2 + x2**2)/(2.0 * sigma[i]**2))
			w = w / np.sum(w)
			ws[i,:,:] = w
		self.filt = torch.from_numpy(ws).unsqueeze_(dim=1)	

class MSSSIML1Loss(nn.Module):
	# A loss layer that calculates alpha * (1 - MS_SSIM) + (1 - alpha) * L1 loss.
	def __init__(self, sigma=[0.5, 1., 2., 4., 8.], alpha=0.025, size_average=True):
		super(MSSSIML1Loss, self).__init__()
		self.alpha = alpha
		self.msssim = MSSSIMLoss(sigma=sigma, size_average=size_average)
		self.l1 = nn.L1Loss(size_average=size_average)

	def forward(self, input, target):
		# ms_ssim loss
		msssim_loss = self.msssim(input, target)
		l1_loss = self.l1(input, target)

		return self.alpha * msssim_loss + (1. - self.alpha) * l1_loss

class MSSSIML2Loss(nn.Module):
	# A loss layer that calculates alpha * (1 - MS_SSIM) + (1 - alpha) * L2 loss.
	def __init__(self, sigma=[0.5, 1., 2., 4., 8.], alpha=0.025, size_average=True):
		super(MSSSIML2Loss, self).__init__()
		self.alpha = alpha
		self.msssim = MSSSIMLoss(sigma=sigma, size_average=size_average)
		self.l2 = nn.MSELoss(size_average=size_average)

	def forward(self, input, target):
		# ms_ssim loss
		msssim_loss = self.msssim(input, target)
		l2_loss = self.l2(input, target)

		return self.alpha * msssim_loss + (1. - self.alpha) * l2_loss	
