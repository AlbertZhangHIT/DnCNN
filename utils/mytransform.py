import torch
import math
import random
import numbers
import numpy as np
from . import functional as F
from torch.nn.modules.utils import _ntuple

class Compose(object):
	def __init__(self, transforms):
		self.transforms = transforms

	def __call__(self, x):
		for t in self.transforms:
			x = t(x)
		return x

class CenterCrop(object):
	"""Crops a give tensor at the center.
	Args:
		size (sequence or int): Desired output size of the crop.
		dim : selected dims to perform cropping
	"""
	def __init__(self, size, dim=(1,2)):
		if isinstance(size, numbers.Number):
			self.size = (int(size), int(size))
		else:
			self.size = size
		self.dim = dim
	def __call__(self, x):
		"""
		Args:
			x (2d or 3d tensor): tensor to be cropped.
		Returns:
			tensor: Cropped tensor
		"""
		return F._CenterCrop(x, self.size, self.dim)

class RandomCrop(object):
	"""Crop the tensor at a random location
	Args:
		size (sequence or int): Desired output size of the crop.
		dim : selected dims to perform cropping
	"""
	def __init__(self, size, dim=(1,2)):
		if isinstance(size, numbers.Number):
			self.size = (int(size), int(size))
		else:
			self.size = size
		self.dim = dim
	def __call__(self, x):
		"""
		Args:
			x (2d or 3d tensor): tensor to be cropped.
		Returns:
			tensor: Cropped tensor
		"""
		return F._RandomCrop(x, self.size, self.dim)

class Scale(object):
	"""Scale the tensor
	Args:
		scale: scale parameter
	"""
	def __init__(self, scale=None):
		self.scale = scale

	def __call__(self, x):
		if self.scale is not None:
			return F._Scale(x, self.scale)
		else:
			scale = x.abs().max()
			return F._Scale(x, scale)

class ToTensor(object):
	def __init__(self, div=1.):
		self.division = div
	def __call__(self, x):
		return torch.from_numpy(x).float()/self.division

class Normalize(object):
	"""Normalize the tensor
	Args:
		dim: expected dimention indexes on which 
			normalization performs
	"""
	def __init__(self, dim=None):
		self.dim = dim

	def __call__(self, x):
		ndim = x.dim()
		if self.dim is None:
			maxValue = x.abs().max()
			return x / maxValue
		else:
			y = x
			dims = _ntuple(ndim)(self.dim)
			for dim in range(ndim):
				if dim in dims:
					maxValue, _ = x.abs().max(dim=dim, keepdim=True)
					y = y.div(maxValue.expand_as(y))
			return y
					


