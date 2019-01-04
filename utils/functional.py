import torch
import math
import numpy as np
from torch.nn.modules.utils import _pair

def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

def _Scale(x, scale=1):
    return x/scale

def _RandomCrop(x, toSize, dim=(1,2)):
	# crop the last two dimensions
	if not(_is_tensor_image(x) or _is_numpy_image(x)):
		raise TypeError('x should be tensor or ndarray. Got {}'.format(type(x)))
	size = _pair(toSize)
	valid = {(0,1), (1,2), (0,2)}
	if not dim in valid:
		raise Exception('dim is not valid. Got {}'.format(dim))

	h, w = x.shape[dim[0]], x.shape[dim[1]]
	sh = h - size[0]
	sw = w - size[1]
	if sh <= 0 or sw <= 0:
		raise Exception('Crop size exceeds the size of image. Got ({})'.format(size))
	i = np.random.randint(0, sh)
	j = np.random.randint(0, sw)
	
	if dim == (0,1):
		return x[i:i+size[0], j:j+size[1], :]
	if dim == (1,2):
		return x[:, i:i+size[0], j:j+size[1]]
	if dim == (0,2):
		return x[i:i+size[0], :, j:j+size[1]]

def _CenterCrop(x, toSize, dim=(1,2)):
	# crop the last two dimensions
	if not(_is_tensor_image(x) or _is_numpy_image(x)):
		raise TypeError('x should be tensor or ndarray. Got {}'.format(type(x)))
	size = _pair(toSize)
	valid = {(0,1), (1,2), (0,2)}
	if not dim in valid:
		raise Exception('dim is not valid. Got {}'.format(dim))

	h, w = x.shape[dim[0]], x.shape[dim[1]]
	ch = math.floor(h/2)
	cw = math.floor(w/2)
	th1 = math.floor(size[0]/2)
	tw1 = math.floor(size[1]/2)
	th2 = math.ceil(size[0]/2)
	tw2 = math.ceil(size[1]/2)

	if dim == (0,1):
		return x[ch-th1:ch+th2, cw-tw1:cw+tw2, :]
	if dim == (1,2):
		return x[:, ch-th1:ch+th2, cw-tw1:cw+tw2]
	if dim == (0,2):
		return x[ch-th1:ch+th2, :, cw-tw1:cw+tw2]


def loadModel(net, device, file):
	checkpoint = torch.load(file)['state_dict']
	if device == 'cuda':
		device_ids = [0]
		net = nn.DataParallel(net, device_ids=device_ids)
		cudnn.benchmark = True
		net.load_state_dict(checkpoint)
	else:
		from collections import OrderedDict
		new_state_dict = OrderedDict()
		for k, v in checkpoint.items():
			name = k[7:]
			new_state_dict[name] = v
		net.load_state_dict(new_state_dict)

	return net