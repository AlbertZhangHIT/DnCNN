import os
import os.path
import math
import numpy as np
import random
import h5py
import torch
import cv2
import torch.utils.data as udata


class DatasetFromH5PY(udata.Dataset):
	def __init__(self, h5Path, transform=None, target_transform=None):
		super(DatasetFromH5PY, self).__init__()

		self.h5f = h5Path
		hf = h5py.File(h5Path, 'r')
		self.keys = list(hf.keys())
		random.shuffle(self.keys)
		hf.close()

		self.transform = transform
		self.target_transform = target_transform
	def __len__(self):
		return len(self.keys)

	def __getitem__(self, index):
		hf = h5py.File(self.h5f, 'r')
		key = self.keys[index]
		g = hf.get(key)
		sample = np.array(g.get('data'))
		label = np.array(g.get('label'))
		hf.close()

		if label == None:
			if self.transform is not None:
				sample = self.transform(sample)
				if sample.dim() < 3:
					sample = sample.unsqueeze(0) 
			return sample
		else:
			if self.transform is not None:
				sample = self.transform(sample)
				if sample.dim() < 3:
					sample = sample.unsqueeze(0) 
			if self.target_transform is not None:
				label = self.target_transform(label)
				if label.dim() < 3:
					label = label.unsqueeze(0) 		
			return sample, label

class DatasetFromH5PYOne(udata.Dataset):
	def __init__(self, h5Path, feature='data', transform=None):
		super(DatasetFromH5PYOne, self).__init__()

		self.h5f = h5Path
		hf = h5py.File(h5Path, 'r')
		self.keys = list(hf.keys())
		random.shuffle(self.keys)
		hf.close()

		self.transform = transform
		self.feature = feature
	def __len__(self):
		return len(self.keys)

	def __getitem__(self, index):
		hf = h5py.File(self.h5f, 'r')
		key = self.keys[index]
		g = hf.get(key)
		sample = np.array(g.get(self.feature))
		hf.close()

		if self.transform is not None:
			sample = self.transform(sample)
		return sample

class DatasetFromHdf5(udata.Dataset):
	def __init__(self, h5Path):
		super(DatasetFromHdf5, self).__init__()
		hf = h5py.File(h5Path)
		self.data = hf.get('data')
		self.target = hf.get('label')

	def __getitem__(self, index):
		return torch.from_numpy(self.data[index,:,:,:]).float(), torch.from_numpy(self.target[index,:,:,:]).float()

	def __len__(self):
		return self.data.shape[0]
