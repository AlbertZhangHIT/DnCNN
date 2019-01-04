import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class L2L1Loss(nn.Module):
	# A loss layer that calculates \sum |x - d*z|_2 + \sum |z|_1
	def __init__(self, beta=0.01, size_average=True):
		super(L2L1Loss, self).__init__()
		self.beta = beta
		self.l2 = nn.MSELoss(size_average=size_average)
		self.l1 = nn.L1Loss(size_average=size_average)

	def forward(input1, sparseCoef, target):
		l2loss = self.l2(input1, target)
		l1target = torch.zeros_like(sparseCoef)
		l1loss = self.l1(sparseCoef, l1target)

		loss = l2loss + self.beta*l1Loss
		return loss