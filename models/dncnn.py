import torch
import torch.nn as nn

def conv(in_channel, out_channel, kernel_size, stride=1, dilation=1, bias=False):
	padding = ((kernel_size-1) * dilation) // 2
	return nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, 
					stride=stride, padding=padding, dilation=dilation, bias=bias)

class convBlock(nn.Module):
	def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, dilation=1, bias=False, nonlinear=True, bn=False):
		super().__init__()
		self.conv = conv(in_channel, out_channel, kernel_size, stride, dilation, bias)
		self.nonlinear = nn.ReLU(inplace=True) if nonlinear else None
		self.bn = nn.BatchNorm2d(out_channel) if bn else None

	def forward(self, x):
		out = self.conv(x)
		if self.bn is not None:
			out = self.bn(out)
		if self.nonlinear is not None:
			out = self.nonlinear(out)

		return out		

class DnCNN(nn.Module):
	def __init__(self, depth=17, n_channels=64, image_channels=1, kernel_size=3):
		super().__init__()
		self.kernel_size = kernel_size
		self.depth = depth
		self.n_channels = n_channels
		self.image_channels = image_channels

		self.dncnn = self._make_layers()
		self._initialize_weights()

	def forward(self, x):
		y = x
		out = self.dncnn(x)
		return y - out

	def _make_layers(self):
		layers = []
		layers.append(convBlock(in_channel=self.image_channels, 
							out_channel=self.n_channels, 
							kernel_size=self.kernel_size,
							bias=True,
							nonlinear=True,
							bn=False))
		for _ in range(self.depth-2):
			layers.append(convBlock(in_channel=self.n_channels, 
								out_channel=self.n_channels,
								kernel_size=self.kernel_size,
								nonlinear=True,
								bn=True))
		layers.append(convBlock(in_channel=self.n_channels,
							out_channel=self.image_channels,
							kernel_size=self.kernel_size,
							nonlinear=False,
							bn=False))

		return nn.Sequential(*layers)

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.orthogonal_(m.weight)
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)