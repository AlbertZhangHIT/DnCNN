import torch
import torch.nn as nn
import torchvision.utils as utils
from PIL import Image
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize

import os
import matplotlib.pyplot as plt

def display_transform():
    return Compose([
        ToPILImage(),
        Resize(300),
        ToTensor()
    ])

def cnn_kernel_visualize(model, selected_layer, out_path='.', nrow=4, normalize=False, saveImage=False, cmap='gray'):
	if list(model.children()) == []:
		print("The selected module has no children. Visualize this module instead.")
		if isinstance(model, nn.Conv2d):
			kernels = model.weight.data.clone().cpu()
			print('kernel size: ', kernels.size())
			channel = 0
			if kernels.size(1) > 1:
				for subkernel in kernels:
					print("channel: ", str(channel))
					subkernel = subkernel.unsqueeze(dim=0).transpose(0, 1)
					sk = torch.stack([display_transform()(x) for x in subkernel], dim=0)
					grid = utils.make_grid(sk, nrow=nrow, padding=5, normalize=normalize, pad_value=0)
					utils.save_image(grid, os.path.join(out_path, 'kernels_channel_' + str(channel) + '.png'))
					channel += 1
					img = grid.mul(255).clamp(0,255).byte().permute(1, 2, 0).cpu().numpy()
					fig = plt.figure(figsize=(12,12))
					ax = fig.add_subplot(111)
					ax.imshow(img, cmap=cmap)
					plt.show()
					if saveImage:
						fig.savefig(os.path.join(out_path, 'kernels_channel_' + str(channel) + '.png'))



			else:
				sk = torch.stack([display_transform()(x) for x in kernels], dim=0)
				grid= utils.make_grid(sk, nrow=nrow, padding=5, normalize=normalize, pad_value=0)
				print(grid.size())
				utils.save_image(grid, os.path.join(out_path, 'kernels_' + str(selected_layer) + '.png'))	

				img = grid.mul(255).clamp(0,255).byte().permute(1, 2, 0).cpu().numpy()
				fig = plt.figure(figsize=(12,12))
				ax = fig.add_subplot(111)
				ax.imshow(img, cmap=cmap)
				plt.show()
				if saveImage:
					fig.savefig(os.path.join(out_path, 'kernels_' + str(channel) + '.png'))
	
	else:
		for name, module in model.named_children():
			print(type(name), name)
			if name == str(selected_layer) and isinstance(module, nn.Conv2d):
				kernels = module.weight.data.clone().cpu()
				print('kernel size: ', kernels.size())
				channel = 0
				if kernels.size(1) > 1:
					for subkernel in kernels:
						print("channel: ", str(channel))
						subkernel = subkernel.unsqueeze(dim=0).transpose(0, 1)
						sk = torch.stack([display_transform()(x) for x in subkernel], dim=0)
						grid = utils.make_grid(sk, nrow=nrow, padding=5, normalize=normalize, pad_value=0)
						utils.save_image(grid, os.path.join(out_path, 'kernels_channel_' + str(channel) + '.png'))
						channel += 1	

						img = grid.mul(255).clamp(0,255).byte().permute(1, 2, 0).cpu().numpy()
						fig = plt.figure(figsize=(12,12))
						ax = fig.add_subplot(111)
						ax.imshow(img, cmap=cmap)
						plt.show()
						if saveImage:
							fig.savefig(os.path.join(out_path, 'kernels_channel_' + str(channel) + '.png'))				
				else:
					sk = [display_transform()(x) for x in kernels]
					grid = utils.make_grid(sk, nrow=nrow, padding=5, normalize=normalize, pad_value=0)
					utils.save_image(grid, os.path.join(out_path, 'kernels_' + str(selected_layer) + '.png'))	

					img = grid.mul(255).clamp(0,255).byte().permute(1, 2, 0).cpu().numpy()
					fig = plt.figure(figsize=(12,12))
					ax = fig.add_subplot(111)
					ax.imshow(img, cmap=cmap)
					plt.show()
					if saveImage:
						fig.savefig(os.path.join(out_path, 'kernels_channel_' + str(channel) + '.png'))
			else:
				pass