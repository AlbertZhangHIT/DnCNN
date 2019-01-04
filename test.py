import os, shutil
import argparse
import time, datetime
import numpy as np
import math
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as tf
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import utils.mytransform as mtf
from utils.measure import batch_PSNR, batch_SNR
from utils.meter import AverageMeter
from utils.functional import loadModel
from utils.imgdataset import ImageDatasetFromFolder
from utils.dataset import DatasetFromH5PY

parser = argparse.ArgumentParser('Test for DnCNN in PyTorch')
parser.add_argument("--dataset", type=str, default="")
parser.add_argument("--imgfile", action='store_true', help='whether the test dataset is image file or folder')
parser.add_argument("--test-batch-size", type=int, default=1)
parser.add_argument("--blind", action='store_true', help='Blind denoising')
parser.add_argument("--add-noise", action='store_true', help='Add noise')
parser.add_argument("--noise-level", type=float, default=0, hlep='noise level')
parser.add_argument("--snr", action='store_true', help='measure method, default psnr')
parser.add_argument("--checkpoint", type=str, default="")
parser.add_argument("--cuda", action='store_true')

group1 = parser.add_argument_group('Model hyperparameters')
group1.add_argument('--depth', type=int, default=17, help='depth of DnCNN, default: 17')
group1.add_argument('--img-channels', type=int, default=1, help='channels of input images, default: 1')
group1.add_argument('--n-channels', type=int, default=64, help='inner channels of DnCNN, default: 64')
group1.add_argument('--kernel-size', type=int, default=3, help='kernel size, default: 3')

blind_noise = [0, 55]
args = parser.parse_args()
args.has_cuda = torch.cuda.is_available()
device = 'cuda' (args.cuda and args.has_cuda) else 'cpu'

# Initialize model
model = DnCNN(args.depth, args.n_channels, args.img_channels, args.kernel_size)
model = loadModel(model, device, args.checkpoint)
# checkpoint = torch.load(args.checkpoint)['state_dict']
# if device == 'cuda':
# 	device_ids = [0]
# 	model = nn.DataParallel(model, device_ids=device_ids)
# 	cudnn.benchmark = True
# 	model.load_state_dict(checkpoint)
# else:
# 	from collections import OrderedDict
# 	new_state_dict = OrderedDict()
# 	for k, v in checkpoint.items():
# 		name = k[7:]
# 		new_state_dict[name] = v
# 	model.load_state_dict(new_state_dict)


if os.path.isdir(args.dataset) and args.imgfile:
	try:
		test_set = ImageDatasetFromFolder(args.dataset, transform=tf.ToTensor())
	except:
		print("Please ensure the dataset path are a folder containing image files.")
		
if os.path.isfile(args.dataset) and not args.imgfile:
	try:
		test_set = DatasetFromH5PY(args.dataset, mtf.ToTensor(), mtf.ToTensor())
	except:
		print("Only files in .h5 format are supported.")



test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
test_bar = tqdm(enumerate(test_loader, 1))


with torch.no_grad():	
	test_measure = AverageMeter()
	start_time = time.time()
	for batch_idx, data in test_bar:
		try:
			x, y = data
		except ValueError:
			y = data
			if args.add_noise:
				if args.blind:
					noise = torch.zeros(y.size())
					stdN = np.random.uniform(blind_noise[0], blind_noise[1], size=noise.size(0))
					for n in range(noise.size(0)):
						sizeN = noise[0, :, :, :].size()
						noise[n, :, :, :] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n]/255.)
				else:
					noise = torch.FloatTensor(y.size()).normal_(mean=0, std=args.noise_level/255.)

				x = y + noise

		x, y = x.to(device), y.to(device)

		y_hat = model(x).clamp(0., 1.)
		l_measure = batch_SNR(y_hat, y, 1.) if args.snr else batch_PSNR(y_hat, y, 1.)
		test_measure.update(l_measure, y.size(0))

		#update the progress.
		test_bar.set_description((
			'progress:[{tested}/{total}({progress:.0f}%)] | '
			'{measure}: {mvalue.val:.3f}({mvalue.avg:.3f})'
			).format(
				tested=batch_idx,
				total=len(test_set),
				progress=(100.*batch_idx/len(test_set)),
				measure=('SNR' if args.snr else 'PSNR'),
				mvalue=test_measure,
			))
elapsed_time = time.time() - start_time
print("Elapsed time: {.2f} s, Average Measure: {.3f} dB".format(elapsed_time, test_measure.avg))