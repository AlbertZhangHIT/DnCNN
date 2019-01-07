import os, shutil
import argparse
import time, datetime
import numpy as np
import math
from tqdm import tqdm
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as tf
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, random_split

import _helpers as helpers
from _parsers import parser
import utils.mytransform as mtf
from utils.dataset import DatasetFromH5PY
from utils.measure import batch_PSNR, batch_SNR
from utils.meter import AverageMeter
from models import *
from loss import *

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

blind_noise = [0, 55]
args = parser.parse_args()

args.has_cuda = torch.cuda.is_available()
device = 'cuda' if args.has_cuda else 'cpu'
args.data_workers=1

if args.log_dir is not None:
	os.makedirs(args.log_dir, exist_ok=True)
else:
	args.log_dir = os.path.join('./logs/', 
		'{0:%Y-%m-%dT%H%M%S}'.format(datetime.datetime.now()))
cudnn.benchmark = True

print('Arguments:')
for p in vars(args).items():
	print('  ', p[0]+': ',p[1])
print('\n')

# Data loaders
data_set = DatasetFromH5PY(args.dataset_train, 
					mtf.Compose([mtf.ToTensor(255)]), 
					mtf.Compose([mtf.ToTensor(255)]))
numValSamples = math.floor(len(data_set)*0.1)
val_set, train_set = random_split(data_set, [numValSamples, len(data_set)-numValSamples])
train_loader = DataLoader(dataset=train_set, num_workers=args.data_workers, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_set, num_workers=args.data_workers, batch_size=args.batch_size, shuffle=False)

print("Train Samples: %d, Validate Samples: %d"%(len(data_set)-numValSamples, numValSamples))

training_logger, testing_logger = helpers.loggers(args)
# Initialize model
model = DnCNN(args.depth, args.n_channels, args.img_channels, args.kernel_size)

# Loss function and regularizers
criterion = nn.MSELoss()

# Move to device
criterion = criterion.to(device)
model = model.to(device)

# Optimizer and learning rate schedule
optimizer = optim.Adam(model.parameters(), lr=args.lr, 
	betas=tuple(args.betas), weight_decay=args.weight_decay)
scheduler = helpers.scheduler(optimizer, args)


class TrainError(Exception):
	"""Exception raised for error during training."""
	pass

def train(epoch, ttot):
	model.train()
	# Run through the training data
	if args.has_cuda:
		torch.cuda.synchronize()
	tepoch = time.perf_counter()
	el_loss = AverageMeter()
	el_measure = AverageMeter()
	data_stream = tqdm(enumerate(train_loader, 1))
	for batch_idx, data in data_stream:
		# unpack the data if needed.
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

		# where are we.
		dataset_size = len(train_set)
		dataset_batches = len(train_loader)
		iteration = (epoch-1) * (dataset_size // args.batch_size) + batch_idx + 1

		x, y = x.to(device), y.to(device)

		optimizer.zero_grad()
		y_hat = model(x)
		loss = criterion(y_hat, y)

		if np.isnan(loss.data.item()):
			raise(TrainError('model returned nan during training'))
		# compute gradient and step
		loss.backward()
		if args.clip_grad:
			nn.utils.clip_grad_norm_(model.parameters(), args.clipping)
		optimizer.step()

		# measure performance and record loss
		model.eval()
		y_tilde = model(x)
		y_tilde = torch.clamp(y_tilde, 0., 1.)
		l_measure = batch_SNR(y_tilde, y, 1.) if args.snr else batch_PSNR(y_tilde, y, 1.)
		el_measure.update(l_measure, y.size(0))
		el_loss.update(loss.data.item(), y.size(0))

		# update the progress.
		data_stream.set_description((
			'epoch: {epoch}/{epochs} | '
			'iteration: {iteration} | '
			'progress: [{trained}/{total}] ({progress:.0f}%) | '
			'loss: {loss.val:.4f}({loss.avg:.4f}) '
			'{measure}: {mvalue.val:.3f}({mvalue.avg:.3f})'
		).format(
			epoch=epoch,
			epochs=args.epochs,
			iteration=iteration,
			trained=batch_idx*args.batch_size,
			total=dataset_size,
			progress=(100.*batch_idx/dataset_batches),
			loss=el_loss,
			measure=('SNR' if args.snr else 'PSNR'),
			mvalue=el_measure,
		))

		if args.log_dir is not None:
			training_logger(el_loss.avg, optimizer, tepoch, ttot)

	return ttot + time.perf_counter() - tepoch

def test(epoch, ttot):
	model.eval()
	with torch.no_grad():
		test_loss = AverageMeter()
		test_measure = AverageMeter()
		for batch_idx, data in enumerate(val_loader, 1):
			# unpack the data if needed.
			try:
				x, y = data
				x, y = x.unsqueeze(dim=1), y.unsqueeze(dim=1)
			except ValueError:
				y = data
				y = y.unsqueeze(dim=1)
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
			# where are we.
			dataset_size = len(val_set)
			dataset_batches = len(val_loader)
			iteration = (epoch-1) * (dataset_size // args.batch_size) + batch_idx + 1

			x, y = x.to(device), y.to(device)
			y_tilde = model(x)
			loss = criterion(y_tilde, y)
			y_tilde = torch.clamp(y_tilde, 0., 1.)
			l_measure = batch_SNR(y_tilde, y, 1.) if args.snr else batch_PSNR(y_tilde, y, 1.)
			test_loss.update(loss.data.item(), y.size(0))
			test_measure.update(l_measure, y.size(0))
			# Report results
			if args.log_dir is not None:
				testing_logger(epoch, test_loss.avg, test_measure.avg, optimizer)

	print('[Epoch %2d] Average test loss: %.3f, Average test %s: %.3f'
		%(epoch, test_loss.avg, 'SNR' if args.snr else 'PSNR', test_measure.avg))

	return test_loss.avg, test_measure.avg

def main():
	save_path = args.log_dir if args.log_dir is not None else '.'

	# Save argument values to yaml file
	args_file_path = os.path.join(save_path, 'args.yaml')
	with open(args_file_path, 'w') as f:
		yaml.dump(vars(args), f, default_flow_style=False)

	save_model_path = os.path.join(save_path, 'checkpoint.pth.tar')
	best_model_path = os.path.join(save_path, 'best.pth.tar')

	best_measure = 0.
	t = 0.
	for e in range(1, args.epochs+1):
		# Update the learning rate
		scheduler.step()

		#try:
		t = train(e, t)

		loss, c_measure = test(e, time)

		torch.save({'epoch': e,
					'state_dict': model.state_dict(),
					'(p)snr': c_measure,
					'loss': loss,
					'optimizer': optimizer.state_dict()}, save_model_path)
		if c_measure >= best_measure:
			shutil.copyfile(save_model_path, best_model_path)
			best_measure = c_measure

	print('Best %s: %.3f' %('SNR' if args.snr else 'PSNR', best_measure))

if __name__ =='__main__':
	try:
		main()
	except KeyboardInterrupt:
		print('Keyboard interrupt; exiting')