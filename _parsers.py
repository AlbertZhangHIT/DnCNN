"""This module parses all command line arguments to main.py"""
import argparse
import numpy as np

parser = argparse.ArgumentParser('Testbed for DNCNN optimization in PyTorch')
parser.add_argument("--dataset-train", type=str, default="", help="Training images root path(.h5)")
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
	help='how many batches to wait before logging training status (default: 100)')
parser.add_argument('--log-dir', type=str, default=None, metavar='DIR',
	help='directory for outputting log files. (default: ./logs/DATASET/MODEL/TIMESTAMP/')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
    help='number of epochs to train (default: 100)')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
    help='input batch size for testing (default: 1)')
parser.add_argument('--blind', action='store_true',
    help="Blind denoising")
parser.add_argument('--add-noise', action='store_true', 
    help="add noise")
parser.add_argument('--noise-level', type=float, default=0, 
    help='noise level')
parser.add_argument('--snr', action='store_true', 
    help='measure performance, default psnr otherwise snr')
parser.add_argument('--visualize', action='store_true',
	help='Visualize the training process')

group0 = parser.add_argument_group('Optimizer hyperparameters')
group0.add_argument('--batch-size', type=int, default=16, 
	help='Input batch size for training. (default: 16)')
group0.add_argument('--lr', type=float, default=0.1, metavar='LR',
	help='Initial step size. (default: 0.1)')
group0.add_argument('--lr-schedule', type=str, metavar='[[epoch,ratio]]',
	default='[[0,1],[50,0.2],[100,0.01],[150,0.005]]', help='List of epochs and multiplier '
	'for changing the learning rate (default: [[0,1],[50,0.2],[100,0.01],[150,0.005]]). ')
group0.add_argument('--betas', nargs=2, type=float, default=[0.9, 0.999])
group0.add_argument('--weight-decay', type=float, default=0)
group0.add_argument('--clip-grad', action='store_true', help='Clip gradient?')
group0.add_argument('--clipping', type=float, default=0.1)

group1 = parser.add_argument_group('Model hyperparameters')
group1.add_argument('--depth', type=int, default=17, help='depth of DnCNN, default: 17')
group1.add_argument('--img-channels', type=int, default=1, help='channels of input images, default: 1')
group1.add_argument('--n-channels', type=int, default=64, help='inner channels of DnCNN, default: 64')
group1.add_argument('--kernel-size', type=int, default=3, help='kernel size, default: 3')
