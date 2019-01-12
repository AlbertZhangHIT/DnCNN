import os
import time
import ast
import bisect
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
import torch.nn as nn
import torch
import math

def scheduler(optimizer,args):
    """Return a hyperparmeter scheduler for the optimizer"""
    lS = np.array(ast.literal_eval(args.lr_schedule))
    llam = lambda e: float(lS[max(bisect.bisect_right(lS[:,0], e)-1,0),1])
    lscheduler = LambdaLR(optimizer, llam)

    return lscheduler

def loggers(args):
    """Training and test loggers from input arguments"""
    os.makedirs(args.log_dir, exist_ok=True)

    training_log = open(os.path.join(args.log_dir, 'training.csv'), 'a+')
    testing_log = open(os.path.join(args.log_dir, 'test.csv'), 'a+')

    print('%s,%s,%s,%s'% ('index','time', 'average loss', 'learning rate'),
            file=training_log)
    print('%s,%s,%s,%s,%s'%('epoch', 'time', 'average loss', 'average SNR/PSNR',
          'learning rate'),
            file=testing_log)



    class training_logger(object):
        def __init__(self):
            self.index = 0
        def __call__(self, loss, optimizer, tepoch, ttot):
            lr_current = optimizer.param_groups[0]["lr"]
            if loss is not None:
                print('%d,%.7g,%.3f,%.3e'% (self.index, time.time() - tepoch + ttot, loss, lr_current),
                    file=training_log, flush=True)
            self.index+=1


    class testing_logger(object):
        def __init__(self):
            super(testing_logger, self).__init__()

        def __call__(self, epoch, loss, mvalue, optimizer):
            lr_current = optimizer.param_groups[0]["lr"]
            print('%d,%.3f,%.3f,%.3e'%
                    (epoch, loss, mvalue, lr_current),
                    file=testing_log, flush=True)

    return training_logger(), testing_logger()


def loadModel(net, device, file, deviceIds=[0], parallel=False):
    checkpoint = torch.load(file)['state_dict']
    if device == 'cuda':
        if parallel:
            net = nn.DataPrallel(net, device_ids=deviceIds)
        net.load_state_dict(checkpoint)
    else:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:]
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)

    return net

def countParam(net):
    count = 0
    for p in net.parameters():
        count += p.data.nelement()
    return count

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant_(m.bias.data, 0.0)