import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchvision.models as models
import torchvision.utils as vutils
from collections import Iterable
import math
import numpy as np

EPSILON = 1e-20

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        n = m.in_features + m.out_features
        m.weight.data.normal_(0.0, math.sqrt(2.0 / n))
        try: # maybe bias=False
            m.bias.data.zero_()
        except Exception as e:
            pass

class OptimizerManager:
    def __init__(self, optims):
        self.optims = optims if isinstance(optims, Iterable) else [optims]
    def __enter__(self):
        for op in self.optims:
            op.zero_grad()
    def __exit__(self, exceptionType, exception, exceptionTraceback):
        for op in self.optims:
            op.step()
        self.optims = None # release reference, to avoid imexplicit reference
        if exceptionTraceback:
            print(exceptionTraceback)
            return False
        return True
    
    
class OptimWithSheduler:
    def __init__(self, optimizer, scheduler_func):
        self.optimizer = optimizer
        self.scheduler_func = scheduler_func
        self.global_step = 0.0
        for g in self.optimizer.param_groups:
            g['initial_lr'] = g['lr']
    def zero_grad(self):
        self.optimizer.zero_grad()
    def step(self):
        for g in self.optimizer.param_groups:
            g['lr'] = self.scheduler_func(step=self.global_step, initial_lr=g['initial_lr'])
        self.optimizer.step()
        self.global_step += 1
    
class TrainingModeManager:
    def __init__(self, nets, train=False):
        self.nets = nets if isinstance(nets, Iterable) else [nets]
        self.modes = [net.training for net in nets]
        self.train = train
    def __enter__(self):
        for net in self.nets:
            net.train(self.train)
    def __exit__(self, exceptionType, exception, exceptionTraceback):
        for (mode, net) in zip(self.modes, self.nets):
            net.train(mode)
        self.nets = None # release reference, to avoid imexplicit reference
        if exceptionTraceback:
            print(exceptionTraceback)
            return False
        return True
    
def variable_to_numpy(x):
    ans = x.cpu().data.numpy()
    if torch.numel(x) == 1:
        # make sure ans has no shape. (float requires number rather than ndarray)
        return float(np.sum(ans))
    return ans

def merge_ncwh_to_one_image(x):
    try:
        # maybe min = max, zero division
        nrow = int(math.ceil(x.size(0) ** 0.5))
        x = vutils.make_grid(x.data, nrow=nrow, padding=0, normalize=True) # torch.cuda.FloatTensor, [3, 224, 224]
        x.unsqueeze_(-1)
        x = x.permute(3, 1, 2, 0)
        return x.cpu().numpy()
    except Exception as e:
        return None

def addkey(diction, key, global_vars):
    diction[key] = global_vars[key]
    
def track_scalars(logger, names, global_vars):
    values = {}
    for name in names:
        addkey(values, name, global_vars)
    for k in values:
        values[k] = variable_to_numpy(values[k])
    for k, v in values.items():
        logger.log_scalar(k, v)
    print(values)

def track_images(logger, names, global_vars):
    values = {}
    for name in names:
        addkey(values, name, global_vars)
    for k in values:
        values[k] = merge_ncwh_to_one_image(values[k])
        if values[k] is not None:
            logger.log_images(k, values[k])
        else:
            print('images generated are of the same value!')

def post_init_module(module):
    module.apply(weight_init)
    return module.cuda()
 
def CrossEntropyLoss(label, predict_prob, class_level_weight=None, instance_level_weight=None, epsilon=EPSILON):
    N, C = label.size()
    N_, C_ = predict_prob.size()
    
    assert N == N_ and C == C_, 'fatal error: dimension mismatch!'
    
    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'
        
    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    ce = -label*torch.log(predict_prob + epsilon)
    return torch.sum(instance_level_weight * ce * class_level_weight) / float(N)