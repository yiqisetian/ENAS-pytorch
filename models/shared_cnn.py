import numpy as np
from collections import defaultdict, deque

import torch as t
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.shared_base import *
from utils import get_logger, get_variable, keydefaultdict

logger = get_logger()


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv5x5(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=1, bias=False)

def conv(kernel, planes):
    if kernel == 3:
        _conv = conv3x3
    elif kernel == 5:
        _conv = conv5x5
    else:
        raise NotImplemented("Unkown kernel size: {0}".format(kernel))

    return nn.Sequential(
            nn.ReLU(inplace=True),
            _conv(planes, planes),
            nn.BatchNorm2d(planes),
    )


class CNN(SharedModel):
    def __init__(self, args, images):
        super(CNN, self).__init__()

        self.args = args
        self.images = images
        #defaultdict的作用是在于，当字典里的key不存在但被查找时，返回的不是keyError而是一个默认值
        self.w_c, self.w_h = defaultdict(dict), defaultdict(dict)
        self.reset_parameters()

        self.conv = defaultdict(dict)
        for idx in range(args.num_blocks):
            for jdx in range(idx+1, args.num_blocks):
                self.conv[idx][jdx] = conv()

        raise NotImplemented("In progress...")

    def forward(self, inputs, dag):
        pass #空语句，pass 不做任何事情，一般用做占位语句。

    def get_f(self, name):
        name = name.lower()
        return name

    def get_num_cell_parameters(self, dag):
        pass

    def reset_parameters(self):
        pass
