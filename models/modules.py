"""
PROFIT quantization with merged modules
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parameter as Parameter
import numpy as np
import torch.nn.init as init
from collections import OrderedDict

class WeightQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale):
        output_int = input.mul(scale[:, None, None, None]).round()
        output_float = output_int.div(scale[:, None, None, None])
        return output_float

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class FMQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale):
        output_int = input.mul(scale[None, :, None, None]).round()
        output_float = output_int.div(scale[None, :, None, None])
        return output_float

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class ConvBN2d(nn.Conv2d):
    r"""
    Conv2d with merged batchnorm
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                padding=0, dilation=1, groups=1, bias=False, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(ConvBN2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        # BN param
        self.eps = eps
        self.num_features = self.weight.data.size(0)
        self.gamma = Parameter(torch.empty(self.num_features))
        self.beta = Parameter(torch.empty(self.num_features))
        self.register_buffer('running_mean', torch.zeros(self.num_features))
        self.register_buffer('running_var', torch.ones(self.num_features))

        self.momentum=momentum
        self.track_running_stats = track_running_stats
        self.register_buffer("num_batches_tracked", torch.tenosr(0))

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
    
    def reset_parameters(self):
        self.reset_running_stats()
        init.uniform_(self.gamma)
        init.zeros_(self.beta)

    def forward(self, input):
        batchsize, channels, height, width = input.size()
        
        # batch norm statistics
        if self.training:
            mean = input.mean([0,2,3])
            var = input.var([0,2,3], unbiased=False)
            n = input.numel() / input.size(1)

            with torch.no_grad():
                self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
                self.running_var = self.momentum * var * n/(n-1) + (1 - self.momentum) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var
        
        std = torch.sqrt(var + self.eps)
        bn_scale = self.gamma / std
        bn_bias = self.beta + mean / std
        
        # merge the BN weight to the weight
        weight = self.weight * bn_scale[:, None, None, None]

        # convolution
        out = F.conv2d(input, weight, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out
