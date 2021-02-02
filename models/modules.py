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
        self.gamma = nn.Parameter(torch.empty(self.num_features))
        self.beta = nn.Parameter(torch.empty(self.num_features))
        self.register_buffer('running_mean', torch.zeros(self.num_features))
        self.register_buffer('running_var', torch.ones(self.num_features))

        self.momentum=momentum
        self.track_running_stats = track_running_stats
        self.register_buffer("num_batches_tracked", torch.tensor(0))

    def forward(self, input):
        def reshape_to_activation(inputs):
            return inputs.reshape(1, -1, 1, 1)

        # batch norm statistics
        if self.training:
            out_ = F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

            batch_mean = out_.mean([0,2,3])
            batch_var = out_.var([0,2,3])
            batch_std = torch.sqrt(batch_var + self.eps)

            with torch.no_grad():
                self.running_mean = self.momentum * batch_mean + (1 - self.momentum) * self.running_mean
                self.running_var = self.momentum * batch_var  + (1 - self.momentum) * self.running_var
        
        running_mean = self.running_mean
        running_var = self.running_var
        running_std = torch.sqrt(running_var + self.eps)

        bn_scale = self.gamma / running_std
        bias = self.beta -  self.gamma * running_mean / running_std
        bn_bias = bias.reshape(-1)
        
        # merge the BN weight to the weight
        weight = self.weight * bn_scale[:, None, None, None]

        # convolution
        out = F.conv2d(input, weight, bn_bias, self.stride, self.padding, self.dilation, self.groups)

        if self.training:
            out *= reshape_to_activation(running_std / batch_std)
            out += reshape_to_activation(self.gamma * (running_mean / running_std - batch_mean / batch_std))
        return out
