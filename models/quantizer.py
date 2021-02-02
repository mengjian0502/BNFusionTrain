"""
quantization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

def to_tensor(sat_val):
    is_scalar = not isinstance(sat_val, torch.Tensor)
    out = torch.tensor(sat_val) if is_scalar else sat_val.clone().detach()
    if not out.is_floating_point():
        out = out.to(torch.float32)
    if out.dim() == 0:
        out = out.unsqueeze(0)
    return is_scalar, out

def linear_quantize(input, scale, zero_point, inplace=False):
    """
    Linearly quantize the input tensor based on scale and zero point.
    https://pytorch.org/docs/stable/quantization.html
    """
    if inplace:
        input.mul_(scale).sub_(zero_point).round_()
        return input
    # print(scale)
    return torch.round(input * scale - zero_point)

def linear_dequantize(input, scale, zero_point, inplace=False):
    if inplace:
        input.add_(zero_point).div_(scale)
        return input
    return (input + zero_point) / scale

def quantizer(num_bits, saturation_min, saturation_max,
                                          integral_zero_point=True, signed=False):
    scalar_min, sat_min = to_tensor(saturation_min)
    scalar_max, sat_max = to_tensor(saturation_max)
    is_scalar = scalar_min and scalar_max

    if scalar_max and not scalar_min:
        sat_max = sat_max.to(sat_min.device)
    elif scalar_min and not scalar_max:
        sat_min = sat_min.to(sat_max.device)

    if any(sat_min > sat_max):
        raise ValueError('saturation_min must be smaller than saturation_max')

    n = 2 ** num_bits - 1

    # Make sure 0 is in the range
    sat_min = torch.min(sat_min, torch.zeros_like(sat_min))
    sat_max = torch.max(sat_max, torch.zeros_like(sat_max))

    diff = sat_max - sat_min
    diff[diff == 0] = n

    scale = n / diff
    zero_point = scale * sat_min
    if integral_zero_point:
        zero_point = zero_point.round()
    if signed:
        zero_point += 2 ** (num_bits - 1)
    if is_scalar:
        return scale.item(), zero_point.item()
    return scale, zero_point

def symmetric_linear_quantization_params(num_bits, saturation_val, restrict_qrange=False):
    is_scalar, sat_val = to_tensor(saturation_val)
    if any(sat_val < 0):
        raise ValueError('Saturation value must be >= 0')

    if restrict_qrange:
        n = 2 ** (num_bits - 1) - 1
    else:
        n = (2 ** num_bits - 1) / 2

    scale = n / sat_val
    zero_point = torch.zeros_like(scale)

    if is_scalar:
        return scale.item(), zero_point.item()
    return scale, zero_point

def get_scale(input, z):
    c1, c2 = 1/z[0], z[1]/z[0]

    std = input.std()
    mean = input.abs().mean()
    
    q_scale = c1 * std - c2 * mean # change the plus sign to minus sign for the correct version of sawb alpha_w
    
    return q_scale 

class STEQuantFM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, zero_point, dequantize, inplace):
        if inplace:
            ctx.mark_dirty(input)

        output = linear_quantize(input, scale, zero_point)
        if dequantize:
            output = linear_dequantize(output, scale, zero_point)  
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Straight Through Estimator
        """
        return grad_output, None, None, None, None

class STEQuantWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, zero_point, dequantize, inplace, nbit, restrict_range):
        if inplace:
            ctx.mark_dirty(input) 
        output = linear_quantize(input, scale, zero_point)
        if restrict_range is False:
            if len(torch.unique(output)) == 2**nbit + 1:
                n = (2 ** nbit) / 2
                output = output.clamp(-n, n-1)
        if dequantize:
            output = linear_dequantize(output, scale, zero_point)  
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Straight Through Estimator
        """
        return grad_output, None, None, None, None, None, None

