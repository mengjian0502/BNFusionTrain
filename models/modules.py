"""
DNN quantization with merged modules
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
        ctx.save_for_backward(input)

        output_int = input.mul(scale[:,None,None,None]).round()
        output_float = output_int.div(scale[:,None,None,None])
        return output_float

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        input, = ctx.saved_tensors
        
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input, None

class RoundQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.save_for_backward(input)
        output_int = input.mul(scale).round()
        output_float = output_int.div(scale)
        return output_float

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        input, = ctx.saved_tensors
        
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input, None

class Dorefa_quant_func(torch.autograd.Function):
    def __init__(self, nbit):
        super(Dorefa_quant_func, self).__init__()
        self.nbit = nbit
    
    def forward(self, x):
        self.save_for_backward(x)
        weight = F.hardtanh(x)

        scale = 2 ** self.nbit - 1
        weight = weight / 2 / torch.max(weight.abs()) + 1/2
        weight_q = torch.round(weight * scale) / scale
        weight_q = 2 * weight_q - 1
        return weight_q

    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class WQPROFIT(nn.Module):
    def __init__(self, wbit, num_features, channel_wise):
        self.wbit = wbit
        self.num_features = num_features
        self.channel_wise
        
        # learnable parameters
        if channel_wise:
            self.a = nn.Parameter(torch.ones(num_features))
            self.c = nn.Parameter(torch.ones(num_features))
        else:
            self.a = nn.Parameter(torch.Tensor(1.))
            self.c = nn.Parameter(torch.Tensor(1.))

    def _upadte_param(self, wbit):
        self.wbit = wbit
        max_val = self.weight.data.abs().max().item()
        self.a.data.fill_(np.log(np.exp(max_val * 0.9)-1))
        self.c.data.fill_(np.log(np.exp(max_val * 0.9)-1))

    def forward(self, input):
        if self.wbit == 32:
            input_q = input
        else:
            n_lv = 2 ** self.wbit
            scale = n_lv // 2 - 1
            
            # gradient friendly
            a = F.softplus(self.a)  # keep the learnable value positive
            c = F.softplus(self.c)

            
            if self.channel_wise:
                input = input.div(a[:, None, None, None])
                input = F.hardtanh(input, -1, 1)

                scale = torch.ones_like(self.a.data).mul(scale)
                input_q = WeightQuant.apply(input, scale)
                input_q = input_q.mul(c[:,None,None,None])
            else:
                input = input.div(a)
                input = F.hardtanh(input, -1, 1)

                input_q = RoundQuant.apply(input, scale)
                input_q = input_q.mul(c)
        return input_q

class WQ(nn.Module):
    def __init__(self, wbit, num_features, channel_wise=True):
        super(WQ, self).__init__()
        self.wbit = wbit
        self.num_features = num_features
        self.register_buffer('alpha_w', torch.ones(num_features))
        self.channel_wise = channel_wise

    def forward(self, input):
        z_typical = {'4bit': [0.077, 1.013], '8bit':[0.027, 1.114]}
        z = z_typical[f'{int(self.wbit)}bit']
        n_lv = 2 ** (self.wbit - 1) - 1

        if self.channel_wise:
            m = input.abs().mean([1,2,3])
            std = input.std([1,2,3])
            
            self.alpha_w = 1/z[0] * std - z[1]/z[0] * m
            lb = self.alpha_w.mul(-1.)
            ub = self.alpha_w

            # channel-wise clamp
            input = torch.max(torch.min(input, ub[:,None,None,None]), lb[:,None,None,None])
            scale = n_lv / self.alpha_w

            w_float = WeightQuant.apply(input, scale)
        else:
            m = input.abs().mean()
            std = input.std()
            
            self.alpha_w = 1/z[0] * std - z[1]/z[0] * m 
            input = input.clamp(-self.alpha_w.item(), self.alpha_w.item())
            scale = n_lv / self.alpha_w

            w_float = RoundQuant.apply(input, scale)
        return w_float

class WQDoreFa(nn.Module):
    def __init__(self, wbit):
        super(WQDoreFa, self).__init__()
        self.wbit = wbit

    def forward(self, input):
        input_q = Dorefa_quant_func(self.wbit)(input)
        return input_q

class AQ(nn.Module):
    def __init__(self, abit, num_features, alpha_init):
        super(AQ, self).__init__()
        self.abit = abit
        self.alpha = nn.Parameter(torch.Tensor([alpha_init]))

    def forward(self, input):
        if input.size(1) > 3:
            input = torch.where(input < self.alpha, input, self.alpha)

            n_lv = 2**self.abit - 1
            scale = n_lv / self.alpha

            a_float = RoundQuant.apply(input, scale)
        else:
            a_float = input
        return a_float

class PROFITAQ(nn.Module):
    def __init__(self, abit):
        self.abit = abit
        self.a = nn.Parameter(torch.Tensor(1.))
        self.c = nn.Parameter(torch.Tensor(1.))

    def _upadte_param(self, abit, offset, diff):
        self.abit = abit

        if offset + diff > 6:
            self.a.data.fill_(np.log(np.exp(6)-1))
            self.c.data.fill_(np.log(np.exp(6)-1))
        else:
            self.a.data.fill_(np.log(np.exp(offset + diff)-1))
            self.c.data.fill_(np.log(np.exp(offset + diff)-1))
        
    def forward(self, input):
        if self.abit == 32:
            input_q = input
        else:
            n_lv = 2**self.abit
            scale = n_lv - 1
        
            a = F.softplus(self.a)
            c = F.softplus(self.c)

            input = F.hardtanh(input / a, 0, 1)
            input_q = RoundQuant.apply(input, scale)
        return input_q


class AQ_Symm(nn.Module):
    r"""
    Quantization function for Hardtanh
    """
    def __init__(self, abit, num_features):
        super(AQ_Symm, self).__init__()
        self.abit = abit
        self.alpha = torch.tensor(1.0).cuda()
    
    def forward(self, input):
        if input.size(1) > 3:
            
            with torch.no_grad():
                n_lv = 2 ** (self.abit - 1) - 1
                scale = n_lv / self.alpha
            a_float = RoundQuant.apply(input, scale)
        else:
            a_float = input
        return a_float

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

class QConvBN2d(ConvBN2d):
    r"""
    Quantized convolutional layer with batchnorm fused
    """
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride=1, 
        padding=0, 
        dilation=1, 
        groups=1, 
        bias=False, 
        eps=1e-5, 
        momentum=0.1, 
        affine=True, 
        track_running_stats=True, 
        wbit=4, 
        abit=4, 
        alpha_init=10.,
    ):
        super(QConvBN2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, 
            groups, bias, eps, momentum, affine, track_running_stats)
        
        # precisions
        self.wbit = wbit
        self.abit = abit
        channels = self.weight.data.size(0) 

        # quantizers
        self.WQ = WQ(wbit=wbit, num_features=channels)
        self.AQ = AQ(abit=abit, num_features=channels, alpha_init=alpha_init)


    def forward(self, input):
        def reshape_to_activation(inputs):
            return inputs.reshape(1, -1, 1, 1)

        # batch norm statistics
        if self.training:
            weight = self.WQ(self.weight)
            input = self.AQ(input)
            out = F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

            batch_mean = out.mean([0,2,3])
            batch_var = out.var([0,2,3])
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
        weight_q = self.WQ(weight)
        input_q = self.AQ(input)
        
        # convolution
        out = F.conv2d(input_q, weight_q, bn_bias, self.stride, self.padding, self.dilation, self.groups)

        if self.training:
            out *= reshape_to_activation(running_std / batch_std)
            out += reshape_to_activation(self.gamma * (running_mean / running_std - batch_mean / batch_std))
        return out

class QConv2d(nn.Conv2d):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride=1, 
        padding=0, 
        dilation=1, 
        groups=1, 
        bias=False, 
        wbit=32, 
        abit=32, 
    ):
        super(QConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, 
            groups, bias
        )
        
        # precisions
        self.abit = abit
        self.wbit = wbit
        num_features = self.weight.data.size(0)

        self.WQ = WQ(wbit=wbit, num_features=num_features, channel_wise=False)
        # self.AQ = AQ_Symm(abit, num_features)

    def forward(self, input):
        weight = self.weight.clone()
        weight_q = self.WQ(weight)
        out = F.conv2d(input, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out


class QLinear(nn.Linear):
    r"""
    Fully connected layer with Quantized weight
    """
    def __init__(self, in_features, out_features, bias=True, wbit=4, abit=4, alpha_init=10.0):
        super(QLinear, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        
        # precisions
        self.wbit = wbit
        self.abit = abit
        self.alpha_init = alpha_init
        channels = self.weight.data.size(0)

        # quantizers
        self.WQ = WQ(wbit=wbit, num_features=channels, channel_wise=False)
        # self.AQ = AQ_Symm(abit=abit, num_features=channels)

    def forward(self, input):
        weight = self.weight.clone()
        weight_q = self.WQ(weight)
        # input_q = self.AQ(input)

        out = F.linear(input, weight_q, self.bias)
        return out
"""
Old version
"""

def to_tensor(sat_val):
    is_scalar = not isinstance(sat_val, torch.Tensor)
    out = torch.tensor(sat_val) if is_scalar else sat_val.clone().detach()
    if not out.is_floating_point():
        out = out.to(torch.float32)
    if out.dim() == 0:
        out = out.unsqueeze(0)
    return is_scalar, out

def linear_quantize(input, scale, zero_point, inplace=False):
    if inplace:
        input.mul_(scale).sub_(zero_point).round_()
        return input
    return torch.round(input * scale - zero_point)

def linear_dequantize(input, scale, zero_point, inplace=False):
    if inplace:
        input.add_(zero_point).div_(scale)
        return input
    return (input + zero_point) / scale


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


class STEQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, zero_point, dequantize, inplace):
        if inplace:
            ctx.mark_dirty(input)

        output = linear_quantize(input, scale, zero_point)
        # print(f'quantized INT = {len(torch.unique(output))}')
        if dequantize:
            output = linear_dequantize(output, scale, zero_point)  
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Straight Through Estimator
        """
        
        return grad_output, None, None, None, None


class DoreFa_Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=False, wbit=4, abit=4):
        super(DoreFa_Conv2d, self).__init__(in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, dilation=dilation, groups=groups,
                bias=bias)
        self.wbit = wbit

    
    def forward(self, input):
        w_l = self.weight.clone()
        weight_q = Dorefa_quant_func(self.wbit)(w_l)
        output = F.conv2d(input, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return output
    
    def extra_repr(self):
        return super(DoreFa_Conv2d, self).extra_repr() + ', nbit={}'.format(self.wbit)


class DoreFa_Linear(nn.Linear):
    def __init__(self, in_channels, out_channels, bias=True, wbit=4, abit=4):
        super(DoreFa_Linear, self).__init__(in_features=in_channels, out_features=out_channels, bias=bias)
        self.wbit = wbit
    
    def forward(self, input):
        w_l = self.weight.clone()
        weight_q = Dorefa_quant_func(self.wbit)(w_l)
        output = F.linear(input, weight_q, self.bias)
        return output
    
    def extra_repr(self):
        super(DoreFa_Linear, self).extra_repr()+ ', nbit={}'.format(self.wbit)

class QHardTanh(nn.Module):
    def __init__(self, num_bits, alpha=1.0, inplace=False, dequantize=True):
        super(QHardTanh, self).__init__()
        self.num_bits = num_bits
        self.inplace = inplace
        self.dequantize = dequantize
        self.alpha = torch.tensor(alpha).cuda()

    def forward(self, input):
        input = F.hardtanh(input)

        with torch.no_grad():
            scale, zero_point = symmetric_linear_quantization_params(self.num_bits, self.alpha, restrict_qrange=True)
        input = STEQuantizer.apply(input, scale, zero_point, self.dequantize, self.inplace)
        return input

def get_scale(input, z):
    c1, c2 = 1/z[0], z[1]/z[0]
    std = input.std()
    mean = input.abs().mean()
    q_scale = c1 * std - c2 * mean # change the plus sign to minus sign for the correct version of sawb alpha_w    
    return q_scale 

class STEQuantizer_weight(torch.autograd.Function):
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

class int_quant_func(torch.autograd.Function):
    def __init__(self, nbit, alpha_w, restrictRange=True):
        super(int_quant_func, self).__init__()
        self.nbit = nbit
        self.restrictRange = restrictRange
        self.alpha_w = alpha_w

    def forward(self, input):
        self.save_for_backward(input)
        output = input.clamp(-self.alpha_w.item(), self.alpha_w.item())
        scale, zero_point = symmetric_linear_quantization_params(self.nbit, self.alpha_w, restrict_qrange=self.restrictRange)
        output = STEQuantizer_weight.apply(output, scale, zero_point, True, False, self.nbit, self.restrictRange)   

        return output

    def backward(self, grad_output):
        grad_input = grad_output.clone()
        input, = self.saved_tensors
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input


class int_conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=False, wbit=4, abit=4, mode='sawb', k=2):
        super(int_conv2d, self).__init__(in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, dilation=dilation, groups=groups,
                bias=bias)
        self.wbit = wbit
        self.mode = mode
        self.k = k
        self.register_buffer('alpha_w', torch.tensor(1.))

    def forward(self, input):
        w_l = self.weight.clone()

        if self.mode == 'mean':
            self.alpha_w = self.k * w_l.abs().mean()
        elif self.mode == 'sawb':
            z_typical = {'4bit': [0.077, 1.013], '8bit':[0.027, 1.114]}               
            self.alpha_w = get_scale(w_l, z_typical[f'{int(self.wbit)}bit'])
        else:
            raise ValueError("Quantization mode must be either 'mean' or 'sawb'! ")
            
        weight_q = int_quant_func(nbit=self.wbit, alpha_w=self.alpha_w, restrictRange=True)(w_l)
        output = F.conv2d(input, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return output
    
    def extra_repr(self):
        return super(int_conv2d, self).extra_repr() + ', nbit={}, mode={}, k={}'.format(self.wbit, self.mode, self.k)


class int_linear(nn.Linear):
    def __init__(self, in_channels, out_channels, bias=True, wbit=4, abit=4, mode='sawb', k=2):
        super(int_linear, self).__init__(in_features=in_channels, out_features=out_channels, bias=bias)
        self.wbit=wbit
        self.mode = mode
        self.k = k

    def forward(self, input):
        w_l = self.weight.clone()

        if self.mode == 'mean':
            self.alpha_w = self.k * w_l.abs().mean()
        elif self.mode == 'sawb':
            z_typical = {'4bit': [0.077, 1.013], '8bit':[0.027, 1.114]}               
            self.alpha_w = get_scale(w_l, z_typical[f'{int(self.wbit)}bit'])
        else:
            raise ValueError("Quantization mode must be either 'mean' or 'sawb'! ")                         
            
        weight_q = int_quant_func(nbit=self.wbit, alpha_w=self.alpha_w, restrictRange=True)(w_l)
        output = F.linear(input, weight_q, self.bias)
        return output
