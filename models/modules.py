"""
DNN quantization with / without merged modules
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
        ub = 6
        lb = -3
        output_int = input.mul(scale[:,None,None,None]).round()
        output_int = output_int.clamp(lb, ub)                       # layer-wise clamp
        output_float = output_int.div(scale[:,None,None,None])
        return output_float

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class RoundQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale):
        output_int = input.mul(scale).round()
        output_float = output_int.div_(scale)
        return output_float

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class RoundQuantClamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale):
        ub = 6
        lb = -3
        output_int = input.mul(scale).round_()
        output_int = output_int.clamp(lb, ub)       # layer-wise clamp
        print("weight_int:{}".format(output_int.unique().cpu().numpy()))
        output_float = output_int.div_(scale)
        return output_float

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

def roundquantclamp(input, scale):
    ub = 6
    lb = -3
    output_int = input.mul(scale).round_()
    output_int = output_int.clamp(lb, ub)       # layer-wise clamp
    output_float = output_int.div(scale)
    return output_int, output_float, scale


class WQ(nn.Module):
    def __init__(self, wbit, num_features, channel_wise=1):
        super(WQ, self).__init__()
        self.wbit = wbit
        self.num_features = num_features

        if channel_wise:
            self.register_buffer('alpha_w', torch.ones(num_features))
        else:
            self.register_buffer('alpha_w', torch.tensor(1.))

        self.channel_wise = channel_wise

    def forward(self, input):
        z_typical = {'4bit': [0.077, 1.013], '8bit':[0.027, 1.114]}
        z = z_typical[f'{int(self.wbit)}bit']
        n_lv = 2 ** (self.wbit - 1) - 1

        if self.channel_wise == 1:
            m = input.abs().mean([1,2,3])
            std = input.std([1,2,3])
            
            self.alpha_w = 1/z[0] * std - z[1]/z[0] * m
            lb = self.alpha_w.mul(-1.)
            ub = self.alpha_w

            # channel-wise clamp
            input = torch.max(torch.min(input, ub[:,None,None,None]), lb[:,None,None,None])
            self.scale = n_lv / self.alpha_w

            w_float = WeightQuant.apply(input, self.scale)
        else:
            m = input.abs().mean()
            std = input.std()
            
            self.alpha_w = 1/z[0] * std - z[1]/z[0] * m 
            input = input.clamp(-self.alpha_w.item(), self.alpha_w.item())
            self.scale = n_lv / self.alpha_w

            w_float = RoundQuant.apply(input, self.scale)
        return w_float
    
    def extra_repr(self):
        return super(WQ, self).extra_repr() + 'wbit={}, channel_wise={}'.format(self.wbit, self.channel_wise)

class WQPROFIT(nn.Module):
    """
    PROFIT: A Novel Training Method for sub-4-bit MobileNet Models
    https://github.com/EunhyeokPark/PROFIT

    Weight Quantization Module
    """
    def __init__(self, wbit=32, num_features=None, channel_wise=False):
        super(WQPROFIT, self).__init__()
        self.wbit = wbit
        self.pbit = 8
        self.num_features = num_features
        self.channel_wise = channel_wise
        self.weight_old = None
        
        # learnable parameters
        if channel_wise:
            self.a = nn.Parameter(torch.ones(num_features))
            self.c = nn.Parameter(torch.ones(num_features))
        else:
            self.a = nn.Parameter(torch.tensor(1.))
            self.c = nn.Parameter(torch.tensor(1.))

    def _update_param(self, wbit, max_val):
        self.wbit = wbit
        # max_val = self.weight.data.abs().max().item()
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

            if self.channel_wise==1:
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
    
    def extra_repr(self):
        return super(WQPROFIT, self).extra_repr() + 'wbit={}, channel_wise={}'.format(self.wbit, self.channel_wise)

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
    
    def extra_repr(self):
        return super(AQ, self).extra_repr() + 'abit={}'.format(self.abit)

class AQ_Symm(nn.Module):
    r"""
    Quantization function for Hardtanh
    """
    def __init__(self, abit, num_features):
        super(AQ_Symm, self).__init__()
        self.abit = abit
    
    def forward(self, input):
        if input.size(1) > 3:
            lb = input.min().item()
            ub = input.max().item()

            n_lv = 2 ** (self.abit - 1) - 1
            scale = n_lv / ub
            a_float = RoundQuant.apply(input, scale)
        else:
            a_float = input
        return a_float

class AQPROFIT(nn.Module):
    """
    PROFIT: A Novel Training Method for sub-4-bit MobileNet Models
    https://github.com/EunhyeokPark/PROFIT

    Activation Quantization Module
    """
    def __init__(self, abit=32):
        super(AQPROFIT, self).__init__()
        self.abit = abit
        self.pbit = 8

        self.a = nn.Parameter(torch.tensor(1.))
        self.c = nn.Parameter(torch.tensor(1.))

    def _update_param(self, abit, offset, diff):
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
            if input.size(1) > 3:
                n_lv = 2**self.abit
                scale = n_lv - 1

                # gradient friendly
                a = F.softplus(self.a)  # keep the learnable value positive
                c = F.softplus(self.c)
                
                input = F.hardtanh(input / a, 0, 1)
                input_q = RoundQuant.apply(input, scale)
                input_q = input_q * c
            else:
                input_q = input
        return input_q

    def extra_repr(self):
        return super(AQPROFIT, self).extra_repr() + 'abit={}'.format(self.abit)

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
        channel_wise=0
    ):
        super(QConvBN2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, 
            groups, bias, eps, momentum, affine, track_running_stats)
        
        # precisions
        self.wbit = wbit
        self.abit = abit
        num_features = self.weight.data.size(0) 

        # quantizers
        self.WQ = WQPROFIT(wbit=wbit, num_features=num_features, channel_wise=channel_wise)
        self.AQ = AQPROFIT(abit)

        # mask
        self.register_buffer("mask", torch.ones(self.weight.data.size()))


    def forward(self, input):
        def reshape_to_activation(inputs):
            return inputs.reshape(1, -1, 1, 1)

        # batch norm statistics
        if self.training:
            weight = self.WQ(self.weight)
            input = self.AQ(input)
            out = F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

            batch_mean = out.mean([0,2,3])
            batch_var = out.var([0,2,3])
            batch_std = torch.sqrt(batch_var + self.eps)

            with torch.no_grad():
                self.running_mean = self.momentum * batch_mean + (1 - self.momentum) * self.running_mean
                self.running_var = self.momentum * batch_var  + (1 - self.momentum) * self.running_var
        
        running_mean = self.running_mean
        running_var = self.running_var
        running_std = torch.sqrt(running_var + self.eps)
        
        # import pdb;pdb.set_trace()
        
        bn_scale = self.gamma / running_std
        bias = self.beta - self.gamma * running_mean / running_std
        bn_bias = bias.reshape(-1)
        
        # merge the BN weight to the weight (channel-wise weight quant -> channel wise bn merge)
        # weight = self.weight.clone()
        # weight_q = self.WQ(weight)
        # weight_q = weight_q * bn_scale[:, None, None, None]
        # input_q = self.AQ(input)

        # merge the BN weight to the weight(channel wise bn merge -> layerwise/channelwise quant)
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
        channel_wise=0
    ):
        super(QConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, 
            groups, bias
        )
        
        # precisions
        self.abit = abit
        self.wbit = wbit
        num_features = self.weight.data.size(0)

        self.WQ = WQPROFIT(wbit=wbit, num_features=num_features, channel_wise=channel_wise)
        self.AQ = AQPROFIT(abit)
        # self.WQ = WQ(wbit=wbit, num_features=num_features, channel_wise=channel_wise)
        # self.AQ = AQ(abit=abit, num_features=num_features, alpha_init=10.0)
        
        # mask
        self.register_buffer("mask", torch.ones(self.weight.data.size()))

    def forward(self, input):
        # if self.weight.size(1)== 512 and self.weight.size(2) == 3:
        #     import pdb;pdb.set_trace()
        weight_q = self.WQ(self.weight)
        input_q = self.AQ(input)

        out = F.conv2d(input_q, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out

class QLinear(nn.Linear):
    r"""
    Fully connected layer with Quantized weight
    """
    def __init__(self, in_features, out_features, bias=True, wbit=32, abit=32, alpha_init=10.0):
        super(QLinear, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        
        # precisions
        self.wbit = wbit
        self.abit = abit
        self.alpha_init = alpha_init
        channels = self.weight.data.size(0)

        # quantizers
        self.WQ = WQPROFIT(wbit=wbit, num_features=channels, channel_wise=0)
        self.AQ = AQPROFIT(abit=abit)
        # self.WQ = WQ(wbit=wbit, num_features=channels, channel_wise=0)
        # self.AQ = AQ(abit=abit, num_features=channels, alpha_init=alpha_init)

        # mask
        self.register_buffer("mask", torch.ones(self.weight.data.size()))

    def forward(self, input):
        weight_q = self.WQ(self.weight)
        input_q = self.AQ(input)

        out = F.linear(input_q, weight_q, self.bias)
        return out


"""
Get column-wise MAC
"""
class ColQConv2d(nn.Conv2d):
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
        channel_wise=0
    ):
        super(ColQConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, 
            groups, bias
        )

        self.layer_idx=0
        
        # precisions
        self.abit = abit
        self.wbit = wbit
        num_features = self.weight.data.size(0)

        # self.WQ = WQPROFIT(wbit=wbit, num_features=num_features, channel_wise=channel_wise)
        # self.AQ = AQPROFIT(abit)
        self.WQ = WQ(wbit=wbit, num_features=num_features, channel_wise=channel_wise)
        self.AQ = AQ(abit=abit, num_features=num_features, alpha_init=10.0)
        
        # mask
        self.register_buffer("mask", torch.ones(self.weight.data.size()))

    def forward(self, input):
        weight_q = self.WQ(self.weight)
        input_q = self.AQ(input)
        out_original = F.conv2d(input_q, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)

        # inference only
        scale_w = self.WQ.scale
        weight_int, weight_float, _ = roundquantclamp(self.weight, scale_w)
        c_out, c_in, h, w = weight_int.size()
        
        target_samples = 10000
        save_mac = False
        mac64 = torch.tensor([]).cuda()
        output = torch.zeros_like(out_original)
        for i in range(c_out):
            channel_out = torch.zeros_like(out_original)
            for k in range(h):
                for m in range(w):
                    mask = torch.zeros_like(weight_int)
                    mask[i, :, k, m] = 1.
                    weight_m = weight_int * mask     # input channel
                    partial = F.conv2d(input_q, weight_m, self.bias, self.stride, self.padding, self.dilation, self.groups)         
                    channel_out += partial
                    
                    # save mac values
                    if len(mac64) < target_samples:
                        mac_matrix = partial[:,i,:,:].reshape(-1)
                        mac64 = torch.cat((mac64, mac_matrix))  
                    else:
                        continue
            output += channel_out
        output = output.div(scale_w)
        
        if save_mac:
            # torch.save(mac64, './mac{}_{}x{}x{}x{}_nonclamp.pt'.format(c_in, c_out, c_in, h, w))
            torch.save(weight_int, './weight_int/weight_q{}_clamp.pt'.format(self.layer_idx))
        return output