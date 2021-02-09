"""
ResNet-18 model with HardTanh Activation Function

https://github.com/itayhubara/BinaryNet.pytorch/blob/master/models/resnet_binary.py
"""

import torch.nn as nn
import math
from .modules import QConv2d, QLinear

__all__ = ['resnet_tanh']

def Qconv3x3(in_planes, out_planes, stride=1, wbit=4, abit=4):
    "3x3 convolution with padding"
    return QConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, wbit=4, abit=4)

def init_model(model):
    for m in model.modules():
        if isinstance(m, BinarizeConv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, wbit=4, abit=4):
        super(BasicBlock, self).__init__()

        self.conv1 = Qconv3x3(inplanes, planes, stride, wbit=wbit, abit=abit)
        self.bn1 = nn.BatchNorm2d(planes)
        self.tanh1 = nn.Hardtanh(inplace=True)
        self.conv2 = Qconv3x3(planes, planes, wbit=wbit, abit=abit)
        self.tanh2 = nn.Hardtanh(inplace=True)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        residual = x.clone()

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.tanh1(out)

        out = self.conv2(out)


        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual
        out = self.bn2(out)
        out = self.tanh2(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, wbit=4, abit=4):
        super(Bottleneck, self).__init__()
        self.conv1 = Qconv3x3(inplanes, planes, kernel_size=1, bias=False, wbit=wbit, abit=abit)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Qconv3x3(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False, wbit=wbit, abit=abit)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = Qconv3x3(planes, planes * 4, kernel_size=1, bias=False, wbit=wbit, abit=abit)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.tanh = nn.Hardtanh(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        import pdb; pdb.set_trace()
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.tanh(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.tanh(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.bn2(out)
        out = self.tanh2(out)
        return out


class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()

    def _make_layer(self, block, planes, blocks, stride=1, wbit=4, abit=4):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Qconv3x3(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False, wbit=wbit, abit=abit),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, wbit=wbit, abit=abit))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks-1):
            layers.append(block(self.inplanes, planes, wbit=wbit, abit=abit))
        layers.append(block(self.inplanes, planes, wbit=wbit, abit=abit))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.bn1(x)
        x = self.tanh1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.tanh2(x)
        x = self.fc(x)
        return x

class ResNet_wide_cifar(ResNet):

    def __init__(self, num_classes=10,
                 block=Bottleneck, layers=[3, 4, 23, 3], wbit=4, abit=4):
        super(ResNet_wide_cifar, self).__init__()
        self.inplanes = 64

        self.conv1 = QConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False, wbit=wbit, abit=abit)
        self.bn1 = nn.BatchNorm2d(64) 
        self.tanh1 = nn.Hardtanh(inplace=True)

        self.maxpool = lambda x: x
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, wbit=wbit, abit=abit)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, wbit=wbit, abit=abit)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, wbit=wbit, abit=abit)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, wbit=wbit, abit=abit)
        self.avgpool = nn.AvgPool2d(4)
        
        self.tanh2 = nn.Hardtanh(inplace=True)
        self.fc = QLinear(512 * block.expansion, num_classes, wbit=wbit, abit=abit)
        init_model(self)

class resnet18_tanh:
    base = ResNet_wide_cifar
    args = list()
    kwargs = {'block': BasicBlock, 'layers': [2, 2, 2, 2]}
