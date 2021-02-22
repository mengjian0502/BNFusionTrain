"""
MobileNetV1 on CIFAR-10
"""
import numpy as np
import torch
import torch.nn as nn
from .modules import QConv2d, QLinear

class Net(nn.Module):
    """
    Full precision mobilenet V1 model for CIFAR10
    """
    def __init__(self, alpha=1.0, num_classes=10, wbit=32, abit=32, channel_wise=0):
        super(Net, self).__init__()
        self.alpha = alpha   # width multiplier of the model

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                QConv2d(inp, oup, 3, stride, 1, bias=False, wbit=wbit, abit=abit, channel_wise=0),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                QConv2d(inp, inp, 3, stride, 1, groups=inp, bias=False, wbit=wbit, abit=abit, channel_wise=channel_wise),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                QConv2d(inp, oup, 1, 1, 0, bias=False, wbit=wbit, abit=abit, channel_wise=channel_wise),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        self.model = nn.Sequential(
            conv_bn(3, int(32*self.alpha), 1), 
            conv_dw(int(32*self.alpha),  int(64*self.alpha), 1),
            conv_dw(int(64*self.alpha), int(128*self.alpha), 2),
            conv_dw(int(128*self.alpha), int(128*self.alpha), 1),
            conv_dw(int(128*self.alpha), int(256*self.alpha), 2),
            conv_dw(int(256*self.alpha), int(256*self.alpha), 1),
            conv_dw(int(256*self.alpha), int(512*self.alpha), 2),
            conv_dw(int(512*self.alpha), int(512*self.alpha), 1),
            conv_dw(int(512*self.alpha), int(512*self.alpha), 1),
            conv_dw(int(512*self.alpha), int(512*self.alpha), 1),
            conv_dw(int(512*self.alpha), int(512*self.alpha), 1),
            conv_dw(int(512*self.alpha), int(512*self.alpha), 1),
            conv_dw(int(512*self.alpha), int(1024*self.alpha), 2),
            conv_dw(int(1024*self.alpha), int(1024*self.alpha), 1),
        )
        self.pool = nn.AvgPool2d(2)
        self.fc = QLinear(int(1024*self.alpha), num_classes, wbit=wbit, abit=abit)

    def forward(self, x):
        x = self.model(x)
        x = self.pool(x)
        x = x.view(-1, int(1024*self.alpha))
        x = self.fc(x)
        return x

class mobilenetv1_Q:
  base=Net
  args = list()
  kwargs = {'alpha': 0.75}