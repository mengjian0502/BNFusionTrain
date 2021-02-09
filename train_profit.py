"""
"""

import os
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.backends.cudnn as cudnn
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='cifar10')
parser.add_argument("--ckpt", required=True, help="checkpoint directory")
parser.add_argument("--model", choices=["mobilenetv1", "mobilenetv2", "mobilenetv3"])
parser.add_argument("--teacher", choices=["none", "self", "resnet18"])

parser.add_argument("--lr", default=0.04, type=float)
parser.add_argument("--decay", default=2e-5, type=float)

parser.add_argument("--warmup", default=3, type=int)
parser.add_argument("--bn_epoch", default=5, type=int)
parser.add_argument("--ft_epoch", default=15, type=int)
parser.add_argument("--sample_epoch", default=5, type=int)

parser.add_argument("--use_ema", action="store_true", default=False)
parser.add_argument("--stabilize", action="store_true", default=False)

parser.add_argument("--w_bit", required=True, type=int, nargs="+")
parser.add_argument("--a_bit", required=True, type=int, nargs="+")
parser.add_argument("--w_profit", required=True, type=int, nargs="+")

args = parser.parse_args()


