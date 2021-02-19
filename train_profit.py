"""
PROFIT: A Novel Training Method for sub-4-bit MobileNet Models

Progressive quantization training 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import argparse
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler
import os
import time
import sys
import pickle
import operator
import models
import logging
from torchsummary import summary

from utils import *
from collections import OrderedDict

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/ImageNet Training')
parser.add_argument('--model', type=str, help='model type')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 64)')

parser.add_argument('--schedule', type=int, nargs='+', default=[60, 120],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1],
                    help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
parser.add_argument('--lr_decay', type=str, default='step', help='mode for learning rate decay')
parser.add_argument('--print_freq', default=1, type=int,
                    metavar='N', help='print frequency (default: 200)')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--log_file', type=str, default=None,
                    help='path to log file')

# dataset
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset: CIFAR10 / ImageNet_1k')
parser.add_argument('--data_path', type=str, default='./data/', help='data directory')

# model saving
parser.add_argument('--save_path', type=str, default='./save/', help='Folder to save checkpoints and log.')
parser.add_argument('--evaluate', action='store_true', help='evaluate the model')

# Acceleration
parser.add_argument('--ngpu', type=int, default=3, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=16,help='number of data loading workers (default: 2)')

# Fine-tuning
parser.add_argument('--fine_tune', dest='fine_tune', action='store_true',
                    help='fine tuning from the pre-trained model, force the start epoch be zero')
parser.add_argument('--resume', default='', type=str, help='path of the pretrained model')

# quantization
parser.add_argument("--teacher", choices=["none", "self", "resnet18"])
parser.add_argument("--warmup", default=3, type=int)
parser.add_argument("--bn_epoch", default=5, type=int)
parser.add_argument("--ft_epoch", default=15, type=int)
parser.add_argument("--sample_epoch", default=5, type=int)

parser.add_argument("--use_ema", action="store_true", default=False)
parser.add_argument("--stabilize", action="store_true", default=False)

parser.add_argument("--w_bit", required=True, type=int, nargs="+")
parser.add_argument("--a_bit", required=True, type=int, nargs="+")
parser.add_argument("--w_profit", required=True, type=int, nargs="+")
parser.add_argument('--channel_wise', type=int, default=0, help='channel_wise quantization flag')

args = parser.parse_args()
args.use_cuda = torch.cuda.is_available()

def main():
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    
    logger = logging.getLogger('training')
    if args.log_file is not None:
        fileHandler = logging.FileHandler(args.save_path+args.log_file)
        fileHandler.setLevel(0)
        logger.addHandler(fileHandler)
    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(0)
    logger.addHandler(streamHandler)
    logger.root.setLevel(0)

    logger.info(args)

    
    # Preparing data
    if args.dataset == 'cifar10':
        data_path = args.data_path + 'cifar'

        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]

        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        trainset = torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=train_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=test_transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        num_classes = 10
    elif args.dataset == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])  # here is actually the validation dataset

        train_dir = os.path.join(args.data_path, 'train')
        test_dir = os.path.join(args.data_path, 'val')

        train_data = torchvision.datasets.ImageFolder(train_dir, transform=train_transform)
        test_data = torchvision.datasets.ImageFolder(test_dir, transform=test_transform)

        trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        testloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        num_classes = 1000
    else:
        raise ValueError("Dataset must be either cifar10 or imagenet_1k")  

    # Prepare the model
    logger.info('==> Building model..\n')
    # student model
    model_cfg = getattr(models, args.model)
    model_cfg.kwargs.update({"num_classes": num_classes, "wbit": 32, "abit":32, "channel_wise":args.channel_wise})
    net = model_cfg.base(*model_cfg.args, **model_cfg.kwargs) 

    # teacher model
    logger.info('==> Building teacher model model {}..\n'.format(args.teacher))
    
    if args.fine_tune:
        new_state_dict = OrderedDict()
        logger.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        for k, v in checkpoint['state_dict'].items():
            name = k[7:]
            new_state_dict[name] = v
        
        state_tmp = net.state_dict()

        if 'state_dict' in checkpoint.keys():
            state_tmp.update(new_state_dict)
        
            net.load_state_dict(state_tmp)
            logger.info("=> loaded checkpoint '{} | Acc={}'".format(args.resume, checkpoint['acc']))
        else:
            raise ValueError('no state_dict found')  

    if args.teacher == "none":
        net_t = None
    elif args.teacher == "self":
        net_t = copy.deepcopy(net)
    else:
        raise NotImplementedError

    logger.info(net)
    start_epoch = 0

    if args.use_cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

        if net_t is not None:
            net_t = net_t.cuda()
            net_t = torch.nn.DataParallel(net_t, device_ids=range(torch.cuda.device_count()))
            cudnn.benchmark = True

    
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    # evaluate
    if args.evaluate:
        set_precision(net, abit=args.a_bit[-1], wbit=args.w_bit[-1], set_a=True, set_w=True)
        test_acc, val_loss = test(testloader, net, criterion, 0)
        logger.info(f'Test accuracy: {test_acc}')
        exit()

    if args.teacher is not None:
        logger.info("Full precision fine-tuning")
        params = categorize_param(net)
        optimizer = get_optimizer(params, train_quant=True, train_weight=True, train_bnbias=True, args=args)
        train_epoch(net, net_t, args.ft_epoch, wbit=32, abit=32, trainloader=trainloader, testloader=testloader, 
                    criterion=criterion, optimizer=optimizer, logger=logger, prefix='ft0', lasso=False)
    
    wbit, abit = 32, 32
    # Progressive activation quantization: 
    init_precision(net, trainloader, args.a_bit[0], args.w_bit[0], set_a=True, set_w=False)
    for abit in args.a_bit:
        logger.info(f"=> Activation quantization: {abit}...")
        set_precision(net, abit=abit, set_a=True)

        if args.stabilize:
            logger.info("=> BN stabilize")
            params = categorize_param(net)
            optimizer = get_optimizer(params, train_quant=True, train_weight=False, train_bnbias=True, args=args)
            train_epoch(net, net_t, epochs=args.bn_epoch, wbit=wbit, abit=abit, start_epoch=0, warmup_epoch=0, trainloader=trainloader, 
                        testloader=testloader, criterion=criterion, optimizer=optimizer, logger=logger, prefix='bn', lasso=False)
    
        logger.info("=> Fine-tune")
        optimizer = get_optimizer(params, train_quant=True, train_weight=False, train_bnbias=True, args=args)
        train_epoch(net, net_t, epochs=args.ft_epoch, wbit=wbit, abit=abit, start_epoch=0, warmup_epoch=args.warmup, trainloader=trainloader, 
                            testloader=testloader, criterion=criterion, optimizer=optimizer, logger=logger, prefix='ft', lasso=False)

        if args.stabilize:
            logger.info("=> BN stabilize 2")
            params = categorize_param(net)
            optimizer = get_optimizer(params, train_quant=True, train_weight=False, train_bnbias=True, args=args)
            train_epoch(net, net_t, epochs=args.bn_epoch, wbit=wbit, abit=abit, start_epoch=0, warmup_epoch=0, trainloader=trainloader, 
                        testloader=testloader, criterion=criterion, optimizer=optimizer, logger=logger, prefix='bn2', lasso=False)
    
    # progressive weight quantization
    with torch.no_grad():
        init_precision(net, trainloader, args.a_bit[-1], wbit=args.w_bit[0], set_w=True)
    for wbit in args.w_bit:
        logger.info(f"=> Weight quantization: {wbit}...")
        set_precision(net, abit=args.a_bit[-1], wbit=wbit, set_a=True, set_w=True)

        if args.stabilize:
            logger.info("=> BN stabilize")
            params = categorize_param(net)
            optimizer = get_optimizer(params, train_quant=True, train_weight=False, train_bnbias=True, args=args)
            train_epoch(net, net_t, epochs=args.bn_epoch, wbit=wbit, abit=abit, start_epoch=0, warmup_epoch=0, trainloader=trainloader, 
                        testloader=testloader, criterion=criterion, optimizer=optimizer, logger=logger, prefix='bn', lasso=False)
        
        if wbit in args.w_profit:
            logger.info("=> Sampling")
            metric_map = {}
            metric_map_name = f"{args.model}_sample_w{wbit}_a{abit}"
            for name, module in net.named_modules():
                if hasattr(module, "WQ") and isinstance(module, nn.Conv2d):
                    metric_map[name] = 0                                    # initialize the metric map

            optimizer = get_optimizer(params, train_quant=True, train_weight=True, train_bnbias=True, args=args)

            for epoch in range(args.sample_epoch):
                train_acc, train_loss, metric_map = train_profit(trainloader, net, net_t, criterion, optimizer, epoch, metric_map, logger)
                val_acc, val_loss = test(testloader, net, criterion, epoch)
            
            with open(os.path.join(args.save_path, metric_map_name + ".pkl"), "wb") as f:
                pickle.dump(metric_map, f)  

            # sampling completed; freezing the parameters
            skip_list = []
            sort = sorted(metric_map.items(), key=operator.itemgetter(1), reverse=True)
            for s in sort[0:int(len(sort)*1/3)]:
                skip_list.append(s[0])
            
            skip_list_next = []
            for s in sort[int(len(sort) * 1/3):int(len(sort) * 2/3)]:
                skip_list_next.append(s[0])
        
            params = categorize_param(net)
            optimizer = get_optimizer(params, train_quant=True, train_weight=True, train_bnbias=True, args=args)
            train_epoch(net, net_t, epochs=args.ft_epoch, wbit=wbit, abit=abit, start_epoch=0, warmup_epoch=args.warmup, trainloader=trainloader, 
                            testloader=testloader, criterion=criterion, optimizer=optimizer, logger=logger, prefix='ft1', lasso=False)

            # train the model with the first 1/3 parameter frozen
            params = categorize_param(net, skip_list)
            optimizer = get_optimizer(params, train_quant=True, train_weight=True, train_bnbias=True, args=args)
            train_epoch(net, net_t, epochs=args.ft_epoch, wbit=wbit, abit=abit, start_epoch=0, warmup_epoch=args.warmup, trainloader=trainloader, 
                            testloader=testloader, criterion=criterion, optimizer=optimizer, logger=logger, prefix='ft2', lasso=False)
            
            # train the model with all the parameter frozen
            params = categorize_param(net, skip_list+skip_list_next)
            optimizer = get_optimizer(params, train_quant=True, train_weight=True, train_bnbias=True, args=args)
            train_epoch(net, net_t, epochs=args.ft_epoch, wbit=wbit, abit=abit, start_epoch=0, warmup_epoch=args.warmup, trainloader=trainloader, 
                            testloader=testloader, criterion=criterion, optimizer=optimizer, logger=logger, prefix='ft3', lasso=False)
            
        else:
            logger.info("=> Fine-tune")
            params = categorize_param(net)
            optimizer = get_optimizer(params, train_quant=True, train_weight=True, train_bnbias=True, args=args)
            train_epoch(net, net_t, epochs=args.ft_epoch, wbit=wbit, abit=abit, start_epoch=0, warmup_epoch=args.warmup, trainloader=trainloader, 
                            testloader=testloader, criterion=criterion, optimizer=optimizer, logger=logger, prefix='ft', lasso=False)

        if args.stabilize:
            logger.info("=> BN stabilize 2")            
            optimizer = get_optimizer(params, train_quant=True, train_weight=False, train_bnbias=True, args=args)
            train_epoch(net, net_t, epochs=args.bn_epoch, wbit=wbit, abit=abit, start_epoch=0, warmup_epoch=0, trainloader=trainloader, 
                        testloader=testloader, criterion=criterion, optimizer=optimizer, logger=logger, prefix='bn2', lasso=False)

def train_epoch(net, net_t, epochs, wbit, abit, start_epoch=0, 
                warmup_epoch=0, trainloader=None, testloader=None, criterion=None, optimizer=None, logger=None, prefix=None, lasso=False):
    start_time = time.time()
    epoch_time = AverageMeter()
    best_acc = 0.
    
    scheduler = CosineWithWarmup(optimizer, warmup_len=warmup_epoch, warmup_start_multiplier=0.1, max_epochs=epochs, eta_min=1e-3, last_epoch=-1)

    for epoch in range(start_epoch, start_epoch+epochs):
        train_acc, train_loss, _ = train_profit(trainloader, net=net, net_t=net_t, criterion=criterion, optimizer=optimizer, epoch=epoch, logger=logger, lasso=lasso)
        val_acc, val_loss = test(testloader, net, criterion, epoch)  
        
        is_best = val_acc > best_acc
        if is_best:
            best_ac = val_acc

        state = {
            'state_dict': net.state_dict(),
            'acc': best_acc,
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
        }

        filename=f'checkpoint_w{wbit}_a{abit}_{prefix}.pth.tar'
        save_checkpoint(state, is_best, args.save_path, filename=filename)

        e_time = time.time() - start_time
        epoch_time.update(e_time)
        start_time = time.time()

        logger.info("\nEpoch [{}]/[{}]: Train Loss: {}; Top1 Train: {} | Val Loss: {}; Val Acc: {}\n".format(
            epoch+1, epochs-start_epoch, train_loss, train_acc, val_loss, val_acc))


class CosineWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    """ Implements a schedule where the first few epochs are linear warmup, and
    then there's cosine annealing after that."""

    def __init__(self, optimizer: torch.optim.Optimizer, warmup_len: int,
                 warmup_start_multiplier: float, max_epochs: int, 
                 eta_min: float = 0.0, last_epoch: int = -1):
        if warmup_len < 0:
            raise ValueError("Warmup can't be less than 0.")
        self.warmup_len = warmup_len
        if not (0.0 <= warmup_start_multiplier <= 1.0):
            raise ValueError(
                "Warmup start multiplier must be within [0.0, 1.0].")
        self.warmup_start_multiplier = warmup_start_multiplier
        if max_epochs < 1 or max_epochs < warmup_len:
            raise ValueError("Max epochs must be longer than warm-up.")
        self.max_epochs = max_epochs
        self.cosine_len = self.max_epochs - self.warmup_len
        self.eta_min = eta_min  # Final LR multiplier of cosine annealing
        super(CosineWithWarmup, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch > self.max_epochs:
            raise ValueError(
                "Epoch may not be greater than max_epochs={}.".format(
                    self.max_epochs))
        if self.last_epoch < self.warmup_len or self.cosine_len == 0:
            # We're in warm-up, increase LR linearly. End multiplier is implicit 1.0.
            slope = (1.0 - self.warmup_start_multiplier) / self.warmup_len
            lr_multiplier = self.warmup_start_multiplier + slope * self.last_epoch
        else:
            cosine_epoch = self.last_epoch - self.warmup_len
            lr_multiplier = self.eta_min + (1.0 - self.eta_min) * (
                1 + math.cos(math.pi * cosine_epoch / self.cosine_len)) / 2
        assert lr_multiplier >= 0.0
        return [base_lr * lr_multiplier for base_lr in self.base_lrs]

if __name__ == '__main__':
    main()
