"""
Utilities of MobileNet training
"""
from models import modules
import os
import sys
import time
import math
import shutil
import tabulate
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial
from models import QConvBN2d
_print_freq = 50


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train(trainloader, net, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        data_time.update(time.time() - end)
        
        targets = targets.cuda(non_blocking=True)
        inputs = inputs.cuda()
    
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        if args.clp:
            reg_alpha = torch.tensor(0.).cuda()
            a_lambda = torch.tensor(args.a_lambda).cuda()

            alpha = []
            for name, param in net.named_parameters():
                if 'alpha' in name:
                    alpha.append(param.item())
                    reg_alpha += param.item() ** 2
            loss += a_lambda * (reg_alpha)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prec1, prec5 = accuracy(outputs.data, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        train_loss += loss.item()
        if args.clp:
            res = {
                'acc':top1.avg,
                'loss':losses.avg,
                'clp_alpha':np.array(alpha)
                }
        else:
            res = {
                'acc':top1.avg,
                'loss':losses.avg,
                } 
    return res


def test(testloader, net, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    net.eval()
    test_loss = 0

    end = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            mean_loader = []
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            prec1, prec5 = accuracy(outputs.data, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            test_loss += loss.item()

            batch_time.update(time.time() - end)
            end = time.time()
            # break
    return top1.avg, losses.avg

def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600*need_hour) / 60)
    need_secs = int(epoch_time - 3600*need_hour - 60*need_mins)
    return need_hour, need_mins, need_secs


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()

def print_table(values, columns, epoch, logger):
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    if epoch == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    logger.info(table)

def adjust_learning_rate_schedule(optimizer, epoch, gammas, schedule, lr, mu):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    if optimizer != "YF":
        assert len(gammas) == len(
            schedule), "length of gammas and schedule should be equal"
        for (gamma, step) in zip(gammas, schedule):
            if (epoch >= step):
                lr = lr * gamma
            else:
                break
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    elif optimizer == "YF":
        lr = optimizer._lr
        mu = optimizer._mu

    return lr, mu


def save_checkpoint(state, is_best, save_path, filename='checkpoint.pth.tar'):
    torch.save(state, save_path+filename)
    if is_best:
        shutil.copyfile(save_path+filename, save_path+'model_best.pth.tar')

def get_alpha_w(model):
    alpha = []
    count = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if not count in [0] and not m.weight.size(2)==1:
                alpha.append(m.alpha_w)
            count += 1
    return alpha

def log2df(log_file_name):
    '''
    return a pandas dataframe from a log file
    '''
    with open(log_file_name, 'r') as f:
        lines = f.readlines() 
    # search backward to find table header
    num_lines = len(lines)
    for i in range(num_lines):
        if lines[num_lines-1-i].startswith('---'):
            break
    header_line = lines[num_lines-2-i]
    num_epochs = i
    columns = header_line.split()
    df = pd.DataFrame(columns=columns)
    for i in range(num_epochs):
        df.loc[i] = [float(x) for x in lines[num_lines-num_epochs+i].split()]
    return df 

"""
PROFIT Util
"""
def categorize_param(model, skip_list=()):
    quant = []
    skip = []
    bnbias = []
    weight = []

    for name, param, in model.named_parameters():
        skip_found = False
        for s in skip_list:
            if name.find(s) != -1:
                skip_found = True
        
        if not param.requires_grad:
            continue
        elif name.endswith(".a") or name.endswith(".c"):
            quant.append(param)
        elif skip_found:
            skip.append(param)
        elif len(param.shape) == 1 or name.endswith(".bias"):
            bnbias.append(param)
        else:
            weight.append(param)

    return (quant, skip, weight, bnbias)

def get_optimizer(params, train_quant, train_weight, train_bnbias, args):
    (quant, skip, weight, bnbias) = params
    optimizer = optim.SGD([
        {'params': skip, 'weight_decay': 0, 'lr': 0},
        {'params': quant, 'weight_decay': 0., 'lr': args.lr * 1e-2 if train_quant else 0},
        {'params': bnbias, 'weight_decay': 0., 'lr': args.lr if train_bnbias else 0},
        {'params': weight, 'weight_decay': args.weight_decay, 'lr': args.lr if train_weight else 0},
    ], momentum=0.9, nesterov=True)
    return optimizer

def reset_weight_copy(model):
    for name, module in model.module.named_modules():
        if hasattr(module, "WQ"):
            if hasattr(module.WQ, "weight_old"):
                del module.WQ.weight_old
            module.WQ.weight_old = None

def lasso_thre(var, thre=1.0):
    thre = torch.tensor(thre).cuda()

    a = var.pow(2).pow(1/2)
    p = torch.max(a, thre)  # penalize or not
    return p

def train_profit(train_loader, net, net_t, criterion, optimizer, epoch, metric_map={}, logger=None, lasso=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    net.train()
    
    # reset weight copy
    reset_weight_copy(net)

    if net_t is not None:
        net_t.train()

    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        # deploy the data
        input = input.cuda()
        target = target.cuda(non_blocking=True)

        if net_t is not None:
            output_t = net_t(input)
        

        # create and attach hook for layer-wise aiwq measure
        hooks = []
        metric_itr_map = {}

        if len(metric_map) > 0:
            def forward_hook(self, input, output):
                if self.WQ.weight_old is not None and input[0].get_device() == 0:
                    with torch.no_grad():
                        out_old = torch.nn.functional.conv2d(input[0], self.WQ.weight_old, self.bias,
                            self.stride, self.padding, self.dilation, self.groups)

                        out_t = torch.transpose(output, 0, 1).contiguous().view(self.out_channels, -1)
                        out_mean = torch.mean(out_t, 1)
                        out_std = torch.std(out_t, 1) # + 1e-8

                        out_old_t = torch.transpose(out_old, 0, 1).contiguous().view(self.out_channels, -1)
                        out_old_mean = torch.mean(out_old_t, 1)
                        out_old_std = torch.std(out_old_t, 1) # + 1e-8

                        out_cond = out_std != 0
                        out_old_cond = out_old_std != 0
                        cond = out_cond & out_old_cond

                        out_mean = out_mean[cond]
                        out_std = out_std[cond]

                        out_old_mean = out_old_mean[cond]
                        out_old_std = out_old_std[cond]

                        KL = torch.log(out_old_std / out_std) + \
                            (out_std ** 2  + (out_mean - out_old_mean) ** 2) / (2 * out_old_std ** 2) - 0.5
                        metric_itr_map[self.name] = KL.mean().data.cpu().numpy()
            
            for name, module in net.module.named_modules():
                if hasattr(module, "WQ") and isinstance(module, torch.nn.Conv2d):
                    module.name = name
                    hooks.append(module.register_forward_hook(forward_hook))
        
        # feed forward
        output = net(input)
        for hook in hooks:
            hook.remove()
        
        loss_s = criterion(output, target)      # student model loss
        if net_t is not None:
            loss_kd = -1 * torch.mean(
                torch.sum(torch.nn.functional.softmax(output_t, dim=1) 
                        * torch.nn.functional.log_softmax(output, dim=1), dim=1))
            loss = loss_s + loss_kd
        else:
            loss = loss_s

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss_s.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        if ((i+1) % _print_freq) == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch+1, i+1, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))
        
        for key, value in metric_itr_map.items():
            if value > 1:
                continue
            metric_map[key] = 0.999 * metric_map[key] + 0.001 * value
    
    return top1.avg, losses.avg, metric_map

def init_precision(model, loader, abit, wbit, set_a=False, set_w=False, eps=0.05):
    def init_hook(module, input, output):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            if not isinstance(input, torch.Tensor):
                input = input[0]
            input = input.detach().cpu()
            input = input.reshape(-1)
            input = input[input > 0]
            input, _ = torch.sort(input)

            if len(input) == 0:
                small, large = 0, 1e-3
            else:
                small, large = input[int(len(input)*eps)], input[int(len(input)*(1-eps))]
            
            if set_a: 
                module.AQ._update_param(abit, small, large-small)
                # import pdb;pdb.set_trace()
            
            if set_w:
                max_val = module.weight.data.abs().max().item()
                module.WQ._update_param(wbit, max_val)

    hooks = []
    for name, module in model.named_modules():
        hook = module.register_forward_hook(init_hook)
        hooks.append(hook)
    
    model.train()
    model.cpu()
    for i, (input, target) in enumerate(loader):
        with torch.no_grad():
            if isinstance(model, nn.DataParallel):
                output = model.module(input)
            else:
                output = model(input)
        break
    
    model.cuda()
    for hook in hooks:
        hook.remove()

def bn_merge(model):
    r"""
    Fuse the batchnorm to the weight given a pretrained model
    """
    for module_name in model._modules:
        block = model._modules[module_name]
        if not isinstance(block, nn.Sequential):
            # import pdb;pdb.set_trace()
            model._modules[module_name] = block
            continue
        else:
            stack = []
            for m in block.children():
                sub_module = []
                for n in m.children():
                    if isinstance(n, nn.BatchNorm2d):
                        if isinstance(sub_module[-1], QConvBN2d):
                            bn_st_dict = n.state_dict()
                            conv_st_dict = sub_module[-1].state_dict()
                            # batchnorm parameters
                            eps = n.eps
                            mu = bn_st_dict['running_mean']
                            var = bn_st_dict['running_var']
                            gamma = bn_st_dict['weight']
                            nb_tr = bn_st_dict['num_batches_tracked']

                            if 'bias' in bn_st_dict:
                                beta = bn_st_dict['bias']
                            else:
                                beta = torch.zeros(gamma.size(0)).float().to(gamma.device)
                            
                            sub_module[-1].gamma.data = gamma
                            sub_module[-1].beta.data = beta
                            sub_module[-1].running_mean.data = mu
                            sub_module[-1].running_var.data = var
                            sub_module[-1].num_batches_tracked.data = nb_tr
                            sub_module[-1].eps = eps
                            # import pdb;pdb.set_trace()
                    else:
                        sub_module.append(n)
                seq_module = nn.Sequential(*sub_module)    
                stack.append(seq_module)
            seq_stack = nn.Sequential(*stack)
            
            model._modules[module_name] = seq_stack
    # import pdb;pdb.set_trace()
    return model

def set_precision(model, abit=32, wbit=32, set_a=False, set_w=False):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            if set_a:
                module.AQ.abit = abit
            else:
                module.AQ.abit = 32
            
            if set_w:
                module.WQ.wbit = wbit
            else:
                module.WQ.wbit = 32

if __name__ == "__main__":
    log = log2df('./save/resnet20_quant_grp8/resnet20_quant_w4_a4_modemean_k2_lambda0.0010_ratio0.7_wd0.0005_lr0.01_swpFalse_groupch8_pushFalse_iter4000_g01/resnet20_quant_w4_a4_modemean_k2_lambda0.0010_ratio0.7_wd0.0005_lr0.01_swpFalse_groupch8_pushFalse_iter4000_tmp_g03.log')
    epoch = log['ep']
    grp_spar = log['grp_spar']
    ovall_spar = log['ovall_spar']
    spar_groups = log['spar_groups']
    penalty_groups = log['penalty_groups']

    table = {
        'epoch': epoch,
        'grp_spar': grp_spar,
        'ovall_spar': ovall_spar,
        'spar_groups':spar_groups,
        'penalty_groups':penalty_groups,
    }

    variable = pd.DataFrame(table, columns=['epoch','grp_spar','ovall_spar', 'spar_groups', 'penalty_groups'])
    variable.to_csv('resnet20_quant_w4_a4_modemean_k2_lambda0.0010_ratio0.7_wd0.0005_lr0.01_swpFalse_groupch8_pushFalse_iter4000_tmp_g03.csv', index=False)
