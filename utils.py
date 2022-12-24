import torch
import torch.nn as nn
import numpy as np
import logging
import os
import sys
import pandas as pd

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
    """Computes the accuracy@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# Training
def train(epoch,net, trainloader, optimizer, criterion, device='cuda'):

    net.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    train_loss = 0
    correct = 0
    total = 0
    num_batches = len(trainloader)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        inputs = inputs.squeeze(1)
        optimizer.zero_grad()
        outputs = net(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        prec1, prec5 = accuracy(outputs.data, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))
        # print(top1.avg)
        # break

    return top1.avg, losses.avg

# Training
def trainNAN(epoch, net, trainloader, optimizer, criterion, device='cuda'):

    net.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    train_loss = 0
    correct = 0
    total = 0
    num_batches = len(trainloader)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs.float())

        loss = criterion(outputs, targets.long())
        loss.backward()
        optimizer.step()

        prec1, prec5 = accuracy(outputs.data, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

    return top1.avg, losses.avg


# logging from: https://github.com/deepinsight/insightface/blob/04da49fbbd8e68745aad970139bbbbf0f15c7b6e/recognition/arcface_torch/utils/utils_logging.py#L30
def init_logging(gid, models_root):
    log_root = logging.getLogger()
    log_root.setLevel(logging.INFO)
    formatter = logging.Formatter("Training: %(asctime)s-%(message)s")
    handler_file = logging.FileHandler(os.path.join(models_root, "training.log"))
    handler_stream = logging.StreamHandler(sys.stdout)
    handler_file.setFormatter(formatter)
    handler_stream.setFormatter(formatter)
    log_root.addHandler(handler_file)
    log_root.addHandler(handler_stream)
    log_root.info(f'gid: {gid}')


def ccsv():
    path = 'sample_data'
    temp = []
    id = 0
    for s in os.listdir(path):
        sdir = os.path.join(path,s)
        for f in os.listdir(sdir):
            if f.endswith('.pt'):
              temp.append([os.path.join(sdir,f),id])
        id +=1
    df = pd.DataFrame(data=temp,columns=['path','id'])
    df.to_csv('sample_data.csv',index=False)
