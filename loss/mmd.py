#!/usr/bin/env python
# encoding: utf-8
import numpy as np
import torch
import torch.nn.functional as F
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):

    # source_size = int(source.size()[0]) if int(source.size()[0]) < 255 else int(len(source.size()))
    # target_size = int(target.size()[0]) if int(target.size()[0]) < 255 else int(len(target.size()))
    source_size = int(source.size()[0])
    target_size = int(target.size()[0])
    n_samples = source_size + target_size
    total = torch.cat([source, target], dim=0)

    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

    # exp(-|x-y|/bandwith)
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

    return sum(kernel_val)


def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]

    return XX, YY, 2 * XY


def mmd_loss(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):

    n = int(source.size()[0])
    m = int(target.size()[0])

    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:n, :n]
    YY = kernels[n:, n:]
    XY = kernels[:n, n:]
    YX = kernels[n:, :n]

    # temp = XX
    # temp = torch.div(temp, n)
    # temp = torch.div(temp, n).sum(dim=-1).view(1, -1)
    # print(temp)
    XX = torch.div(XX, n * n).sum(dim=1).view(1, -1)  # K_ss，Source<->Source
    # print(XX)

    XY = torch.div(XY, -n * m).sum(dim=1).view(1, -1)  # K_st，Source<->Target

    YX = torch.div(YX, -m * n).sum(dim=1).view(1, -1)  # K_ts,Target<->Source
    YY = torch.div(YY, m * m).sum(dim=1).view(1, -1)  # K_tt,Target<->Target

    loss = (XX + XY).sum() + (YX + YY).sum()

    return loss




