import torch
import torch.nn as nn

from rexnet.model import Swish

def _add_conv(out, in_channels, channels, kernel=1, stride=1, pad=0,
             num_group=1, active=True, relu6=False):
    out.append(nn.Conv2d(in_channels, channels, kernel, stride, pad, groups=num_group, bias=False))
    out.append(nn.BatchNorm2d(channels))
    if active:
        out.append(nn.ReLU6(inplace=True) if relu6 else nn.ReLU(inplace=True))

def _add_conv_swish(out, in_channels, channels, kernel=1, stride=1, pad=0, num_group=1):
    out.append(nn.Conv2d(in_channels, channels, kernel, stride, pad, groups=num_group, bias=False))
    out.append(nn.BatchNorm2d(channels))
    out.append(Swish())