#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Nov-21-20 16:25
# @Update  : Dec-09-20 20:03
# @Author  : Kelly Hwong (dianhuangkan@gmail.com)

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable


class TwoConvNet(nn.Module):
    """简单二层卷积网络，使用了 Dropout 层
    # @RefLink : https://github.com/pytorch/examples/blob/master/mnist/main.py
    """

    def __init__(self):
        super(TwoConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class MLP(nn.Module):
    """MLP with BatchNorm, ReLU and Dropout
    """

    def __init__(self):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(784, 548)
        self.bn1 = nn.BatchNorm1d(548)

        self.fc2 = nn.Linear(548, 252)
        self.bn2 = nn.BatchNorm1d(252)

        self.fc3 = nn.Linear(252, 10)

    def forward(self, x):
        x = x.view((-1, 784))
        h = self.fc1(x)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)

        h = self.fc2(h)
        h = self.bn2(h)
        h = F.relu(h)
        h = F.dropout(h, p=0.2, training=self.training)

        h = self.fc3(h)
        out = F.log_softmax(h)
        return out
