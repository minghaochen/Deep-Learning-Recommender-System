# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 09:01:03 2021

@author: mhchen
"""

import torch

# d is the dimension of x
# m is the dimension of pieces

class MLR_Model(torch.nn.Module):
    def __init__(self, m, d):
        super(MLR_Model, self).__init__()
        self.m = m
        self.softmax = torch.nn.Sequential(
                torch.nn.Linear(d, m),
                torch.nn.Softmax(dim=1)
                )
        self.logistic = torch.nn.Sequential(
                torch.nn.Linear(d, m),
                torch.nn.Sigmoid()
                )
    
    def forward(self, x):
        LR_out = self.logistic(x)
        Class_out = self.softmax(x)
        output = torch.mul(LR_out, Class_out)
        return output.sum(1)


MLR = MLR_Model(3,10)
x = torch.randn(5,10)
output = MLR(x)