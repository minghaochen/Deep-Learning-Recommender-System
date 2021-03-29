# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 09:54:09 2021

@author: mhchen
"""

import torch

# n is the dimension of x
# k is the dimension of v_i
class FM_Model(torch.nn.Module):
    def __init__(self, n, k):
        super(FM_Model, self).__init__()
        self.n = n
        self.k = k
        self.linear = torch.nn.Linear(self.n, 1)
        self.V = torch.nn.Parameter(torch.randn(self.n, self.k))
        torch.nn.init.uniform_(self.V, -0.1, 0.1) # 均匀分布初始化

    def forward(self,x):
        linear_part = self.linear(x)
        interaction_1 = torch.mm(x, self.V) # torch.mm is matrix multiply
        interaction_1 = torch.pow(interaction_1, 2) # element-wise pow
        interaction_2 = torch.mm(torch.pow(x,2), torch.pow(self.V, 2))
        interaction_part = 0.5*torch.sum(interaction_1 - interaction_2)
        output = linear_part + interaction_part
        return output

FM = FM_Model(10,5)
x = torch.randn(2,10)
output = FM(x)