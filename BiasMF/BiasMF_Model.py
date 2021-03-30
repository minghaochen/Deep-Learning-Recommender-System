# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 09:31:10 2021

@author: mhchen
"""

import torch

# user_num is the number of user
# k is the dimension of embedding
# mu is the global bias
# u_bias/i_bias are the user and item bias

class BiasMF_Model(torch.nn.Module):
    def __init__(self, user_num, item_num, k):
        super(BiasMF_Model, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.k = k
        self.U_embedding = torch.nn.Embedding(self.user_num, self.k)
        self.I_embedding = torch.nn.Embedding(self.item_num, self.k)
        self.mu = torch.nn.parameter.Parameter(torch.zeros(1))
        self.u_bias = torch.nn.Embedding(self.user_num, 1)
        self.i_bias = torch.nn.Embedding(self.item_num, 1)
        torch.nn.init.constant_(self.u_bias.weight, 0.0)
        torch.nn.init.constant_(self.i_bias.weight, 0.0)
        
    
    def forward(self, user_indices, item_indices):
        U = self.U_embedding(user_indices)
        I = self.I_embedding(item_indices)
        inner_product = torch.mul(U,I).sum(1)
        
        Bias = self.mu + self.u_bias(user_indices).view(-1) + self.i_bias(item_indices).view(-1)
        
        rating = inner_product + Bias
        return rating


BiasMF = BiasMF_Model(10,20,5)
x = torch.tensor([1,2,3])
y = torch.tensor([5,7,8])
output = BiasMF(x,y)

