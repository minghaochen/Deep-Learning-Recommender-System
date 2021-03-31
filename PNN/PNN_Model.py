# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 16:51:02 2021

@author: mhchen
"""

import torch

# field_size is the input dimension
# field_num is feature num in each field (list)

class PNN_Model(torch.nn.Module):
    def __init__(self, field_size, field_num, embedding_dim, D1, D2):
        super(PNN_Model, self).__init__()
        self.field_size = field_size
        self.field_num = field_num
        self.embedding_dim = embedding_dim
        self.embedding = [torch.nn.Embedding(self.field_num[i], self.embedding_dim) for i in range(self.field_size)]
        self.D1 = D1
        self.W1 = torch.nn.parameter.Parameter(torch.randn((self.D1, self.field_size, self.embedding_dim)))
        self.W2 = torch.nn.parameter.Parameter(torch.randn((self.D1, self.field_size, self.field_size)))
        self.W3 = torch.nn.parameter.Parameter(torch.randn((self.D1, self.embedding_dim, self.embedding_dim)))
        self.D2 = D2
        self.hidden1 = torch.nn.Linear(self.D1*3, self.D2)
        self.hidden2 = torch.nn.Linear(self.D2, 1)
        self.prob = torch.nn.Sigmoid()
        

    def forward(self, x):
        # x is Batch*field_size
        input_embedding = []
        for i in range(self.field_size):
            input_embedding.append(self.embedding[i](x[:,i]))
        # Batch*field_size*embedding_dim
        input_embedding = torch.stack(input_embedding,1) 

        # product layer - part z
        z = []
        for i in range(self.D1):
            z.append(torch.sum(torch.mul(input_embedding,self.W1[i]),(2,1)))
        # batch*D1
        z = torch.stack(z,1)
        
        # product layer - part p
        # inner product
        p = torch.bmm(input_embedding,input_embedding.view(-1,self.embedding_dim,self.field_size))
        
        p1 = []
        for i in range(self.D1):
            p1.append(torch.sum(torch.mul(p,self.W2[i]),(2,1)))
        p1 = torch.stack(p1,1)
        
        # outer product
        f_Sigma = input_embedding.sum(dim=1)
        p = torch.bmm(f_Sigma.unsqueeze(2), f_Sigma.unsqueeze(1))
        p2 = []
        for i in range(self.D1):
            p2.append(torch.sum(torch.mul(p,self.W3[i]),(2,1)))
        p2 = torch.stack(p2,1)
        
        # concate
        Product_output = torch.nn.ReLU()(torch.cat([z,p1,p2],1))
        # hidden layer
        hidden1_output = self.hidden1(Product_output)
        hidden2_output = self.hidden2(hidden1_output)
        # prob
        output = self.prob(hidden2_output)
        
        
        
        return output


# model test
field_num = list([5,3,5])
embedding_dim = 10
field_size = 3
D1 = 10
D2 = 10
x = torch.tensor([[1,2,3],[1,2,3],[3,0,4]])
model = PNN_Model(field_size, field_num, embedding_dim, D1, D2)
output = model(x)