# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 10:35:15 2021

@author: mhchen
"""

import torch
from FM_Model import FM_Model
import pandas as pd
import numpy as np
from itertools import count 
from scipy.sparse import csr
from collections import defaultdict


# csr_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
# where data, row_ind and col_ind satisfy the relationship:
# a[row_ind[k], col_ind[k]] = data[k]

def vectorize_dic(dic, label2index=None, hold_num=None):
    if (label2index == None):
        d = count(0)
        label2index = defaultdict(lambda: next(d)) 
        
    sample_num = len(list(dic.values())[0]) # num samples
    feat_num = len(list(dic.keys())) # num of features
    total_value_num = sample_num * feat_num # number of non-zeros
    
    col_ix = np.empty(total_value_num, dtype=int) 
    
    i = 0
    for k, lis in dic.items():     
        col_ix[i::feat_num] = [label2index[str(el) + str(k)] for el in lis]
        i += 1
        
    row_ix = np.repeat(np.arange(sample_num), feat_num)
    data = np.ones(total_value_num)
    
    if (hold_num == None):
        hold_num = len(label2index)
        
    left_data_index = np.where(col_ix < hold_num)

    return csr.csr_matrix(
            (data[left_data_index],(row_ix[left_data_index], col_ix[left_data_index])),
            shape=(sample_num, hold_num)), label2index



cols = ['user','item','rating','timestamp']
train = pd.read_csv('MovieLens/ua.base',delimiter = '\t',names = cols)
test = pd.read_csv('MovieLens/ua.test',delimiter = '\t',names = cols)

x_train,ix = vectorize_dic({'users':train['user'].values,
                            'items':train['item'].values})
x_test,ix = vectorize_dic({'users':test['user'].values,
                           'items':test['item'].values},ix,x_train.shape[1])
x_train = x_train.todense()
x_test = x_test.todense()

print(x_train.shape)

# label
y_train = train['rating'].values
y_test = test['rating'].values

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
batch_size = 64
x_train = torch.tensor(x_train).float().to(device)
x_test = torch.tensor(x_test).float().to(device)
y_train = torch.tensor(y_train).float().to(device)
y_test = torch.tensor(y_test).float().to(device)
train_dataset = torch.utils.data.TensorDataset(x_train,y_train)
train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0)
test_dataset = torch.utils.data.TensorDataset(x_test,y_test)
test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0)



samples,n = x_train.shape
k = 10
model = FM_Model(n,k).to(device)
criterion = torch.nn.MSELoss()
optimizer =torch.optim.SGD(model.parameters(),lr=0.0001,weight_decay=0.001)
epochs = 10


for epoch in range(epochs):
    model.train()
    for step,(batch_x,batch_y) in enumerate(train_loader):
        x = batch_x.view(-1,n)
        y = batch_y.view(-1,1)
        y_pred = model(x)
        loss = criterion(y_pred, y)
        if step % 100 == 1:
            print('Epoch:',epoch,'|step:',step)
            print(epoch,loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    eval_loss = []
    for step,(batch_x,batch_y) in enumerate(test_loader):
        x = batch_x.view(-1,n)
        y = batch_y.view(-1,1)
        y_pred = model(x)
        loss = criterion(y_pred, y)
        eval_loss.append(loss)
        if step % 100 == 1:
            print('Epoch:',epoch,'|step:',step)
            print(epoch,loss.item())
    print('Eval loss:',torch.mean(torch.tensor(eval_loss)))


