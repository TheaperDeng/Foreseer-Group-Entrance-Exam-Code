# !/usr/bin/python3
# coding:utf-8
# author:Junwei Deng

from torch_geometric.datasets import Planetoid
dataset = Planetoid(root='D:/ResearchApp/Foreseer/pygcn_otherloss/data', name='Cora') # change to your dir

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.nn as nn
import numpy as np
import model as Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model.Net().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=5e-4)
lambda_loss_1 = 0.25
# prepare the adjacency matrix
A = np.zeros((data.x.shape[0],data.x.shape[0]))
for i in range(data.edge_index.shape[1]):
    x = int(data.edge_index[0,i])
    y = int(data.edge_index[1,i])
    A[x,y] = 1
    A[y,x] = 1
D = np.zeros((data.x.shape[0],data.x.shape[0]))
for i in range(data.x.shape[0]):
    sumval = 0
    for j in range(data.x.shape[0]):
        sumval += A[i,j]
    D[i,i] = sumval
A = torch.from_numpy(A).type(torch.FloatTensor)
D = torch.from_numpy(D).type(torch.FloatTensor)
lossfun = Model.MyLoss()

model.train()
count = 0
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data) # 2708,7
    loss = lossfun(data, out, data.y, A, D, lambda_loss_1)
    loss.backward()
    optimizer.step()
    count+=1
    print("epoch: ", count)
    print("done!")

model.eval()
_, pred = model(data).max(dim=1)
correct = float (pred[data.test_mask.type(torch.bool)].eq(data.y[data.test_mask.type(torch.bool)]).sum().item())
acc = correct / data.test_mask.sum().item()
print('Accuracy: {:.4f}'.format(acc))