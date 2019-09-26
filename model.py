import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.nn as nn
import numpy as np

from torch_geometric.datasets import Planetoid
dataset = Planetoid(root='D:/ResearchApp/Foreseer/pygcn_otherloss/data', name='Cora')

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
    def forward(self, data, pred, truth, A, D, lambda_1 = 0.25):

        temp = F.softmax(pred) # softmax for loss
        loss_1 = (pred.t().mm(D-A)).mm(pred) # pred' * (D-A) * pred
        pred = F.log_softmax(pred, dim=1) # Multi-class Cross Engtropy
        loss_0 = F.nll_loss(pred[data.train_mask.type(torch.bool)], truth[data.train_mask.type(torch.bool)])
        
        
        print("L_0: ", loss_0)
        print("L_reg(with no regulation coefficient): ", torch.trace(loss_1)/((data.x.shape[0])**2))
        print("Totel Loss: ", loss_0 + lambda_1 * (torch.trace(loss_1)/((data.x.shape[0])**2)))
        return loss_0 + lambda_1 * (torch.trace(loss_1)/((data.x.shape[0])**2))
        # return loss_0

